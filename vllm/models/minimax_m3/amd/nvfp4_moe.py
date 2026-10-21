# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 routed experts for MiniMax-M3.1 checkpoints on ROCm.

The checkpoint's quantization_config is ``mxfp8`` with
``moe_quant_algo == "NVFP4"`` (``nvfp4-pack-quantized``, group 16): every
routed expert matrix ships as ``weight_packed`` (uint8, two E2M1 codes per
byte, even element in the low nibble), ``weight_scale`` (E4M3 per 16),
``weight_global_scale`` (fp32 scalar, dequant = code * scale /
weight_global_scale) and ``input_global_scale`` (unused, activations stay
dynamic).

Loading: custom weight loaders slice the TP shard (w1/w3 rows, w2 columns)
into ``w13_*`` / ``w2_*`` NVFP4 parameters. After loading, up to three
execution paths are prepared; per call the first that applies is used:

  1. HIP NVFP4 decode kernels (VLLM_M3_NVFP4_MOE_HIP=1) for
     T <= VLLM_M3_NVFP4_MOE_HIP_MAX_T (default 1024, max 1024).
  2. NVFP4 FlyDSL MoE (VLLM_M3_NVFP4_MOE_PREFILL=nv): nv2 for
     NV_MIN_T < T <= NV2_MAX_T, VLLM_M3_NVFP4_MOE_NV_LARGE (nv8w | xdeca |
     baseline) above. See ops/nvfp4_moe_pf.py.
  3. Baseline copy: the exact NVFP4 weight (global scale included) re-quantized
     to VLLM_M3_NVFP4_MOE_BASELINE_FMT (mxfp8 | ptpc_fp8 | mxfp4), all on the
     AITER MoE path (aiter.fused_moe). See ops/moe_baselines.py.

The baseline copy is built if any call can reach it, or always with
VLLM_M3_NVFP4_MOE_KEEP_MXFP8=1 (default).
"""

import functools
import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8FusedMoE
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)  # fmt: skip
_BASELINE_ALIASES = ("baseline", "mxfp8")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _env_choice(name: str, default: str, choices) -> str:
    v = _env(name, default)
    if v not in choices:
        raise ValueError(f"{name}={v!r}: expected one of {choices}")
    return v


def nvfp4_moe_hip_enabled() -> bool:
    return os.environ.get("VLLM_M3_NVFP4_MOE_HIP", "0") == "1"


def nvfp4_moe_hip_max_t() -> int:
    return min(int(os.environ.get("VLLM_M3_NVFP4_MOE_HIP_MAX_T", "1024")), 1024)


def nvfp4_moe_baseline_fmt() -> str:
    return _env_choice(
        "VLLM_M3_NVFP4_MOE_BASELINE_FMT", "mxfp8", ("mxfp8", "ptpc_fp8", "mxfp4")
    )


def nvfp4_moe_prefill() -> str:
    v = _env_choice(
        "VLLM_M3_NVFP4_MOE_PREFILL", "mxfp8", _BASELINE_ALIASES + ("nv", "hip")
    )
    return v if v in ("nv", "hip") else "baseline"


def nvfp4_moe_nv_large() -> str:
    v = _env_choice(
        "VLLM_M3_NVFP4_MOE_NV_LARGE", "nv8w", ("nv8w", "xdeca") + _BASELINE_ALIASES
    )
    return "baseline" if v in _BASELINE_ALIASES else v


def _qc_moe_quant_algo(qc) -> str | None:
    if isinstance(qc, dict) and qc.get("moe_quant_algo"):
        return str(qc["moe_quant_algo"])
    if qc is not None and getattr(qc, "moe_quant_algo", None):
        return str(qc.moe_quant_algo)
    return None


@functools.cache
def _config_json_moe_quant_algo(model: str, revision: str | None) -> str | None:
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    try:
        raw = get_hf_file_to_dict("config.json", model, revision)
    except Exception:  # noqa: BLE001
        logger.warning("MiniMax-M3: could not read config.json for moe_quant_algo")
        return None
    if not raw:
        return None
    for cfg in (raw, raw.get("text_config") or {}):
        algo = _qc_moe_quant_algo(cfg.get("quantization_config"))
        if algo is not None:
            return algo
    return None


def hf_moe_quant_algo(model_config) -> str | None:
    """``moe_quant_algo`` of the checkpoint quantization_config, if any.

    The VL wrapper builds the language model with ``hf_config`` replaced by
    ``text_config`` (no quantization_config), so fall back to config.json.
    """
    hf_config = model_config.hf_config
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        algo = _qc_moe_quant_algo(getattr(cfg, "quantization_config", None))
        if algo is not None:
            return algo
    return _config_json_moe_quant_algo(
        str(model_config.model), getattr(model_config, "revision", None)
    )


def dequant_nvfp4(packed, sc, gmul):
    """packed uint8 [e, N, K/2], sc E4M3 [e, N, K/16], gmul fp32 [e, N] -> fp32."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    v = lut[codes]
    e, N, K = v.shape
    s = sc.view(torch.float8_e4m3fn).float().view(e, N, K // 16, 1)
    return (v.view(e, N, K // 16, 16) * s).view(e, N, K) * gmul.view(e, N, 1)


def requant_mxfp8(v):
    """fp32 [e, N, K] -> (fp8 e4m3 [e, N, K], E8M0 uint8 [e, N, K/32]).

    The block exponent is the smallest one that keeps the block amax within
    448 (no clipping); values are rounded to nearest by the fp8 cast.
    """
    e, N, K = v.shape
    vb = v.view(e, N, K // 32, 32)
    amax = vb.abs().amax(dim=-1)
    ex = torch.ceil(torch.log2(amax / 448.0))
    ex = torch.where(amax > 0, ex, torch.full_like(ex, -127.0)).clamp(-127, 127)
    q = (vb / torch.exp2(ex).unsqueeze(-1)).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).view(e, N, K)
    return q, (ex + 127).to(torch.uint8)


def requant_baseline(fmt, w13p, w13s, w2p, w2s, g13, g2, step=8):
    """Exact NVFP4 (incl. 1 / global) -> fmt, raw (unshuffled) tensors."""
    from vllm.models.minimax_m3.amd.ops import moe_baselines as mb

    E, N13, H2 = w13p.shape
    I, H = N13 // 2, H2 * 2
    dev = w13p.device
    fp8 = torch.float8_e4m3fn
    if fmt == "mxfp8":
        q = (fp8, fp8, torch.uint8, torch.uint8)
        shp = ((E, N13, H), (E, H, I), (E, N13, H // 32), (E, H, I // 32))
        fn = requant_mxfp8
    elif fmt == "ptpc_fp8":
        q = (fp8, fp8, torch.float32, torch.float32)
        shp = ((E, N13, H), (E, H, I), (E, N13, 1), (E, H, 1))
        fn = mb.requant_ptpc_fp8
    else:
        q = (torch.uint8,) * 4
        shp = ((E, N13, H // 2), (E, H, I // 2), (E, N13, H // 32), (E, H, I // 32))
        fn = functools.partial(mb.requant_mxfp4, mode=mb.mxfp4_scale_mode())
    w13, w2, s13, s2 = (torch.empty(s, dtype=d, device=dev) for s, d in zip(shp, q))
    rows13 = g13.repeat_interleave(I, dim=1)  # [E, 2I]
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        v = dequant_nvfp4(w13p[e0:e1], w13s[e0:e1], rows13[e0:e1])
        w13[e0:e1], s13[e0:e1] = fn(v)
        gm = g2[e0:e1, None].expand(e1 - e0, H)
        v = dequant_nvfp4(w2p[e0:e1], w2s[e0:e1], gm)
        w2[e0:e1], s2[e0:e1] = fn(v)
        del v
    return w13, s13, w2, s2


def _param(data: torch.Tensor, loader) -> nn.Parameter:
    p = nn.Parameter(data, requires_grad=False)
    set_weight_attrs(p, {"weight_loader": loader})
    return p


def _max_num_batched_tokens() -> int | None:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    sched = getattr(cfg, "scheduler_config", None)
    return getattr(sched, "max_num_batched_tokens", None)


_NVFP4_PARAMS = (
    "w13_weight_packed",
    "w13_weight_scale",
    "w13_weight_global_scale",
    "w13_input_global_scale",
    "w2_weight_packed",
    "w2_weight_scale",
    "w2_weight_global_scale",
    "w2_input_global_scale",
)


class MiniMaxM3NvFp4MoEMethod(ModelOptMxFp8FusedMoE):
    """NVFP4 checkpoint experts: HIP decode, NVFP4 FlyDSL MoE and an AITER
    baseline copy (mxfp8 / ptpc_fp8 / mxfp4)."""

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(quant_config, moe_config)
        self.use_hip = nvfp4_moe_hip_enabled()
        self.hip_max_t = nvfp4_moe_hip_max_t()
        self.baseline_fmt = nvfp4_moe_baseline_fmt()
        self.prefill = nvfp4_moe_prefill()
        self.nv_large = nvfp4_moe_nv_large()
        self.nv2_max_t = int(os.environ.get("VLLM_M3_NVFP4_MOE_NV2_MAX_T", "14336"))
        self.nv_min_t = int(os.environ.get("VLLM_M3_NVFP4_MOE_NV_MIN_T", "0"))
        self.keep_baseline = os.environ.get("VLLM_M3_NVFP4_MOE_KEEP_MXFP8", "1") == "1"
        self.max_tokens = _max_num_batched_tokens()
        self.hip_weights = None
        self.nv_weights = None
        self.base_weights = None  # ptpc_fp8 / mxfp4 BaselineWeights
        self.has_baseline = False
        backend = getattr(self.moe, "moe_backend", "auto")
        if self.baseline_fmt != "mxfp8" and backend not in ("auto", "aiter"):
            raise ValueError(
                f"MiniMax-M3 NVFP4 baseline {self.baseline_fmt} runs on the AITER "
                f"MoE path only; got --moe-backend {backend}"
            )

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        # The HIP / nv / ptpc / mxfp4 paths bypass the modular kernel, so the
        # runner must always own the shared expert.
        return False

    # ------------------------------------------------------------ planning
    def _plan(self):
        """(nv modes to build, whether the baseline copy is needed)."""
        if self.prefill == "hip":
            # HIP kernel for every T (chunked above hip_max_t): one weight copy.
            if not self.use_hip:
                raise ValueError("VLLM_M3_NVFP4_MOE_PREFILL=hip needs VLLM_M3_NVFP4_MOE_HIP=1")
            return [], self.keep_baseline
        big = self.max_tokens is None or self.max_tokens > self.nv2_max_t
        modes = []
        need_base = self.prefill != "nv"
        if self.prefill == "nv":
            modes.append("nv2")
            if big:
                if self.nv_large == "baseline":
                    need_base = True
                else:
                    modes.append(self.nv_large)
            if self.nv_min_t > (self.hip_max_t if self.use_hip else 0):
                need_base = True
        return modes, need_base or self.keep_baseline

    # -------------------------------------------------------------- loading
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert layer.intermediate_size_per_partition == intermediate_size_per_partition
        assert layer.hidden_size == hidden_size
        if layer.moe_config.moe_parallel_config.use_ep:
            raise NotImplementedError("MiniMax-M3 NVFP4 experts: EP is unsupported")
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        if H % 128 or I % 128:
            raise ValueError(f"MiniMax-M3 NVFP4 experts need H, I % 128 == 0: {H} {I}")
        layer.orig_dtype = params_dtype
        layer.num_experts = num_experts
        u8 = torch.uint8
        fp8 = torch.float8_e4m3fn
        f32 = torch.float32
        mk = self._make_loader
        specs = {
            "w13_weight_packed": ((E, 2 * I, H // 2), u8, "rows"),
            "w13_weight_scale": ((E, 2 * I, H // 16), fp8, "rows"),
            "w13_weight_global_scale": ((E, 2), f32, "scalar"),
            "w13_input_global_scale": ((E, 2), f32, "scalar"),
            "w2_weight_packed": ((E, H, I // 2), u8, "cols"),
            "w2_weight_scale": ((E, H, I // 16), fp8, "cols"),
            "w2_weight_global_scale": ((E,), f32, "scalar"),
            "w2_input_global_scale": ((E,), f32, "scalar"),
        }
        for name, (shape, dtype, kind) in specs.items():
            fill = torch.ones if dtype == f32 else torch.empty
            layer.register_parameter(
                name, _param(fill(shape, dtype=dtype), mk(layer, kind))
            )

    @staticmethod
    def _make_loader(layer, kind: str):
        def loader(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str = "",
            shard_id: str = "",
            expert_id: int = 0,
            return_success: bool = False,
        ):
            e = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if e == -1:
                return False if return_success else None
            r = layer.moe_config.tp_rank
            dst = param.data[e]
            w = loaded_weight
            if kind == "scalar":
                val = w.reshape(()).to(dst.dtype)
                if shard_id == "w2":
                    dst.copy_(val)
                else:
                    dst[0 if shard_id == "w1" else 1].copy_(val)
            elif kind == "rows":
                assert shard_id in ("w1", "w3"), shard_id
                n = dst.shape[0] // 2
                off = 0 if shard_id == "w1" else n
                dst[off : off + n].copy_(w.narrow(0, r * n, n).view(dst.dtype))
            else:
                assert shard_id == "w2", shard_id
                n = dst.shape[1]
                dst.copy_(w.narrow(1, r * n, n).view(dst.dtype))
            return True if return_success else None

        return loader

    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        w13p, w13s = layer.w13_weight_packed.data, layer.w13_weight_scale.data
        w2p, w2s = layer.w2_weight_packed.data, layer.w2_weight_scale.data
        g13 = 1.0 / layer.w13_weight_global_scale.data.float()  # [E, 2]
        g2 = 1.0 / layer.w2_weight_global_scale.data.float()  # [E]
        modes, need_base = self._plan()

        if self.use_hip:
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_dec import (
                NvFp4DecodeWeights,
            )

            self.hip_weights = NvFp4DecodeWeights(w13p, w13s, w2p, w2s, g13, g2)

        if modes:
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_pf import (
                NvPrefillWeights,
                install,
            )

            # One fp32 multiplier per expert for gate and up (load-time check).
            if not bool((g13[:, 0] == g13[:, 1]).all()):
                raise ValueError(
                    "MiniMax-M3 NVFP4 prefill (nv) needs w1/w3 global scales "
                    "equal per expert; use VLLM_M3_NVFP4_MOE_PREFILL=mxfp8"
                )
            install()
            self.nv_weights = NvPrefillWeights(
                w13p, w13s, w2p, w2s, g13[:, 0].contiguous(), g2, modes
            )

        base = None
        if need_base:
            base = requant_baseline(self.baseline_fmt, w13p, w13s, w2p, w2s, g13, g2)

        for name in _NVFP4_PARAMS:
            delattr(layer, name)
        del w13p, w13s, w2p, w2s

        if base is not None:
            w13, s13, w2, s2 = base
            del base
            # Set before super(): ModelOpt builds its kernel from
            # get_fused_moe_quant_config(), which checks has_baseline.
            self.has_baseline = True
            if self.baseline_fmt == "mxfp8":
                for name, t in (
                    ("w13_weight", w13),
                    ("w2_weight", w2),
                    ("w13_weight_scale", s13),
                    ("w2_weight_scale", s2),
                ):
                    layer.register_parameter(name, nn.Parameter(t, requires_grad=False))
                del w13, s13, w2, s2
                super().process_weights_after_loading(layer)
            else:
                from vllm.models.minimax_m3.amd.ops.moe_baselines import (
                    BaselineWeights,
                )

                self.base_weights = BaselineWeights(self.baseline_fmt, w13, s13, w2, s2)
                del w13, s13, w2, s2
        layer._already_called_process_weights_after_loading = True

        if self.baseline_fmt != "mxfp8" and self.has_baseline:
            from vllm.models.minimax_m3.amd.ops.moe_baselines import (
                aiter_backend_line,
            )

            logger.info_once(aiter_backend_line(self.baseline_fmt))
        logger.info_once(
            "MiniMax-M3 NVFP4 experts: baseline %s copy %s; HIP decode %s "
            "(T <= %d); prefill %s (nv2 %d < T <= %d, above: %s; built %s); "
            "max_num_batched_tokens %s",
            self.baseline_fmt,
            "built" if self.has_baseline else "not built",
            "on" if self.use_hip else "off",
            self.hip_max_t,
            self.prefill,
            self.nv_min_t,
            self.nv2_max_t,
            self.nv_large,
            ",".join(modes) or "-",
            self.max_tokens,
        )

    def get_fused_moe_quant_config(self, layer):
        if self.baseline_fmt == "mxfp8" and self.has_baseline:
            return super().get_fused_moe_quant_config(layer)
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )

        # Placeholder: these paths call aiter / HIP directly (no modular kernel).
        return FUSED_MOE_UNQUANTIZED_CONFIG

    # -------------------------------------------------------------- forward
    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input,
    ) -> torch.Tensor:
        T = x.shape[0]
        if self.hip_weights is not None and T <= self.hip_max_t:
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_dec import (
                nvfp4_moe_decode,
            )

            return nvfp4_moe_decode(self.hip_weights, x, topk_ids, topk_weights)
        if self.hip_weights is not None and self.prefill == "hip":
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_dec import (
                nvfp4_moe_decode,
            )

            m = self.hip_max_t
            return torch.cat(
                [
                    nvfp4_moe_decode(
                        self.hip_weights,
                        x[i : i + m],
                        topk_ids[i : i + m],
                        topk_weights[i : i + m],
                    )
                    for i in range(0, T, m)
                ]
            )
        nv = self.nv_weights
        if nv is not None:
            mode = None
            if T <= self.nv2_max_t:
                if T > self.nv_min_t or not self.has_baseline:
                    mode = "nv2"
            elif self.nv_large in nv.modes:
                mode = self.nv_large
            if mode is not None:
                from vllm.models.minimax_m3.amd.ops.nvfp4_moe_pf import nv_moe

                return nv_moe(
                    nv, mode, x, topk_weights, topk_ids, getattr(layer, "swiglu_limit", None)
                )
        if not self.has_baseline:
            raise RuntimeError(
                f"MiniMax-M3 NVFP4: no path for T={T} (baseline copy not built; "
                "set VLLM_M3_NVFP4_MOE_KEEP_MXFP8=1)"
            )
        if self.base_weights is not None:
            from vllm.models.minimax_m3.amd.ops.moe_baselines import baseline_moe

            return baseline_moe(
                self.base_weights, x, topk_weights, topk_ids, getattr(layer, "swiglu_limit", None)
            )
        return super().apply(
            layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
