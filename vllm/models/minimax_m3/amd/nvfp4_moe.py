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
into ``w13_*`` / ``w2_*`` NVFP4 parameters.

After loading, the NVFP4 weights are always re-quantized to MXFP8 (E8M0 per
32, the per-expert global multiplier folded in) and handed to the stock
ModelOpt MXFP8 MoE path (AITER MXFP8 MoE kernels). That is the reference
fallback and the prefill path ("xdeca-folded").

VLLM_M3_NVFP4_MOE_HIP=1 additionally keeps the NVFP4 weights in the HIP F16
decode-kernel layout and runs batches with T <= VLLM_M3_NVFP4_MOE_HIP_MAX_T
(default 1024, max 1024) on them.
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


def nvfp4_moe_hip_enabled() -> bool:
    return os.environ.get("VLLM_M3_NVFP4_MOE_HIP", "0") == "1"


def nvfp4_moe_hip_max_t() -> int:
    return min(int(os.environ.get("VLLM_M3_NVFP4_MOE_HIP_MAX_T", "1024")), 1024)


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


def _param(data: torch.Tensor, loader) -> nn.Parameter:
    p = nn.Parameter(data, requires_grad=False)
    set_weight_attrs(p, {"weight_loader": loader})
    return p


class MiniMaxM3NvFp4MoEMethod(ModelOptMxFp8FusedMoE):
    """NVFP4 checkpoint experts; MXFP8 (AITER) execution plus HIP decode."""

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(quant_config, moe_config)
        self.use_hip = nvfp4_moe_hip_enabled()
        self.hip_max_t = nvfp4_moe_hip_max_t()
        self.hip_weights = None

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        # The HIP path bypasses the modular kernel, so the runner must always
        # own the shared expert.
        return False

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
        E, N13, H2 = w13p.shape
        I, H = N13 // 2, H2 * 2
        dev = w13p.device

        if self.use_hip:
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_dec import (
                NvFp4DecodeWeights,
            )

            self.hip_weights = NvFp4DecodeWeights(w13p, w13s, w2p, w2s, g13, g2)

        w13 = torch.empty(E, N13, H, dtype=torch.float8_e4m3fn, device=dev)
        s13 = torch.empty(E, N13, H // 32, dtype=torch.uint8, device=dev)
        w2 = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=dev)
        s2 = torch.empty(E, H, I // 32, dtype=torch.uint8, device=dev)
        rows13 = g13.repeat_interleave(I, dim=1)  # [E, 2I]
        step = 8
        for e0 in range(0, E, step):
            e1 = min(E, e0 + step)
            v = dequant_nvfp4(w13p[e0:e1], w13s[e0:e1], rows13[e0:e1])
            w13[e0:e1], s13[e0:e1] = requant_mxfp8(v)
            gm = g2[e0:e1, None].expand(e1 - e0, H)
            v = dequant_nvfp4(w2p[e0:e1], w2s[e0:e1], gm)
            w2[e0:e1], s2[e0:e1] = requant_mxfp8(v)
            del v

        for name in (
            "w13_weight_packed",
            "w13_weight_scale",
            "w13_weight_global_scale",
            "w13_input_global_scale",
            "w2_weight_packed",
            "w2_weight_scale",
            "w2_weight_global_scale",
            "w2_input_global_scale",
        ):
            delattr(layer, name)
        del w13p, w13s, w2p, w2s
        for name, t in (
            ("w13_weight", w13),
            ("w2_weight", w2),
            ("w13_weight_scale", s13),
            ("w2_weight_scale", s2),
        ):
            layer.register_parameter(name, nn.Parameter(t, requires_grad=False))
        super().process_weights_after_loading(layer)
        logger.info_once(
            "MiniMax-M3 NVFP4 experts: MXFP8 re-quantized copy for the AITER "
            "MXFP8 MoE path; HIP NVFP4 decode %s (T <= %d)",
            "on" if self.use_hip else "off",
            self.hip_max_t,
        )

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
        if self.hip_weights is not None and x.shape[0] <= self.hip_max_t:
            from vllm.models.minimax_m3.amd.ops.nvfp4_moe_dec import (
                nvfp4_moe_decode,
            )

            return nvfp4_moe_decode(self.hip_weights, x, topk_ids, topk_weights)
        return super().apply(
            layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
