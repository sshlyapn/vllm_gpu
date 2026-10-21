# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecodeBenchConnector: A KV Connector for decode instance performance testing.

This connector emulates a prefill-decode disaggregated setting by filling
the KV cache with dummy values, allowing measurement of decoder performance
under larger input sequence lengths (ISL) in resource-limited environments.

Usage:
    To use this connector for benchmarking, configure it in the kv_transfer_config:

Example:
        vllm serve <model> --kv-transfer-config '{
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0
            }
        }'

    Then run your benchmark with desired input/output lengths:
        vllm bench serve --base-url http://127.0.0.1:8000 --model <model> \\
            --dataset-name random --random-input-len 40000 \\
            --random-output-len 100 --max-concurrency 10

    Configuration options (via kv_connector_extra_config):
        - fill_mean (float): Mean value for random normal fill (default: 0.015)
        - fill_std (float): Standard deviation for random fill (default: 0.0)
          Set to 0 for constant values, >0 for random sampling
        - startup_fill (bool): Fill the whole KV cache once at startup instead
          of filling each new request's blocks in the step that schedules it
          (default: False). Circular-buffer caches are still zeroed per
          request, and attention caches whose new blocks the engine zeroes
          before use (hybrid Mamba or mixed-precision KV caches) are still
          filled per request. The fill is not redone after sleep mode
          discards the KV cache. Not supported with fill_mode "m3_pool".
        - fill_mode (str): "const" (default, the behavior above) or "m3_pool"
          (MiniMax-M3 study overlay, see _M3Pools): every filled block gets a
          copy of a pre-encoded template block. Index-key templates are exact
          in fp8 and NVFP4, so a request has the same top-k winner blocks in
          both formats. Compact NVFP4 pages
          (MINIMAX_M3_NVFP4_COMPACT_PAGES=1) get compact templates.
        - fill_index_winners (int): winner blocks per request (default 15).
        - fill_seed (int): template seed (default 1234).

    Study-only env (m3_pool): M3FILL_TIMING=1 logs the synced fill time per
    step; M3FILL_DUMP_DIR=<dir> writes fill records and patches the AITER
    indexer to dump decode top-k (needs --enforce-eager to see decode steps).

    The dummy KV cache is only meant for performance measurement; outputs are
    not meaningful for accuracy evaluation.

"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import (
    resolve_dcp_kv_block_size,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    KVCacheSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _get_fp8_dtype(spec: KVCacheSpec, cache_dtype: str) -> torch.dtype | None:
    """The fp8 dtype of a KV cache that stores plain fp8 values as uint8.

    Returns None for other layouts, including packed ones that embed scales
    or other metadata in the uint8 page.
    """
    if (
        not isinstance(spec, AttentionSpec)
        or spec.dtype != torch.uint8
        or spec.kv_quant_mode != KVQuantMode.FP8_PER_TENSOR
        or spec.state_content_bytes is not None
    ):
        return None
    cache_dtype = getattr(spec, "cache_dtype_str", None) or cache_dtype
    if cache_dtype in ("fp8", "fp8_e4m3"):
        return current_platform.fp8_dtype()
    if cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    return None


def _get_startup_fill_group_ids(kv_cache_config: "KVCacheConfig") -> set[int]:
    """KV cache groups that startup fill covers.

    Circular buffers are zero-filled per request. Attention groups whose new
    blocks the engine zeroes before use are filled per request, since the
    zeroing would wipe the startup fill.
    """
    group_ids = set()
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if any(isinstance(s, CircularBufferSpec) for s in iter_layer_specs(spec)):
            continue
        if kv_cache_config.needs_kv_cache_zeroing and isinstance(spec, AttentionSpec):
            continue
        group_ids.add(group_idx)
    return group_ids


# ---------------------------------------------------------------------------
# m3_pool fill: template encoders (pure torch, no model imports)
# ---------------------------------------------------------------------------

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
# Bytes per token of a compact NVFP4 page (D=128): D/2 codes + D/16 scales.
_NVFP4_COMPACT_DIM = 72


def m3_spec(gen, shape, scales):
    """Random e2m1 codes with one +-6 per 16-group of the last dim.

    Returns (code uint8 0..7, neg bool, sc float32 [..., D // 16]). The value
    of an element is (-1)^neg * e2m1[code] * sc; with sc a power of two (or
    1.25 / 1.5 times one) every value is exact in e4m3 and NVFP4 (scale =
    amax / 6 = sc exactly).
    """
    code = torch.randint(0, 8, shape, generator=gen, dtype=torch.uint8)
    neg = torch.randint(0, 2, shape, generator=gen, dtype=torch.bool)
    grp = code.view(*shape[:-1], shape[-1] // 16, 16)
    pos = torch.randint(0, 16, grp.shape[:-1] + (1,), generator=gen)
    grp.scatter_(-1, pos, 7)
    neg &= code != 0
    sidx = torch.randint(0, len(scales), grp.shape[:-1], generator=gen)
    sc = torch.tensor(scales, dtype=torch.float32)[sidx]
    return code, neg, sc


def m3_values(code, neg, sc):
    mag = _E2M1[code.long()]
    val = torch.where(neg, -mag, mag) * sc.repeat_interleave(16, dim=-1)
    return val + 0.0  # no negative zeros


def _e4m3_bytes(x):
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def enc_idx_fp8(code, neg, sc):
    """[T, bs, D] -> [T, bs * D] bytes, row-major token then dim."""
    t = code.shape[0]
    return _e4m3_bytes(m3_values(code, neg, sc)).reshape(t, -1)


def enc_idx_nvfp4(code, neg, sc, compact=False):
    """AITER msa nvfp4 index page: e2m1 nibbles (low = even dim, code | sign
    << 3) at tok * D / 2, then one e4m3 scale per 16 dims at bs * D / 2 + tok *
    D / 16; zero padded to bs * D bytes (compact: no pad, bs * 9 * D / 16)."""
    t, bs, d = code.shape
    nib = code | (neg.to(torch.uint8) << 3)
    nib = nib.view(t, bs, d // 2, 2)
    packed = (nib[..., 0] | (nib[..., 1] << 4)).reshape(t, -1)
    scales = _e4m3_bytes(sc).reshape(t, -1)
    if compact:
        return torch.cat([packed, scales], dim=1)
    pad = torch.zeros(t, bs * d - packed.shape[1] - scales.shape[1], dtype=torch.uint8)
    return torch.cat([packed, scales, pad], dim=1)


def enc_kv_fp8(kspec, vspec):
    """Packed AITER asm sparse-PA block: K side [page, H, D/16, 16 tok, 16],
    then V side [page, H, D, 16 tok]. Specs are [T, bs, H, D]."""
    kv = m3_values(*kspec)
    vv = m3_values(*vspec)
    t, bs, h, d = kv.shape
    p = bs // 16
    kb = _e4m3_bytes(kv).view(t, p, 16, h, d // 16, 16).permute(0, 1, 3, 4, 2, 5)
    vb = _e4m3_bytes(vv).view(t, p, 16, h, d).permute(0, 1, 3, 4, 2)
    return torch.cat([kb.reshape(t, -1), vb.reshape(t, -1)], dim=1)


def _nvfp4_kv_codes(code, neg):
    """[T, P, 16 tok, H, 128] -> [T, P, H, 1024] sparse_pa NVFP4 code bytes.

    Byte j of word g of dim-block s at s * 128 + tok * 8 + g * 4 + j holds
    |A_j| | sign(A_j) << 7 | |B_j| << 4 | sign(B_{j-1 mod 4}) << 3 with A =
    elements s * 16 + g * 8 + 0..3 and B the next 4.
    """
    t, p, _, h, _ = code.shape
    c = code.view(t, p, 16, h, 8, 2, 2, 4)
    n = neg.view(t, p, 16, h, 8, 2, 2, 4).to(torch.uint8)
    a, b = c[..., 0, :], c[..., 1, :]
    na, nb = n[..., 0, :], n[..., 1, :]
    byte = a | (na << 7) | (b << 4) | (torch.roll(nb, 1, dims=-1) << 3)
    # [T, P, tok, H, s, g, j] -> [T, P, H, s, tok, g, j]
    return byte.permute(0, 1, 3, 4, 2, 5, 6).reshape(t, p, h, 1024)


def _nvfp4_kv_parts(kspec, vspec):
    """[T, P, H, 1024] K codes, [T, P, H, 256] K/V scales (per token [s % 4]
    [K_s, K_s+4, V_s, V_s+4]), [T, P, H, 1024] V codes."""
    kc, kn, ks = kspec
    vc, vn, vs = vspec
    t, bs, h, d = kc.shape
    assert d == 128
    p = bs // 16
    kcode = _nvfp4_kv_codes(kc.view(t, p, 16, h, d), kn.view(t, p, 16, h, d))
    vcode = _nvfp4_kv_codes(vc.view(t, p, 16, h, d), vn.view(t, p, 16, h, d))

    def sc4(s):  # [T, bs, H, 8] -> [T, P, H, tok, w=4, hi=2]
        s = _e4m3_bytes(s).view(t, p, 16, h, 2, 4).permute(0, 1, 3, 2, 5, 4)
        return s

    scales = torch.cat([sc4(ks), sc4(vs)], dim=-1).reshape(t, p, h, 256)
    return kcode, scales, vcode


def enc_kv_nvfp4_compact(kspec, vspec):
    """Compact NVFP4 sparse KV (MINIMAX_M3_NVFP4_COMPACT_PAGES=1, one head):
    per 16-token page one 2304 B chunk [K codes 1024 | K/V scales 256 | V
    codes 1024]."""
    kcode, scales, vcode = _nvfp4_kv_parts(kspec, vspec)
    t, p, h, _ = kcode.shape
    assert h == 1
    return torch.cat([kcode, scales, vcode], dim=-1).reshape(t, -1)


def enc_kv_nvfp4(kspec, vspec):
    """sparse_pa NVFP4 sparse KV in the packed fp8 block: per (page, head)
    2048 B K chunk = K codes 1024 + scales 256 + 768 zero; V chunk = V codes
    1024 + 1024 zero."""
    kcode, scales, vcode = _nvfp4_kv_parts(kspec, vspec)
    t, p, h, _ = kcode.shape
    kz = torch.zeros(t, p, h, 2048 - 1024 - 256, dtype=torch.uint8)
    vz = torch.zeros(t, p, h, 1024, dtype=torch.uint8)
    kside = torch.cat([kcode, scales, kz], dim=-1).reshape(t, -1)
    vside = torch.cat([vcode, vz], dim=-1).reshape(t, -1)
    return torch.cat([kside, vside], dim=1)


def m3_index_winner_positions(total_blocks: int, num_winners: int) -> list[int]:
    """Logical block positions that get winner templates: spread over
    [0, total_blocks - 2]; the last filled block (local at decode) is never
    one."""
    n = total_blocks - 1
    if n <= num_winners:
        return list(range(max(n, 0)))
    return [int((i + 0.5) * n / num_winners) for i in range(num_winners)]


class _M3Pools:
    """Pre-encoded template blocks for the MiniMax-M3 caches.

    Index keys: winner template w holds, for direction u_i (shared by all
    winners), tokens 2i = +s_w * u_i and 2i + 1 = -s_w * u_i, so its block max
    score is s_w * max_i |q . u_i| > 0 for any q. Winner scales are 64/48/40/32
    and losers use independent directions at scale 2^-6, so every winner beats
    every loser by >= ~500x whatever q is. All values are e2m1 * scale with
    exact e4m3 products, i.e. the fp8 bytes and the NVFP4 decode feed the same
    e4m3 operands to the score MFMA.

    Sparse KV: random e2m1 * {1/8, 1/4}, exact in fp8 and NVFP4.
    """

    WIN_SCALES = (64.0, 48.0, 40.0, 32.0)
    LOSE_SCALE = 2.0**-6
    NUM_LOSERS = 32
    NUM_KV = 64

    def __init__(self, seed: int, num_winners: int, idx_fmt: str, kv_fmt: str):
        self.seed = seed
        self.num_winners = num_winners
        self.idx_fmt = idx_fmt
        self.kv_fmt = kv_fmt
        self.pools: dict[tuple, torch.Tensor] = {}

    def index_specs(self, bs: int, d: int = 128):
        gen = torch.Generator().manual_seed(self.seed)
        uc, un, _ = m3_spec(gen, (bs // 2, d), (1.0,))
        nw = self.num_winners
        code = torch.empty((nw, bs, d), dtype=torch.uint8)
        neg = torch.empty((nw, bs, d), dtype=torch.bool)
        code[:, 0::2] = uc
        code[:, 1::2] = uc
        neg[:, 0::2] = un
        neg[:, 1::2] = ~un & (uc != 0)
        ws = torch.tensor(
            [self.WIN_SCALES[i % len(self.WIN_SCALES)] for i in range(nw)]
        )
        sc = ws.view(nw, 1, 1).expand(nw, bs, d // 16).contiguous()
        lc, ln, ls = m3_spec(gen, (self.NUM_LOSERS, bs, d), (self.LOSE_SCALE,))
        return (
            torch.cat([code, lc]),
            torch.cat([neg, ln]),
            torch.cat([sc, ls]),
        )

    def kv_specs(self, bs: int, h: int, d: int = 128):
        gen = torch.Generator().manual_seed(self.seed + 1)
        k = m3_spec(gen, (self.NUM_KV, bs, h, d), (0.125, 0.25))
        v = m3_spec(gen, (self.NUM_KV, bs, h, d), (0.125, 0.25))
        return k, v

    def get(self, kind: tuple, device) -> torch.Tensor:
        pool = self.pools.get(kind)
        if pool is None:
            compact = kind[-1] == "compact"
            if kind[0] == "idx":
                spec = self.index_specs(kind[1])
                if compact:
                    assert self.idx_fmt == "nvfp4", "compact index needs nvfp4"
                    pool = enc_idx_nvfp4(*spec, compact=True)
                else:
                    enc = enc_idx_nvfp4 if self.idx_fmt == "nvfp4" else enc_idx_fp8
                    pool = enc(*spec)
            else:
                k, v = self.kv_specs(kind[1], kind[2])
                if compact:
                    assert self.kv_fmt == "nvfp4", "compact KV needs nvfp4"
                    pool = enc_kv_nvfp4_compact(k, v)
                else:
                    enc = enc_kv_nvfp4 if self.kv_fmt == "nvfp4" else enc_kv_fp8
                    pool = enc(k, v)
            # int64 rows: index_copy_ moves 8 B per element instead of 1 B.
            pool = pool.contiguous().view(torch.int64).to(device)
            self.pools[kind] = pool
        return pool

    def template_ids(self, kind: tuple, first: int, n: int) -> list[int]:
        if kind[0] == "kv":
            return [(p * 40503) % self.NUM_KV for p in range(first, first + n)]
        nw = self.num_winners
        tids = [nw + (p * 7919) % self.NUM_LOSERS for p in range(first, first + n)]
        for w, p in enumerate(m3_index_winner_positions(first + n, nw)):
            if p >= first:
                tids[p - first] = w
        return tids


def _dist_rank() -> int:
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _install_topk_dump(dump_dir: str, layers: set[int], max_rec: int) -> None:
    """Study-only: wrap MiniMaxM3IndexerAiterImpl.forward to append the decode
    top-k of the chosen layers to <dump_dir>/topk_r<rank>_<pid>.jsonl."""
    from vllm.forward_context import get_forward_context
    from vllm.models.minimax_m3.amd import indexer_aiter as ia

    cls = ia.MiniMaxM3IndexerAiterImpl
    if getattr(cls, "_m3fill_dump", False):
        return
    orig = cls.forward
    state: dict[str, Any] = {"n": 0, "f": None}

    def forward(self, index_query, **kwargs):
        out = orig(self, index_query, **kwargs)
        dec = out[0]
        if dec is None or state["n"] >= max_rec:
            return out
        m = re.search(r"layers\.(\d+)\.", self.index_cache.prefix)
        if m is None or int(m.group(1)) not in layers:
            return out
        md = get_forward_context().attn_metadata[self.index_cache.prefix]
        d = md.decode
        if state["f"] is None:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(
                dump_dir, f"topk_r{_dist_rank()}_{os.getpid()}.jsonl"
            )
            state["f"] = open(path, "a")
        rec = {
            "layer": int(m.group(1)),
            "qlen": int(d.decode_query_len),
            "seq_lens": d.seq_lens[: dec.shape[1]].tolist(),
            "topk": dec.tolist(),
        }
        state["f"].write(json.dumps(rec) + "\n")
        state["f"].flush()
        state["n"] += 1
        return out

    cls.forward = forward
    cls._m3fill_dump = True
    logger.warning("DecodeBenchConnector m3_pool: top-k dump hook -> %s", dump_dir)


@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Metadata for DecodeBenchConnector.

    Contains information about which requests need their KV cache filled
    with dummy values for benchmarking purposes.
    """

    # request_id -> (block_ids_per_group, num_tokens_to_fill)
    # block_ids_per_group is a tuple of lists, one per KV cache group
    # One group: ([1, 2, 3],)
    # Multiple groups: ([1, 2], [5, 6])
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]
    # request_id -> logical index of the first filled block, per group
    first_blocks: dict[str, tuple[int, ...]] = field(default_factory=dict)


class DecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """A KV Connector for decode instance performance testing.

    This connector fills the KV cache with dummy values to emulate a
    prefill-decode disaggregated setting, enabling performance testing of the
    decoder with larger input sequence lengths.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self.connector_scheduler: DecodeBenchConnectorScheduler | None = None
        self.connector_worker: DecodeBenchConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(
                vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(
                vllm_config, kv_cache_config
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, DecodeBenchConnectorMetadata)
        self.connector_worker.start_fill_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # All operations are synchronous, so nothing to wait for
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # This connector doesn't save KV cache (benchmarking only)
        pass

    def wait_for_save(self):
        # This connector doesn't save KV cache (benchmarking only)
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # HMA-enabled path: same cleanup as the single-group variant since
        # this connector owns no external state per block.
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class DecodeBenchConnectorScheduler:
    """Scheduler-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.startup_fill = kv_transfer_config.get_from_extra_config(
            "startup_fill", False
        )
        self._startup_fill_group_ids = (
            _get_startup_fill_group_ids(kv_cache_config) if self.startup_fill else set()
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.group_block_sizes = tuple(
            resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_world_size)
            for group in self.kv_cache_groups
        )

        # Track which requests have already been filled
        self._filled_requests: set[str] = set()

        # Track pending fills for the current scheduler step
        # request_id -> (block_ids_per_group, num_tokens_to_fill)
        # Note: _pending_fills doesn't need explicit cleanup - it's cleared
        # after build_connector_meta() is called in the same scheduler step
        self._pending_fills: dict[str, tuple[tuple[list[int], ...], int]] = {}
        self._pending_first_blocks: dict[str, tuple[int, ...]] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """For new requests, return the number of tokens that should be filled
        with dummy KV cache values.

        Returns:
            (num_tokens_to_fill, is_async)
            - num_tokens_to_fill: number of uncomputed tokens minus 1
                (we fill everything except the last token for decode)
            - is_async: False (synchronous filling)

        """
        req_id = request.request_id

        # Only fill once per request on first scheduling
        if req_id in self._filled_requests:
            return 0, False

        # Calculate how many tokens we need to fill
        # Fill all uncomputed tokens except the last one (which will be decoded)
        # This simulates having processed a long prefill
        num_uncomputed_tokens = request.num_tokens - num_computed_tokens
        num_tokens_to_fill = max(0, num_uncomputed_tokens - 1)

        if num_tokens_to_fill == 0:
            return 0, False

        # Return False for synchronous operation - the fill is fast enough
        # that async overhead isn't worth it
        return num_tokens_to_fill, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Called after blocks are allocated. Store the block IDs so we can
        fill them with dummy values.

        Supports both single- and multi-group KV cache configurations.
        """
        req_id = request.request_id

        if num_external_tokens == 0:
            return

        total_computed_tokens = request.num_tokens - 1
        num_local_computed_tokens = total_computed_tokens - num_external_tokens
        block_ids_per_group_list: list[list[int]] = []
        first_blocks: list[int] = []
        for group_idx, (
            group,
            group_blocks,
            group_block_size,
        ) in enumerate(
            zip(
                self.kv_cache_groups,
                blocks.blocks,
                self.group_block_sizes,
                strict=True,
            )
        ):
            if group_idx in self._startup_fill_group_ids:
                block_ids_per_group_list.append([])
                first_blocks.append(0)
                continue
            is_circular_buffer = all(
                isinstance(spec, CircularBufferSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
            if not is_circular_buffer:
                num_computed_blocks = cdiv(total_computed_tokens, group_block_size)
                external_block_start = num_local_computed_tokens // group_block_size
                assert (
                    0
                    <= external_block_start
                    <= num_computed_blocks
                    <= len(group_blocks)
                ), (
                    "DecodeBenchConnector block range exceeds allocated blocks: "
                    f"request={req_id}, group={group_idx}, "
                    f"range=[{external_block_start}, {num_computed_blocks}), "
                    f"allocated={len(group_blocks)}"
                )
                selected_blocks = group_blocks[external_block_start:num_computed_blocks]
            else:
                external_block_start = 0
                selected_blocks = group_blocks

            block_ids = [
                block.block_id for block in selected_blocks if not block.is_null
            ]
            if not block_ids:
                logger.warning(
                    "DecodeBenchConnector: No blocks selected for KV cache group "
                    "%d with %d external tokens for request %s",
                    group_idx,
                    num_external_tokens,
                    req_id,
                )
            block_ids_per_group_list.append(block_ids)
            first_blocks.append(external_block_start)
        block_ids_per_group = tuple(block_ids_per_group_list)
        self._filled_requests.add(req_id)
        if self.startup_fill and not any(block_ids_per_group):
            return

        # Store the blocks to fill for all group. _pending_fills doesn't need cleanup
        # as it's cleared after build_connector_meta
        self._pending_fills[req_id] = (
            block_ids_per_group,
            num_external_tokens,
        )
        self._pending_first_blocks[req_id] = tuple(first_blocks)

        block_counts = tuple(len(group) for group in block_ids_per_group)
        logger.debug(
            "DecodeBenchConnector: Selected %d total blocks across %d KV cache "
            "groups for request %s (per-group counts: %s)",
            sum(block_counts),
            len(blocks.blocks),
            req_id,
            ", ".join(map(str, block_counts)),
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """Build metadata containing information about which blocks to fill
        with dummy KV values.
        """
        meta = DecodeBenchConnectorMetadata(
            reqs_to_fill=self._pending_fills.copy(),
            first_blocks=self._pending_first_blocks.copy(),
        )

        # Clear pending fills after building metadata
        self._pending_fills.clear()
        self._pending_first_blocks.clear()

        return meta

    def request_finished(self, request: "Request"):
        """Called when a request has finished. Clean up any state."""
        self._filled_requests.discard(request.request_id)


class DecodeBenchConnectorWorker:
    """Worker-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        # Get fill parameters from extra config
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)
        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)
        self.startup_fill = kv_transfer_config.get_from_extra_config(
            "startup_fill", False
        )
        self._startup_fill_group_ids = (
            _get_startup_fill_group_ids(kv_cache_config) if self.startup_fill else set()
        )
        self.fill_mode = str(
            kv_transfer_config.get_from_extra_config("fill_mode", "const")
        )
        if self.fill_mode not in ("const", "m3_pool"):
            raise ValueError(f"DecodeBenchConnector: unknown fill_mode {self.fill_mode}")
        if self.fill_mode == "m3_pool" and self.startup_fill:
            raise ValueError(
                "DecodeBenchConnector: fill_mode m3_pool does not support startup_fill"
            )

        # Will be populated via register_kv_caches
        self.kv_caches: dict[str, torch.Tensor] | None = None

        # Layers whose uint8 KV cache stores plain fp8 values
        self._fp8_dtypes: dict[str, torch.dtype] = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                fp8_dtype = _get_fp8_dtype(spec, vllm_config.cache_config.cache_dtype)
                if fp8_dtype is not None:
                    self._fp8_dtypes[layer_name] = fp8_dtype

        # Mapping from KV cache group index to list of layer names in that group
        self.group_to_layers = {
            group_idx: list(group.layer_names)
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self._zero_fill_group_ids = {
            group_idx
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            if any(
                isinstance(spec, CircularBufferSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
        }
        num_zeroed_groups = len(
            self.group_to_layers.keys()
            - self._startup_fill_group_ids
            - self._zero_fill_group_ids
        )
        if self.startup_fill and num_zeroed_groups:
            logger.info(
                "DecodeBenchConnector: the engine zeroes new blocks of %d KV "
                "cache groups before use, so they are still filled per request.",
                num_zeroed_groups,
            )

        self._m3_kinds: dict[str, tuple] = {}
        self._m3_flat: dict[str, torch.Tensor] = {}
        self._m3_pools: _M3Pools | None = None
        self._m3_timing = os.environ.get("M3FILL_TIMING", "0") == "1"
        self._m3_dump_dir = os.environ.get("M3FILL_DUMP_DIR", "")
        if self.fill_mode == "m3_pool":
            idx_fmt = (
                os.environ.get("AITER_MSA_INDEX_KV_FMT", "fp8").strip().lower()
                or "fp8"
            )
            kv_fmt = os.environ.get("MINIMAX_M3_SPARSE_KV_FMT", "fp8").lower()
            self._m3_pools = _M3Pools(
                seed=int(kv_transfer_config.get_from_extra_config("fill_seed", 1234)),
                num_winners=int(
                    kv_transfer_config.get_from_extra_config("fill_index_winners", 15)
                ),
                idx_fmt=idx_fmt,
                kv_fmt=kv_fmt,
            )
            logger.info(
                "DecodeBenchConnector m3_pool: idx_fmt=%s kv_fmt=%s winners=%d",
                idx_fmt,
                kv_fmt,
                self._m3_pools.num_winners,
            )
            if self._m3_dump_dir:
                layers = {
                    int(x)
                    for x in os.environ.get("M3FILL_DUMP_LAYERS", "0,30,59").split(",")
                }
                _install_topk_dump(
                    self._m3_dump_dir,
                    layers,
                    int(os.environ.get("M3FILL_DUMP_MAX", "20000")),
                )

    @staticmethod
    def _m3_classify(t: torch.Tensor) -> tuple | None:
        """("kv", block_size, heads) for the packed sparse-PA KV cache,
        ("idx", block_size) for the fp8 index-key cache, a trailing "compact"
        for MINIMAX_M3_NVFP4_COMPACT_PAGES=1 caches (last dim 72), else None."""
        if not isinstance(t, torch.Tensor) or t.dim() < 3 or t.element_size() != 1:
            return None
        if not t.is_contiguous():
            return None
        if t.dim() == 4 and t.shape[1] == 2 and t.shape[2] % 16 == 0:
            if t.shape[3] % 128 == 0:
                return ("kv", int(t.shape[2]), int(t.shape[3]) // 128)
            if t.shape[3] == _NVFP4_COMPACT_DIM:
                return ("kv", int(t.shape[2]), 1, "compact")
        if t.dtype in _FP8_DTYPES and t.shape[-1] == 128:
            per = t.numel() // t.shape[0]
            return ("idx", per // 128)
        if t.dtype in _FP8_DTYPES and t.shape[-1] == _NVFP4_COMPACT_DIM:
            per = t.numel() // t.shape[0]
            return ("idx", per // _NVFP4_COMPACT_DIM, "compact")
        return None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Store references to the KV cache tensors."""
        self.kv_caches = kv_caches

        logger.debug(
            "DecodeBenchConnector: Registered %d KV cache layers",
            len(kv_caches),
        )
        if self.fill_mode != "m3_pool":
            return
        assert self._m3_pools is not None
        counts: dict[Any, int] = {}
        examples: dict[Any, str] = {}
        for name, t in kv_caches.items():
            kind = self._m3_classify(t)
            counts[kind] = counts.get(kind, 0) + 1
            if isinstance(t, torch.Tensor):
                examples.setdefault(
                    kind, f"{name} {tuple(t.shape)} {t.stride()} {t.dtype}"
                )
            if kind is None:
                continue
            self._m3_kinds[name] = kind
            self._m3_flat[name] = (
                t.view(torch.uint8).view(t.shape[0], -1).view(torch.int64)
            )
            self._m3_pools.get(kind, t.device)
        for kind, n in counts.items():
            logger.info(
                "DecodeBenchConnector m3_pool: %d layers kind=%s e.g. %s",
                n,
                kind,
                examples.get(kind),
            )

        if self.startup_fill:
            self._fill_all_kv_caches()

    def _fill_all_kv_caches(self):
        """Fill the KV cache tensors of the startup-filled groups once, in
        place, with dummy values, so that no per-request fill is needed.
        """
        assert self.kv_caches is not None

        num_layers = 0
        for group_idx in sorted(self._startup_fill_group_ids):
            for layer_name in self.group_to_layers[group_idx]:
                kv_cache = self.kv_caches.get(layer_name)
                if isinstance(kv_cache, torch.Tensor):
                    tensors = [kv_cache]
                elif isinstance(kv_cache, (list, tuple)) and all(
                    isinstance(t, torch.Tensor) for t in kv_cache
                ):
                    tensors = list(kv_cache)
                else:
                    logger.warning_once(
                        "DecodeBenchConnector: skipping startup fill for layer "
                        "%s whose KV cache is %s, not a tensor or a list/tuple "
                        "of tensors.",
                        layer_name,
                        type(kv_cache).__name__,
                    )
                    continue
                for tensor in tensors:
                    fill_dtype = self._fp8_dtypes.get(layer_name, tensor.dtype)
                    self._fill_tensor(tensor, self.fill_mean, self.fill_std, fill_dtype)
                num_layers += 1

        logger.info(
            "DecodeBenchConnector: Filled %d KV cache layers at startup with %s "
            "values (mean=%.3f, std=%.3f)",
            num_layers,
            "random" if self.fill_std > 0 else "constant",
            self.fill_mean,
            self.fill_std,
        )

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """Fill the allocated KV cache blocks with dummy values.

        This simulates having a populated KV cache from a prefill phase,
        allowing decode performance testing with larger context sizes.

        Supports both single- and multi-group KV cache configurations.
        """
        if not metadata.reqs_to_fill:
            return

        assert self.kv_caches is not None, "KV caches must be registered before filling"

        if self.fill_mode == "m3_pool":
            self._m3_fill(metadata)
            return

        for req_id, (block_ids_per_group, num_tokens) in metadata.reqs_to_fill.items():
            # Fill blocks for each KV cache group
            for group_idx, block_ids in enumerate(block_ids_per_group):
                self._fill_blocks(group_idx, block_ids, num_tokens)

            block_counts = tuple(len(group) for group in block_ids_per_group)
            logger.debug(
                "DecodeBenchConnector: Filled %d total blocks (%d tokens) across "
                "%d groups for request %s (per-group counts: %s)",
                sum(block_counts),
                num_tokens,
                len(block_ids_per_group),
                req_id,
                ", ".join(map(str, block_counts)),
            )

    def _m3_fill(self, metadata: DecodeBenchConnectorMetadata):
        """One H2D, one template gather and one index_copy_ per layer, per
        group and layer kind, for all requests of the step."""
        assert self.kv_caches is not None and self._m3_pools is not None
        pools = self._m3_pools
        if self._m3_timing:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        reqs = list(metadata.reqs_to_fill.items())
        num_groups = max(len(b) for _, (b, _) in reqs)
        dump_rows: list[dict] = []
        for g in range(num_groups):
            layers = self.group_to_layers.get(g, [])
            by_kind: dict[tuple, list[str]] = {}
            for name in layers:
                kind = self._m3_kinds.get(name)
                if kind is None or g in self._zero_fill_group_ids:
                    # Unknown layout: original per-layer fill for this layer.
                    for _, (bpg, ntok) in reqs:
                        if g < len(bpg) and bpg[g] and name in self.kv_caches:
                            self._fill_layer(g, name, bpg[g])
                    continue
                by_kind.setdefault(kind, []).append(name)
            for kind, names in by_kind.items():
                nblk = self._m3_flat[names[0]].shape[0]
                ids: list[int] = []
                tids: list[int] = []
                for req_id, (bpg, _) in reqs:
                    if g >= len(bpg) or not bpg[g]:
                        continue
                    first = metadata.first_blocks.get(req_id, (0,) * len(bpg))[g]
                    t = pools.template_ids(kind, first, len(bpg[g]))
                    for b, tid in zip(bpg[g], t):
                        if b < nblk:
                            ids.append(b)
                            tids.append(tid)
                    if self._m3_dump_dir and kind[0] == "idx":
                        dump_rows.append(
                            {
                                "req": req_id,
                                "group": g,
                                "first": first,
                                "nblocks": len(bpg[g]),
                                "winners": m3_index_winner_positions(
                                    first + len(bpg[g]), pools.num_winners
                                ),
                            }
                        )
                if not ids:
                    continue
                pool = pools.pools[kind]
                n = len(ids)
                both = torch.tensor(ids + tids, dtype=torch.long).to(
                    pool.device, non_blocking=True
                )
                src = pool.index_select(0, both[n:])
                dst_ids = both[:n]
                for name in names:
                    self._m3_flat[name].index_copy_(0, dst_ids, src)
        if self._m3_timing:
            torch.cuda.synchronize()
            logger.info(
                "DecodeBenchConnector m3_pool: filled %d reqs in %.3f ms",
                len(reqs),
                (time.perf_counter() - t0) * 1e3,
            )
        if dump_rows and _dist_rank() <= 0:
            os.makedirs(self._m3_dump_dir, exist_ok=True)
            with open(os.path.join(self._m3_dump_dir, "fills.jsonl"), "a") as f:
                for row in dump_rows:
                    f.write(json.dumps(row) + "\n")

    def _fill_layer(self, group_idx: int, layer_name: str, block_ids: list[int]):
        fill_mean, fill_std = (
            (0.0, 0.0)
            if group_idx in self._zero_fill_group_ids
            else (self.fill_mean, self.fill_std)
        )
        assert self.kv_caches is not None
        kv_cache = self.kv_caches[layer_name]
        if isinstance(kv_cache, torch.Tensor):
            fill_dtype = self._fp8_dtypes.get(layer_name, kv_cache.dtype)
            self._fill_block_tensor(
                kv_cache, block_ids, fill_mean, fill_std, fill_dtype
            )
        elif isinstance(kv_cache, (list, tuple)):
            for state_tensor in kv_cache:
                self._fill_tensor(
                    state_tensor, fill_mean, fill_std, state_tensor.dtype
                )

    def _fill_blocks(self, group_idx: int, block_ids: list[int], num_tokens: int):
        """Fill specified blocks with dummy values for a specific KV cache group.

        Args:
            group_idx: The KV cache group index to fill
            block_ids: List of block IDs to fill in this group
            num_tokens: Total number of tokens to fill across these blocks

        """
        if not block_ids:
            return

        assert self.kv_caches is not None

        # Circular buffers may pack non-floating metadata alongside their
        # floating-point state, so arbitrary fill values are not representation-safe.
        fill_mean, fill_std = (
            (0.0, 0.0)
            if group_idx in self._zero_fill_group_ids
            else (self.fill_mean, self.fill_std)
        )

        # Get the layers that belong to this group
        layer_names = self.group_to_layers.get(group_idx, [])

        # Fill only the layers in this group
        for layer_name in layer_names:
            if layer_name not in self.kv_caches:
                logger.warning(
                    "DecodeBenchConnector: Layer %s not found in KV caches", layer_name
                )
                continue

            kv_cache = self.kv_caches[layer_name]

            # Attention layers store KV as a single block-indexed tensor whose
            # first dim is num_blocks; fill the requested block rows. Hybrid /
            # linear-attention layers (e.g. Mamba, Kimi Delta Attention) store
            # their state as a list/tuple of tensors that are NOT block-indexed
            # -- each tensor is a single state buffer with no num_blocks
            # dimension -- so fill each tensor in its entirety with the same
            # dummy values.
            if isinstance(kv_cache, torch.Tensor):
                fill_dtype = self._fp8_dtypes.get(layer_name, kv_cache.dtype)
                self._fill_block_tensor(
                    kv_cache, block_ids, fill_mean, fill_std, fill_dtype
                )
            elif isinstance(kv_cache, (list, tuple)) and all(
                isinstance(t, torch.Tensor) for t in kv_cache
            ):
                for state_tensor in kv_cache:
                    self._fill_tensor(
                        state_tensor, fill_mean, fill_std, state_tensor.dtype
                    )
            else:
                logger.warning_once(
                    "DecodeBenchConnector: skipping fill for layer %s whose KV "
                    "cache is %s, not a tensor or a list/tuple of tensors.",
                    layer_name,
                    type(kv_cache).__name__,
                )
                continue

        logger.debug(
            "DecodeBenchConnector: Filled %d blocks in group %d with %s values "
            "(mean=%.3f, std=%.3f)",
            len(block_ids),
            group_idx,
            "random" if fill_std > 0 else "constant",
            fill_mean,
            fill_std,
        )

    def _fill_block_tensor(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        fill_mean: float,
        fill_std: float,
        fill_dtype: torch.dtype,
    ):
        """Fill the requested block rows of a block-indexed KV cache tensor.

        Args:
            kv_cache: A KV cache tensor whose first dim is num_blocks.
            block_ids: Block IDs to fill. IDs that are out of range for this
                tensor's first dim are ignored.
            fill_mean: Mean value for the fill.
            fill_std: Standard deviation for the fill.
            fill_dtype: Dtype the values are encoded as, e.g. the fp8 dtype
                of a uint8 fp8 cache.

        """
        # Convert block_ids to tensor on device
        block_ids_tensor = torch.tensor(
            block_ids, dtype=torch.long, device=kv_cache.device
        )

        # Filter invalid block IDs
        valid_mask = block_ids_tensor < kv_cache.shape[0]
        valid_block_ids = block_ids_tensor[valid_mask]

        if len(valid_block_ids) == 0:
            return

        fill_values = self._make_fill_values(
            (len(valid_block_ids),) + kv_cache.shape[1:],
            kv_cache.device,
            fill_dtype,
            fill_mean,
            fill_std,
        )

        # Batch fill operation
        kv_cache[valid_block_ids] = fill_values.view(kv_cache.dtype)

    def _make_fill_values(
        self,
        size: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        fill_mean: float,
        fill_std: float,
    ) -> torch.Tensor:
        """Create constant or random fill values of ``dtype``, clamped to its
        finite range. Non-floating dtypes (packed layouts) are filled with
        zeros.
        """
        if not dtype.is_floating_point:
            if fill_mean or fill_std:
                logger.warning_once(
                    "DecodeBenchConnector: %s KV caches do not hold plain "
                    "floating-point values; filling them with zeros.",
                    dtype,
                )
            return torch.zeros(size, dtype=dtype, device=device)

        finfo = torch.finfo(dtype)
        if fill_std > 0:
            sample_dtype = torch.float32
            if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                sample_dtype = dtype
            values = torch.normal(
                fill_mean, fill_std, size=size, device=device, dtype=sample_dtype
            )
            return values.clamp_(finfo.min, finfo.max).to(dtype)
        fill_mean = min(max(fill_mean, finfo.min), finfo.max)
        return torch.full(size, fill_mean, dtype=dtype, device=device)

    def _fill_tensor(
        self,
        kv_cache: torch.Tensor,
        fill_mean: float,
        fill_std: float,
        fill_dtype: torch.dtype,
    ):
        """Fill an entire tensor in place with dummy values.

        Used for startup fills, and for hybrid / linear-attention layers (e.g.
        Mamba, Kimi Delta Attention) whose per-layer state tensors are filled
        in their entirety with the same constant or random values used for
        block fills, rather than selected block rows.

        Args:
            kv_cache: A tensor to fill in its entirety.
            fill_mean: Mean value for the fill.
            fill_std: Standard deviation for the fill.
            fill_dtype: Dtype the values are encoded as, e.g. the fp8 dtype
                of a uint8 fp8 cache.

        """
        view = kv_cache.view(fill_dtype)
        if fill_std == 0 or not fill_dtype.is_floating_point:
            view.fill_(
                self._make_fill_values(
                    (), torch.device("cpu"), fill_dtype, fill_mean, fill_std
                )
            )
            return
        # Sample in chunks to bound the temporary memory.
        rows = max(1, (1 << 24) // max(1, view[0].numel()))
        for chunk in view.split(rows):
            chunk.copy_(
                self._make_fill_values(
                    chunk.shape, chunk.device, fill_dtype, fill_mean, fill_std
                )
            )
