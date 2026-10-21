# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER page-16 sparse paged-attention helpers for MiniMax-M3 on ROCm."""

import os
from functools import cache

import torch

try:
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:
    import triton
    import triton.language as tl

from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE

_FP8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
}


def _is_fp8_kv_cache_tensor(kv_cache: torch.Tensor) -> bool:
    return kv_cache.dtype in _FP8_DTYPES


@cache
def minimax_m3_sparse_kv_fmt_requested() -> str | None:
    """MINIMAX_M3_SPARSE_KV_FMT (fp8 | nvfp4), or None when unset."""
    fmt = os.environ.get("MINIMAX_M3_SPARSE_KV_FMT", "").strip().lower()
    if fmt not in ("", "fp8", "nvfp4"):
        raise ValueError(f"MINIMAX_M3_SPARSE_KV_FMT must be fp8 or nvfp4, got {fmt}")
    return fmt or None


# Set by the sparse layers at init (minimax_m3_resolve_sparse_kv_nvfp4).
_sparse_kv_nvfp4: bool | None = None


def minimax_m3_resolve_sparse_kv_nvfp4(supported: bool) -> bool:
    """Pick the sparse layers' KV format: MINIMAX_M3_SPARSE_KV_FMT if set,
    else nvfp4 when the layer supports it (AITER sparse PA, fp8 KV cache)."""
    global _sparse_kv_nvfp4
    fmt = minimax_m3_sparse_kv_fmt_requested()
    nvfp4 = supported if fmt is None else fmt == "nvfp4"
    if _sparse_kv_nvfp4 is not None and _sparse_kv_nvfp4 != nvfp4:
        raise ValueError("MiniMax-M3 sparse layers disagree on the KV format")
    _sparse_kv_nvfp4 = nvfp4
    return nvfp4


def minimax_m3_sparse_kv_nvfp4() -> bool:
    """NVFP4 KV for the sparse layers (default where supported).

    The fp8 KV cache keeps its shape and size. The first 1024 B of each
    2048 B (page, head) chunk of the K side hold the NVFP4 K codes and the
    next 256 B the K and V scales; the first 1024 B of the V side chunk hold
    the NVFP4 V codes. That is the layout pa_decode_gluon reads with
    kv_block_scale, and only those 2304 of 4096 B per page are touched.
    Before any layer resolved it, only an explicit nvfp4 counts.
    """
    if _sparse_kv_nvfp4 is None:
        return minimax_m3_sparse_kv_fmt_requested() == "nvfp4"
    return _sparse_kv_nvfp4


@cache
def _compact_pages_requested() -> bool:
    return os.environ.get("MINIMAX_M3_NVFP4_COMPACT_PAGES", "0") == "1"


def minimax_m3_sparse_kv_compact() -> bool:
    """MINIMAX_M3_NVFP4_COMPACT_PAGES=1 (NVFP4 sparse KV only): allocate real
    NVFP4 pages instead of fp8 ones. A 128-token block is 8 pages of 2304 B,
    [K codes 1024 | K/V scales 256 | V codes 1024], so K and V of a page share
    its page id with no page-16 rebase (see NVFP4_COMPACT_PAGE_BYTES).
    """
    return minimax_m3_sparse_kv_nvfp4() and _compact_pages_requested()


# Per (page, head) chunk: [head_size // 16, 16 tokens, 8 B] codes, then on the
# K side [16 tokens, head_size // 32, 4 B] E4M3 scales.
_NVFP4_CODE_BYTES = 128 // 2 * ASM_PAGE_SIZE
_NVFP4_SCALE_BYTES = 128 // 16 * 2 * ASM_PAGE_SIZE
# One compact page: K codes, K/V scales, then V codes.
NVFP4_COMPACT_V_OFFSET = _NVFP4_CODE_BYTES + _NVFP4_SCALE_BYTES
NVFP4_COMPACT_PAGE_BYTES = NVFP4_COMPACT_V_OFFSET + _NVFP4_CODE_BYTES


def minimax_m3_compact_kv_views(
    kv_cache: torch.Tensor,  # fp8 [num_blocks, 2, block_size, C] with C = 72
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 K / V code views [pages, 1, 8, 16, 8] of a compact NVFP4 cache.

    Both views address page ``p`` at ``p * NVFP4_COMPACT_PAGE_BYTES``; V starts
    NVFP4_COMPACT_V_OFFSET bytes in, and K's scales sit right after its codes
    where _nvfp4_kv_views expects them.
    """
    page = NVFP4_COMPACT_PAGE_BYTES
    num_blocks = kv_cache.shape[0]
    block_bytes = kv_cache[0].numel() * kv_cache.element_size()
    if (
        not kv_cache.is_contiguous()
        or kv_cache.element_size() != 1
        or block_bytes % page
    ):
        raise RuntimeError(
            "MiniMax-M3 compact NVFP4 KV needs contiguous 1-byte blocks of whole "
            f"{page} B pages, got {tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
    num_pages = num_blocks * (block_bytes // page)
    shape = (num_pages, 1, 128 // 16, ASM_PAGE_SIZE, 8)
    stride = (page, page, 8 * ASM_PAGE_SIZE, 8, 1)
    base = kv_cache.storage_offset()
    return (
        kv_cache.as_strided(shape, stride, base),
        kv_cache.as_strided(shape, stride, base + NVFP4_COMPACT_V_OFFSET),
    )


def _nvfp4_kv_views(
    k_cache: torch.Tensor,  # fp8 [pages, 1, head_size // 16, 16, 16]
    v_cache: torch.Tensor,  # fp8 [pages, 1, 1, head_size, 16]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 K codes, V codes and K/V scales inside the fp8 pages."""
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    page = k8.stride(0)
    if v8.stride(0) != page or page < _NVFP4_CODE_BYTES + _NVFP4_SCALE_BYTES:
        raise ValueError("MiniMax-M3 NVFP4 sparse KV needs equal fp8 page strides")
    code_shape = (128 // 16, ASM_PAGE_SIZE, 8)
    code_stride = (page, page, 8 * ASM_PAGE_SIZE, 8, 1)
    return (
        k8.as_strided((k8.shape[0], 1, *code_shape), code_stride),
        v8.as_strided((v8.shape[0], 1, *code_shape), code_stride),
        k8.as_strided(
            (k8.shape[0], 1, ASM_PAGE_SIZE, 128 // 32, 4),
            (page, page, 16, 4, 1),
            k8.storage_offset() + _NVFP4_CODE_BYTES,
        ),
    )


@triton.jit
def _nvfp4_quant_row(src, scale):
    """NVFP4 words and E4M3 scales of one 128-wide head row.

    Returns int32 words [8 blocks, 2]; byte j of word g of block s holds, for
    A = elements s * 16 + g * 8 + 0..3 and B = the next 4, |A_j| | sign(A_j)
    << 7 | |B_j| << 4 | sign(B_{j-1 mod 4}) << 3. Also returns the [8] scale
    bytes. Rounding matches the reference quantizer: scale E4M3_rn(amax / 6)
    clamped to [2^-9, 448], codes RNE onto the E2M1 grid.
    """
    s = tl.arange(0, 8)[:, None, None]
    g = tl.arange(0, 2)[None, :, None]
    j = tl.arange(0, 4)[None, None, :]
    a = tl.math.div_rn(tl.load(src + s * 16 + g * 8 + j).to(tl.float32), scale)
    b = tl.math.div_rn(tl.load(src + s * 16 + g * 8 + 4 + j).to(tl.float32), scale)
    amax = tl.max(tl.max(tl.maximum(tl.abs(a), tl.abs(b)), axis=2), axis=1)
    sf = tl.math.div_rn(tl.maximum(amax, 1e-12), 6.0)
    sf = tl.minimum(sf, 448.0).to(tl.float8e4nv).to(tl.float32)
    sf = tl.maximum(sf, 1.0 / 512).to(tl.float8e4nv)
    sf32 = sf.to(tl.float32)[:, None, None]

    ma = tl.minimum(tl.abs(tl.math.div_rn(a, sf32)), 6.0)
    mb = tl.minimum(tl.abs(tl.math.div_rn(b, sf32)), 6.0)
    a_mag = (
        (ma > 0.25).to(tl.uint32)
        + (ma >= 0.75).to(tl.uint32)
        + (ma > 1.25).to(tl.uint32)
        + (ma >= 1.75).to(tl.uint32)
        + (ma > 2.5).to(tl.uint32)
        + (ma >= 3.5).to(tl.uint32)
        + (ma > 5.0).to(tl.uint32)
    )
    b_mag = (
        (mb > 0.25).to(tl.uint32)
        + (mb >= 0.75).to(tl.uint32)
        + (mb > 1.25).to(tl.uint32)
        + (mb >= 1.75).to(tl.uint32)
        + (mb > 2.5).to(tl.uint32)
        + (mb >= 3.5).to(tl.uint32)
        + (mb > 5.0).to(tl.uint32)
    )
    a_neg = (a.to(tl.int32, bitcast=True) < 0).to(tl.uint32)
    b_neg = (b.to(tl.int32, bitcast=True) < 0).to(tl.uint32)
    shift = (j * 8).to(tl.uint32)
    word = tl.sum((a_mag | (a_neg << 7) | (b_mag << 4)) << shift, axis=2)
    b_sign = tl.sum((b_neg << 3) << shift, axis=2)
    word = word | (b_sign << 8) | (b_sign >> 24)
    return word.to(tl.int32, bitcast=True), sf.to(tl.uint8, bitcast=True)


@triton.jit
def _insert_kv_nvfp4_kernel(
    k_ptr,  # [num_tokens, num_kv_heads, 128]
    v_ptr,
    k_cache_ptr,  # uint8, page-16 K side
    v_cache_ptr,  # uint8, page-16 V side
    slot_mapping_ptr,  # [num_tokens], page-16 slots
    k_scale_ptr,
    v_scale_ptr,
    stride_k_t,
    stride_k_h,
    stride_v_t,
    stride_v_h,
    stride_c_page,
    stride_c_head,
    SCALE_OFFSET: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + pid_t)
    if slot < 0:
        return
    page = slot // 16
    tok = slot % 16
    chunk = page.to(tl.int64) * stride_c_page + pid_h * stride_c_head

    k_words, k_sf = _nvfp4_quant_row(
        k_ptr + pid_t * stride_k_t + pid_h * stride_k_h, tl.load(k_scale_ptr)
    )
    v_words, v_sf = _nvfp4_quant_row(
        v_ptr + pid_t * stride_v_t + pid_h * stride_v_h, tl.load(v_scale_ptr)
    )
    # Words of block s for this token at int32 offset s * 32 + tok * 2.
    w = tl.arange(0, 8)[:, None] * 32 + tok * 2 + tl.arange(0, 2)[None, :]
    tl.store((k_cache_ptr + chunk).to(tl.pointer_type(tl.int32)) + w, k_words)
    tl.store((v_cache_ptr + chunk).to(tl.pointer_type(tl.int32)) + w, v_words)
    # Scale bytes of this token: [s % 4, :] = K_s, K_s+4, V_s, V_s+4.
    s = tl.arange(0, 8)
    sf_dst = k_cache_ptr + chunk + SCALE_OFFSET + tok * 16 + (s % 4) * 4 + s // 4
    tl.store(sf_dst, k_sf)
    tl.store(sf_dst + 2, v_sf)


@torch.no_grad()
def minimax_m3_insert_kv_nvfp4(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, 128]
    v: torch.Tensor,
    k_cache: torch.Tensor,  # fp8 [pages, num_kv_heads, 8, 16, 16]
    v_cache: torch.Tensor,  # fp8 [pages, num_kv_heads, 1, 128, 16]
    slot_mapping: torch.Tensor,  # [num_tokens], page-16 slots
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    """Quantize K/V to NVFP4 in place of fp8 (see minimax_m3_sparse_kv_nvfp4)."""
    num_tokens, num_kv_heads, head_size = k.shape
    if num_tokens == 0:
        return
    if head_size != 128 or k.stride(2) != 1 or v.stride(2) != 1:
        raise ValueError("MiniMax-M3 NVFP4 KV insert needs contiguous head_dim 128")
    if not _is_fp8_kv_cache_tensor(k_cache) or (
        k_cache.stride()[:2] != v_cache.stride()[:2]
    ):
        raise ValueError("MiniMax-M3 NVFP4 KV insert needs matching fp8 caches")
    if k_cache.stride(1) != 128 * ASM_PAGE_SIZE and not (
        num_kv_heads == 1 and k_cache.stride(1) == NVFP4_COMPACT_PAGE_BYTES
    ):
        raise ValueError(
            "MiniMax-M3 NVFP4 KV insert needs 2048 B page chunks or compact pages"
        )
    one = torch.ones(1, dtype=torch.float32, device=k.device)
    _insert_kv_nvfp4_kernel[(num_tokens, num_kv_heads)](
        k,
        v,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        slot_mapping,
        one if k_scale is None else k_scale,
        one if v_scale is None else v_scale,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        SCALE_OFFSET=_NVFP4_CODE_BYTES,
        num_warps=1,
    )


@triton.jit
def _write_sparse_block_table_row_from_values(
    blk,  # [BLOCK_SIZE_T] int32 selected logical block ids for this query
    bt_row,  # [max_blocks] int32, this request's logical page table
    sbt_row,  # [topk * PAGES_PER_BLOCK] int32, physical 16-page table
    ctx_ptr,  # int32, this query's attended context length
    abs_pos,  # absolute position of this query token, may be negative
    max_topk,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Compact one query's selected logical blocks into physical page-16 ids.

    ``BLOCK_PAGE_STRIDE`` is how many page ids a block spans, which is
    ``PAGES_PER_BLOCK`` when each K/V side is its own dense plane and twice
    that when both sides share a block. Padded rows carry a negative
    ``abs_pos``, which clamps the causal range to empty.
    """
    causal_len = tl.maximum(abs_pos + 1, 0)
    self_blk = abs_pos // SPARSE_BLOCK_SIZE_C

    off_t = tl.arange(0, BLOCK_SIZE_T)
    valid = (off_t < max_topk) & (causal_len > 0) & (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)

    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)

    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * BLOCK_PAGE_STRIDE
    dst_base = slot * PAGES_PER_BLOCK

    for j in tl.static_range(PAGES_PER_BLOCK):
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=valid)

    n_used = n_valid * PAGES_PER_BLOCK
    off_w = tl.arange(0, BLOCK_SIZE_T * PAGES_PER_BLOCK)
    row_width = max_topk * PAGES_PER_BLOCK
    tl.store(
        sbt_row + off_w,
        tl.zeros_like(off_w),
        mask=(off_w >= n_used) & (off_w < row_width),
    )

    tail_tokens = causal_len - self_blk * SPARSE_BLOCK_SIZE_C
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * SPARSE_BLOCK_SIZE_C + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * SPARSE_BLOCK_SIZE_C, causal_len))
    tl.store(ctx_ptr, ctx)


@triton.jit
def _write_sparse_block_table_row(
    topk_row,  # [topk] int32, selected logical block ids for this query
    bt_row,  # [max_blocks] int32, this request's logical page table
    sbt_row,  # [topk * PAGES_PER_BLOCK] int32, physical 16-page table
    ctx_ptr,  # int32, this query's attended context length
    abs_pos,  # absolute position of this query token, may be negative
    max_topk,
    stride_topk_k,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_k, mask=off_t < max_topk, other=-1)
    _write_sparse_block_table_row_from_values(
        blk,
        bt_row,
        sbt_row,
        ctx_ptr,
        abs_pos,
        max_topk,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


@triton.jit
def _build_sparse_block_table_kernel(
    topk_ptr,  # [1, batch * DECODE_QUERY_LEN, topk] int32, logical block ids
    block_table_ptr,  # [batch, max_blocks] int32, logical 128-page table
    seq_lens_ptr,  # [batch] int32
    sparse_bt_ptr,  # [batch * DECODE_QUERY_LEN, topk * 8] int32
    sparse_ctx_ptr,  # [batch * DECODE_QUERY_LEN] int32
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_b,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
):
    # One program per query token. The decode indexer flattens speculative rows
    # as ``req_id * DECODE_QUERY_LEN + local_q``, and a request's query tokens
    # are the last DECODE_QUERY_LEN positions of its sequence, so both the
    # request id and the absolute position follow from the program id.
    pid_n = tl.program_id(0)
    req_id = pid_n // DECODE_QUERY_LEN
    local_q = pid_n % DECODE_QUERY_LEN
    seq_len = tl.load(seq_lens_ptr + req_id)
    abs_pos = seq_len - DECODE_QUERY_LEN + local_q

    _write_sparse_block_table_row(
        topk_ptr + pid_n * stride_topk_n,
        block_table_ptr + req_id * stride_bt_b,
        sparse_bt_ptr + pid_n * stride_sbt_b,
        sparse_ctx_ptr + pid_n,
        abs_pos,
        max_topk,
        stride_topk_k,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


@triton.jit
def _build_sparse_block_table_prefill_kernel(
    topk_ptr,  # [1, total_q, topk]
    block_table_ptr,  # [batch, max_blocks]
    req_id_ptr,  # [total_q]
    abs_pos_ptr,  # [total_q]
    sparse_bt_ptr,  # [total_q, topk * 8]
    sparse_ctx_ptr,  # [total_q]
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_n,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    req_id = tl.load(req_id_ptr + pid_n)
    abs_pos = tl.load(abs_pos_ptr + pid_n)

    _write_sparse_block_table_row(
        topk_ptr + pid_n * stride_topk_n,
        block_table_ptr + req_id * stride_bt_b,
        sparse_bt_ptr + pid_n * stride_sbt_n,
        sparse_ctx_ptr + pid_n,
        abs_pos,
        max_topk,
        stride_topk_k,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


def minimax_m3_alloc_sparse_block_table(
    topk_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_idx.shape[0] == 1, "AITER sparse PA requires num_kv_heads == 1"
    total_q = topk_idx.shape[1]
    width = topk_idx.shape[-1] * PAGES_PER_SPARSE_BLOCK
    return (
        torch.empty((total_q, width), dtype=torch.int32, device=topk_idx.device),
        torch.empty((total_q,), dtype=torch.int32, device=topk_idx.device),
    )


@torch.no_grad()
def minimax_m3_build_sparse_block_table_prefill(
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    query_req_id: torch.Tensor,  # [total_q]
    query_abs_pos: torch.Tensor,  # [total_q]
    block_page_stride: int = PAGES_PER_SPARSE_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one page-16 sparse block table row per prefill query token."""
    sparse_bt, sparse_ctx = minimax_m3_alloc_sparse_block_table(topk_idx)
    topk = topk_idx.shape[-1]
    _build_sparse_block_table_prefill_kernel[(topk_idx.shape[1],)](
        topk_idx,
        block_table,
        query_req_id,
        query_abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_PAGE_STRIDE=block_page_stride,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@torch.no_grad()
def minimax_m3_build_sparse_block_table_decode(
    topk_idx: torch.Tensor,  # [1, batch * decode_query_len, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    decode_query_len: int = 1,
    block_page_stride: int = PAGES_PER_SPARSE_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one page-16 sparse block table row per decode query token."""
    total_q = topk_idx.shape[1]
    expected_q = block_table.shape[0] * decode_query_len
    assert total_q == expected_q, (
        "MiniMax-M3 decode top-k rows must equal batch * decode_query_len: "
        f"{total_q} != {expected_q}"
    )
    sparse_bt, sparse_ctx = minimax_m3_alloc_sparse_block_table(topk_idx)
    topk = topk_idx.shape[-1]
    _build_sparse_block_table_kernel[(total_q,)](
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_PAGE_STRIDE=block_page_stride,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        DECODE_QUERY_LEN=decode_query_len,
    )
    return sparse_bt, sparse_ctx


# Retained for the decode_query_len == 1 callers (and tests) that predate the
# speculative-decode rows.
minimax_m3_build_sparse_block_table = minimax_m3_build_sparse_block_table_decode


@triton.jit
def _insert_index_cache_kernel(
    index_k_ptr,  # [num_tokens, head_dim]
    index_cache_ptr,  # [num_blocks, block_size, head_dim]
    slot_mapping_ptr,  # [num_tokens]
    stride_k_t,
    stride_k_d,
    stride_c_b,
    stride_c_t,
    stride_c_d,
    stride_slot_t,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    slot = tl.load(slot_mapping_ptr + pid_t * stride_slot_t)
    block_id = slot // CACHE_BLOCK_SIZE
    block_off = slot - block_id * CACHE_BLOCK_SIZE

    src = index_k_ptr + pid_t * stride_k_t + offs_d * stride_k_d
    dst = (
        index_cache_ptr
        + block_id * stride_c_b
        + block_off * stride_c_t
        + offs_d * stride_c_d
    )
    mask = (slot >= 0) & (offs_d < HEAD_DIM)
    value = tl.load(src, mask=offs_d < HEAD_DIM, other=0.0)
    # The side cache carries its own dtype (bf16, or e4m3 for the fp8 indexer),
    # so convert rather than leaning on the store's implicit cast.
    tl.store(dst, value.to(dst.dtype.element_ty), mask=mask)


@torch.no_grad()
def minimax_m3_insert_index_cache(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    index_slot_mapping: torch.Tensor,
) -> None:
    """Scatter index keys into MiniMax-M3's key-only side cache."""
    if index_k.numel() == 0 or index_cache.numel() == 0:
        return
    if index_k.dim() != 2 or index_cache.dim() != 3:
        raise ValueError("MiniMax-M3 index cache insert expects [N,D] and [B,T,D]")
    if index_k.shape[1] != index_cache.shape[2]:
        raise ValueError("MiniMax-M3 index key dim must match index cache head dim")
    if index_slot_mapping.dim() != 1 or index_slot_mapping.shape[0] != index_k.shape[0]:
        raise ValueError("MiniMax-M3 index slot mapping must be a length-N vector")
    if index_cache.stride(2) != 1:
        raise ValueError("MiniMax-M3 index cache requires contiguous head dimension")

    head_dim = index_k.shape[1]
    _insert_index_cache_kernel[(index_k.shape[0],)](
        index_k,
        index_cache,
        index_slot_mapping,
        index_k.stride(0),
        index_k.stride(1),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(2),
        index_slot_mapping.stride(0),
        CACHE_BLOCK_SIZE=index_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def _sides_are_packed(k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    """Whether a block holds both its K and V pages instead of one side."""
    return v_cache.shape[0] != k_cache.shape[0]


def minimax_m3_sparse_block_page_stride(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> int:
    """How many page ids one sparse block spans.

    Each side owns ``PAGES_PER_SPARSE_BLOCK`` pages. When the sides are dense
    planes a block is exactly that wide; when they interleave, a block covers
    both sides' pages and V is reached from the same page id at a fixed offset.
    """
    return PAGES_PER_SPARSE_BLOCK * (2 if _sides_are_packed(k_cache, v_cache) else 1)


# K/V sides sharing one block, which is how many sides' pages a block spans.
# The attend publishes two head slots and the indexer's side cache one, and
# specs that mix HNC shapes narrow the resolved layout to a block-compact one
# (LHBNC is not), so this path always lands on an interleaved block.
PAGE16_SIDES_PER_BLOCK = 2


def minimax_m3_page16_sides_per_block() -> int:
    """PAGE16_SIDES_PER_BLOCK, or 1 for compact NVFP4 pages, whose K and V
    share one page id per 16 tokens."""
    return 1 if minimax_m3_sparse_kv_compact() else PAGE16_SIDES_PER_BLOCK


def minimax_m3_rebase_block_table_to_page16(
    block_table: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rebase a logical page table onto AITER's page-16 page numbering.

    The page ids AITER's top-k emits for the attend are
    ``block_table[blk] * pages_per_block + j``, and ``pages_per_block`` is fixed
    at AITER build time to one side's pages. An interleaved block spans both
    sides, so the only way to reach its pages through that kernel is to hand it
    a table already scaled to the wider stride. The caller does this once per
    step, since every sparse layer resolves its selection through the same
    table.
    """
    if out is None:
        out = torch.empty_like(block_table)
    return torch.mul(block_table, minimax_m3_page16_sides_per_block(), out=out)


def _gluon_scale_arg(
    scale: torch.Tensor | None,
    *,
    num_phys_pages: int,
    num_kv_heads: int,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.numel() == 1:
        return scale
    if scale.dim() != 2 or scale.shape[0] != num_kv_heads:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA supports scalar KV scales or "
            "[num_kv_heads, max_kv_tokens] scales"
        )
    max_tokens = num_phys_pages * ASM_PAGE_SIZE
    if scale.shape[1] < max_tokens:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA KV scale tensor is smaller than the "
            f"cache token capacity ({scale.shape[1]} < {max_tokens})"
        )
    scale = scale[:, :max_tokens]
    return (
        scale.transpose(0, 1)
        .contiguous()
        .view(num_phys_pages, ASM_PAGE_SIZE, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
        .unsqueeze(-1)
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode_aiter(
    q: torch.Tensor,  # [batch * decode_query_len, num_heads, head_dim]
    k_cache: torch.Tensor,  # [phys16, num_kv_heads, head_dim // x, 16, x]
    v_cache: torch.Tensor,  # [phys16, num_kv_heads, 16 // x, head_dim, x]
    topk_idx: torch.Tensor,  # [1, batch * decode_query_len, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [batch * decode_query_len, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    decode_query_len: int = 1,
    sparse_block_table: torch.Tensor | None = None,
    sparse_context_lens: torch.Tensor | None = None,
) -> None:
    if (sparse_block_table is None) != (sparse_context_lens is None):
        raise ValueError(
            "MiniMax-M3 prepared sparse block table and context lengths "
            "must be provided together"
        )
    if sparse_block_table is None:
        sparse_block_table, sparse_context_lens = (
            minimax_m3_build_sparse_block_table_decode(
                topk_idx,
                block_table,
                seq_lens,
                decode_query_len,
                minimax_m3_sparse_block_page_stride(k_cache, v_cache),
            )
        )
    assert sparse_context_lens is not None
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_block_table,
        sparse_context_lens,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )


@cache
def minimax_m3_sparse_kv_prefill_dq() -> bool:
    """MINIMAX_M3_SPARSE_KV_PREFILL_DQ (default 1, NVFP4 KV only): prefill
    dequantizes every context page of the step's requests to fp8 once (dqC)
    and runs the fp8 kernel on that workspace; 0 runs the fused NVFP4 kernel.
    """
    return os.environ.get("MINIMAX_M3_SPARSE_KV_PREFILL_DQ", "1") == "1"


def minimax_m3_dqc_enabled() -> bool:
    return minimax_m3_sparse_kv_nvfp4() and minimax_m3_sparse_kv_prefill_dq()


@cache
def minimax_m3_dqc_reserve_rows() -> int:
    """MINIMAX_M3_DQC_RESERVE_ROWS (default 4): the dqC fp8 workspace is
    reserved during memory profiling for this many max_model_len contexts;
    prefill rows are dequantized and attended in groups that fit it.
    """
    rows = int(os.environ.get("MINIMAX_M3_DQC_RESERVE_ROWS", "4"))
    if rows < 1:
        raise ValueError(f"MINIMAX_M3_DQC_RESERVE_ROWS must be >= 1, got {rows}")
    return rows


def minimax_m3_dqc_budget_blocks(max_model_len: int) -> int:
    """Workspace capacity in 128-token blocks. One spare block per row covers
    seq_lens_cpu_upper_bound running past max_model_len."""
    return minimax_m3_dqc_reserve_rows() * (
        triton.cdiv(max_model_len, SPARSE_BLOCK_SIZE) + 1
    )


def minimax_m3_dqc_groups(
    seq_lens_cpu: list[int], query_start_cpu: list[int], budget_blocks: int
) -> list[tuple[int, int, int, int, int]]:
    """Greedy consecutive grouping of prefill rows under ``budget_blocks``.

    query_start_cpu: prefill-relative cumulative query lengths (len rows + 1).
    Returns (row_start, row_end, q_start, q_end, kv_blocks) per group.
    """
    groups = []
    r0 = 0
    blocks = 0
    for r, s in enumerate(seq_lens_cpu):
        nb = (s + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        if nb > budget_blocks:
            raise ValueError(
                f"MiniMax-M3 dqC: a {s}-token prefill row needs {nb} blocks, "
                f"the workspace holds {budget_blocks}"
            )
        if blocks + nb > budget_blocks:
            groups.append((r0, r, query_start_cpu[r0], query_start_cpu[r], blocks))
            r0, blocks = r, 0
        blocks += nb
    n = len(seq_lens_cpu)
    if n > r0:
        groups.append((r0, n, query_start_cpu[r0], query_start_cpu[n], blocks))
    return groups


@cache
def minimax_m3_dqc_min_qlen() -> int:
    """MINIMAX_M3_DQC_MIN_QLEN (default 0, dqC only): prefill rows with fewer
    new tokens run the fused NVFP4 kernel instead of dqC, whose whole-context
    dequant does not pay off for short chunks. 0 sends every row to dqC.
    """
    n = int(os.environ.get("MINIMAX_M3_DQC_MIN_QLEN", "0"))
    if n < 0:
        raise ValueError(f"MINIMAX_M3_DQC_MIN_QLEN must be >= 0, got {n}")
    return n


def minimax_m3_dqc_hybrid_plan(
    seq_lens_cpu: list[int],
    query_start_cpu: list[int],
    budget_blocks: int,
    min_qlen: int,
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Split prefill rows by query length: >= min_qlen to dqC, others fused.

    Returns (dqC groups as minimax_m3_dqc_groups, over the dqC rows only;
    fused runs as (row_start, row_end, q_start, q_end)). Both index the whole
    prefill batch; runs of consecutive rows of one mode stay together.
    """
    groups: list[tuple[int, int, int, int, int]] = []
    fused: list[tuple[int, int, int, int]] = []
    qs = query_start_cpu
    n = len(seq_lens_cpu)
    r = 0
    while r < n:
        dq = qs[r + 1] - qs[r] >= min_qlen
        e = r + 1
        while e < n and (qs[e + 1] - qs[e] >= min_qlen) == dq:
            e += 1
        if dq:
            for a, b, q0, q1, blk in minimax_m3_dqc_groups(
                seq_lens_cpu[r:e], qs[r : e + 1], budget_blocks
            ):
                groups.append((a + r, b + r, q0, q1, blk))
        else:
            fused.append((r, e, qs[r], qs[e]))
        r = e
    return groups, fused


def minimax_m3_dqc_mix_block_table(
    ws_block_table: torch.Tensor,
    page16_block_table: torch.Tensor,
    groups: list[tuple[int, int, int, int, int]],
) -> torch.Tensor:
    """Rows of ``groups`` from the workspace table, the rest from the page-16
    table (physical ids), so one top-k emits both kinds of rows."""
    rows, c16 = page16_block_table.shape
    cws = ws_block_table.shape[1]
    out = torch.empty(
        (rows, max(c16, cws)), dtype=torch.int32, device=page16_block_table.device
    )
    out[:, :c16] = page16_block_table
    if cws > c16:
        out[:, c16:] = 0
    for r0, r1, *_ in groups:
        out[r0:r1, :cws] = ws_block_table[r0:r1]
    return out


def minimax_m3_dqc_ws_block_table(
    seq_lens_cpu: list[int],
    groups: list[tuple[int, int, int, int, int]],
    device: torch.device,
) -> torch.Tensor:
    """[rows, max blocks] int32: workspace block of each row's logical block.

    Row r's blocks follow the blocks of the earlier rows of its group, from
    the same lengths ``groups`` was budgeted on, so every group fits its
    workspace. A sparse table built through it (in place of the page-16
    table) addresses the dqC workspace directly. One H2D copy per step.
    """
    nblk = [triton.cdiv(s, SPARSE_BLOCK_SIZE) for s in seq_lens_cpu]
    start = [0] * len(nblk)
    for r0, r1, *_ in groups:
        acc = 0
        for r in range(r0, r1):
            start[r] = acc
            acc += nblk[r]
    cols = max(nblk, default=0)
    host = torch.tensor(start, dtype=torch.int32)[:, None] + torch.arange(
        cols, dtype=torch.int32
    )
    return host.pin_memory().to(device, non_blocking=True)


def minimax_m3_dqc_workspace(ws_pages: int) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 K [ws_pages, 1, 8, 16, 16] / V [ws_pages, 1, 1, 128, 16] views of
    vLLM's workspace (sized by the profiling run, fixed after lock)."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    shapes = (
        ((ws_pages, 1, 8, 16, 16), torch.uint8),
        ((ws_pages, 1, 1, 128, 16), torch.uint8),
    )
    if is_workspace_manager_initialized():
        k8, v8 = current_workspace_manager().get_simultaneous(*shapes)
        return k8, v8
    # Standalone op tests: grow-only module buffers.
    key = ("ws", torch.cuda.current_device())
    st = _DQC_STATE.get(key)
    if st is None or st[0].shape[0] < ws_pages:
        st = tuple(torch.empty(s, dtype=d, device="cuda") for s, d in shapes)
        _DQC_STATE[key] = st
    return st[0][:ws_pages], st[1][:ws_pages]


@triton.jit
def _dqc_page(src, dst, kw, vw, ksc, lut, k8, v8, page_w):
    """Dequantize NVFP4 page ``src`` (K and V) into fp8 workspace page ``dst``.

    kw / vw: int32 views of the K / V sides, ksc: uint8 view of the K side,
    page_w: page stride in int32 words. Output layouts: k8 [S, 8, 16, 16]
    (dim group, token, element), v8 [S, 128, 16] (dim, token).
    """
    j = tl.arange(0, 8)[:, None, None]
    t = tl.arange(0, 16)[None, :, None]
    e = tl.arange(0, 16)[None, None, :]
    grp = e // 8
    hi = (e // 4) % 2
    jj = e % 4
    src = src.to(tl.int64)
    dst = dst.to(tl.int64)
    word_off = src * page_w + j * 32 + t * 2 + grp
    sf_off = src * page_w * 4 + 1024 + t * 16 + (j % 4) * 4 + j // 4
    for side in tl.static_range(2):
        if side == 0:
            w = tl.load(kw + word_off)
            s = tl.load(ksc + sf_off)
        else:
            w = tl.load(vw + word_off)
            s = tl.load(ksc + sf_off + 2)
        b = (w >> (8 * jj)) & 0xFF
        nb = (w >> (8 * ((jj + 1) % 4))) & 0xFF
        mag = tl.where(hi == 0, b & 7, (b >> 4) & 7)
        sign = tl.where(hi == 0, b >> 7, (nb >> 3) & 1)
        lv = tl.load(lut + s.to(tl.int32))
        val = ((lv >> (mag * 8).to(tl.int64)) & 0xFF).to(tl.int32) | (sign << 7)
        val = val.to(tl.uint8)
        if side == 0:
            tl.store(k8 + dst * 2048 + j * 256 + t * 16 + e, val)
        else:
            tl.store(v8 + dst * 2048 + (j * 16 + e) * 16 + t, val)


@triton.jit
def _dqc_context_kernel(
    block_table,  # [num_prefills, max_blocks] int32, logical 128-token blocks
    stride_bt,
    seq_lens,  # [num_prefills] int32
    blk_start,  # [num_prefills] int32, exclusive cumsum of blocks per request
    stride_bs,
    slot_of,  # [num_pages] int32, page id -> workspace page
    kw,
    vw,
    ksc,
    lut,
    k8,
    v8,
    page_w,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    STORE_SLOT: tl.constexpr,
):
    r = tl.program_id(0)
    b = tl.program_id(1)
    jp = tl.program_id(2)
    if b * (PAGES_PER_BLOCK * 16) < tl.load(seq_lens + r):
        blk = tl.load(block_table + r * stride_bt + b)
        page = blk * BLOCK_PAGE_STRIDE + jp
        dst = (tl.load(blk_start + r * stride_bs) + b) * PAGES_PER_BLOCK + jp
        _dqc_page(page, dst, kw, vw, ksc, lut, k8, v8, page_w)
        if STORE_SLOT:
            tl.store(slot_of + page, dst)


@triton.jit
def _dqc_remap_kernel(
    sparse_bt,  # [rows, cols] int32 page ids
    stride_sbt,
    sparse_ctx,  # [rows] int32 attended tokens
    new_bt,
    stride_nbt,
    slot_of,
    cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_row = c < cols
    used = c < (tl.load(sparse_ctx + row) + 15) // 16
    page = tl.load(sparse_bt + row * stride_sbt + c, mask=used, other=0)
    slot = tl.load(slot_of + page, mask=used, other=0)
    tl.store(new_bt + row * stride_nbt + c, tl.where(used, slot, 0), mask=in_row)


_DQC_STATE: dict = {}


def _nvfp4_lut(device: torch.device) -> torch.Tensor:
    """int64 [256]: byte m of entry s = E4M3(|E2M1[m]| * E4M3(s)), clamped."""
    key = ("lut", device)
    if key not in _DQC_STATE:
        e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        sf = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        sf = torch.nan_to_num(sf, nan=0.0)
        v = (e2m1[None, :] * sf[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        lut = v.view(torch.uint8).contiguous().view(torch.int64).view(256)
        _DQC_STATE[key] = lut.to(device)
    return _DQC_STATE[key]


def _one(device: torch.device) -> torch.Tensor:
    key = ("one", device)
    if key not in _DQC_STATE:
        _DQC_STATE[key] = torch.ones(1, dtype=torch.float32, device=device)
    return _DQC_STATE[key]


def _dqc_slot_of(device, num_pages: int) -> torch.Tensor:
    # int32 per physical KV page (KV pool pages * 4 B), outside the workspace:
    # its size is only known once the KV cache exists.
    key = ("slot_of", device)
    slot_of = _DQC_STATE.get(key)
    if slot_of is None or slot_of.numel() < num_pages:
        slot_of = torch.zeros(num_pages, dtype=torch.int32, device=device)
        _DQC_STATE[key] = slot_of
    return slot_of


def minimax_m3_sparse_kv_dequant_context(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,  # [num_prefills, max_blocks] logical blocks
    seq_lens: torch.Tensor,  # [num_prefills] int32
    max_seq_len: int,
    total_kv_blocks: int,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ws_block_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dqC: NVFP4 context pages -> fp8 workspace pages + remapped page table.

    workspace: uint8 K / V page buffers with >= total_kv_blocks * 8 pages
    (default: minimax_m3_dqc_workspace of exactly that size).
    ws_block_table: these rows of minimax_m3_dqc_ws_block_table; sparse_bt
    then already holds workspace ids and is returned as is (no remap).
    Returns (fp8 K workspace, fp8 V workspace, page table in workspace ids).
    """
    nphys16, hkv = k_cache.shape[0], k_cache.shape[1]
    if hkv != 1:
        raise ValueError("MiniMax-M3 NVFP4 sparse KV needs one KV head")
    k_view = k_cache.view(nphys16, 1, *k_cache.shape[2:])
    v_view = v_cache.view(v_cache.shape[0], 1, *v_cache.shape[2:])
    k_u8, v_u8 = k_view.view(torch.uint8), v_view.view(torch.uint8)
    page = k_u8.stride(0)
    if v_u8.stride(0) != page or page % 4 or page < 1280:
        raise ValueError("MiniMax-M3 NVFP4 sparse KV needs equal fp8 page strides")
    stride = minimax_m3_sparse_block_page_stride(k_cache, v_cache)
    dev = k_cache.device
    ws_pages = total_kv_blocks * PAGES_PER_SPARSE_BLOCK
    if workspace is None:
        workspace = minimax_m3_dqc_workspace(ws_pages)
    k8, v8 = workspace
    if k8.shape[0] < ws_pages or v8.shape[0] < ws_pages:
        raise ValueError(
            f"MiniMax-M3 dqC workspace holds {k8.shape[0]} pages, "
            f"{ws_pages} needed"
        )
    remap = ws_block_table is None
    if remap:
        slot_of = _dqc_slot_of(dev, nphys16)
        nblk = (seq_lens + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        blk_start = (torch.cumsum(nblk, 0) - nblk).to(torch.int32)
    else:
        # Column 0 is the row's first workspace block; slot_of is unused.
        blk_start = slot_of = ws_block_table
    max_blocks = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    num_prefills = seq_lens.shape[0]
    if num_prefills and max_blocks:
        _dqc_context_kernel[(num_prefills, max_blocks, PAGES_PER_SPARSE_BLOCK)](
            block_table,
            block_table.stride(0),
            seq_lens,
            blk_start,
            blk_start.stride(0),
            slot_of,
            k_u8.view(torch.int32),
            v_u8.view(torch.int32),
            k_u8,
            _nvfp4_lut(dev),
            k8,
            v8,
            page // 4,
            BLOCK_PAGE_STRIDE=stride,
            PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
            STORE_SLOT=remap,
            num_warps=2,
        )
    fp8 = k_cache.dtype
    if not remap:
        return k8.view(fp8), v8.view(fp8), sparse_bt
    rows, cols = sparse_bt.shape
    new_bt = torch.empty((rows, cols), dtype=torch.int32, device=dev)
    BLOCK = 256
    if rows:
        _dqc_remap_kernel[(rows, triton.cdiv(cols, BLOCK))](
            sparse_bt,
            sparse_bt.stride(0),
            sparse_ctx,
            new_bt,
            new_bt.stride(0),
            slot_of,
            cols,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return k8.view(fp8), v8.view(fp8), new_bt


@torch.no_grad()
def minimax_m3_sparse_attn_prefill_aiter(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,
    query_req_id: torch.Tensor,  # [total_q] int32, owning request per token
    query_abs_pos: torch.Tensor,  # [total_q] int32, absolute position per token
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    sparse_bt: torch.Tensor | None = None,
    sparse_ctx: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    total_kv_blocks: int | None = None,
    dqc_groups: list[tuple[int, int, int, int, int]] | None = None,
    dqc_ws_pages: int | None = None,
    dqc_ws_block_table: torch.Tensor | None = None,
    dqc_fused_runs: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """dqc_groups (from minimax_m3_dqc_groups): prefill rows are dequantized
    and attended group by group in one fixed workspace of dqc_ws_pages pages.
    None: one group of total_kv_blocks.

    dqc_ws_block_table (minimax_m3_dqc_ws_block_table over dqc_groups): a
    given sparse_bt was emitted through it and holds workspace page ids, and
    one built here is built through it; no per-layer remap. None: sparse_bt
    holds physical page ids and dqC remaps them per group.

    dqc_fused_runs (from minimax_m3_dqc_hybrid_plan): these rows run the
    fused NVFP4 kernel and dqc_groups cover only the rest. Their rows of
    dqc_ws_block_table hold physical page-16 ids
    (minimax_m3_dqc_mix_block_table).
    """
    if (sparse_bt is None) != (sparse_ctx is None):
        raise ValueError(
            "MiniMax-M3 prepared sparse block table and context lengths "
            "must be provided together"
        )
    dqc = (
        seq_lens is not None
        and max_seq_len is not None
        and total_kv_blocks is not None
        and _is_fp8_kv_cache_tensor(k_cache)
        and minimax_m3_sparse_kv_nvfp4()
        and minimax_m3_sparse_kv_prefill_dq()
        and all(s is None or s.numel() == 1 for s in (k_scale, v_scale))
    )
    if dqc_ws_block_table is not None and not dqc:
        raise ValueError(
            "MiniMax-M3 dqC workspace block table given, but dqC does not "
            "apply to this KV cache"
        )
    if sparse_bt is None:
        sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
            topk_idx,
            block_table if dqc_ws_block_table is None else dqc_ws_block_table,
            query_req_id,
            query_abs_pos,
            (
                minimax_m3_sparse_block_page_stride(k_cache, v_cache)
                if dqc_ws_block_table is None
                else PAGES_PER_SPARSE_BLOCK
            ),
        )
    assert sparse_ctx is not None
    if dqc:
        assert seq_lens is not None and max_seq_len is not None
        assert total_kv_blocks is not None
        total_q = q.shape[0]
        if dqc_groups is None:
            dqc_groups = [(0, seq_lens.shape[0], 0, total_q, total_kv_blocks)]
        kvh = num_kv_heads
        for _, _, q0, q1 in dqc_fused_runs or ():
            if q1 > q0:
                _run_gluon_decode(
                    q[q0:q1],
                    k_cache,
                    v_cache,
                    sparse_bt[q0 * kvh : q1 * kvh],
                    sparse_ctx[q0 * kvh : q1 * kvh],
                    num_kv_heads,
                    sm_scale,
                    output[q0:q1],
                    k_scale,
                    v_scale,
                    split_num_seqs=total_q * kvh,
                )
        if not dqc_groups:
            return
        if dqc_ws_pages is None:
            dqc_ws_pages = max(g[4] for g in dqc_groups) * PAGES_PER_SPARSE_BLOCK
        # Requested at the same size every call: the profiled reservation
        # never grows.
        workspace = minimax_m3_dqc_workspace(dqc_ws_pages)
        one = _one(q.device)
        for r0, r1, q0, q1, blocks in dqc_groups:
            if q1 == q0:
                continue
            g_ctx = sparse_ctx[q0 * kvh : q1 * kvh]
            k_ws, v_ws, ws_bt = minimax_m3_sparse_kv_dequant_context(
                k_cache,
                v_cache,
                block_table[r0:r1],
                seq_lens[r0:r1],
                max_seq_len,
                blocks,
                sparse_bt[q0 * kvh : q1 * kvh],
                g_ctx,
                workspace,
                None if dqc_ws_block_table is None else dqc_ws_block_table[r0:r1],
            )
            # Split count from the whole prefill batch keeps grouped results
            # bit-identical to one group.
            _run_gluon_decode(
                q[q0:q1],
                k_ws,
                v_ws,
                ws_bt,
                g_ctx,
                num_kv_heads,
                sm_scale,
                output[q0:q1],
                one if k_scale is None else k_scale,
                one if v_scale is None else v_scale,
                nvfp4=False,
                split_num_seqs=total_q * kvh,
            )
        return
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_bt,
        sparse_ctx,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )


def _run_gluon_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    nvfp4: bool | None = None,
    split_num_seqs: int | None = None,
) -> None:
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )

    if not q.is_contiguous():
        q = q.contiguous()
    if not output.is_contiguous():
        raise ValueError("MiniMax-M3 AITER sparse PA output must be contiguous")

    total_q, num_q_heads, head_size = q.shape
    if head_size != 128:
        raise ValueError("MiniMax-M3 AITER sparse PA requires head_dim == 128")
    group_size = num_q_heads // num_kv_heads

    nphys16, hkv = k_cache.shape[0], k_cache.shape[1]
    # V is offset by half a block when the sides interleave, so it holds fewer
    # pages than K even though both are addressed by the same page id.
    v_nphys16 = v_cache.shape[0]
    k_cache_view = k_cache.view(nphys16 * hkv, 1, *k_cache.shape[2:])
    v_cache_view = v_cache.view(v_nphys16 * hkv, 1, *v_cache.shape[2:])
    q_view = q.view(total_q * num_kv_heads, group_size, head_size)
    out_view = output.view(total_q * num_kv_heads, group_size, head_size)

    num_seqs = total_q * num_kv_heads
    max_context_partition_num = get_recommended_splits(
        num_seqs if split_num_seqs is None else split_num_seqs, 1
    )
    context_partition_size = 256
    intermediate_shape = (num_seqs, 1, max_context_partition_num, group_size)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        *intermediate_shape, head_size, dtype=q.dtype, device=q.device
    )

    is_fp8 = _is_fp8_kv_cache_tensor(k_cache)
    compute_type = aiter_dtypes.fp8 if is_fp8 else q.dtype
    if is_fp8:
        if _sides_are_packed(k_cache, v_cache) and not all(
            scale is None or scale.numel() == 1 for scale in (k_scale, v_scale)
        ):
            raise NotImplementedError(
                "MiniMax-M3 AITER sparse PA supports only scalar KV scales "
                "when K and V pages share a block; per-token scales would "
                "need the same page renumbering."
            )
        k_scale_arg = _gluon_scale_arg(
            k_scale, num_phys_pages=nphys16, num_kv_heads=hkv
        )
        v_scale_arg = _gluon_scale_arg(
            v_scale, num_phys_pages=v_nphys16, num_kv_heads=hkv
        )
    else:
        k_scale_arg = v_scale_arg = None

    # Only passed when set: stock AITER has no kv_block_scale argument.
    nvfp4_kwargs: dict[str, torch.Tensor] = {}
    if nvfp4 is None:
        nvfp4 = minimax_m3_sparse_kv_nvfp4()
    if is_fp8 and nvfp4:
        if hkv != 1:
            raise ValueError("MiniMax-M3 NVFP4 sparse KV needs one KV head")
        k_cache_view, v_cache_view, nvfp4_kwargs["kv_block_scale"] = (
            _nvfp4_kv_views(k_cache_view, v_cache_view)
        )
        one = _one(q.device)
        k_scale_arg = one if k_scale_arg is None else k_scale_arg
        v_scale_arg = one if v_scale_arg is None else v_scale_arg
        if k_scale_arg.numel() != 1 or v_scale_arg.numel() != 1:
            raise ValueError("MiniMax-M3 NVFP4 sparse KV needs scalar KV scales")

    pa_decode_gluon(
        output=out_view,
        query=q_view,
        key_cache=k_cache_view,
        value_cache=v_cache_view,
        context_lengths=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        query_length=1,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        query_scale=None,
        key_scale=k_scale_arg,
        value_scale=v_scale_arg,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        sliding_window=-1,
        ps=True,
        **nvfp4_kwargs,
    )
