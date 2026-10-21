# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 AITER fused QK-norm + RoPE + NVFP4 insert vs the unfused path.

The reference is what the model runs on AITER builds without NVFP4 output: the
HIP fused_minimax_m3_qknorm_rope_kv_insert op (q / index_q out, k / v / index_k
written back into qkv), then minimax_m3_insert_kv_nvfp4 or AITER
reshape_and_cache(asm_layout=True) for K/V, and the AITER MSA NVFP4 insert or
minimax_m3_insert_index_cache for the index key. Every output must match bit
for bit.

Run:  pytest tests/kernels/test_minimax_m3_aiter_qknorm_nvfp4.py -v
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 AMD fused ops require ROCm.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

import vllm._custom_ops as ops  # noqa: E402
from vllm._aiter_ops import (  # noqa: E402
    _aiter_fused_qknorm_supports_nvfp4,
    rocm_aiter_ops,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import (  # noqa: E402
    minimax_m3_compact_kv_views,
    minimax_m3_insert_index_cache,
    minimax_m3_insert_kv_nvfp4,
)

if not _aiter_fused_qknorm_supports_nvfp4():
    pytest.skip("AITER fused QK-norm without NVFP4 output", allow_module_level=True)

from aiter import reshape_and_cache  # noqa: E402
from aiter.ops.msa_block_select import KV_FMT_NVFP4  # noqa: E402
from aiter.ops.triton.msa_index_cache import msa_index_cache_insert  # noqa: E402

DEVICE = "cuda"
NQ, NKV, NIQ, D, ROT = 16, 1, 1, 128, 64
EPS = 1e-6
BLOCK = 128  # main KV block size
NUM_BLOCKS = 80
IDX_BLOCK = 64
IDX_PAGES = 160
FP8 = torch.float8_e4m3fn


def _slots(num_tokens, capacity, gen):
    s = torch.randperm(capacity, generator=gen)[:num_tokens]
    s[torch.rand(num_tokens, generator=gen) < 0.2] = -1
    return s.to(DEVICE)


def _kv_caches(kv_fmt):
    if kv_fmt == "nvfp4_compact":
        kv = torch.zeros(NUM_BLOCKS, 2, BLOCK, 72, dtype=torch.uint8, device=DEVICE)
        kv.random_()
        k, v = minimax_m3_compact_kv_views(kv.view(FP8))
        return kv, k, v
    # Block-contiguous K/V sides, V offset by half a block (see the model's
    # _ensure_aiter_sparse_pa_kv_cache).
    kv = torch.zeros(NUM_BLOCKS, 2, BLOCK, NKV * D, dtype=torch.uint8, device=DEVICE)
    kv.random_()
    kv8 = kv.view(FP8)
    pages = NUM_BLOCKS * 2 * BLOCK // 16
    k = kv8.view(pages, NKV, D // 16, 16, 16)
    v = kv8.view(pages, NKV, 1, D, 16)[BLOCK // 16 :]
    return kv, k, v


def _index_cache(idx_fmt):
    if idx_fmt == "bf16":
        c = torch.randn(IDX_PAGES, IDX_BLOCK, D, device=DEVICE)
        return c.to(torch.bfloat16), IDX_BLOCK
    if idx_fmt == "nvfp4_2d":
        # Padded pages of the 128-token 2D layout.
        shape, block = (IDX_PAGES // 2, 128 * 72 + 64), 128
    else:
        shape, block = (IDX_PAGES, IDX_BLOCK, D), IDX_BLOCK
    c = torch.zeros(shape, dtype=torch.uint8, device=DEVICE)
    c.random_()
    return c.view(FP8), block


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128, 1031])
@pytest.mark.parametrize("kv_fmt", ["nvfp4", "nvfp4_compact", "fp8"])
@pytest.mark.parametrize("idx_fmt", ["nvfp4_3d", "nvfp4_2d", "fp8", "bf16", "skip"])
@pytest.mark.parametrize("scales", ["none", "ones", "custom"])
@torch.inference_mode()
def test_aiter_qknorm_nvfp4_bitwise(num_tokens, kv_fmt, idx_fmt, scales):
    kv_nvfp4 = kv_fmt != "fp8"
    index_nvfp4 = idx_fmt.startswith("nvfp4")
    if not kv_nvfp4 and scales == "none":
        pytest.skip("fp8 K/V needs scales")
    gen = torch.Generator().manual_seed(num_tokens * 131 + len(kv_fmt + idx_fmt))
    torch.manual_seed(num_tokens)
    row = (NQ + 2 * NKV + NIQ + 1) * D
    qkv = (torch.randn(num_tokens, row, device=DEVICE) * 3).to(torch.bfloat16)
    # Some rows far outside the fp8 / NVFP4 grids, some near zero.
    qkv[::5] *= 300
    qkv[1::7] *= 1e-4
    w = [(torch.randn(D, device=DEVICE) * 0.5).to(torch.bfloat16) for _ in range(4)]
    cos_sin = torch.randn(70000, ROT, device=DEVICE).to(torch.bfloat16)
    positions = torch.randint(0, 70000, (num_tokens,), device=DEVICE)
    k_scale = v_scale = None
    if scales == "ones":
        k_scale = v_scale = torch.ones(1, device=DEVICE)
    elif scales == "custom":
        k_scale = torch.tensor([0.37], device=DEVICE)
        v_scale = torch.tensor([1.9], device=DEVICE)

    kv_ref, k_ref, v_ref = _kv_caches(kv_fmt)
    kv_new = kv_ref.clone()
    k_new, v_new = (
        kv_new.view(FP8).as_strided(t.shape, t.stride(), t.storage_offset())
        for t in (k_ref, v_ref)
    )
    slots = _slots(num_tokens, NUM_BLOCKS * BLOCK, gen)
    if kv_fmt == "nvfp4_compact":
        page16 = slots
    else:
        page16 = torch.where(slots >= 0, slots + slots // BLOCK * BLOCK, slots)

    skip = idx_fmt == "skip"
    if skip:
        idx_ref = idx_new = idx_slots = None
    else:
        idx_ref, idx_block = _index_cache(idx_fmt)
        idx_new = idx_ref.clone()
        idx_slots = _slots(num_tokens, idx_ref.shape[0] * idx_block, gen)
    iq_dtype = torch.bfloat16 if idx_fmt == "bf16" else FP8

    # Reference: HIP op, then the separate inserts.
    norm_args = (w[0], w[1], cos_sin, positions, NQ, NKV, ROT, EPS)
    qkv_ref = qkv.clone()
    q_ref = torch.empty(num_tokens, NQ * D, dtype=torch.bfloat16, device=DEVICE)
    iq_ref = torch.empty(num_tokens, NIQ * D, dtype=iq_dtype, device=DEVICE)
    if skip:
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv_ref,
            *norm_args,
            num_index_heads=NIQ,
            q_out=q_ref,
            skip_index_branch=True,
        )
    else:
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv_ref,
            *norm_args,
            w[2],
            w[3],
            NIQ,
            q_out=q_ref,
            index_q_out=iq_ref,
            kv_cache_dtype="fp8",
        )
    k = qkv_ref[:, NQ * D : (NQ + NKV) * D].view(num_tokens, NKV, D)
    v = qkv_ref[:, (NQ + NKV) * D : (NQ + 2 * NKV) * D].view(num_tokens, NKV, D)
    if kv_nvfp4:
        minimax_m3_insert_kv_nvfp4(k, v, k_ref, v_ref, page16, k_scale, v_scale)
    else:
        reshape_and_cache(
            k.contiguous(),
            v.contiguous(),
            k_ref,
            v_ref,
            page16,
            kv_cache_dtype="fp8",
            k_scale=k_scale,
            v_scale=v_scale,
            asm_layout=True,
        )
    if not skip:
        index_k = qkv_ref[:, (NQ + 2 * NKV + NIQ) * D :]
        if index_nvfp4:
            msa_index_cache_insert(index_k, idx_ref, idx_slots, KV_FMT_NVFP4)
        else:
            minimax_m3_insert_index_cache(index_k, idx_ref, idx_slots)

    q_new = torch.empty_like(q_ref)
    iq_new = torch.empty_like(iq_ref)
    qkv_in = qkv.clone()
    rocm_aiter_ops.fused_qknorm_idxrqknorm(
        qkv,
        *norm_args,
        page16,
        k_new,
        v_new,
        q_new,
        "fp8",
        k_scale,
        v_scale,
        w[2],
        w[3],
        NIQ,
        idx_new,
        None if skip else iq_new,
        idx_slots,
        skip_index_branch=skip,
        kv_nvfp4=kv_nvfp4,
        index_nvfp4=index_nvfp4,
    )

    def _bits(t):
        return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)

    assert torch.equal(qkv, qkv_in)
    assert torch.equal(_bits(q_new), _bits(q_ref))
    if not skip:
        assert torch.equal(_bits(iq_new), _bits(iq_ref))
        assert torch.equal(_bits(idx_new), _bits(idx_ref))
    mism = (kv_new != kv_ref).sum().item()
    assert mism == 0, f"{mism} KV cache bytes differ"


def test_aiter_qknorm_nvfp4_enabled(monkeypatch):
    import vllm._aiter_ops as aiter_ops

    one = torch.ones(1)
    enabled = rocm_aiter_ops.fused_qknorm_idxrqknorm_enabled
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    assert enabled("fp8", None, None, kv_nvfp4=True, index_nvfp4=True)
    assert enabled("fp8", one, one, kv_nvfp4=True)
    assert not enabled("fp8", one, None, kv_nvfp4=True)
    assert enabled("fp8", one, one, index_nvfp4=True)
    assert not enabled("fp8", None, None, index_nvfp4=True)
    monkeypatch.setattr(aiter_ops, "_aiter_fused_qknorm_supports_nvfp4", lambda: False)
    assert not enabled("fp8", one, one, kv_nvfp4=True)
    assert not enabled("fp8", one, one, index_nvfp4=True)
    assert enabled("fp8", one, one)
