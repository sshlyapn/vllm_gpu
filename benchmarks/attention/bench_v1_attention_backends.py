"""
Benchmark v1 attention backends using main logic from tests, with optional
accuracy validation and torch.profiler-based timing.

The test builds synthetic workloads similar to
tests/v1/attention/test_attention_backends.py:
- Generates batch shapes (seq_lens, query_lens)
- Computes SDPA baseline via torch FlexAttention with appropriate masks
- Simulates paged KV cache and realistic slot_mapping
- Runs each backend implementation
- Optionally validates accuracy vs baseline
- Profiles runtime using torch.profiler (CUDA)

Usage example:
  HF_TOKEN=<token> python -m bench_v1_attention_backends.py \
    --model meta-llama/Meta-Llama-3-8B --batch-spec custom_single_prefill_small \
    --backends ROCM_AITER_FA --dtype float16 \
    --accuracy --profile --print-kernels-table
"""

from __future__ import annotations

import argparse
from functools import partial
import os
import time
import re
from typing import Optional, Union

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

try:
    from vllm.attention.backends.registry import (
        _Backend,
        backend_name_to_enum,
    )
except:
    from vllm.platforms import _Backend
    from vllm.attention.selector import backend_name_to_enum

from vllm.config import ModelConfig
from vllm.platforms import current_platform

try:
    from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv, is_torch_equal_or_newer
except:
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, is_torch_equal_or_newer
    from vllm.utils import cdiv

from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    set_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

# if import failed, just copy utils.py near to the current script
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    try_get_attention_backend,
)


def _convert_dtype_to_torch(dtype):
    """Convert ModelDType to torch.dtype."""
    if isinstance(dtype, str):
        if dtype == "float16":
            return torch.float16
        elif dtype in STR_DTYPE_TO_TORCH_DTYPE:
            return STR_DTYPE_TO_TORCH_DTYPE[dtype]
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


# ----------------------------- Debugging helpers -----------------------------


def _tensor_stats_str(name: str, t: torch.Tensor) -> str:
    t32 = t.detach().float()
    finite_mask = torch.isfinite(t32)
    if finite_mask.any():
        t32f = t32[finite_mask]
        t_min = t32f.min().item()
        t_max = t32f.max().item()
        t_mean = t32f.mean().item()
        # std can be nan for 1 element; guard with max(numel, 2)
        t_std = (t32f.std(unbiased=False) if t32f.numel() > 1 else torch.tensor(0.0, device=t32f.device)).item()
        t_absmax = t32f.abs().max().item()
        finite_frac = float(finite_mask.float().mean().item())
    else:
        t_min = float("nan")
        t_max = float("nan")
        t_mean = float("nan")
        t_std = float("nan")
        t_absmax = float("nan")
        finite_frac = 0.0
    return (
        f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, "
        f"min={t_min:.4g}, max={t_max:.4g}, mean={t_mean:.4g}, std={t_std:.4g}, absmax={t_absmax:.4g}, "
        f"finite={finite_frac*100:.2f}%"
    )


def _compare_tensors_str(label: str, test_t: torch.Tensor, ref_t: torch.Tensor, rtol: float, atol: float) -> str:
    # Compute error metrics on finite elements only
    test32 = test_t.detach().float()
    ref32 = ref_t.detach().float()
    finite_mask = torch.isfinite(test32) & torch.isfinite(ref32)
    if not finite_mask.any():
        return f"{label}: no finite elements to compare"

    a = test32[finite_mask]
    b = ref32[finite_mask]
    diff = (a - b)
    abs_diff = diff.abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    ref_absmax = b.abs().max().item()
    rel_max = (max_abs / (ref_absmax + 1e-12)) if ref_absmax != 0.0 else float("inf")

    return (
        f"{label}: max_abs={max_abs:.4g}, mean_abs={mean_abs:.4g}, rmse={rmse:.4g}, "
        f"rel_max={rel_max*100:.2f}% of |ref_max|"
    )

# Define common batch configurations
BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1, 1, 1, 1, 1],
    ),
    "medium_prefill": BatchSpec(
        seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]
    ),
    "mixed_medium": BatchSpec(
        seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[1, 1, 1, 7, 7, 7]
    ),
    "large_decode": BatchSpec(seq_lens=[4096] * 32, query_lens=[1] * 32),
    "large_prefill": BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[4096], query_lens=[64]),

    "custom_single_prefill_small": BatchSpec(seq_lens=[128], query_lens=[128]),
    "custom_single_prefill_medium": BatchSpec(seq_lens=[2048], query_lens=[2048]),
    "custom_single_prefill_large": BatchSpec(seq_lens=[8192], query_lens=[8192]),
    "custom_single_decode_small": BatchSpec(seq_lens=[128], query_lens=[1]),
    "custom_single_decode_medium": BatchSpec(seq_lens=[2048], query_lens=[1]),
    "custom_single_decode_large": BatchSpec(seq_lens=[8192], query_lens=[1]),

    "custom_medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1, 1, 1, 1, 1],
    ),
    "custom_medium_prefill": BatchSpec(
        seq_lens=[256, 512, 1024, 2048], query_lens=[256, 512, 1024, 2048]
    ),

    "custom_mixed_medium": BatchSpec(
        seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[1, 1, 1, 128, 512, 1024]
    ),
    "custom_mixed_large": BatchSpec(
        seq_lens=[2048, 4096, 4096, 4096, 4096, 4096], query_lens=[1, 1, 1, 2048, 2048, 1024]
    ),
}


# ---------------------------- Metadata + KV cache ----------------------------


SUPPORTED_BACKENDS = [
    "ROCM_AITER_FA",
    "ROCM_AITER_UNIFIED_ATTN",
    "ROCM_ATTN",
    "TRITON_ATTN",
    "FLEX_ATTENTION",
]

def create_and_prepopulate_kv_cache(
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: CommonAttentionMetadata,
    randomize_blocks: bool = True,
) -> torch.Tensor:
    """Create and prepopulate a KV cache with context data.

    Args:
        k_contexts: List of key context tensors for each sequence
        v_contexts: List of value context tensors for each sequence
        seq_lens: List of sequence lengths
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        block_table: Block table tensor to populate
        randomize_blocks: Whether to randomly permute blocks
                          or use sequential order

    Returns:
        Tuple of (kv_cache, updated_block_table)
    """
    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # Create KV cache
    kv_cache = torch.empty(
        2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    kv_cache_flat = kv_cache.view(2, -1, num_kv_heads, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[0, start:end, ...] = k_context
        kv_cache_flat[1, start:end, ...] = v_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        # Random permutation starting from block 1
        perm = torch.randperm(blocks_end - 1) + 1
    else:
        # Sequential order starting from block 1
        perm = torch.arange(1, blocks_end)

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    # Add 1 to account for starting from block 1
    inv_perm[1:] = torch.argsort(perm) + 1
    kv_cache[:, 1:blocks_end, ...] = kv_cache[:, perm, ...]

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i, block_indices
        ] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def run_attention_backend(
    backend: _Backend | str,
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    sliding_window: Optional[int] = None,
):
    actual_backend = backend
    use_direct_block_mask = is_torch_equal_or_newer("2.9.0.dev0")
    if backend == "FLEX_ATTENTION_SLOW":
        actual_backend = _Backend.FLEX_ATTENTION
        use_direct_block_mask = False

    if isinstance(actual_backend, str):
        enum_val = backend_name_to_enum(actual_backend)
        if enum_val is None:
            raise ValueError(f"Unknown backend: {actual_backend}")
        actual_backend = enum_val

    builder_cls, impl_cls = try_get_attention_backend(actual_backend)

    if actual_backend == _Backend.FLASHINFER:
        import unittest.mock
        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            head_size = vllm_config.model_config.get_head_size()
            return {
                layer_name: PerLayerParameters(
                    window_left=-1,
                    logits_soft_cap=0.0,
                    sm_scale=1.0 / (head_size**0.5),
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            mock_get_per_layer_parameters,
        ):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
    else:
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        if actual_backend == _Backend.FLEX_ATTENTION:
            setattr(builder, "direct_build", use_direct_block_mask)
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)

    num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)

    return impl.forward(mock_layer, query, key, value, kv_cache, attn_metadata, output=output)


# -------------------------------- Baseline SDPA ------------------------------


def compute_sdpa_baseline(
    *,
    batch_spec: BatchSpec,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    mask_mod_builder,
):
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    k_contexts, v_contexts = [], []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)

        q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            repeats = num_q_heads // num_kv_heads
            k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
            v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)

        kv_len = s_len
        final_mask_mod = partial(mask_mod_builder, context_len=context_len)
        block_mask = create_block_mask(final_mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)

        scale = 1.0 / (head_size**0.5)
        sdpa_out_i = flex_attention(
            q_sdpa_in,
            k_sdpa_in,
            v_sdpa_in,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
        )

        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)
    return query_vllm, key_vllm, value_vllm, k_contexts, v_contexts, sdpa_output


# --------------------------------- Main bench --------------------------------


def build_vllm_config(model: str, max_model_len: int, requests_num : int, block_size : int, dtype: torch.dtype):
    gpu_blocks = (cdiv(max_model_len, block_size) * requests_num) + 1
    
    return create_vllm_config(
        model_name=model,
        max_model_len=max_model_len,
        dtype=dtype,
        num_gpu_blocks=gpu_blocks,
        block_size=block_size,
    )


def build_kv_cache_spec(vllm_config) -> FullAttentionSpec:
    return create_standard_kv_cache_spec(vllm_config)


def make_common_metadata(batch_spec: BatchSpec, vllm_config, device: torch.device) -> CommonAttentionMetadata:
    return create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)


def get_mask_mod_builder(mode: str, sliding_window: Optional[int]):
    if mode == "causal":
        def causal_mask_mod(b, h, q_idx, kv_idx, *, context_len: int):
            return (q_idx + context_len) >= kv_idx

        return causal_mask_mod

    if mode == "sliding":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError("sliding_window must be > 0 for sliding mode")

        def sliding_window_mask_mod(b, h, q_idx, kv_idx, *, context_len: int, sliding_window: int):
            causal_mask = q_idx + context_len >= kv_idx
            window_mask = q_idx + context_len - kv_idx < sliding_window
            return causal_mask & window_mask

        return partial(sliding_window_mask_mod, sliding_window=sliding_window)

    raise ValueError(f"Unknown mode: {mode}")


def kv_cache_for_backend(original_kv_cache: torch.Tensor, backend: _Backend) -> torch.Tensor:
    kv_cache = original_kv_cache
    if backend == _Backend.FLASHINFER:
        kv_cache = kv_cache.transpose(0, 1)
        kv_cache = kv_cache.transpose(2, 3).contiguous().transpose(2, 3)
        set_kv_cache_layout("HND")
    elif backend == _Backend.TRITON_ATTN:
        kv_cache = kv_cache.transpose(0, 1).contiguous()
    return kv_cache


def print_attention_config_for_models(models: list[str]) -> None:
    """Group models by unique attention configuration and print the mapping.

    """
    config_to_models: dict[tuple, list[str]] = {}

    def _dtype_to_short_str(dtype) -> str:
        if isinstance(dtype, str):
            return dtype
        s = str(dtype)
        if s.startswith("torch."):
            return s.split(".", 1)[1]
        return s

    for model_name in models:
        print(f"Processing model: {model_name}")
        block_size = 16
        max_model_len = 1
        requests_num = 1
        vcfg = build_vllm_config(model_name, max_model_len, requests_num, block_size, "auto")
        num_q_heads = vcfg.model_config.get_num_attention_heads(vcfg.parallel_config)
        num_kv_heads = vcfg.model_config.get_num_kv_heads(vcfg.parallel_config)
        head_size = vcfg.model_config.get_head_size()
        sw = vcfg.model_config.get_sliding_window()
        dtype_str = _dtype_to_short_str(vcfg.model_config.dtype)
        cfg_key = (
            dtype_str,
            head_size,
            num_q_heads,
            num_kv_heads,
            -1 if sw is None else int(sw),
        )
        config_to_models.setdefault(cfg_key, []).append(model_name)

    print("\n================ Unique attention configurations ================")
    for idx, (cfg, model_list) in enumerate(config_to_models.items(), start=1):
        dtype_str, head_size, num_heads, num_kv_heads, sw = cfg
        sw_str = "None" if sw == -1 else str(sw)
        print(f"[{idx}] dtype={dtype_str}, head_size={head_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, sliding_window={sw_str}")
        print("    models:")
        for m in model_list:
            print(f"      - {m}")
        # Print a representative command part to reproduce this configuration
        representative_model = model_list[0]
        mode_part = "--mode causal" if sw == -1 else f"--mode sliding --sliding-window {sw}"
        print(f"    cmd: --model {representative_model} {mode_part} --dtype {dtype_str}")
    print("================================================================\n")


def profile_backend(
    *,
    label: str,
    iters: int,
    warmup: int,
    fn,
    trace_file: Optional[str] = None,
    print_table: bool = False,
) -> dict:
    activities = [torch.profiler.ProfilerActivity.CUDA]
    # Only include active iterations in the exported trace
    schedule = torch.profiler.schedule(wait=0, warmup=warmup, active=iters, repeat=1)

    prof = torch.profiler.profile(activities=activities, schedule=schedule)
    prof.__enter__()
    try:
        # Warmup iterations (excluded from trace by schedule)
        for _ in range(warmup):
            with torch.profiler.record_function(label):
                torch.cuda.synchronize()
                fn()
                torch.cuda.synchronize()
            prof.step()

        # Active iterations (included in trace)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iters):
            with torch.profiler.record_function(label):
                torch.cuda.synchronize()
                fn()
                torch.cuda.synchronize()
            prof.step()
        torch.cuda.synchronize()
        total_s = time.perf_counter() - start_time
    finally:
        prof.__exit__(None, None, None)

    if trace_file:
        try:
            print(f"Exporting chrome trace to {trace_file}")
            prof.export_chrome_trace(trace_file)
        except Exception:
            pass

    if print_table:
        perf_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        cpu_total_match = re.search(r"Self CPU time total:\s*([\d\.]+[a-z]+)", perf_table)
        cuda_total_match = re.search(r"Self CUDA time total:\s*([\d\.]+[a-z]+)", perf_table)

        cpu_total = cpu_total_match.group(1) if cpu_total_match else None
        cuda_total = cuda_total_match.group(1) if cuda_total_match else None

        print(f"Self CPU time total: {cpu_total}")
        print(f"Self CUDA time total: {cuda_total}")

        print(perf_table)

    try:
        per_iter_ms = (total_s / max(iters, 1)) * 1000.0 * 1000.0
        print(f"E2E [{label}] time: total={total_s:.3f}s, per_iter={per_iter_ms:.3f}us (iters={iters}, warmup={warmup})")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Benchmark v1 attention backends with optional accuracy and profiling")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--batch-spec", type=str, default="small_decode", choices=list(BATCH_SPECS.keys()))
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--backends",
        type=str,
        default=SUPPORTED_BACKENDS[0],
        help=f"Comma-separated backend names from _Backend enum. Supported: {','.join(SUPPORTED_BACKENDS)} (default: {SUPPORTED_BACKENDS[0]})",
    )
    parser.add_argument("--mode", type=str, choices=["causal", "sliding"], default="causal")
    parser.add_argument("--sliding-window", type=int, default=-1, help="Sliding window length if mode=sliding")
    parser.add_argument("--randomize-blocks", action="store_true", help="Randomize KV block physical layout")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--accuracy", action="store_true", help="Validate outputs vs SDPA baseline")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-file", type=str, default="", help="Export chrome trace to this path if provided")
    parser.add_argument("--print-kernels-table", action="store_true", help="Print per-op kernel statistics from profiler")
    parser.add_argument("--dry-run", action="store_true", help="Only load model configs, print attention configuration, then exit")
    args = parser.parse_args()

    # We run synthetic attention benchmark that can exceed the model's
    # derived max position (e.g., max_position_embeddings), while not actually
    # executing full model forward. Allow overriding for benchmarking only.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    device = torch.device("cuda:0")
    current_platform.seed_everything(42)
    dtype = _convert_dtype_to_torch(args.dtype) if args.dtype != "auto" else "auto"

    # Dry-run: print info for models and exit
    if args.dry_run:
        models = [
            "facebook/opt-125m",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",  
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct", 
            "mistralai/Magistral-Small-2506",
            "mistralai/Mistral-7B-v0.1",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "microsoft/phi-4",
            "microsoft/Phi-3-medium-4k-instruct",
            "microsoft/Phi-3.5-MoE-instruct",
            "LLM360/K2-Think",
            "deepseek-ai/deepseek-llm-7b-base",
            "deepseek-ai/deepseek-llm-67b-base",
        ]
        print_attention_config_for_models(models)
        return

    batch_spec = BATCH_SPECS[args.batch_spec]
    vllm_config = build_vllm_config(args.model, max(batch_spec.seq_lens) + max(batch_spec.query_lens), len(batch_spec.seq_lens), args.block_size, dtype)
    kv_cache_spec = build_kv_cache_spec(vllm_config)

    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    sliding_window = None if args.mode == "causal" else (args.sliding_window if args.sliding_window > 0 else None)

    mask_mod_builder = get_mask_mod_builder(args.mode, sliding_window)

    query_vllm, key_vllm, value_vllm, k_contexts, v_contexts, sdpa_output = compute_sdpa_baseline(
        batch_spec=batch_spec,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        mask_mod_builder=mask_mod_builder,
    )

    common_attn_metadata = make_common_metadata(batch_spec, vllm_config, device)

    print(f"vllm_config.cache_config.num_gpu_blocks: {vllm_config.cache_config.num_gpu_blocks}")

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=kv_cache_spec.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1000,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=args.randomize_blocks,
    )

    context_lens_list = [int(s) - int(q) for s, q in zip(batch_spec.seq_lens, batch_spec.query_lens)]
    print("\n=========================================================")
    print("============== Attention layer information ==============")
    print(f"  Model: {args.model}")
    print(f"  Head size: {head_size}, Num heads: {num_q_heads}, Num KV heads: {num_kv_heads}")
    print(f"  Block size: {kv_cache_spec.block_size}")
    print(f"  Q dtype: {query_vllm.dtype}, K dtype: {key_vllm.dtype}, V dtype: {value_vllm.dtype}")
    print(f"  KV cache dtype: {kv_cache.dtype}, shape: {tuple(kv_cache.shape)}")
    print(f"  Masking: causal={args.mode == 'causal'}, sliding_window={sliding_window if sliding_window is not None else 'None'}")
    print(f"  Batch spec: {batch_spec.name}, batch_size: {batch_spec.batch_size}")
    print(f"  Seq lens: {list(batch_spec.seq_lens)}")
    print(f"  Query lens: {list(batch_spec.query_lens)}")
    print(f"  Context lens: {context_lens_list}")
    print("=========================================================")

    backend_names = [name.strip() for name in args.backends.split(",") if name.strip()]

    for backend_name in backend_names:
        backend_enum = backend_name_to_enum(backend_name)
        if backend_enum is None:
            print(f"[WARN] Unknown backend '{backend_name}', skipping")
            continue

        print(f"\n>> Running backend {backend_name} with batch={batch_spec.name}, block_size={kv_cache_spec.block_size}, dtype={dtype}")

        kv_cache_for_run = kv_cache_for_backend(kv_cache, backend_enum) if backend_enum else kv_cache
        print(f">> Backend {backend_name}: KV cache shape for run: {tuple(kv_cache_for_run.shape)}")

        def do_forward():
            return run_attention_backend(
                backend=backend_name if backend_enum is None else backend_enum,
                kv_cache_spec=kv_cache_spec,
                layer_names=["placeholder"],
                vllm_config=vllm_config,
                device=device,
                common_attn_metadata=common_attn_metadata,
                query=query_vllm,
                key=key_vllm,
                value=value_vllm,
                kv_cache=kv_cache_for_run,
                sliding_window=sliding_window,
            )

        with torch.no_grad():
            out = do_forward()

        if args.accuracy:
            ANSI_GREEN = "\033[92m"
            ANSI_RED = "\033[91m"
            ANSI_RESET = "\033[0m"

            print("\n-- Accuracy diagnostics --")
            try:
                print(_tensor_stats_str("Q", query_vllm))
                print(_tensor_stats_str("K", key_vllm))
                print(_tensor_stats_str("V", value_vllm))
                print(_tensor_stats_str("Output(test)", out))
                print(_tensor_stats_str("Output(ref)", sdpa_output))
                print(_compare_tensors_str("Output vs Ref", out, sdpa_output, args.rtol, args.atol))
            except Exception as diag_e:
                print(f"[WARN] Failed to print diagnostics: {diag_e}")

            try:
                torch.testing.assert_close(out, sdpa_output, rtol=args.rtol, atol=args.atol)
                acc_msg = f"{ANSI_GREEN}OK{ANSI_RESET}"
            except AssertionError as e:
                acc_msg = f"{ANSI_RED}FAIL{ANSI_RESET}: {e}"
            print(f"Accuracy [{backend_name}]: {acc_msg}")

        if args.profile:
            profile_backend(
                label=f"attn_forward_profiling/{backend_name}",
                iters=args.iters,
                warmup=args.warmup,
                fn=do_forward,
                trace_file=(args.trace_file or None),
                print_table=args.print_kernels_table,
            )

if __name__ == "__main__":
    main()
