# MiniMax-M3.1 MI355X image

`vllm/vllm-openai-rocm:v0.32.0` plus the M3.1 vLLM and AITER commits.
Building needs Docker and git, but no GPU. Allow a few minutes: about 80 s of
that is compiling the kernels. If the base image is not already local, it is
pulled first (about 34 GB).

## Build

```bash
# 1. Get both sources. Use full clones, not --depth.
git clone -b sshliapn/features/minimax-m31-optimizations <vllm-repo-url> vllm
git clone -b sshliapn/features/minimax-m31-flash <aiter-repo-url> aiter

# 2. Build.
AITER_REPO=$PWD/aiter vllm/docker/minimax_m31/build.sh minimax-m31:v032

# 3. Optional: check the result against the reference image.
vllm/docker/minimax_m31/verify.sh minimax-m31:v032
```

`verify.sh` prints `matches all 53 reference md5s` when the build is
identical to the reference image.

## Notes

- **Pinned commits.** `build.sh` uses fixed commits by default, so a later
  push to either branch does not change the image. You can override them:
  - `VLLM_REF` / `AITER_REF` set the commit to install from each repo.
  - `VLLM_BASE` / `AITER_BASE` must stay the base image's package commits:
    v0.32.0 for vLLM and the v0.1.24 release commit for AITER.
- **Repo locations.** `AITER_REPO` and `VLLM_REPO` take a local path or a git
  URL. `VLLM_REPO` defaults to the repo that contains this script.
- **No local edits.** Only committed files are used. Uncommitted edits are
  ignored.
- **Entrypoint.** The image's entrypoint is `/usr/bin/env`, so a container
  must be started with the full command:

  ```bash
  docker run ... minimax-m31:v032 vllm serve <model> ...
  ```

## Serving: prefill (P) and decode (D) servers

These are the best configurations from the 2026-10-21/22 MI355X sweeps, all
at TP4 (decode: 1k-90k context, C1-128; prefill: 1k-90k context, C1-256).

- **P** is for prefill-heavy traffic. It uses AITER MXFP4 MoE with prefix
  caching on.
- **D** is for decode-heavy traffic. It uses the HIP NVFP4 MoE and turns the
  shared-expert stream off.

Both use NVFP4 sparse KV and NVFP4 indexer KV with compact pages, so their
KV cache layouts match. P/D disaggregation (KV transfer between the two
servers) has not been tested.

```bash
IMG=minimax-m31:v032
MODEL=<path to the MiniMax-M3.1 checkpoint>
CACHE=<host dir>  # kernel caches; reuse it across restarts to skip recompiles

COMMON_RUN=(
  --network=host --ipc=host --shm-size=64g
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render
  -v "$MODEL:$MODEL:ro" -v "$CACHE:$CACHE"
  -e FLYDSL_RUNTIME_CACHE_DIR=$CACHE/flydsl -e TRITON_CACHE_DIR=$CACHE/triton
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 -e SAFETENSORS_FAST_GPU=1
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
  -e VLLM_ROCM_MXFP8_AITER_BPRESHUFFLE=1
  -e AITER_MSA_INDEX_KV_FMT=nvfp4
  -e AITER_MSA_PREFILL_CFG=2,8,6 -e AITER_MSA_PREFILL_CHUNK_BLOCKS=4
  -e MINIMAX_M3_SPARSE_KV_FMT=nvfp4 -e MINIMAX_M3_NVFP4_COMPACT_PAGES=1
  -e VLLM_M3_NVFP4_MOE_HIP_MAX_T=1024
  -e VLLM_M3_NVFP4_MOE_MXFP4_SCALE=roundup
  -e GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0
)
COMMON_SERVE=(
  --language-model-only --tensor-parallel-size=4
  --max-model-len=131072 --block-size=128 --kv-cache-dtype=fp8
  --gpu-memory-utilization=0.85 --max-num-seqs=256
  --max-num-batched-tokens=32768
  --attention-config='{"backend":"ROCM_AITER_UNIFIED_ATTN","indexer_kv_dtype":"fp8"}'
  --moe-backend=aiter
  --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}'
  --renderer-num-workers=8
)

# P server
docker run -d --name m31-p "${COMMON_RUN[@]}" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_M3_NVFP4_MOE_HIP=0 -e VLLM_M3_NVFP4_MOE_BASELINE_FMT=mxfp4 \
  -e VLLM_M3_NVFP4_MOE_PREFILL=mxfp8 -e VLLM_M3_NVFP4_MOE_KEEP_MXFP8=1 \
  "$IMG" vllm serve "$MODEL" --port 8000 "${COMMON_SERVE[@]}" \
  --enable-prefix-caching

# D server
docker run -d --name m31-d "${COMMON_RUN[@]}" \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 \
  -e VLLM_M3_NVFP4_MOE_HIP=1 -e VLLM_M3_NVFP4_MOE_BASELINE_FMT=mxfp8 \
  -e VLLM_M3_NVFP4_MOE_PREFILL=hip -e VLLM_M3_NVFP4_MOE_KEEP_MXFP8=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
  "$IMG" vllm serve "$MODEL" --port 8001 "${COMMON_SERVE[@]}" \
  --no-enable-prefix-caching
```

Notes:

- **KV formats come from env.** `MINIMAX_M3_SPARSE_KV_FMT` and
  `AITER_MSA_INDEX_KV_FMT` select NVFP4. Keep `--kv-cache-dtype=fp8` and
  `"indexer_kv_dtype":"fp8"` on the command line as shown.
- **NUMA.** Pin each server to the NUMA node of its GPUs with `--cpuset-cpus`
  and `--cpuset-mems`. The sweeps used HIP 0-3 on node 0 and HIP 4-7 on
  node 1.
