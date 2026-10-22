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

`verify.sh` prints `matches all 52 reference md5s` when the build is
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
