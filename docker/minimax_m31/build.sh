#!/bin/bash
# Build docker/Dockerfile.rocm_minimax_m31: vllm/vllm-openai-rocm:v0.32.0 plus
# the M3.1 optimization commits of vLLM and AITER.
#
# The overlay is every file changed between the base commit (the source of the
# package already in the base image) and the target commit, restricted to the
# installed package trees (vllm/ for vLLM, aiter/ and csrc/ for AITER). Files
# are taken from the commits with `git archive`, so uncommitted worktree edits
# never reach the image.
#
# Usage: docker/minimax_m31/build.sh [image_tag]
# Env (defaults reproduce minimax-m3-mi355x:m31-v032-final-20261021):
#   VLLM_REPO   vLLM git repo, local path or URL (default: this repo)
#   VLLM_BASE   commit of the vllm package in BASE_IMAGE (v0.32.0)
#   VLLM_REF    vLLM commit to install
#   AITER_REPO  AITER git repo, local path or URL
#   AITER_BASE  commit of the aiter package in BASE_IMAGE (v0.1.24 release)
#   AITER_REF   AITER commit to install
#   BASE_IMAGE  base image (default vllm/vllm-openai-rocm:v0.32.0)
#   KEEP_CTX=1  keep the staged build context (path is printed)
set -euo pipefail

TAG=${1:-minimax-m3-mi355x:m31-v032-repro}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)

VLLM_REPO=${VLLM_REPO:-$VLLM_ROOT}
VLLM_BASE=${VLLM_BASE:-5df63a7203cdb9328d824a27b682e77c7b9928be}
VLLM_REF=${VLLM_REF:-9d170f540586751a7f7a2f55ff06547ab47f6628}
AITER_REPO=${AITER_REPO:-$HOME/code/aiter}
AITER_BASE=${AITER_BASE:-04896172e283ecc389f0dfff6479c3379ed63fa1}
AITER_REF=${AITER_REF:-c27755dfa7f423723eb89c565b3d9aaeedf2e0de}
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai-rocm:v0.32.0}

CTX=$(mktemp -d "${TMPDIR:-/tmp}/m31_image_ctx.XXXXXX")
[ "${KEEP_CTX:-0}" = 1 ] || trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/overlay/vllm" "$CTX/overlay/aiter"

# Local path -> use as is; URL -> blobless clone into the context dir.
repo_dir() {
    local name=$1 src=$2
    if [ -d "$src" ]; then
        echo "$src"
    else
        git clone -q --bare --filter=blob:none "$src" "$CTX/$name.git"
        echo "$CTX/$name.git"
    fi
}

# stage <name> <repo> <base> <ref> <pathspec...>
stage() {
    local name=$1 repo=$2 base=$3 ref=$4
    shift 4
    git -C "$repo" cat-file -e "$base^{commit}"
    git -C "$repo" cat-file -e "$ref^{commit}"
    git -C "$repo" diff --name-only --diff-filter=d "$base" "$ref" -- "$@" \
        > "$CTX/overlay/${name}_files.txt"
    git -C "$repo" diff --name-only --diff-filter=D "$base" "$ref" -- "$@" \
        > "$CTX/overlay/${name}_deleted.txt"
    if [ -s "$CTX/overlay/${name}_files.txt" ]; then
        tr '\n' '\0' < "$CTX/overlay/${name}_files.txt" \
            | xargs -0 git -C "$repo" archive "$ref" -- \
            | tar -x -C "$CTX/overlay/$name"
    fi
    echo "$name: $(git -C "$repo" rev-parse --short "$base")..$(git -C "$repo" \
        rev-parse --short "$ref"): $(wc -l < "$CTX/overlay/${name}_files.txt")" \
        "files, $(wc -l < "$CTX/overlay/${name}_deleted.txt") deleted"
}

VREPO=$(repo_dir vllm "$VLLM_REPO")
AREPO=$(repo_dir aiter "$AITER_REPO")
stage vllm "$VREPO" "$VLLM_BASE" "$VLLM_REF" vllm
stage aiter "$AREPO" "$AITER_BASE" "$AITER_REF" aiter csrc
cp "$HERE/build_jit.py" "$CTX/overlay/"
cp "$VLLM_ROOT/docker/Dockerfile.rocm_minimax_m31" "$CTX/Dockerfile"
[ "${KEEP_CTX:-0}" = 1 ] && echo "context: $CTX"

docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --label org.opencontainers.image.base.name="$BASE_IMAGE" \
    --label vllm.ref="$(git -C "$VREPO" rev-parse "$VLLM_REF")" \
    --label aiter.ref="$(git -C "$AREPO" rev-parse "$AITER_REF")" \
    -t "$TAG" "$CTX"
