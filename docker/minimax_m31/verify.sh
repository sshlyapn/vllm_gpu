#!/bin/bash
# Compare the overlaid files and rebuilt JIT .so of an image against
# reference_md5.txt (minimax-m3-mi355x:m31-v032-final-20261021 with the
# fused_qknorm_idxrqknorm kernel from AITER c27755dfa).
# Needs no GPU. Exit status is non-zero on any mismatch.
#
# Usage: docker/minimax_m31/verify.sh <image>
set -euo pipefail
IMAGE=${1:?usage: verify.sh <image>}
REF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reference_md5.txt

out=$(docker run --rm -i --entrypoint bash "$IMAGE" -c '
SP=/usr/local/lib/python3.12/dist-packages
while read -r kind sum name; do
  case $kind:$name in
    OK:aiter:aiter/*) p=$SP/${name#aiter:} ;;
    OK:aiter:csrc/*) p=$SP/aiter_meta/${name#aiter:} ;;
    OK:vllm:*) p=$SP/${name#vllm:} ;;
    SO:*) p=$SP/aiter/jit/$name ;;
  esac
  echo "$kind $(md5sum < "$p" 2>/dev/null | cut -c1-32) $name"
done' < "$REF")

if diff "$REF" <(echo "$out"); then
    echo "verify: $IMAGE matches all $(wc -l < "$REF") reference md5s"
else
    echo "verify: $IMAGE differs from reference (diff above)" >&2
    exit 1
fi
