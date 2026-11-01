#!/bin/sh
# Compiles the CUDA kernels to PTX with clang (no CUDA toolkit needed) for a
# Maxwell and an Ada GPU, and fails if any kernel would spill to local memory:
# the walk's registers, program and BLAKE3 state must all stay in registers.
set -eu
cd "$(dirname "$0")"
CLANG=$(command -v clang++-18 || command -v clang++)
"$CLANG" --version | head -n 1
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
for arch in sm_52 sm_89; do
  "$CLANG" -x cuda --cuda-device-only --cuda-gpu-arch="$arch" -nocudainc -nocudalib \
    -O3 -std=c++14 -Wall -Wextra -S kernels.cu -o "$out/$arch.ptx"
  kernels=$(grep -c '\.entry' "$out/$arch.ptx")
  local_lines=$(grep -cE '\.local|__local_depot' "$out/$arch.ptx" || true)
  echo "$arch: $kernels kernels, $local_lines lines of local memory"
  [ "$kernels" -eq 5 ] || { echo "expected 5 kernels"; exit 1; }
  [ "$local_lines" -eq 0 ] || { echo "a kernel uses local memory"; exit 1; }
done
