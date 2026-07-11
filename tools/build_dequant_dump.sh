#!/usr/bin/env bash
# build_dequant_dump.sh -- reproduce the ggml-side harness of the GGUF crosscheck.
#
#   usage: tools/build_dequant_dump.sh [llamacpp_dir]
#
# 1. clones llama.cpp into <llamacpp_dir> (default ./llamacpp) if not present
# 2. builds ONLY the ggml shared libs (Release, Metal off -- CPU reference path)
# 3. compiles tools/dequant_dump.c against the built libggml-base/-cpu
#    (mirrors the original binary's link line: -Lbuild/bin -lggml-base -lggml-cpu
#     + rpath to build/bin, verified via `otool -L dequant_dump`)
#
# Result: <llamacpp_dir>/dequant_dump, which tools/gguf_crosscheck.py invokes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${1:-llamacpp}"

if [ ! -d "$LLAMA" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
fi
LLAMA="$(cd "$LLAMA" && pwd)"

cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_METAL=OFF
cmake --build "$LLAMA/build" --target ggml -j

cc -O2 "$REPO_DIR/tools/dequant_dump.c" -o "$LLAMA/dequant_dump" \
    -L"$LLAMA/build/bin" -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LLAMA/build/bin"

# smoke: binary must load its dylibs and print usage (exit 2 with no args)
out="$("$LLAMA/dequant_dump" 2>&1 || true)"
case "$out" in
    *usage*) echo "OK: built $LLAMA/dequant_dump" ;;
    *) echo "FAIL: $LLAMA/dequant_dump did not run: $out"; exit 1 ;;
esac
