#!/usr/bin/env bash
# =============================================================================
#  CS6023: GPU Programming -- Assignment 2
#  compile.sh : build one .cu file and, optionally, run it on one input file.
#
#  This is a convenience helper for debugging a single test case.  To check
#  your submission against all six public test cases, use run_tests.sh instead.
#
#  Usage:
#     ./scripts/compile.sh submit/cs22d003.cu
#         -> builds submit/cs22d003.cu into ./cs22d003
#
#     ./scripts/compile.sh submit/cs22d003.cu testcases/input/input2.txt out2.txt
#         -> builds it, then runs it on input2.txt and writes out2.txt
#
#  Paths are interpreted relative to the assignment folder.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ $# -lt 1 ]; then
    awk 'NR>1 && /^#/ { sub(/^#[[:space:]]?/, ""); print; next } NR>1 { exit }' "$0"
    exit 2
fi

SRC="$1"
INPUT="${2:-}"
OUTPUT="${3:-output.txt}"
BIN="$(basename "${SRC%.cu}")"

if [ ! -f "$SRC" ]; then
    echo "ERROR: source file '$SRC' not found (looked under $ROOT)." >&2
    exit 2
fi
if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    echo "ERROR: 'nvcc' was not found on your PATH." >&2
    echo "  Install the CUDA Toolkit, or on Aqua submit a job instead:" >&2
    echo "      qsub scripts/aqua_job.cmd" >&2
    exit 127
fi

FLAGS=(-O3 -std=c++11 -lineinfo)
[ -n "${CUDA_ARCH:-}" ] && FLAGS+=("-arch=${CUDA_ARCH}")

"${NVCC:-nvcc}" "${FLAGS[@]}" "$SRC" -o "$BIN" || exit 1
echo "Build successful: ./$BIN"

if [ -n "$INPUT" ]; then
    if [ ! -f "$INPUT" ]; then
        echo "ERROR: input file '$INPUT' not found." >&2
        exit 2
    fi
    "./$BIN" "$INPUT" "$OUTPUT" || exit 1
    echo "Output written to: $OUTPUT"
fi
