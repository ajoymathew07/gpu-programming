#!/usr/bin/env bash
# =============================================================================
#  GPU Programming -- Assignment 1 : "The Wizard's Lens"
#  compile.sh -- build the pipeline, and optionally run it on one input file.
#
#  Usage:
#     ./compile.sh                              build pipeline.cu -> ./pipeline
#     ./compile.sh mysource.cu                  build a different source file
#     ./compile.sh pipeline.cu input_01.txt     build, then run on that input
#
#  Environment overrides:
#     CUDA_ARCH=sm_75   force a target architecture
#     NVCC=/path/to/nvcc
# =============================================================================
set -u

SRC="${1:-pipeline.cu}"
INPUT="${2:-}"
NVCC="${NVCC:-nvcc}"

BIN="$(basename "${SRC%.*}")"

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'; BLD=$'\033[1m'; OFF=$'\033[0m'
if [ ! -t 1 ]; then RED=""; GRN=""; YEL=""; BLD=""; OFF=""; fi

if [ ! -f "$SRC" ]; then
    echo "${RED}ERROR${OFF}: source file '$SRC' not found." >&2
    exit 2
fi

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "${RED}ERROR${OFF}: 'nvcc' was not found on your PATH." >&2
    echo "        Load the CUDA module or add /usr/local/cuda/bin to PATH:" >&2
    echo "            export PATH=/usr/local/cuda/bin:\$PATH" >&2
    exit 127
fi

FLAGS=(-O3 -std=c++11 -lineinfo -Xcompiler -Wall)
if [ -n "${CUDA_ARCH:-}" ]; then
    FLAGS+=("-arch=${CUDA_ARCH}")
fi

echo "${BLD}Compiling${OFF} $SRC  ->  ./$BIN"
echo "  $NVCC ${FLAGS[*]} $SRC -o $BIN"
if ! "$NVCC" "${FLAGS[@]}" "$SRC" -o "$BIN"; then
    echo "${RED}BUILD FAILED${OFF}" >&2
    exit 1
fi
echo "${GRN}Build successful.${OFF}"

if [ -n "$INPUT" ]; then
    if [ ! -f "$INPUT" ]; then
        echo "${RED}ERROR${OFF}: input file '$INPUT' not found." >&2
        exit 2
    fi
    echo
    echo "${BLD}Running${OFF} ./$BIN < $INPUT"
    echo "${YEL}------------------------------ output ------------------------------${OFF}"
    "./$BIN" < "$INPUT"
    rc=$?
    echo "${YEL}--------------------------------------------------------------------${OFF}"
    echo "exit code: $rc"
    exit $rc
fi
