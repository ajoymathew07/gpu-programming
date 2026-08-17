#!/usr/bin/env bash
# =============================================================================
#  GPU Programming -- Assignment 1 : "The Wizard's Lens"
#  run_tests.sh -- build your pipeline and run it against every test case.
#
#  Usage:
#     ./run_tests.sh                     build pipeline.cu, run all test cases
#     ./run_tests.sh -s mysource.cu      use a different source file
#     ./run_tests.sh -t testcases/public run only one test directory
#     ./run_tests.sh -c 03               run only the test case numbered 03
#     ./run_tests.sh -k                  keep the produced output files
#     ./run_tests.sh -v                  verbose: show the first mismatch
#     ./run_tests.sh --cpu               emulate CUDA on the CPU (no GPU needed;
#                                        instructor toolkit only, see below)
#
#  A test case is a pair of files inside a test directory:
#       input_NN.txt      fed to the program on stdin
#       expected_NN.txt   compared against the program's stdout
#
#  Exit code 0 if every test passed, 1 otherwise.
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SRC="pipeline.cu"
TESTDIRS=()
ONLY=""
KEEP=0
VERBOSE=0
CPUMODE=0
TIMEOUT="${TIMEOUT:-60}"
ATOL="${ATOL:-1e-3}"
RTOL="${RTOL:-1e-4}"

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--source)  SRC="$2"; shift 2 ;;
        -t|--tests)   TESTDIRS+=("$2"); shift 2 ;;
        -c|--case)    ONLY="$2"; shift 2 ;;
        -k|--keep)    KEEP=1; shift ;;
        -v|--verbose) VERBOSE=1; shift ;;
        --cpu)        CPUMODE=1; shift ;;
        -h|--help)    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "Unknown option: $1  (try --help)" >&2; exit 2 ;;
    esac
done

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
CYN=$'\033[0;36m'; BLD=$'\033[1m'; OFF=$'\033[0m'
if [ ! -t 1 ]; then RED=""; GRN=""; YEL=""; CYN=""; BLD=""; OFF=""; fi

# ---------------------------------------------------------------- checks ----
if [ ! -f "$SRC" ]; then
    echo "${RED}ERROR${OFF}: source file '$SRC' not found." >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "${RED}ERROR${OFF}: python3 is required by the comparator." >&2
    exit 2
fi
if [ ! -f compare.py ]; then
    echo "${RED}ERROR${OFF}: compare.py not found next to this script." >&2
    exit 2
fi

# Default test directories: whichever of these exist.
if [ ${#TESTDIRS[@]} -eq 0 ]; then
    for d in testcases/public testcases/hidden; do
        [ -d "$d" ] && TESTDIRS+=("$d")
    done
fi
if [ ${#TESTDIRS[@]} -eq 0 ]; then
    echo "${RED}ERROR${OFF}: no test directories found under ./testcases/" >&2
    exit 2
fi

WORK="$(mktemp -d)"
cleanup() { [ "$KEEP" -eq 0 ] && rm -rf "$WORK"; }
trap cleanup EXIT

# ----------------------------------------------------------------- build ----
BIN="$WORK/pipeline_bin"
echo "${BLD}=== Building ===${OFF}"

if [ "$CPUMODE" -eq 1 ]; then
    if [ ! -f tools/cu2cpp.py ] || [ ! -f tools/cuda_cpu_shim.h ]; then
        echo "${RED}ERROR${OFF}: --cpu needs tools/cu2cpp.py and tools/cuda_cpu_shim.h," >&2
        echo "        which ship only with the instructor toolkit." >&2
        exit 2
    fi
    echo "${YEL}CPU EMULATION MODE${OFF} - kernels run sequentially on the CPU."
    echo "${YEL}This checks your arithmetic and indexing, NOT your concurrency.${OFF}"
    echo "${YEL}Always confirm on a real GPU before submitting.${OFF}"
    python3 tools/cu2cpp.py "$SRC" "$WORK/translated.cpp" || exit 1
    if ! g++ -O2 -w -I tools -o "$BIN" "$WORK/translated.cpp"; then
        echo "${RED}BUILD FAILED${OFF}" >&2
        exit 1
    fi
else
    if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
        echo "${RED}ERROR${OFF}: 'nvcc' was not found on your PATH." >&2
        echo "        export PATH=/usr/local/cuda/bin:\$PATH" >&2
        exit 127
    fi
    ARCHFLAG=()
    [ -n "${CUDA_ARCH:-}" ] && ARCHFLAG=("-arch=${CUDA_ARCH}")
    if ! "${NVCC:-nvcc}" -O3 -std=c++11 "${ARCHFLAG[@]}" -Xcompiler -Wall \
            "$SRC" -o "$BIN"; then
        echo "${RED}BUILD FAILED${OFF}" >&2
        exit 1
    fi
fi
echo "${GRN}Build OK${OFF}  ($SRC)"
echo

# ------------------------------------------------------------------ run -----
total=0; passed=0; failed=0
declare -a FAILED_NAMES=()

printf "%-28s %-10s %s\n" "TEST" "RESULT" "DETAIL"
printf -- "%s\n" "----------------------------------------------------------------------"

for dir in "${TESTDIRS[@]}"; do
    for inp in "$dir"/input_*.txt; do
        [ -e "$inp" ] || continue
        base="$(basename "$inp")"
        num="${base#input_}"; num="${num%.txt}"
        [ -n "$ONLY" ] && [ "$ONLY" != "$num" ] && continue

        exp="$dir/expected_$num.txt"
        label="$(basename "$dir")/$num"
        total=$((total + 1))

        if [ ! -f "$exp" ]; then
            printf "%-28s ${YEL}%-10s${OFF} %s\n" "$label" "SKIP" "no expected file"
            total=$((total - 1))
            continue
        fi

        act="$WORK/out_$(basename "$dir")_$num.txt"
        err="$WORK/err_$(basename "$dir")_$num.txt"

        if command -v timeout >/dev/null 2>&1; then
            timeout "$TIMEOUT" "$BIN" < "$inp" > "$act" 2> "$err"; rc=$?
        else
            "$BIN" < "$inp" > "$act" 2> "$err"; rc=$?
        fi

        if [ $rc -eq 124 ]; then
            printf "%-28s ${RED}%-10s${OFF} %s\n" "$label" "TIMEOUT" "> ${TIMEOUT}s"
            failed=$((failed + 1)); FAILED_NAMES+=("$label"); continue
        fi
        if [ $rc -ne 0 ]; then
            printf "%-28s ${RED}%-10s${OFF} %s\n" "$label" "CRASH" "exit code $rc"
            [ "$VERBOSE" -eq 1 ] && sed 's/^/      /' "$err" | head -5
            failed=$((failed + 1)); FAILED_NAMES+=("$label"); continue
        fi

        if msg=$(python3 compare.py "$exp" "$act" "$ATOL" "$RTOL" 2>&1); then
            printf "%-28s ${GRN}%-10s${OFF} %s\n" "$label" "PASS" "$msg"
            passed=$((passed + 1))
        else
            printf "%-28s ${RED}%-10s${OFF} %s\n" "$label" "FAIL" "$msg"
            if [ "$VERBOSE" -eq 1 ]; then
                echo "      ${CYN}expected (first 3 lines):${OFF}"
                head -3 "$exp" | sed 's/^/        /'
                echo "      ${CYN}your output (first 3 lines):${OFF}"
                head -3 "$act" | sed 's/^/        /'
            fi
            failed=$((failed + 1)); FAILED_NAMES+=("$label")
        fi
    done
done

echo
printf -- "%s\n" "----------------------------------------------------------------------"
if [ "$total" -eq 0 ]; then
    echo "${YEL}No test cases were run.${OFF}"
    exit 2
fi
echo "${BLD}Summary:${OFF} $passed/$total passed, $failed failed"
if [ "$failed" -gt 0 ]; then
    echo "${RED}Failed:${OFF} ${FAILED_NAMES[*]}"
    echo "Tip: re-run with -v to see the first differing values."
fi
[ "$KEEP" -eq 1 ] && echo "Outputs kept in: $WORK"
echo

[ "$failed" -eq 0 ] || exit 1
echo "${GRN}All tests passed.${OFF}"
exit 0
