#!/usr/bin/env bash
# =============================================================================
#  CS6023: GPU Programming -- Assignment 2
#  run_tests.sh : compile every submission in submit/ and check it against the
#                 six public test cases.
#
#  This is the ONLY test engine.  Running it directly (on a machine with an
#  NVIDIA GPU) and submitting scripts/aqua_job.cmd on Aqua both execute this
#  same file, so the two paths always give identical results.
#
#  Usage:
#     ./scripts/run_tests.sh              compile and test every submit/*.cu
#     ./scripts/run_tests.sh -c 3         run only test case 3
#     ./scripts/run_tests.sh -s FILE.cu   test one specific file
#     ./scripts/run_tests.sh -v           on failure, show the first differences
#     ./scripts/run_tests.sh -k           keep the generated output files
#     ./scripts/run_tests.sh -h           show this help
#
#  Results are printed to the screen AND written to results.txt in the
#  assignment folder.
#
#  Exit status: 0 if every test of every submission passed, 1 otherwise.
# =============================================================================
set -u

# Always work from the assignment folder (the parent of scripts/), no matter
# which directory the script was invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# These four can be overridden from the environment (the course staff use that
# to point the same engine at the private test cases).  Students never need to.
INPUT_DIR="${A2_INPUT_DIR:-testcases/input}"
EXPECTED_DIR="${A2_EXPECTED_DIR:-testcases/output}"
SUBMIT_DIR="${A2_SUBMIT_DIR:-submit}"
RESULTS="${A2_RESULTS:-$ROOT/results.txt}"

SRC=""
ONLY=""
KEEP=0
VERBOSE=0
TIMEOUT="${TIMEOUT:-120}"

# Print the header comment block (everything from line 2 up to the first
# non-comment line), so the help text can never drift out of sync with it.
usage() { awk 'NR>1 && /^#/ { sub(/^#[[:space:]]?/, ""); print; next } NR>1 { exit }' "$0"; }

need_arg() {
    if [ "$2" -lt 2 ]; then
        echo "ERROR: option $1 needs a value (try -h)." >&2
        exit 2
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--source)  need_arg "$1" $#; SRC="$2";  shift 2 ;;
        -c|--case)    need_arg "$1" $#; ONLY="$2"; shift 2 ;;
        -k|--keep)    KEEP=1;    shift ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1' (try -h)." >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------ colours --
if [ -t 1 ]; then
    RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
    BLD=$'\033[1m';    OFF=$'\033[0m'
else
    RED=""; GRN=""; YEL=""; BLD=""; OFF=""
fi

# ------------------------------------------------------------ sanity checks --
if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    echo "${RED}ERROR${OFF}: 'nvcc' was not found on your PATH." >&2
    echo "  On your own machine : install the CUDA Toolkit, then" >&2
    echo "                        export PATH=/usr/local/cuda/bin:\$PATH" >&2
    echo "  On Aqua             : do not run this script on the login node." >&2
    echo "                        Submit it as a job instead:" >&2
    echo "                            qsub scripts/aqua_job.cmd" >&2
    exit 127
fi

for d in "$INPUT_DIR" "$EXPECTED_DIR"; do
    if [ ! -d "$d" ]; then
        echo "${RED}ERROR${OFF}: '$d' not found under $ROOT." >&2
        echo "  Run this script from the assignment folder and keep the" >&2
        echo "  testcases/ directory where it is." >&2
        exit 2
    fi
done

# Discover the available test cases from the input directory rather than
# hard-coding a count, so adding or removing a case needs no script edit.
ALL_CASES=()
for f in "$INPUT_DIR"/input*.txt; do
    [ -e "$f" ] || continue
    n="$(basename "$f")"; n="${n#input}"; n="${n%.txt}"
    case "$n" in (*[!0-9]*|"") continue ;; esac
    ALL_CASES+=("$n")
done
if [ ${#ALL_CASES[@]} -eq 0 ]; then
    echo "${RED}ERROR${OFF}: no inputN.txt files found in $INPUT_DIR/." >&2
    exit 2
fi
# Numeric sort, so input10 comes after input9.
IFS=$'\n' ALL_CASES=($(printf '%s\n' "${ALL_CASES[@]}" | sort -n)); unset IFS
NUM_CASES=${#ALL_CASES[@]}

if [ -n "$ONLY" ]; then
    found=0
    for n in "${ALL_CASES[@]}"; do [ "$n" = "$ONLY" ] && found=1; done
    if [ "$found" -eq 0 ]; then
        echo "${RED}ERROR${OFF}: no test case '$ONLY'. Available: ${ALL_CASES[*]}" >&2
        exit 2
    fi
    CASES=("$ONLY")
else
    CASES=("${ALL_CASES[@]}")
fi

# Which source files to test.
SOURCES=()
if [ -n "$SRC" ]; then
    if [ ! -f "$SRC" ]; then
        echo "${RED}ERROR${OFF}: source file '$SRC' not found." >&2
        exit 2
    fi
    SOURCES=("$SRC")
else
    for f in "$SUBMIT_DIR"/*.cu; do
        [ -e "$f" ] && SOURCES+=("$f")
    done
    if [ ${#SOURCES[@]} -eq 0 ]; then
        echo "${RED}ERROR${OFF}: no .cu file found in $SUBMIT_DIR/." >&2
        echo "  Copy main.cu to submit/<YourRollNumber>.cu, implement compute()," >&2
        echo "  and run this script again.  Example:" >&2
        echo "      cp main.cu submit/cs22d003.cu" >&2
        exit 2
    fi
    if [ -f "$SUBMIT_DIR/main.cu" ]; then
        echo "${YEL}WARNING${OFF}: $SUBMIT_DIR/main.cu is still named main.cu."
        echo "  Rename it to your roll number, or Moodle will not accept it:"
        echo "      mv $SUBMIT_DIR/main.cu $SUBMIT_DIR/<YourRollNumber>.cu"
        echo
    fi
fi

# --------------------------------------------------------------- work area --
WORK="$ROOT/.work"
rm -rf "$WORK"
mkdir -p "$WORK"
cleanup() { [ "$KEEP" -eq 0 ] && rm -rf "$WORK"; }
trap cleanup EXIT

# No -std flag on purpose: your code is graded with plain "nvcc", so let nvcc
# use its own default C++ standard.  Pinning one here would make code that
# builds on a recent local toolkit fail on the older one installed on Aqua.
NVCC_FLAGS=(-O3)
[ -n "${CUDA_ARCH:-}" ] && NVCC_FLAGS+=("-arch=${CUDA_ARCH}")
# The Aqua job script sets this to e.g. "-ccbin /path/to/g++" when the default
# host compiler does not match the libstdc++ headers on the include path.
# Intentionally unquoted so multiple flags split into separate words.
# shellcheck disable=SC2206
[ -n "${NVCC_EXTRA_FLAGS:-}" ] && NVCC_FLAGS+=(${NVCC_EXTRA_FLAGS})

: > "$RESULTS"
{
    echo "CS6023 Assignment 2 -- test results"
    echo "Date : $(date)"
    echo "Host : $(hostname)"
    echo
} >> "$RESULTS"

overall_fail=0
build_failed=0

# ------------------------------------------------------------------- run it --
for src in "${SOURCES[@]}"; do
    name="$(basename "$src" .cu)"
    bin="$WORK/$name.bin"
    buildlog="$WORK/$name.build.log"

    echo "${BLD}=== $src ===${OFF}"
    echo "Compiling with: ${NVCC:-nvcc} ${NVCC_FLAGS[*]} $src"
    if ! "${NVCC:-nvcc}" "${NVCC_FLAGS[@]}" "$src" -o "$bin" >"$buildlog" 2>&1; then
        echo "${RED}BUILD FAILED${OFF}  (compiler output below)"
        sed 's/^/    /' "$buildlog"
        # Record the compiler output in results.txt as well.  The work
        # directory is deleted when this script exits -- and on Aqua the whole
        # scratch directory goes with it -- so results.txt must carry the
        # reason for the failure, not just the fact of it.
        {
            echo "$name : BUILD FAILED : 0/${#CASES[@]} test cases passed"
            echo "--- compiler output ---"
            sed 's/^/    /' "$buildlog"
            echo "--- end of compiler output ---"
            echo
        } >> "$RESULTS"
        echo
        overall_fail=1
        build_failed=1
        continue
    fi
    echo "${GRN}Build OK${OFF}"
    # Surface warnings without burying the result table.
    if [ -s "$buildlog" ]; then
        echo "${YEL}(compiler warnings -- see $WORK/$name.build.log)${OFF}"
    fi
    echo

    printf "%-12s %-9s %-12s %s\n" "TEST" "RESULT" "TIME" "DETAIL"
    printf '%s\n' "----------------------------------------------------------------"

    passed=0; total=0
    for n in "${CASES[@]}"; do
        input="$INPUT_DIR/input${n}.txt"
        expected="$EXPECTED_DIR/output${n}.txt"
        actual="$WORK/${name}.output${n}.txt"
        stdout="$WORK/${name}.stdout${n}.txt"
        stderr="$WORK/${name}.stderr${n}.txt"

        if [ ! -f "$input" ] || [ ! -f "$expected" ]; then
            printf "%-12s ${YEL}%-9s${OFF} %-12s %s\n" \
                   "input${n}" "SKIP" "-" "test case files missing"
            continue
        fi

        total=$((total + 1))
        if command -v timeout >/dev/null 2>&1; then
            timeout "$TIMEOUT" "$bin" "$input" "$actual" >"$stdout" 2>"$stderr"
            rc=$?
        else
            "$bin" "$input" "$actual" >"$stdout" 2>"$stderr"
            rc=$?
        fi

        # main.cu prints "Time taken (ms): ..." on stdout.
        ms="$(sed -n 's/.*Time taken (ms): *//p' "$stdout" | head -1)"
        [ -n "$ms" ] && ms="${ms} ms" || ms="-"

        if [ "$rc" -eq 124 ]; then
            printf "%-12s ${RED}%-9s${OFF} %-12s %s\n" \
                   "input${n}" "TIMEOUT" "-" "exceeded ${TIMEOUT}s"
        elif [ "$rc" -ne 0 ] || [ ! -f "$actual" ]; then
            printf "%-12s ${RED}%-9s${OFF} %-12s %s\n" \
                   "input${n}" "CRASH" "-" "exit code $rc"
            [ "$VERBOSE" -eq 1 ] && sed 's/^/      /' "$stderr" | head -10
        elif diff -w "$expected" "$actual" >/dev/null 2>&1; then
            printf "%-12s ${GRN}%-9s${OFF} %-12s %s\n" \
                   "input${n}" "PASS" "$ms" "output matches"
            passed=$((passed + 1))
        else
            first="$(diff -w "$expected" "$actual" | head -1)"
            printf "%-12s ${RED}%-9s${OFF} %-12s %s\n" \
                   "input${n}" "FAIL" "$ms" "output differs (first diff at ${first:-line 1})"
            if [ "$VERBOSE" -eq 1 ]; then
                echo "      lines starting with '-' are EXPECTED, '+' are YOURS:"
                diff -uw "$expected" "$actual" | tail -n +3 | head -20 | sed 's/^/      /'
            fi
        fi
    done

    echo
    echo "${BLD}$name: $passed/$total test cases passed${OFF}"
    echo "$name : $passed/$total test cases passed" >> "$RESULTS"
    echo
    [ "$passed" -eq "$total" ] || overall_fail=1
done

echo "Results also written to: $RESULTS"
[ "$KEEP" -eq 1 ] && echo "Generated output files kept in: $WORK"

if [ "$overall_fail" -eq 0 ]; then
    echo "${GRN}All public test cases passed.${OFF}"
    echo "All public test cases passed." >> "$RESULTS"
    exit 0
fi
if [ "$build_failed" -eq 1 ]; then
    echo "${YEL}Fix the compilation errors listed above, then run this script again.${OFF}"
else
    echo "${YEL}Some test cases did not pass. Re-run with -v to see the differences:${OFF}"
    echo "    ./scripts/run_tests.sh -v"
fi
exit 1
