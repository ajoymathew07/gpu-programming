#!/bin/bash
#PBS -N cs6023_a2
#PBS -o logfile.log
#PBS -e errorfile.err
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq
#PBS -l walltime=00:20:00

# =============================================================================
#  CS6023: GPU Programming -- Assignment 2
#  aqua_job.cmd : PBS job script for the Aqua cluster at IIT Madras.
#
#  Use this if you do NOT have an NVIDIA GPU on your own machine.
#
#  Submit it from the assignment folder (the one that contains main.cu):
#      qsub scripts/aqua_job.cmd
#
#  When the job finishes you will find, in that same folder:
#      results.txt    the PASS/FAIL table for your submission
#      logfile.log    everything the job printed
#      errorfile.err  error messages, if any
#
#  This job runs scripts/run_tests.sh -- exactly the same test engine used on
#  a local NVIDIA machine -- so the results are identical either way.
# =============================================================================

# Deliberately no 'set -e': if a test case fails we still want the results
# copied back to your folder instead of the job dying silently.

job_id="${PBS_JOBID%%.*}"
scratch_dir="$HOME/scratch/cs6023_a2_${job_id}"

mkdir -p "$scratch_dir" || {
    echo "ERROR: could not create $scratch_dir" >&2
    exit 1
}

# Work on a copy in scratch so compilation and generated files stay off the
# shared network filesystem.
cp -R "$PBS_O_WORKDIR"/* "$scratch_dir"/ 2>/dev/null
cd "$scratch_dir" || exit 1

chmod +x scripts/*.sh 2>/dev/null

echo "Job $PBS_JOBID running on $(hostname)"
echo

# --------------------------------------------------------------------------
#  Pick a host compiler that actually matches the libstdc++ headers nvcc will
#  use.  Loading a GCC module whose version is older than the headers on the
#  include path makes every program that includes <iostream> fail deep inside
#  <string> with errors like
#      bits/move.h: identifier "__builtin_addressof" is undefined
#  which looks like a bug in your code but is not.  So instead of hard-coding
#  a module, compile a tiny probe and keep the first combination that works.
# --------------------------------------------------------------------------
#  Note: the Assignment 1 script loaded "gcc640" on top of "cuda11.4".  That
#  combination now breaks, because the CUDA module puts GCC 9.2.0's libstdc++
#  headers on the include path while gcc640 makes the compiler GCC 6.4.0, and
#  6.4.0 does not implement __builtin_addressof.  So we do not hard-code any
#  GCC module here.
module load cuda11.4 2>/dev/null

cat > .probe.cu <<'PROBE'
#include <iostream>
#include <cstdio>
#include <cstdlib>
__global__ void probe_kernel(int *a) { if (threadIdx.x == 0) a[0] = 1; }
int main() { std::cout << ""; printf(""); return 0; }
PROBE

# $1 = extra nvcc flags to test with (may be empty)
probe_ok() { nvcc -O3 $1 .probe.cu -o .probe.bin >.probe.log 2>&1; }

CHOSEN=""
NVCC_EXTRA_FLAGS=""

# 1. The default host compiler.  Normally this is the one whose headers are on
#    the include path, so nothing else is needed.
if probe_ok ""; then
    CHOSEN="default host compiler"
fi

# 2. Still broken?  The compiler and the libstdc++ headers disagree.  The error
#    names the header directory, e.g. /lfs/sware/gcc9.2.0/include/bits/move.h,
#    so point nvcc at the g++ that owns those very headers.
if [ -z "$CHOSEN" ]; then
    hdr_dir=$(grep -o '/[^ :]*/gcc[0-9][^ :/]*/include' .probe.log 2>/dev/null | head -1)
    if [ -n "$hdr_dir" ]; then
        cand="${hdr_dir%/include}/bin/g++"
        if [ -x "$cand" ] && probe_ok "-ccbin $cand"; then
            CHOSEN="-ccbin $cand (matched to its own headers)"
            NVCC_EXTRA_FLAGS="-ccbin $cand"
        fi
    fi
fi

# 3. Last resort: try GCC modules explicitly, newest first.
if [ -z "$CHOSEN" ]; then
    for m in gcc920 gcc102 gcc740 gcc640; do
        module load "$m" 2>/dev/null || continue
        if probe_ok ""; then CHOSEN="module $m"; break; fi
        module unload "$m" 2>/dev/null
    done
fi

export NVCC_EXTRA_FLAGS

echo "nvcc          : $(command -v nvcc)"
echo "host compiler : $(command -v g++)  [$(g++ -dumpversion 2>/dev/null)]"
if [ -n "$CHOSEN" ]; then
    echo "toolchain     : $CHOSEN  (verified by a test compile)"
else
    echo "toolchain     : could not compile even a trivial CUDA program."
    echo "                This is an Aqua toolchain problem, NOT a bug in your"
    echo "                code.  Probe output:"
    sed 's/^/                /' .probe.log 2>/dev/null | head -12
    echo "                Please report this on the course forum."
fi
rm -f .probe.cu .probe.bin .probe.log
echo

# TEST_ARGS is optional.  Example, to see the differences on a failing case:
#     qsub -v TEST_ARGS="-v" scripts/aqua_job.cmd
./scripts/run_tests.sh ${TEST_ARGS:-}
status=$?

# Always copy the results back, whether the tests passed or not.
cp -f results.txt "$PBS_O_WORKDIR"/ 2>/dev/null

echo
echo "============================================================"
echo "Job finished with status $status."
echo "Your results are in:  $PBS_O_WORKDIR/results.txt"
echo "The full output of this job is in: $PBS_O_WORKDIR/logfile.log"
if [ $status -ne 0 ]; then
    echo
    echo "Something did not pass.  results.txt says what, and for a"
    echo "compilation failure it also contains the compiler's error"
    echo "messages.  Read it first:"
    echo "    cat results.txt"
fi
echo "(Ignore the scratch path printed above -- that directory is"
echo " temporary and has now been deleted.)"
echo "============================================================"

cd "$HOME" || exit $status
rm -rf "$scratch_dir"
exit $status
