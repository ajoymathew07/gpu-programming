# CS6023: GPU Programming — Assignment 2

Compute `E = Aᵀ·B + C·Dᵀ` on the GPU, using **coalesced memory accesses** and
**shared memory**.

**Released:** 23 August 2026 | **Due:** 13 September 2026, 23:59 IST | **Marks:** 12

Read `Assignment2.pdf` for the problem statement, the input/output format and the
constraints. This file tells you how to build, test and submit.

---

## 1. What is in this folder

```
Assignment2/
├── Assignment2.pdf     the problem statement — read this first
├── README.md           this file
├── main.cu             starter code — you implement compute() here
├── submit/             put your <RollNumber>.cu in here
├── scripts/
│   ├── run_tests.sh    runs the 6 public test cases  (needs an NVIDIA GPU)
│   ├── aqua_job.cmd    job script for the Aqua cluster (if you have no GPU)
│   └── compile.sh      builds a single file, for debugging
└── testcases/
    ├── input/          input1.txt … input6.txt
    └── output/         output1.txt … output6.txt   ← the expected results
```

Keep this layout as it is. The scripts find everything relative to this folder,
so moving files around will break them.

The six test cases here are the **public** ones, given to you so that you can
check your own work. Your marks are computed on a larger set of **private** test
cases. They use the same format and obey the same constraints, so code that is
correct and handles the boundaries properly will pass those too.

---

## 2. Step 1 — set up (do this first, whichever machine you use)

Copy the starter file into `submit/`, renamed to your roll number,
and write your solution in that copy. If your roll number is `cs22d003`:

```bash
cd Assignment2
cp main.cu submit/cs22d003.cu
chmod +x scripts/*.sh
```

Now implement the `compute()` function inside `submit/cs22d003.cu`.

Leave `main()` alone, and do not add any print statements — the test scripts
compare your output file byte for byte against the expected one.

Then follow **either** section 3 (you have an NVIDIA GPU) **or** section 4
(you do not). You do not need both.

---

## 3. Option A — a machine with an NVIDIA GPU

**Requirements:** an NVIDIA GPU, a working NVIDIA driver, and the CUDA Toolkit,
so that `nvcc` is on your `PATH`. Check with `nvcc --version` and `nvidia-smi`.

Run all six public test cases:

```bash
./scripts/run_tests.sh
```

While debugging:

```bash
./scripts/run_tests.sh -c 3    # run only test case 3
./scripts/run_tests.sh -v      # on failure, show expected vs. your output
./scripts/run_tests.sh -k      # keep the generated output files
./scripts/run_tests.sh -h      # list every option
```

To build and run one case by hand:

```bash
./scripts/compile.sh submit/cs22d003.cu testcases/input/input2.txt myout2.txt
diff -w testcases/output/output2.txt myout2.txt
```

---

## 4. Option B — the Aqua cluster (no NVIDIA GPU of your own)

Aqua is the IIT Madras HPC facility. You do not run your program directly there;
you submit it as a **job** and the scheduler runs it on a GPU node for you.

### 4.1 Copy the folder to Aqua

Run this **on your own machine**, from the directory that *contains* the
`Assignment2` folder:

```bash
scp -P 40826 -r Assignment2 <your-roll-no>@aqua.iitm.ac.in:~/
```

Aqua does not use the default SSH port, so the port option is required every
time. Note the case: `scp` uses a **capital** `-P` (before the file names),
`ssh` uses a **small** `-p`. Aqua is reachable only from inside the institute
network — if you are off campus, connect to the IITM VPN first.

### 4.2 Log in and submit the job

```bash
ssh <your-roll-no>@aqua.iitm.ac.in -p 40826
cd Assignment2
chmod +x scripts/*.sh
qsub scripts/aqua_job.cmd
```

`qsub` prints a job id such as `123456.aqua` and returns immediately. Your job
is now **queued**, not finished.

### 4.3 Wait for it, then read the results

```bash
qstat -u <your-roll-no>     # shows your job; empty output = it has finished
```

Once the job is gone from `qstat`, three files appear in the `Assignment2`
folder on Aqua:

```bash
cat results.txt     # the PASS/FAIL table — this is what you want
cat logfile.log     # everything the job printed
cat errorfile.err   # error messages, if there were any
```

To copy the results back to your own machine, run this **on your machine**:

```bash
scp -P 40826 <your-roll-no>@aqua.iitm.ac.in:~/Assignment2/results.txt .
```

### 4.4 After you edit your code

Copy just the changed file up again and resubmit:

```bash
scp -P 40826 submit/cs22d003.cu <your-roll-no>@aqua.iitm.ac.in:~/Assignment2/submit/
ssh <your-roll-no>@aqua.iitm.ac.in -p 40826 "cd Assignment2 && qsub scripts/aqua_job.cmd"
```

> **Do not run `nvcc` or `run_tests.sh` directly on the Aqua login node.**
> The login node has no GPU and is shared by everyone. Always use `qsub`.

---

## 5. Reading the results

Both options run the same script, `scripts/run_tests.sh`, so the output is
identical either way:

```
=== submit/cs22d003.cu ===
Build OK

TEST         RESULT    TIME         DETAIL
----------------------------------------------------------------
input1       PASS      0.412 ms     output matches
input2       PASS      0.395 ms     output matches
input3       FAIL      0.401 ms     output differs (first diff at 12c12)
...

cs22d003: 5/6 test cases passed
```

* **PASS** — your output matches the expected output.
* **FAIL** — it does not. Re-run with `-v` to see the first differing lines.
* **CRASH** — your program exited with an error (often an illegal memory access).
* **TIMEOUT** — it ran longer than 120 seconds. A correct solution finishes the
  largest public case in well under a second.

The same table is written to `results.txt`.

---

## 6. Submission

Submit **one zip file** on Moodle, named `<YourRollNumber>.zip`, containing **exactly
one** file: your `<YourRollNumber>.cu`, with no folder around it. Do not include
the `submit/` folder, the test cases, or anything else.

Create it from inside `submit/`, so the `.cu` sits at the top of the zip:

```bash
cd submit
zip CS22D003.zip CS22D003.cu
```

* The roll number may be **upper or lower case** — both are accepted. Use the
  same spelling for the zip and for the `.cu` inside it.
* After submitting, download the zip again, open it, and confirm it holds the
  file you meant to submit.
* Your file must compile with `nvcc` and run as it is, with no manual fixes by
  the evaluator.

---

## 7. Troubleshooting

**`nvcc: command not found`**
The CUDA Toolkit is not on your `PATH`. On a machine that has it installed:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```
If you do not have an NVIDIA GPU at all, use Option B (Aqua) instead. Having a
GPU from another vendor, or Apple Silicon, does not work — CUDA is NVIDIA only.

**`Permission denied` when running a script**
```bash
chmod +x scripts/*.sh
```

**`no .cu file found in submit/`**
You have not copied your solution into `submit/` yet. See section 2.

**The job stays in the queue on Aqua**
That is normal when the cluster is busy. `qstat -u <your-roll-no>` shows the
state: `Q` = queued, `R` = running. Wait; do not submit the same job repeatedly.

**`results.txt` did not appear on Aqua**
The job has probably not finished. Check `qstat`. If the job is gone but there
is no `results.txt`, read `logfile.log` and `errorfile.err`.

**`results.txt` says `BUILD FAILED` on Aqua**
`results.txt` also contains the compiler's error messages, right underneath
that line — read those first. Note that `errorfile.err` is usually *empty*
even when the build fails, because the compiler output is captured by the test
script rather than left on the job's stderr.

**It compiles on my machine but `BUILD FAILED` on Aqua**
Aqua uses CUDA 11.4, which is older than a current desktop toolkit. Two things
usually cause this:

* *Headers you did not include.* Your local compiler may pull them in for you.
  Include what you use — `#include <cstdio>` for `printf`/`fopen`,
  `#include <cstdlib>` for `malloc`/`free`.
* *Newer C++ syntax.* Stick to C++14 and earlier.

Aqua is the reference environment: if it fails there, fix it there.

**Errors inside `bits/move.h`, `basic_string.h` or `__builtin_addressof`**
These come from the C++ standard library, not from your file, and they mean the
compiler and its headers did not match on the compute node. `aqua_job.cmd`
detects and works around this automatically, and prints the `toolchain :` line
it settled on. If you see this anyway, report the `toolchain :` line on the
course forum — it is not something you can fix in your code.

**All test cases FAIL with completely different numbers**
Check the order in which the matrices are read and their shapes. `A` is stored
as `q × p` (not `p × q`), and `D` is stored as `r × q`. It is easy to transpose
the wrong one.

**Small cases PASS but larger ones FAIL**
Almost always a boundary bug. `p`, `q` and `r` are deliberately *not* multiples
of 32 in several test cases, so every kernel needs correct guards. Run under
`compute-sanitizer ./yourbinary input.txt out.txt` to find out-of-bounds
accesses.

**My timings vary between runs**
That is expected and is not part of the correctness check. Only the output
values are compared.

**Do trailing spaces at the end of a line matter?**
No. The comparison ignores whitespace differences, so the trailing space that
`main.cu` writes after each number is fine. Do not change how `main.cu` prints.

---

## 8. Where to ask

If something here is unclear or a script misbehaves, post on the course Moodle
forum with the exact command you ran and the full output — that is much faster
than describing it in words.
