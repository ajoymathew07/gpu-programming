# CS6023: GPU Programming — Assignment 1
## The Wizard's Lens: An Image Preprocessing Pipeline on the GPU

Read **`assignment1.pdf`** for the problem statement, formulas, worked example
and submission rules. This file covers building and testing.

---

## Contents

```
assignment1.pdf          The assignment
pipeline.cu              Your starting point. Fill in the 9 TODOs.
compile.sh               Builds pipeline.cu -> ./pipeline
run_tests.sh             Builds and runs all public test cases
compare.py               Output comparator used by run_tests.sh
testcases/public/        5 test cases: input_NN.txt + expected_NN.txt
```

---

## Building

```bash
chmod +x compile.sh run_tests.sh          # first time only

# If nvcc isn't on your PATH:
export PATH=/usr/local/cuda/bin:$PATH

./compile.sh                              # build pipeline.cu -> ./pipeline
./compile.sh pipeline.cu testcases/public/input_01.txt   # build, then run once
```

## Testing

```bash
./run_tests.sh                # build + run all public test cases
./run_tests.sh -v             # show the first differing values on a failure
./run_tests.sh -c 03          # run only test case 03
./run_tests.sh -s CS24S009.cu # test a renamed file — do this before submitting
./run_tests.sh -k             # keep the generated output files
```

A passing run looks like:

```
TEST                         RESULT     DETAIL
----------------------------------------------------------------------
public/01                    PASS       OK (4 values, exact match)
...
Summary: 5/5 passed, 0 failed
All tests passed.
```

Before you write any code, `./run_tests.sh` should build successfully and fail
all 5 tests with zeros. That confirms your toolchain works.

---

## Instructions

- **Do not modify anything outside the TODO markers.** The parser, the printer
  and the kernel signatures are checked automatically during grading.
- Passing the public tests is necessary, not sufficient.
- **Sequential codes will not fetch marks.** The work must happen in your
  kernels, on the device.
- If you have any general questions, please post on Moodle.
- Write your own code. You are welcome to discuss the ideas with your
  classmates, but the code you submit must be yours, written by you.

---

## Submitting

1. Rename to `<YourRollNumber>.cu` (e.g. `CS24S009.cu`).
2. Verify: `./run_tests.sh -s CS24S009.cu`
3. Create a folder named with your roll number (e.g., CS24S009),
   place the <RollNumber>.cu file inside it, and compress the folder into <RollNumber>.zip
4. Submit only the <RollNumber>.zip file. Do not include any additional files, reports, test outputs, or nested archives.

Deadline is in the PDF.
