#!/usr/bin/env python3
"""
compare.py -- compare a program's output against the expected output.

    python3 compare.py <expected_file> <actual_file> [abs_tol] [rel_tol]

Exit code 0  -> match      (prints "OK")
Exit code 1  -> mismatch   (prints a one-line diagnosis on stderr)
Exit code 2  -> usage / IO error

Comparison rules
----------------
  * The files are tokenised on whitespace, so trailing spaces, extra blank
    lines and CRLF endings are all harmless.
  * The first two tokens must be the output height and width, and must match
    exactly.
  * The remaining H*W tokens are compared numerically with the tolerance
        |actual - expected| <= abs_tol + rel_tol * |expected|
    Default: abs_tol = 1e-3, rel_tol = 1e-4. This is deliberately looser than
    float32 precision so that a different-but-equivalent order of arithmetic
    operations, or a different GPU architecture, still passes.
  * NaN and Inf in the actual output always fail.
"""

import math
import sys

DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-4


def tokens(path):
    with open(path, "r", errors="replace") as f:
        return f.read().split()


def fail(msg):
    sys.stderr.write("MISMATCH: %s\n" % msg)
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 2

    exp_path, act_path = sys.argv[1], sys.argv[2]
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ATOL
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_RTOL

    try:
        exp = tokens(exp_path)
    except OSError as e:
        sys.stderr.write("ERROR: cannot read expected file: %s\n" % e)
        return 2
    try:
        act = tokens(act_path)
    except OSError as e:
        sys.stderr.write("ERROR: cannot read actual file: %s\n" % e)
        return 2

    if len(act) == 0:
        fail("output is empty")
    if len(exp) < 2:
        sys.stderr.write("ERROR: expected file is malformed\n")
        return 2
    if len(act) < 2:
        fail("output has fewer than 2 tokens (missing the 'H W' header line)")

    try:
        eh, ew = int(exp[0]), int(exp[1])
    except ValueError:
        sys.stderr.write("ERROR: expected file header is not integral\n")
        return 2
    try:
        ah, aw = int(act[0]), int(act[1])
    except ValueError:
        fail("first line should be two integers 'H W', found '%s %s'"
             % (act[0], act[1]))

    if (ah, aw) != (eh, ew):
        fail("output dimensions are %dx%d, expected %dx%d" % (ah, aw, eh, ew))

    n = eh * ew
    if len(exp) != 2 + n:
        sys.stderr.write("ERROR: expected file has %d values, header says %d\n"
                         % (len(exp) - 2, n))
        return 2
    if len(act) != 2 + n:
        fail("output has %d values, expected %d" % (len(act) - 2, n))

    worst = 0.0
    worst_at = -1
    for i in range(n):
        try:
            e = float(exp[2 + i])
            a = float(act[2 + i])
        except ValueError:
            fail("value %d ('%s') is not a number" % (i, act[2 + i]))
        if math.isnan(a) or math.isinf(a):
            fail("value at row %d col %d is %s" % (i // ew, i % ew, act[2 + i]))
        d = abs(a - e)
        if d > atol + rtol * abs(e):
            fail("value at row %d col %d is %.6f, expected %.6f (diff %.3e)"
                 % (i // ew, i % ew, a, e, d))
        if d > worst:
            worst, worst_at = d, i

    if worst_at >= 0:
        print("OK (%d values, max abs error %.3e at row %d col %d)"
              % (n, worst, worst_at // ew, worst_at % ew))
    else:
        print("OK (%d values, exact match)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
