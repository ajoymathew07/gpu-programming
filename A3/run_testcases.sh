#!/bin/bash

# Configuration
SOLVER_EXEC="./main"
INPUT_DIR="input"
EXPECTED_DIR="output"
TEMP_DIR="temp_output"

# ANSI Color Codes for UI
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if executable exists
if [ ! -f "$SOLVER_EXEC" ]; then
    echo -e "${RED}Error: Executable '$SOLVER_EXEC' not found! Build your code first.${NC}"
    exit 1
fi

# Create temporary folder for actual outputs
mkdir -p "$TEMP_DIR"

TOTAL=0
PASSED=0
FAILED=0

echo "=========================================="
echo "    Running CUDA Delta-Stepping Tests     "
echo "=========================================="

for input_file in "$INPUT_DIR"/*; do
    # Handle case where no files match
    [ -e "$input_file" ] || continue

    # Extract testcase number/filename (e.g., input/input6.txt -> output6.txt)
    base_name=$(basename "$input_file")
    expected_file="$EXPECTED_DIR/$base_name"
    temp_file="$TEMP_DIR/$base_name"

    ((TOTAL++))

    # Check if expected output exists
    if [ ! -f "$expected_file" ]; then
        echo -e "[${YELLOW}SKIP${NC}] $(basename "$input_file") -> Missing expected output '$expected_file'"
        continue
    fi

    # Run solver
    $SOLVER_EXEC "$input_file" "$temp_file" > /dev/null 2>&1

    # Compare outputs ignoring whitespace/blank lines (-w -B)
    if diff -w -B "$temp_file" "$expected_file" > /dev/null 2>&1; then
        echo -e "[${GREEN}PASS${NC}] $(basename "$input_file")"
        ((PASSED++))
    else
        echo -e "[${RED}FAIL${NC}] $(basename "$input_file")"
        ((FAILED++))
    fi
done

echo "=========================================="
echo -e "Summary: Total: $TOTAL | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
echo "=========================================="

# Cleanup temp dir if all tests pass
if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    rm -rf "$TEMP_DIR"
fi