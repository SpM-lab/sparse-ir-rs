#!/bin/bash

# Timing benchmark runner for sparse-ir-rs Fortran bindings (local version)
#
# Test patterns (8 patterns = 2 statistics × 2 lreal × 2 lsize_ir):
#   - Fermion (F): positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F
#   - Boson (B):   positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F
#   - lsize_ir: 1, 10
#   - num: 185640
#
# Output files:
#   - timing_raw_<compiler>.txt : Raw output from test_timing (all output preserved)
#   - timing_results.tsv        : Extracted TSV data for Excel
#
# Usage:
#   ./run_timing_benchmark.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Move to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${SCRIPT_DIR}"

# Base directory for builds
BASE_DIR="${SCRIPT_DIR}/.."

# Output files
OUTPUT_FILE="${SCRIPT_DIR}/timing_results.tsv"

echo -e "${BLUE}SCRIPT_DIR: $SCRIPT_DIR${NC}"
echo -e "${BLUE}BASE_DIR: $BASE_DIR${NC}"
echo -e "${BLUE}OUTPUT_FILE: $OUTPUT_FILE${NC}"

# Define compilers and their build directories
COMPILERS=("gfortran" "ifx" "ifort")

# Define base test patterns (without lsize_ir): nlambda ndigit positive_only statistics lreal_ir lreal_tau num
BASE_PATTERNS=(
    "6 8 T F T T 185640"
    "6 8 T F F F 185640"
    "6 8 F F F F 185640"
    "6 8 T B T T 185640"
    "6 8 T B F F 185640"
    "6 8 F B F F 185640"
)

# Define lsize_ir values to test
LSIZE_IR_VALUES=(1 10)

# Print header to output file
echo -e "compiler\tnlambda\tndigit\tpositive_only\tstatistics\tlreal_ir\tlreal_tau\tnum\tlsize_ir\tIR_size\tnfreq\tntau\tfit_matsu(s)\teval_tau(s)\tfit_tau(s)\teval_matsu(s)\ttotal(s)\tper_vector(s)\trel_error" > "$OUTPUT_FILE"

# Track which compilers were tested
TESTED_COMPILERS=()

# Run benchmarks for each compiler
for COMPILER in "${COMPILERS[@]}"; do
    BUILD_DIR="${BASE_DIR}/_build_${COMPILER}"
    TEST_TIMING="${BUILD_DIR}/test/test_timing"
    RAW_OUTPUT="${SCRIPT_DIR}/timing_raw_${COMPILER}.txt"
    
    # Check if executable exists
    if [ ! -f "$TEST_TIMING" ]; then
        echo -e "${YELLOW}Skipping ${COMPILER}: executable not found at ${TEST_TIMING}${NC}"
        continue
    fi
    
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Running benchmarks for ${COMPILER}...${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo -e "Raw output will be saved to: ${RAW_OUTPUT}"
    TESTED_COMPILERS+=("$COMPILER")
    
    # Set library path
    RUST_CAPI_INSTALL_DIR="${BUILD_DIR}/_rust_capi_install"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH="${RUST_CAPI_INSTALL_DIR}/lib:${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="${RUST_CAPI_INSTALL_DIR}/lib:${BUILD_DIR}:${LD_LIBRARY_PATH:-}"
    fi
    
    # Clear raw output file
    > "$RAW_OUTPUT"
    
    # Run each base pattern with each lsize_ir value
    for PATTERN in "${BASE_PATTERNS[@]}"; do
        # Parse pattern into arguments
        read -r NLAMBDA NDIGIT POSITIVE_ONLY STATISTICS LREAL_IR LREAL_TAU NUM <<< "$PATTERN"
        
        for LSIZE_IR in "${LSIZE_IR_VALUES[@]}"; do
            echo -e "  Running: statistics=${STATISTICS}, positive_only=${POSITIVE_ONLY}, lreal=${LREAL_IR}, lsize_ir=${LSIZE_IR}..."
            
            # Write separator to raw output
            echo "================== RUN: ${COMPILER} stat=${STATISTICS} pos=${POSITIVE_ONLY} lreal=${LREAL_IR} lsize=${LSIZE_IR} ==================" >> "$RAW_OUTPUT"
            
            # Run test_timing and save ALL output to raw file
            "$TEST_TIMING" "$NLAMBDA" "$NDIGIT" "$POSITIVE_ONLY" "$STATISTICS" \
                     "$LREAL_IR" "$LREAL_TAU" "$NUM" "$LSIZE_IR" >> "$RAW_OUTPUT" 2>&1
            
            EXIT_CODE=$?
            echo "EXIT_CODE: $EXIT_CODE" >> "$RAW_OUTPUT"
            echo "" >> "$RAW_OUTPUT"
            
            if [ $EXIT_CODE -ne 0 ]; then
                echo -e "${RED}    Error running test (exit code: $EXIT_CODE)${NC}"
            else
                echo -e "    Done"
            fi
        done
    done
    
    echo ""
    echo -e "${BLUE}Extracting data from raw output...${NC}"
    
    # Extract data lines from raw output file
    # Look for lines after "--- Data ---" markers
    grep -A1 -- "--- Data ---" "$RAW_OUTPUT" | grep -v -- "--- Data ---" | grep -v "^--$" | while read -r DATA_LINE; do
        if [ -n "$DATA_LINE" ]; then
            echo -e "${COMPILER}\t${DATA_LINE}" >> "$OUTPUT_FILE"
        fi
    done
    
    EXTRACTED_COUNT=$(grep -c "^${COMPILER}	" "$OUTPUT_FILE" 2>/dev/null || echo "0")
    echo -e "Extracted ${EXTRACTED_COUNT} data lines for ${COMPILER}"
    echo ""
done

# Summary
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}Benchmark Summary${NC}"
echo -e "${BLUE}==========================================${NC}"

if [ ${#TESTED_COMPILERS[@]} -eq 0 ]; then
    echo -e "${RED}No compilers were tested. Please build the project first.${NC}"
    exit 1
fi

echo -e "Tested compilers: ${TESTED_COMPILERS[*]}"
echo -e "Base patterns: ${#BASE_PATTERNS[@]}"
echo -e "lsize_ir values: ${#LSIZE_IR_VALUES[@]}"
echo -e "Results saved to: $OUTPUT_FILE"
echo ""
echo -e "Raw output files:"
for COMPILER in "${TESTED_COMPILERS[@]}"; do
    RAW_FILE="${SCRIPT_DIR}/timing_raw_${COMPILER}.txt"
    if [ -f "$RAW_FILE" ]; then
        echo -e "  - $RAW_FILE"
    fi
done
echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
