#!/bin/bash
# ============================================================================
# PBS/QSUB DIRECTIVES (uncomment these lines for qsub usage)
# ============================================================================
# #PBS -l select=1:ncpus=1:mem=4gb
# #PBS -l walltime=02:00:00
# #PBS -q your_queue_name
# #PBS -N timing-benchmark
# #PBS -j oe
# #PBS -o timing_benchmark.out

# ============================================================================
# Timing benchmark runner for sparse-ir-rs Fortran bindings
# ============================================================================
# This script can be used both locally and with qsub/PBS.
#
# For qsub usage:
#   1. Uncomment the PBS directives above (remove "#" from lines starting with "#PBS")
#   2. Uncomment the module commands below if needed
#   3. Submit with: qsub run_timing_benchmark.sh
#
# For local usage:
#   1. Keep PBS directives commented out
#   2. Keep module commands commented out (or adjust for your environment)
#   3. Run with: ./run_timing_benchmark.sh
# ============================================================================

# ============================================================================
# CONFIGURATION: Test Duration
# ============================================================================
# Set to "short" for quick tests or "long" for comprehensive benchmarks
# SHORT: 6 patterns × 2 lsize_ir values = 12 runs per compiler (num=185640)
# LONG:  6 patterns × 11 lsize_ir values = 66 runs per compiler (num=2784600)
TEST_DURATION="short"  # Change to "long" for comprehensive benchmarks

# ============================================================================
# MODULE COMMANDS (uncomment and adjust for your environment if needed)
# ============================================================================
# module purge
# module load oneapi/XXXXXX
# Or load other compiler modules as needed for your environment

set -euo pipefail

# Colors for output (only used in local mode)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect if running under PBS/qsub
if [ -n "${PBS_O_WORKDIR:-}" ]; then
    # Running under PBS/qsub
    SCRIPT_DIR="$PBS_O_WORKDIR"
    cd "$SCRIPT_DIR"
    USE_COLORS=false
else
    # Running locally
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    cd "${SCRIPT_DIR}"
    USE_COLORS=true
fi

# Base directory for builds
BASE_DIR="${SCRIPT_DIR}/.."

# Output files
OUTPUT_FILE="${SCRIPT_DIR}/timing_results.tsv"

# Helper function for colored output
print_msg() {
    local color=$1
    shift
    if [ "$USE_COLORS" = true ]; then
        echo -e "${color}$@${NC}"
    else
        echo "$@"
    fi
}

print_msg "${BLUE}" "SCRIPT_DIR: $SCRIPT_DIR"
print_msg "${BLUE}" "BASE_DIR: $BASE_DIR"
print_msg "${BLUE}" "OUTPUT_FILE: $OUTPUT_FILE"
print_msg "${BLUE}" "TEST_DURATION: $TEST_DURATION"

# Define compilers and their build directories
COMPILERS=("gfortran" "ifx" "ifort")

# ============================================================================
# TEST PATTERNS CONFIGURATION
# ============================================================================
# Base test patterns (without lsize_ir): nlambda ndigit positive_only statistics lreal_ir lreal_tau num
# Test patterns:
#   - Fermion (F): positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F
#   - Boson (B):   positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F

if [ "$TEST_DURATION" = "long" ]; then
    # Long test: comprehensive benchmark
    BASE_PATTERNS=(
        "6 8 T F T T 2784600"
        "6 8 T F F F 2784600"
        "6 8 F F F F 2784600"
        "6 8 T B T T 2784600"
        "6 8 T B F F 2784600"
        "6 8 F B F F 2784600"
    )
    LSIZE_IR_VALUES=(1 4 8 10 12 13 14 15 17 18 20)
    print_msg "${YELLOW}" "Running LONG test: 6 patterns × 11 lsize_ir values = 66 runs per compiler"
else
    # Short test: quick validation
    BASE_PATTERNS=(
        "6 8 T F T T 185640"
        "6 8 T F F F 185640"
        "6 8 F F F F 185640"
        "6 8 T B T T 185640"
        "6 8 T B F F 185640"
        "6 8 F B F F 185640"
    )
    LSIZE_IR_VALUES=(1 10)
    print_msg "${YELLOW}" "Running SHORT test: 6 patterns × 2 lsize_ir values = 12 runs per compiler"
fi

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
        print_msg "${YELLOW}" "Skipping ${COMPILER}: executable not found at ${TEST_TIMING}"
        continue
    fi
    
    print_msg "${GREEN}" "=========================================="
    print_msg "${GREEN}" "Running benchmarks for ${COMPILER}..."
    print_msg "${GREEN}" "=========================================="
    print_msg "" "Raw output will be saved to: ${RAW_OUTPUT}"
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
            print_msg "" "  Running: statistics=${STATISTICS}, positive_only=${POSITIVE_ONLY}, lreal_ir=${LREAL_IR}, lreal_tau=${LREAL_TAU}, lsize_ir=${LSIZE_IR}..."
            
            # Write separator to raw output
            echo "================== RUN: ${COMPILER} stat=${STATISTICS} pos=${POSITIVE_ONLY} lreal_ir=${LREAL_IR} lreal_tau=${LREAL_TAU} lsize=${LSIZE_IR} ==================" >> "$RAW_OUTPUT"
            
            # Run test_timing and save ALL output to raw file
            "$TEST_TIMING" "$NLAMBDA" "$NDIGIT" "$POSITIVE_ONLY" "$STATISTICS" \
                     "$LREAL_IR" "$LREAL_TAU" "$NUM" "$LSIZE_IR" >> "$RAW_OUTPUT" 2>&1
            
            EXIT_CODE=$?
            echo "EXIT_CODE: $EXIT_CODE" >> "$RAW_OUTPUT"
            echo "" >> "$RAW_OUTPUT"
            
            if [ $EXIT_CODE -ne 0 ]; then
                print_msg "${RED}" "    Error running test (exit code: $EXIT_CODE)"
            else
                print_msg "" "    Done"
            fi
        done
    done
    
    echo ""
    print_msg "${BLUE}" "Extracting data from raw output..."
    
    # Extract data lines from raw output file
    # Look for lines after "--- Data ---" markers
    grep -A1 -- "--- Data ---" "$RAW_OUTPUT" | grep -v -- "--- Data ---" | grep -v "^--$" | while read -r DATA_LINE; do
        if [ -n "$DATA_LINE" ]; then
            echo -e "${COMPILER}\t${DATA_LINE}" >> "$OUTPUT_FILE"
        fi
    done
    
    EXTRACTED_COUNT=$(grep -c "^${COMPILER}	" "$OUTPUT_FILE" 2>/dev/null || echo "0")
    print_msg "" "Extracted ${EXTRACTED_COUNT} data lines for ${COMPILER}"
    echo ""
done

# Summary
print_msg "${BLUE}" "=========================================="
print_msg "${BLUE}" "Benchmark Summary"
print_msg "${BLUE}" "=========================================="

if [ ${#TESTED_COMPILERS[@]} -eq 0 ]; then
    print_msg "${RED}" "No compilers were tested. Please build the project first."
    exit 1
fi

print_msg "" "Tested compilers: ${TESTED_COMPILERS[*]}"
print_msg "" "Base patterns: ${#BASE_PATTERNS[@]}"
print_msg "" "lsize_ir values: ${#LSIZE_IR_VALUES[@]}"
print_msg "" "Results saved to: $OUTPUT_FILE"
echo ""
print_msg "" "Raw output files:"
for COMPILER in "${TESTED_COMPILERS[@]}"; do
    RAW_FILE="${SCRIPT_DIR}/timing_raw_${COMPILER}.txt"
    if [ -f "$RAW_FILE" ]; then
        print_msg "" "  - $RAW_FILE"
    fi
done
echo ""
print_msg "${GREEN}" "Benchmark complete!"
