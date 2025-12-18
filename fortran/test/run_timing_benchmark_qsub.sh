#!/bin/sh
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=02:00:00
#PBS -q CP_001
#PBS -N timing-benchmark
#PBS -j oe
#PBS -o timing_benchmark.out

# Timing benchmark runner for sparse-ir-rs Fortran bindings (qsub version)
#
# Test patterns (6 base patterns × 11 lsize_ir values = 66 runs per compiler):
#   - Fermion (F): positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F
#   - Boson (B):   positive_only=T/lreal=T, positive_only=T/lreal=F, positive_only=F/lreal=F
#   - lsize_ir: 1, 4, 8, 10, 12, 13, 14, 15, 17, 18, 20
#   - num: 2784600
#
# Output files:
#   - timing_raw_<compiler>.txt : Raw output from test_timing (all output preserved)
#   - timing_results.tsv        : Extracted TSV data for Excel

module purge
module load oneapi/2024.2.1

cd $PBS_O_WORKDIR

export OMP_NUM_THREADS=1

# Use PBS_O_WORKDIR as the script directory (where qsub was submitted from)
# Note: $(dirname "$0") doesn't work in PBS because the script is copied to a temp location
SCRIPT_DIR="$PBS_O_WORKDIR"
BASE_DIR="${SCRIPT_DIR}/.."

# Output files
OUTPUT_FILE="${SCRIPT_DIR}/timing_results.tsv"

echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BASE_DIR: $BASE_DIR"
echo "OUTPUT_FILE: $OUTPUT_FILE"

# Define compilers and their build directories
COMPILERS="gfortran ifx ifort"

# Define base test patterns (without lsize_ir): nlambda ndigit positive_only statistics lreal_ir lreal_tau num
BASE_PATTERNS="
6 8 T F T T 2784600
6 8 T F F F 2784600
6 8 F F F F 2784600
6 8 T B T T 2784600
6 8 T B F F 2784600
6 8 F B F F 2784600
"

# Define lsize_ir values to test
LSIZE_IR_VALUES="1 4 8 10 12 13 14 15 17 18 20"

# Print header to output file
printf "compiler\tnlambda\tndigit\tpositive_only\tstatistics\tlreal_ir\tlreal_tau\tnum\tlsize_ir\tIR_size\tnfreq\tntau\tfit_matsu(s)\teval_tau(s)\tfit_tau(s)\teval_matsu(s)\ttotal(s)\tper_vector(s)\trel_error\n" > "$OUTPUT_FILE"

# Track which compilers were tested
TESTED_COMPILERS=""

# Run benchmarks for each compiler
for COMPILER in $COMPILERS; do
    BUILD_DIR="${BASE_DIR}/_build_${COMPILER}"
    TEST_TIMING="${BUILD_DIR}/test/test_timing"
    RAW_OUTPUT="${SCRIPT_DIR}/timing_raw_${COMPILER}.txt"
    
    # Check if executable exists
    if [ ! -f "$TEST_TIMING" ]; then
        echo "Skipping ${COMPILER}: executable not found at ${TEST_TIMING}"
        continue
    fi
    
    echo "=========================================="
    echo "Running benchmarks for ${COMPILER}..."
    echo "=========================================="
    echo "Raw output will be saved to: ${RAW_OUTPUT}"
    TESTED_COMPILERS="${TESTED_COMPILERS} ${COMPILER}"
    
    # Set library path
    RUST_CAPI_INSTALL_DIR="${BUILD_DIR}/_rust_capi_install"
    export LD_LIBRARY_PATH="${RUST_CAPI_INSTALL_DIR}/lib:${BUILD_DIR}:${LD_LIBRARY_PATH:-}"
    
    # Clear raw output file
    > "$RAW_OUTPUT"
    
    # Run each base pattern with each lsize_ir value
    echo "$BASE_PATTERNS" | while read -r PATTERN; do
        # Skip empty lines
        [ -z "$PATTERN" ] && continue
        
        # Parse pattern into arguments
        NLAMBDA=$(echo "$PATTERN" | awk '{print $1}')
        NDIGIT=$(echo "$PATTERN" | awk '{print $2}')
        POSITIVE_ONLY=$(echo "$PATTERN" | awk '{print $3}')
        STATISTICS=$(echo "$PATTERN" | awk '{print $4}')
        LREAL_IR=$(echo "$PATTERN" | awk '{print $5}')
        LREAL_TAU=$(echo "$PATTERN" | awk '{print $6}')
        NUM=$(echo "$PATTERN" | awk '{print $7}')
        
        for LSIZE_IR in $LSIZE_IR_VALUES; do
            echo "  Running: statistics=${STATISTICS}, positive_only=${POSITIVE_ONLY}, lreal=${LREAL_IR}, lsize_ir=${LSIZE_IR}..."
            
            # Write separator to raw output
            echo "================== RUN: ${COMPILER} stat=${STATISTICS} pos=${POSITIVE_ONLY} lreal=${LREAL_IR} lsize=${LSIZE_IR} ==================" >> "$RAW_OUTPUT"
            
            # Run test_timing and save ALL output to raw file
            "$TEST_TIMING" "$NLAMBDA" "$NDIGIT" "$POSITIVE_ONLY" "$STATISTICS" \
                     "$LREAL_IR" "$LREAL_TAU" "$NUM" "$LSIZE_IR" >> "$RAW_OUTPUT" 2>&1
            
            EXIT_CODE=$?
            echo "EXIT_CODE: $EXIT_CODE" >> "$RAW_OUTPUT"
            echo "" >> "$RAW_OUTPUT"
            
            if [ $EXIT_CODE -ne 0 ]; then
                echo "    Error running test (exit code: $EXIT_CODE)"
            else
                echo "    Done"
            fi
        done
    done
    
    echo ""
    echo "Extracting data from raw output..."
    
    # Extract data lines from raw output file
    # Look for lines after "--- Data ---" markers
    grep -A1 -- "--- Data ---" "$RAW_OUTPUT" | grep -v -- "--- Data ---" | grep -v "^--$" | while read -r DATA_LINE; do
        if [ -n "$DATA_LINE" ]; then
            printf "%s\t%s\n" "${COMPILER}" "${DATA_LINE}" >> "$OUTPUT_FILE"
        fi
    done
    
    EXTRACTED_COUNT=$(grep -c "^${COMPILER}	" "$OUTPUT_FILE" 2>/dev/null || echo "0")
    echo "Extracted ${EXTRACTED_COUNT} data lines for ${COMPILER}"
    echo ""
done

# Summary
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="

if [ -z "$TESTED_COMPILERS" ]; then
    echo "No compilers were tested. Please build the project first."
    exit 1
fi

echo "Tested compilers:${TESTED_COMPILERS}"
echo "Base patterns: 6"
echo "lsize_ir values: 11"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Raw output files:"
for COMPILER in $TESTED_COMPILERS; do
    RAW_FILE="${SCRIPT_DIR}/timing_raw_${COMPILER}.txt"
    if [ -f "$RAW_FILE" ]; then
        echo "  - $RAW_FILE"
    fi
done
echo ""
echo "Benchmark complete!"
