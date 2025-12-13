#!/bin/bash
# Git bisect test script for SVE memory issue
# Returns 0 if problem exists (bad), 1 if no problem (good)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build library if needed
if [ ! -f "../target/release/libsparse_ir_capi.dylib" ] && [ ! -f "../target/release/libsparse_ir_capi.so" ]; then
    cd ..
    cargo build --release -p sparse-ir-capi 2>&1 | tail -5
    cd "$SCRIPT_DIR"
fi

# Install library
mkdir -p _install/lib _install/include/sparseir
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi
LIB_NAME="libsparse_ir_capi.${LIB_EXT}"
cp "../target/release/${LIB_NAME}" _install/lib/ 2>/dev/null || true
cp "../sparse-ir-capi/include/sparseir/sparseir.h" _install/include/sparseir/ 2>/dev/null || true

# Build test program
mkdir -p _build
cd _build
cmake .. >/dev/null 2>&1 || true
make test_sve_memory >/dev/null 2>&1 || {
    # If test_sve_memory doesn't exist, try to build a simple test
    cat > simple_sve_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <sparseir/sparseir.h>
#include <sys/resource.h>

int main() {
    int32_t status;
    double beta = 1e+5;
    double omega_max = 1.0;
    double epsilon = 1e-10;
    
    // Create kernel
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    if (status != 0 || kernel == NULL) {
        return 1;
    }
    
    // Compute SVE - this is where the memory issue occurs
    int lmax = -1;
    int n_gauss = -1;
    int Twork = -1; // SPIR_TWORK_AUTO
    spir_sve_result *sve = spir_sve_result_new(kernel, epsilon, lmax, n_gauss, Twork, &status);
    
    // Check memory usage
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        long mem_mb = usage.ru_maxrss / 1024;
        // Problem exists if memory usage > 100GB
        if (mem_mb > 100000) {
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
            return 0; // Problem exists (bad commit)
        }
    }
    
    if (sve) spir_sve_result_release(sve);
    if (kernel) spir_kernel_release(kernel);
    return 1; // No problem (good commit)
}
EOF
    gcc -I../_install/include -L../_install/lib -lsparse_ir_capi -o simple_sve_test simple_sve_test.c 2>/dev/null || return 125
    ./simple_sve_test
    exit_code=$?
    cd ..
    return $exit_code
}

# Run test
if [ -f "./test_sve_memory" ]; then
    # Use test_sve_memory if available
    OUTPUT=$(./test_sve_memory 1e+5 1e-10 2>&1 | grep "After SVE computation" || echo "")
    if echo "$OUTPUT" | grep -q "MB"; then
        MEM_MB=$(echo "$OUTPUT" | grep -oE '[0-9]+ MB' | head -1 | grep -oE '[0-9]+')
        if [ "$MEM_MB" -gt 100000 ]; then
            echo "BAD: Memory usage is ${MEM_MB} MB (> 100GB)"
            cd ..
            return 0  # Problem exists
        else
            echo "GOOD: Memory usage is ${MEM_MB} MB (< 100GB)"
            cd ..
            return 1  # No problem
        fi
    fi
fi

# Fallback: try simple test
if [ -f "./simple_sve_test" ]; then
    ./simple_sve_test
    exit_code=$?
    cd ..
    return $exit_code
fi

cd ..
return 125  # Skip this commit
