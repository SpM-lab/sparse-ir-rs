#!/bin/bash
# Helper script for manual bisection testing

set -e

COMMIT=$1
if [ -z "$COMMIT" ]; then
    echo "Usage: $0 <commit-hash>"
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Testing commit: $COMMIT"
git checkout "$COMMIT" >/dev/null 2>&1

# Build library
echo "Building library..."
cargo build --release -p sparse-ir-capi 2>&1 | tail -3

# Install library
cd capi_benchmark
mkdir -p _install/lib _install/include/sparseir
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi
LIB_NAME="libsparse_ir_capi.${LIB_EXT}"
cp "../target/release/${LIB_NAME}" _install/lib/ 2>/dev/null || echo "Library not found"
cp "../sparse-ir-capi/include/sparseir/sparseir.h" _install/include/sparseir/ 2>/dev/null || echo "Header not found"

# Build and run simple test
cat > _build/simple_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <sparseir/sparseir.h>
#include <sys/resource.h>

int main() {
    int32_t status;
    double beta = 1e+5;
    double omega_max = 1.0;
    double epsilon = 1e-10;
    
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    if (status != 0 || kernel == NULL) {
        fprintf(stderr, "Failed to create kernel\n");
        return 1;
    }
    
    int lmax = -1;
    int n_gauss = -1;
    int Twork = -1;
    spir_sve_result *sve = spir_sve_result_new(kernel, epsilon, lmax, n_gauss, Twork, &status);
    
    struct rusage usage;
    long mem_mb = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        mem_mb = usage.ru_maxrss / 1024;
    }
    
    printf("Memory usage: %ld MB\n", mem_mb);
    
    if (sve) spir_sve_result_release(sve);
    if (kernel) spir_kernel_release(kernel);
    
    // Return 0 if problem exists (> 100GB), 1 if no problem
    if (mem_mb > 100000) {
        return 0; // BAD
    }
    return 1; // GOOD
}
EOF

mkdir -p _build
cd _build
gcc -I../_install/include -L../_install/lib -lsparse_ir_capi -o simple_test ../simple_test.c 2>&1 | head -5
if [ -f simple_test ]; then
    ./simple_test
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "RESULT: BAD (memory issue exists)"
    else
        echo "RESULT: GOOD (no memory issue)"
    fi
    exit $RESULT
else
    echo "Failed to build test"
    exit 125
fi
