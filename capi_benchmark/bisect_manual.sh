#!/bin/bash
# Manual bisection helper - test a specific commit

COMMIT=$1
if [ -z "$COMMIT" ]; then
    echo "Usage: $0 <commit-hash>"
    echo "Example: $0 24b1992"
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=========================================="
echo "Testing commit: $COMMIT"
echo "=========================================="

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Checkout commit
git checkout "$COMMIT" >/dev/null 2>&1 || {
    echo "Failed to checkout commit $COMMIT"
    exit 1
}

# Build library
echo "Building library..."
cargo build --release -p sparse-ir-capi 2>&1 | tail -5

# Install library
cd capi_benchmark
mkdir -p _install/lib _install/include/sparseir
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi
LIB_NAME="libsparse_ir_capi.${LIB_EXT}"
cp "../target/release/${LIB_NAME}" _install/lib/ 2>/dev/null || {
    echo "Warning: Library not found, may need to build"
    cd ..
    git checkout "$CURRENT_BRANCH" >/dev/null 2>&1
    exit 125
}

cp "../sparse-ir-capi/include/sparseir/sparseir.h" _install/include/sparseir/ 2>/dev/null || {
    echo "Warning: Header not found"
}

# Build simple test
mkdir -p _build
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
        fprintf(stderr, "Failed to create kernel: status=%d\n", status);
        return 125;
    }
    
    int lmax = -1;
    int n_gauss = -1;
    int Twork = -1; // SPIR_TWORK_AUTO
    spir_sve_result *sve = spir_sve_result_new(kernel, epsilon, lmax, n_gauss, Twork, &status);
    
    if (status != 0) {
        fprintf(stderr, "SVE computation failed: status=%d\n", status);
        if (kernel) spir_kernel_release(kernel);
        return 125;
    }
    
    struct rusage usage;
    long mem_mb = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        mem_mb = usage.ru_maxrss / 1024;
    }
    
    printf("Memory usage after SVE: %ld MB (%.2f GB)\n", mem_mb, mem_mb / 1024.0);
    
    if (sve) spir_sve_result_release(sve);
    if (kernel) spir_kernel_release(kernel);
    
    // Return 0 if problem exists (> 100GB), 1 if no problem
    if (mem_mb > 100000) {
        printf("RESULT: BAD (memory issue exists - %ld MB > 100GB)\n", mem_mb);
        return 0;
    } else {
        printf("RESULT: GOOD (no memory issue - %ld MB < 100GB)\n", mem_mb);
        return 1;
    }
}
EOF

cd _build
gcc -I../_install/include -L../_install/lib -lsparse_ir_capi -o simple_test simple_test.c 2>&1 | head -10
if [ ! -f simple_test ]; then
    echo "Failed to build test"
    cd ../..
    git checkout "$CURRENT_BRANCH" >/dev/null 2>&1
    exit 125
fi

# Run test
echo "Running test..."
./simple_test
RESULT=$?

cd ../..
git checkout "$CURRENT_BRANCH" >/dev/null 2>&1

exit $RESULT
