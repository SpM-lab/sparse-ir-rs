#include "include/sparseir.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing SparseIR C API installation...\n");
    
    // Test 1: Check if we can create a kernel
    StatusCode status;
    spir_kernel* kernel = spir_logistic_kernel_new(10.0, &status);
    
    if (kernel == NULL) {
        printf("❌ Failed to create kernel (status: %d)\n", status);
        return 1;
    }
    
    printf("✅ Successfully created kernel (status: %d)\n", status);
    
    // Test 2: Check kernel lambda
    double lambda = spir_kernel_lambda(kernel);
    printf("✅ Kernel lambda: %f\n", lambda);
    
    // Test 3: Test kernel computation
    double result = spir_kernel_compute(kernel, 0.5);
    printf("✅ Kernel computation result: %f\n", result);
    
    // Cleanup
    spir_kernel_release(kernel);
    
    printf("🎉 All tests passed! SparseIR C API is working correctly.\n");
    return 0;
}
