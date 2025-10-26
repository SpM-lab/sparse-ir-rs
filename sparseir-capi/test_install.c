#include "include/sparseir_capi.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing SparseIR C API installation...\n");
    
    // Test 1: Check if we can create a kernel
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(10.0, &status);
    
    if (kernel == NULL) {
        printf("❌ Failed to create kernel (status: %d)\n", status);
        return 1;
    }
    
    printf("✅ Successfully created kernel (status: %d)\n", status);
    
    // Test 2: Check kernel lambda
    double lambda;
    status = spir_kernel_lambda(kernel, &lambda);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        printf("❌ Failed to get kernel lambda (status: %d)\n", status);
        spir_kernel_release(kernel);
        return 1;
    }
    
    printf("✅ Kernel lambda: %f\n", lambda);
    
    // Test 3: Test kernel computation
    double result;
    status = spir_kernel_compute(kernel, 0.5, 0.5, &result);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        printf("❌ Failed to compute kernel (status: %d)\n", status);
        spir_kernel_release(kernel);
        return 1;
    }
    
    printf("✅ Kernel computation result: %f\n", result);
    
    // Cleanup
    spir_kernel_release(kernel);
    
    printf("🎉 All tests passed! SparseIR C API is working correctly.\n");
    return 0;
}
