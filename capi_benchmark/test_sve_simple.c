#include <stdio.h>
#include <stdlib.h>
#include <sparseir/sparseir.h>

int main(int argc, char *argv[]) {
    double beta = argc > 1 ? atof(argv[1]) : 1e+5;
    double epsilon = argc > 2 ? atof(argv[2]) : 1e-10;
    
    printf("Testing SVE: beta=%e, epsilon=%e\n", beta, epsilon);
    
    int32_t status;
    spir_kernel *kernel = spir_logistic_kernel_new(beta, &status);
    if (status != 0 || kernel == NULL) {
        printf("Kernel creation failed\n");
        return 1;
    }
    printf("Kernel created\n");
    
    printf("Computing SVE...\n");
    spir_sve_result *sve = spir_sve_result_new(kernel, epsilon, -1, -1, -1, &status);
    if (status != 0 || sve == NULL) {
        printf("SVE failed with status: %d\n", status);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("SVE completed successfully\n");
    
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
    return 0;
}
