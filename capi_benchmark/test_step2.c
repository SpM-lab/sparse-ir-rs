#include <stdio.h>
#include <stdlib.h>
#include <sparseir/sparseir.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

static long get_mem_mb(void) {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS)
        return info.resident_size / (1024 * 1024);
#endif
    return -1;
}

int main() {
    double beta = 1e+5, omega_max = 1.0, epsilon = 1e-10;
    int32_t status;
    
    printf("Initial: %ld MB\n", get_mem_mb());
    
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    spir_sve_result *sve = spir_sve_result_new(kernel, epsilon, -1, -1, -1, &status);
    spir_basis *basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, 
                                        epsilon, kernel, sve, -1, &status);
    
    int32_t n_basis, n_tau;
    spir_basis_get_size(basis, &n_basis);
    spir_basis_get_n_default_taus(basis, &n_tau);
    printf("n_basis=%d, n_tau=%d\n", n_basis, n_tau);
    
    double *tau_points = malloc(n_tau * sizeof(double));
    spir_basis_get_default_taus(basis, tau_points);
    spir_sampling *tau_sampling = spir_tau_sampling_new(basis, n_tau, tau_points, &status);
    
    printf("After setup: %ld MB\n", get_mem_mb());
    
    // Test with extra_size=1000 (like benchmark1.c)
    int extra_size = 1000;
    double *g_tau = malloc(n_tau * extra_size * sizeof(double));
    double *g_basis = malloc(n_basis * extra_size * sizeof(double));
    for (int i = 0; i < n_tau * extra_size; i++) g_tau[i] = 0.0;
    
    int32_t dims[2] = {n_tau, extra_size};
    
    // Run fit multiple times like benchmark1.c does
    printf("Running 100 iterations of fit_dd...\n");
    for (int run = 0; run < 100; run++) {
        status = spir_sampling_fit_dd(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR, 2, dims, 0, g_tau, g_basis);
        if (run % 10 == 0) {
            printf("  Run %d: %ld MB\n", run, get_mem_mb());
        }
    }
    printf("After 100 runs: %ld MB\n", get_mem_mb());
    
    free(g_tau);
    free(g_basis);
    free(tau_points);
    spir_sampling_release(tau_sampling);
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
    
    printf("Done: %ld MB\n", get_mem_mb());
    return 0;
}
