#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>

// Get memory usage in MB
static long get_memory_usage_mb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024; // On macOS, ru_maxrss is in KB
    }
    return -1;
}

// Print memory usage
static void print_memory_usage(const char *label) {
    long mem_mb = get_memory_usage_mb();
    printf("[MEMORY] %s: %ld MB\n", label, mem_mb);
}

// Check status and print error message
static bool check_status(int32_t status, const char *operation) {
    if (status != SPIR_COMPUTATION_SUCCESS) {
        printf("[ERROR] %s failed with status: %d\n", operation, status);
        switch (status) {
            case SPIR_GET_IMPL_FAILED:
                printf("  -> SPIR_GET_IMPL_FAILED\n");
                break;
            case SPIR_INVALID_DIMENSION:
                printf("  -> SPIR_INVALID_DIMENSION\n");
                break;
            case SPIR_INPUT_DIMENSION_MISMATCH:
                printf("  -> SPIR_INPUT_DIMENSION_MISMATCH\n");
                break;
            case SPIR_OUTPUT_DIMENSION_MISMATCH:
                printf("  -> SPIR_OUTPUT_DIMENSION_MISMATCH\n");
                break;
            case SPIR_NOT_SUPPORTED:
                printf("  -> SPIR_NOT_SUPPORTED\n");
                break;
            case SPIR_INVALID_ARGUMENT:
                printf("  -> SPIR_INVALID_ARGUMENT\n");
                break;
            case SPIR_INTERNAL_ERROR:
                printf("  -> SPIR_INTERNAL_ERROR\n");
                break;
            default:
                printf("  -> Unknown error code\n");
                break;
        }
        return false;
    }
    return true;
}

// Calculate memory size in MB
static double calculate_memory_mb(size_t bytes) {
    return bytes / (1024.0 * 1024.0);
}

int debug_benchmark(double beta, double omega_max, double epsilon, int extra_size, int nrun, bool positive_only)
{
    printf("\n========================================\n");
    printf("DEBUG BENCHMARK\n");
    printf("========================================\n");
    printf("beta: %f\n", beta);
    printf("omega_max: %f\n", omega_max);
    printf("epsilon: %f\n", epsilon);
    printf("Extra size: %d\n", extra_size);
    printf("Number of runs: %d\n", nrun);
    printf("Positive only: %s\n", positive_only ? "true" : "false");
    printf("========================================\n\n");

    int32_t status;
    int ndim = 2;

    print_memory_usage("Initial");

    // Step 1: Kernel creation
    printf("\n[STEP 1] Creating kernel...\n");
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    if (!check_status(status, "spir_logistic_kernel_new")) {
        return 1;
    }
    if (kernel == NULL) {
        printf("[ERROR] kernel is NULL\n");
        return 1;
    }
    printf("[OK] Kernel created\n");
    print_memory_usage("After kernel creation");

    // Step 2: SVE computation
    printf("\n[STEP 2] Computing SVE...\n");
    int lmax = -1;
    int n_gauss = -1;
    int Twork = SPIR_TWORK_AUTO;
    spir_sve_result *sve_logistic = spir_sve_result_new(
        kernel, epsilon, lmax, n_gauss, Twork, &status);
    if (!check_status(status, "spir_sve_result_new")) {
        spir_kernel_release(kernel);
        return 1;
    }
    if (sve_logistic == NULL) {
        printf("[ERROR] sve_logistic is NULL\n");
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] SVE computed\n");
    print_memory_usage("After SVE computation");

    // Step 3: Basis creation
    printf("\n[STEP 3] Creating basis...\n");
    int max_size = -1;
    double epsilon_basis = 1e-10;
    spir_basis *basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon_basis,
                                       kernel, sve_logistic, max_size, &status);
    if (!check_status(status, "spir_basis_new")) {
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    if (basis == NULL) {
        printf("[ERROR] basis is NULL\n");
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Basis created\n");
    print_memory_usage("After basis creation");

    // Step 4: Get basis size
    printf("\n[STEP 4] Getting basis size...\n");
    int32_t n_basis;
    status = spir_basis_get_size(basis, &n_basis);
    if (!check_status(status, "spir_basis_get_size")) {
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("n_basis: %d\n", n_basis);
    print_memory_usage("After getting basis size");

    // Step 5: Get tau points
    printf("\n[STEP 5] Getting tau points...\n");
    int32_t n_tau;
    status = spir_basis_get_n_default_taus(basis, &n_tau);
    if (!check_status(status, "spir_basis_get_n_default_taus")) {
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("n_tau: %d\n", n_tau);

    double *tau_points = (double *)malloc(n_tau * sizeof(double));
    if (tau_points == NULL) {
        printf("[ERROR] Failed to allocate tau_points\n");
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("Allocated tau_points: %.2f MB\n", calculate_memory_mb(n_tau * sizeof(double)));

    status = spir_basis_get_default_taus(basis, tau_points);
    if (!check_status(status, "spir_basis_get_default_taus")) {
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Tau points retrieved\n");
    print_memory_usage("After getting tau points");

    // Step 6: Create tau sampling
    printf("\n[STEP 6] Creating tau sampling...\n");
    spir_sampling *tau_sampling = spir_tau_sampling_new(basis, n_tau, tau_points, &status);
    if (!check_status(status, "spir_tau_sampling_new")) {
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    if (tau_sampling == NULL) {
        printf("[ERROR] tau_sampling is NULL\n");
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Tau sampling created\n");
    print_memory_usage("After tau sampling creation");

    // Step 7: Get Matsubara indices
    printf("\n[STEP 7] Getting Matsubara indices...\n");
    int32_t n_matsubara;
    status = spir_basis_get_n_default_matsus(basis, positive_only, &n_matsubara);
    if (!check_status(status, "spir_basis_get_n_default_matsus")) {
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("n_matsubara: %d\n", n_matsubara);

    int64_t *matsubara_indices = (int64_t *)malloc(n_matsubara * sizeof(int64_t));
    if (matsubara_indices == NULL) {
        printf("[ERROR] Failed to allocate matsubara_indices\n");
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("Allocated matsubara_indices: %.2f MB\n", calculate_memory_mb(n_matsubara * sizeof(int64_t)));

    status = spir_basis_get_default_matsus(basis, positive_only, matsubara_indices);
    if (!check_status(status, "spir_basis_get_default_matsus")) {
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Matsubara indices retrieved\n");
    print_memory_usage("After getting Matsubara indices");

    // Step 8: Create Matsubara sampling
    printf("\n[STEP 8] Creating Matsubara sampling...\n");
    spir_sampling *matsubara_sampling = spir_matsu_sampling_new(
        basis, positive_only, n_matsubara, matsubara_indices, &status);
    if (!check_status(status, "spir_matsu_sampling_new")) {
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    if (matsubara_sampling == NULL) {
        printf("[ERROR] matsubara_sampling is NULL\n");
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Matsubara sampling created\n");
    print_memory_usage("After Matsubara sampling creation");

    // Step 9: Verify n_matsubara
    printf("\n[STEP 9] Verifying n_matsubara...\n");
    status = spir_sampling_get_npoints(matsubara_sampling, &n_matsubara);
    if (!check_status(status, "spir_sampling_get_npoints")) {
        spir_sampling_release(matsubara_sampling);
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("Verified n_matsubara: %d\n", n_matsubara);
    print_memory_usage("After verifying n_matsubara");

    // Step 10: Allocate arrays
    printf("\n[STEP 10] Allocating arrays...\n");
    size_t g_matsu_z_size = n_matsubara * extra_size * sizeof(Complex64);
    size_t g_tau_z_size = n_tau * extra_size * sizeof(Complex64);
    size_t g_basis_d_size = n_basis * extra_size * sizeof(double);
    size_t g_basis_z_size = n_basis * extra_size * sizeof(Complex64);

    printf("g_matsu_z size: %.2f MB\n", calculate_memory_mb(g_matsu_z_size));
    printf("g_tau_z size: %.2f MB\n", calculate_memory_mb(g_tau_z_size));
    printf("g_basis_d size: %.2f MB\n", calculate_memory_mb(g_basis_d_size));
    printf("g_basis_z size: %.2f MB\n", calculate_memory_mb(g_basis_z_size));
    printf("Total array size: %.2f MB\n", calculate_memory_mb(g_matsu_z_size + g_tau_z_size + g_basis_d_size + g_basis_z_size));

    Complex64 *g_matsu_z = (Complex64 *)malloc(g_matsu_z_size);
    Complex64 *g_tau_z = (Complex64 *)malloc(g_tau_z_size);
    double *g_basis_d = (double *)malloc(g_basis_d_size);
    Complex64 *g_basis_z = (Complex64 *)malloc(g_basis_z_size);

    if (g_matsu_z == NULL || g_tau_z == NULL || g_basis_d == NULL || g_basis_z == NULL) {
        printf("[ERROR] Failed to allocate arrays\n");
        if (g_matsu_z) free(g_matsu_z);
        if (g_tau_z) free(g_tau_z);
        if (g_basis_d) free(g_basis_d);
        if (g_basis_z) free(g_basis_z);
        spir_sampling_release(matsubara_sampling);
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }

    // Initialize arrays with test data
    printf("Initializing arrays with test data...\n");
    for (int i = 0; i < n_matsubara * extra_size; i++) {
        g_matsu_z[i].re = 1.0 / (1.0 + i);
        g_matsu_z[i].im = 0.5 / (1.0 + i);
    }
    for (int i = 0; i < n_tau * extra_size; i++) {
        g_tau_z[i].re = 0.5;
        g_tau_z[i].im = 0.0;
    }
    memset(g_basis_d, 0, g_basis_d_size);
    memset(g_basis_z, 0, g_basis_z_size);

    printf("[OK] Arrays allocated and initialized\n");
    print_memory_usage("After array allocation");

    // Step 11: Test fit_zz (Matsubara) - single run first
    printf("\n[STEP 11] Testing fit_zz (Matsubara) - single run...\n");
    int32_t target_dim = 0;
    int32_t dims[2] = {n_matsubara, extra_size};
    status = spir_sampling_fit_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                  ndim, dims, target_dim, g_matsu_z, g_basis_z);
    if (!check_status(status, "spir_sampling_fit_zz (Matsubara, single run)")) {
        free(g_basis_z);
        free(g_basis_d);
        free(g_tau_z);
        free(g_matsu_z);
        spir_sampling_release(matsubara_sampling);
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    printf("[OK] Single fit_zz (Matsubara) succeeded\n");
    print_memory_usage("After single fit_zz (Matsubara)");

    // Step 12: Test fit_zz (Matsubara) - multiple runs
    printf("\n[STEP 12] Testing fit_zz (Matsubara) - %d runs...\n", nrun);
    printf("Running iterations (checking every 1000 iterations)...\n");
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_fit_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, dims, target_dim, g_matsu_z, g_basis_z);
        if (!check_status(status, "spir_sampling_fit_zz (Matsubara)")) {
            printf("[ERROR] Failed at iteration %d\n", i);
            free(g_basis_z);
            free(g_basis_d);
            free(g_tau_z);
            free(g_matsu_z);
            spir_sampling_release(matsubara_sampling);
            free(matsubara_indices);
            spir_sampling_release(tau_sampling);
            free(tau_points);
            spir_basis_release(basis);
            spir_sve_result_release(sve_logistic);
            spir_kernel_release(kernel);
            return 1;
        }
        if ((i + 1) % 1000 == 0) {
            printf("  Completed %d iterations\n", i + 1);
            print_memory_usage("During fit_zz loop");
        }
    }
    printf("[OK] All %d fit_zz (Matsubara) iterations succeeded\n", nrun);
    print_memory_usage("After fit_zz (Matsubara) loop");

    // Step 13: Test eval_zz (Matsubara)
    printf("\n[STEP 13] Testing eval_zz (Matsubara)...\n");
    dims[0] = n_basis;
    dims[1] = extra_size;
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_z, g_matsu_z);
        if (!check_status(status, "spir_sampling_eval_zz (Matsubara)")) {
            printf("[ERROR] Failed at iteration %d\n", i);
            free(g_basis_z);
            free(g_basis_d);
            free(g_tau_z);
            free(g_matsu_z);
            spir_sampling_release(matsubara_sampling);
            free(matsubara_indices);
            spir_sampling_release(tau_sampling);
            free(tau_points);
            spir_basis_release(basis);
            spir_sve_result_release(sve_logistic);
            spir_kernel_release(kernel);
            return 1;
        }
    }
    printf("[OK] All eval_zz (Matsubara) iterations succeeded\n");
    print_memory_usage("After eval_zz (Matsubara)");

    // Step 14: Test eval_dz (Matsubara)
    printf("\n[STEP 14] Testing eval_dz (Matsubara)...\n");
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_dz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_d, g_matsu_z);
        if (!check_status(status, "spir_sampling_eval_dz (Matsubara)")) {
            printf("[ERROR] Failed at iteration %d\n", i);
            free(g_basis_z);
            free(g_basis_d);
            free(g_tau_z);
            free(g_matsu_z);
            spir_sampling_release(matsubara_sampling);
            free(matsubara_indices);
            spir_sampling_release(tau_sampling);
            free(tau_points);
            spir_basis_release(basis);
            spir_sve_result_release(sve_logistic);
            spir_kernel_release(kernel);
            return 1;
        }
    }
    printf("[OK] All eval_dz (Matsubara) iterations succeeded\n");
    print_memory_usage("After eval_dz (Matsubara)");

    // Step 15: Test fit_zz (Tau)
    printf("\n[STEP 15] Testing fit_zz (Tau)...\n");
    dims[0] = n_tau;
    dims[1] = extra_size;
    status = spir_sampling_fit_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                  ndim, dims, target_dim, g_tau_z, g_basis_z);
    if (!check_status(status, "spir_sampling_fit_zz (Tau, single run)")) {
        free(g_basis_z);
        free(g_basis_d);
        free(g_tau_z);
        free(g_matsu_z);
        spir_sampling_release(matsubara_sampling);
        free(matsubara_indices);
        spir_sampling_release(tau_sampling);
        free(tau_points);
        spir_basis_release(basis);
        spir_sve_result_release(sve_logistic);
        spir_kernel_release(kernel);
        return 1;
    }
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_fit_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, dims, target_dim, g_tau_z, g_basis_z);
        if (!check_status(status, "spir_sampling_fit_zz (Tau)")) {
            printf("[ERROR] Failed at iteration %d\n", i);
            free(g_basis_z);
            free(g_basis_d);
            free(g_tau_z);
            free(g_matsu_z);
            spir_sampling_release(matsubara_sampling);
            free(matsubara_indices);
            spir_sampling_release(tau_sampling);
            free(tau_points);
            spir_basis_release(basis);
            spir_sve_result_release(sve_logistic);
            spir_kernel_release(kernel);
            return 1;
        }
    }
    printf("[OK] All fit_zz (Tau) iterations succeeded\n");
    print_memory_usage("After fit_zz (Tau)");

    // Step 16: Test eval_zz (Tau)
    printf("\n[STEP 16] Testing eval_zz (Tau)...\n");
    dims[0] = n_basis;
    dims[1] = extra_size;
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_z, g_tau_z);
        if (!check_status(status, "spir_sampling_eval_zz (Tau)")) {
            printf("[ERROR] Failed at iteration %d\n", i);
            free(g_basis_z);
            free(g_basis_d);
            free(g_tau_z);
            free(g_matsu_z);
            spir_sampling_release(matsubara_sampling);
            free(matsubara_indices);
            spir_sampling_release(tau_sampling);
            free(tau_points);
            spir_basis_release(basis);
            spir_sve_result_release(sve_logistic);
            spir_kernel_release(kernel);
            return 1;
        }
    }
    printf("[OK] All eval_zz (Tau) iterations succeeded\n");
    print_memory_usage("After eval_zz (Tau)");

    // Clean up
    printf("\n[CLEANUP] Releasing resources...\n");
    free(g_basis_z);
    free(g_basis_d);
    free(g_tau_z);
    free(g_matsu_z);
    spir_sampling_release(matsubara_sampling);
    free(matsubara_indices);
    spir_sampling_release(tau_sampling);
    free(tau_points);
    spir_basis_release(basis);
    spir_sve_result_release(sve_logistic);
    spir_kernel_release(kernel);
    printf("[OK] All resources released\n");
    print_memory_usage("After cleanup");

    printf("\n========================================\n");
    printf("DEBUG BENCHMARK COMPLETED SUCCESSFULLY\n");
    printf("========================================\n");

    return 0;
}

int main(int argc, char *argv[])
{
    double beta = 1e+5;
    double epsilon = 1e-10;
    int extra_size = 1000;
    int nrun = 10000;
    bool positive_only = false;

    // Parse command line arguments
    if (argc > 1) {
        beta = atof(argv[1]);
    }
    if (argc > 2) {
        epsilon = atof(argv[2]);
    }
    if (argc > 3) {
        extra_size = atoi(argv[3]);
    }
    if (argc > 4) {
        nrun = atoi(argv[4]);
    }
    if (argc > 5) {
        positive_only = (atoi(argv[5]) != 0);
    }

    printf("Debug Benchmark Configuration:\n");
    printf("  beta: %e\n", beta);
    printf("  epsilon: %e\n", epsilon);
    printf("  extra_size: %d\n", extra_size);
    printf("  nrun: %d\n", nrun);
    printf("  positive_only: %s\n", positive_only ? "true" : "false");
    printf("\n");

    double omega_max = 1.0;
    return debug_benchmark(beta, omega_max, epsilon, extra_size, nrun, positive_only);
}
