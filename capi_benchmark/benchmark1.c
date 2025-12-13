#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <sparseir/sparseir.h>
#include <time.h>
#include <stdbool.h>

// CHECK macro that works in both Debug and Release modes
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Simple benchmark utilities
typedef struct
{
    struct timespec start;
    const char *name;
} Benchmark;

static inline void benchmark_start(Benchmark *bench, const char *name)
{
    bench->name = name;
    clock_gettime(CLOCK_MONOTONIC, &bench->start);
}

static inline double benchmark_end(Benchmark *bench)
{
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - bench->start.tv_sec);
    elapsed += (end.tv_nsec - bench->start.tv_nsec) / 1e9;

    printf("%-30s: %10.6f ms\n", bench->name, elapsed * 1000.0);
    return elapsed;
}

// Helper to set dims for 3D array with target_dim
// dims = [sqrt_extra, sqrt_extra, sqrt_extra] with dims[target_dim] = target_size
void set_dims_3d(int32_t dims[3], int target_size, int sqrt_extra, int target_dim) {
    for (int i = 0; i < 3; i++) {
        dims[i] = (i == target_dim) ? target_size : sqrt_extra;
    }
}

int benchmark(double beta, double omega_max, double epsilon, int sqrt_extra_size, int nrun, bool positive_only)
{
    Benchmark bench;

    int32_t status;
    int ndim = 3;

    printf("beta: %f\n", beta);
    printf("omega_max: %f\n", omega_max);
    printf("epsilon: %f\n", epsilon);
    printf("sqrt_extra_size: %d (total extra = %d)\n", sqrt_extra_size, sqrt_extra_size * sqrt_extra_size);
    printf("Number of runs: %d\n", nrun);

    benchmark_start(&bench, "Kernel creation");
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    CHECK(kernel != NULL);
    benchmark_end(&bench);

    // Create a pre-computed SVE result
    int lmax = -1;
    int n_gauss = -1;
    int Twork = SPIR_TWORK_AUTO;  // Auto-select: FLOAT64 for epsilon >= 1e-8, FLOAT64X2 for epsilon < 1e-8
    benchmark_start(&bench, "SVE computation");
    spir_sve_result *sve_logistic = spir_sve_result_new(
        kernel, epsilon, lmax, n_gauss, Twork, &status);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    CHECK(sve_logistic != NULL);
    benchmark_end(&bench);

    // Create fermionic and bosonic finite temperature bases with pre-computed
    // SVE result
    int max_size = -1;
    double epsilon_basis = 1e-10;
    spir_basis *basis =
        spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon_basis,
                       kernel, sve_logistic, max_size, &status);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    CHECK(basis != NULL);

    // Get basis size
    int32_t n_basis;
    status = spir_basis_get_size(basis, &n_basis);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_basis: %d\n", n_basis);

    // Get imaginary-time sampling points
    int32_t n_tau;
    status = spir_basis_get_n_default_taus(basis, &n_tau);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_tau: %d\n", n_tau);

    double *tau_points = (double *)malloc(n_tau * sizeof(double));
    status = spir_basis_get_default_taus(basis, tau_points);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);

    // Create sampling object for imaginary-time domain
    spir_sampling *tau_sampling =
        spir_tau_sampling_new(basis, n_tau, tau_points, &status);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    CHECK(tau_sampling != NULL);

    // Get Matsubara frequency indices
    int32_t n_matsubara;
    status =
        spir_basis_get_n_default_matsus(basis, positive_only, &n_matsubara);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    int64_t *matsubara_indices =
        (int64_t *)malloc(n_matsubara * sizeof(int64_t));
    status =
        spir_basis_get_default_matsus(basis, positive_only, matsubara_indices);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_matsubara: %d\n", n_matsubara);

    // Create sampling object for Matsubara domain
    spir_sampling *matsubara_sampling = spir_matsu_sampling_new(
        basis, positive_only, n_matsubara, matsubara_indices, &status);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);
    CHECK(matsubara_sampling != NULL);

    // Create Green's function with a pole at 0.5*omega_max
    status = spir_sampling_get_npoints(matsubara_sampling, &n_matsubara);
    CHECK(status == SPIR_COMPUTATION_SUCCESS);

    // Allocate buffers large enough for all cases
    // Max size needed: max(n_matsubara, n_tau, n_basis) * sqrt_extra_size^2
    int max_target_size = n_matsubara > n_tau ? n_matsubara : n_tau;
    max_target_size = max_target_size > n_basis ? max_target_size : n_basis;
    size_t total_size = (size_t)max_target_size * sqrt_extra_size * sqrt_extra_size;

    Complex64 *g_matsu_z = (Complex64 *)malloc(total_size * sizeof(Complex64));
    Complex64 *g_tau_z = (Complex64 *)malloc(total_size * sizeof(Complex64));
    double *g_basis_d = (double *)malloc(total_size * sizeof(double));
    Complex64 *g_basis_z = (Complex64 *)malloc(total_size * sizeof(Complex64));

    int32_t dims[3];
    char bench_name[64];

    // Test all target_dim values: 0, 1, 2
    for (int target_dim = 0; target_dim < 3; target_dim++) {
        printf("\n--- target_dim = %d ---\n", target_dim);

        // Test: matsubara, fit_zz
        set_dims_3d(dims, n_matsubara, sqrt_extra_size, target_dim);
        status = spir_sampling_fit_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, dims, target_dim, g_matsu_z, g_basis_z);
        snprintf(bench_name, sizeof(bench_name), "fit_zz (Matsubara) dim=%d", target_dim);
        benchmark_start(&bench, bench_name);
        for (int i = 0; i < nrun; ++i) {
            status = spir_sampling_fit_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                          ndim, dims, target_dim, g_matsu_z, g_basis_z);
            CHECK(status == SPIR_COMPUTATION_SUCCESS);
        }
        benchmark_end(&bench);

        // Test: matsubara, eval_zz
        set_dims_3d(dims, n_basis, sqrt_extra_size, target_dim);
        status = spir_sampling_eval_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_z, g_matsu_z);
        if (status != SPIR_COMPUTATION_SUCCESS) {
            printf("eval_zz (Matsubara) dim=%d    : SKIPPED (status=%d)\n", target_dim, status);
        } else {
            snprintf(bench_name, sizeof(bench_name), "eval_zz (Matsubara) dim=%d", target_dim);
            benchmark_start(&bench, bench_name);
            for (int i = 0; i < nrun; ++i) {
                status = spir_sampling_eval_zz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                               ndim, dims, target_dim, g_basis_z, g_matsu_z);
            }
            benchmark_end(&bench);
        }

        // Test: matsubara, eval_dz
        snprintf(bench_name, sizeof(bench_name), "eval_dz (Matsubara) dim=%d", target_dim);
        benchmark_start(&bench, bench_name);
        for (int i = 0; i < nrun; ++i) {
            status = spir_sampling_eval_dz(matsubara_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                           ndim, dims, target_dim, g_basis_d, g_matsu_z);
            CHECK(status == SPIR_COMPUTATION_SUCCESS);
        }
        benchmark_end(&bench);

        // Test: tau, fit_zz
        set_dims_3d(dims, n_tau, sqrt_extra_size, target_dim);
        status = spir_sampling_fit_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, dims, target_dim, g_tau_z, g_basis_z);
        snprintf(bench_name, sizeof(bench_name), "fit_zz (Tau) dim=%d", target_dim);
        benchmark_start(&bench, bench_name);
        for (int i = 0; i < nrun; ++i) {
            status = spir_sampling_fit_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                          ndim, dims, target_dim, g_tau_z, g_basis_z);
            CHECK(status == SPIR_COMPUTATION_SUCCESS);
        }
        benchmark_end(&bench);

        // Test: tau, eval_zz
        set_dims_3d(dims, n_basis, sqrt_extra_size, target_dim);
        snprintf(bench_name, sizeof(bench_name), "eval_zz (Tau) dim=%d", target_dim);
        benchmark_start(&bench, bench_name);
        for (int i = 0; i < nrun; ++i) {
            status = spir_sampling_eval_zz(tau_sampling, NULL, SPIR_ORDER_COLUMN_MAJOR,
                                           ndim, dims, target_dim, g_basis_z, g_tau_z);
            CHECK(status == SPIR_COMPUTATION_SUCCESS);
        }
        benchmark_end(&bench);
    }
    
    // Clean up (order is arbitrary)
    free(matsubara_indices);
    free(g_matsu_z);
    free(g_basis_z);
    free(g_tau_z);
    free(tau_points);
    spir_basis_release(basis);
    spir_sampling_release(tau_sampling);
    spir_sampling_release(matsubara_sampling);

    return 0;
}


int benchmark_internal(double beta, double epsilon)
{
    double omega_max = 1.0; // Ultraviolet cutoff

    int sqrt_extra_size = 32; // sqrt of extra dimension size (total = 32^2 = 1024)

    int nrun = 1000; // Number of runs to average over
    
    printf("Benchmark (positive only = false)\n");
    benchmark(beta, omega_max, epsilon, sqrt_extra_size, nrun, false);
    printf("\n");

    printf("Benchmark (positive only = true)\n");
    benchmark(beta, omega_max, epsilon, sqrt_extra_size, nrun, true);
    printf("\n");

    return 0;
}


int main()
{
    printf("Benchmark (beta = 1e+3, epsilon = 1e-6)\n");
    benchmark_internal(1e+3, 1e-6);
    printf("\n");

    //printf("Benchmark (beta = 1e+5, epsilon = 1e-10)\n");
    //benchmark_internal(1e+5, 1e-10);

    return 0;
}

