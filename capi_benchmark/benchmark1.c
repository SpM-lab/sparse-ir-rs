#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>
#include <time.h>
#include <stdbool.h>

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

int benchmark(double beta, double omega_max, double epsilon, int extra_size, int nrun, bool positive_only)
{
    Benchmark bench;

    int32_t status;
    int ndim = 2;

    printf("beta: %f\n", beta);
    printf("omega_max: %f\n", omega_max);
    printf("epsilon: %f\n", epsilon);
    printf("Extra size: %d\n", extra_size);
    printf("Number of runs: %d\n", nrun);

    benchmark_start(&bench, "Kernel creation");
    spir_kernel *kernel = spir_logistic_kernel_new(beta * omega_max, &status);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    assert(kernel != NULL);
    benchmark_end(&bench);

    // Create a pre-computed SVE result
    int lmax = -1;
    int n_gauss = -1;
    int Twork = SPIR_TWORK_AUTO;  // Auto-select: FLOAT64 for epsilon >= 1e-8, FLOAT64X2 for epsilon < 1e-8
    benchmark_start(&bench, "SVE computation");
    spir_sve_result *sve_logistic = spir_sve_result_new(
        kernel, epsilon, lmax, n_gauss, Twork, &status);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    assert(sve_logistic != NULL);
    benchmark_end(&bench);

    // Create fermionic and bosonic finite temperature bases with pre-computed
    // SVE result
    int max_size = -1;
    double epsilon_basis = 1e-10;
    spir_basis *basis =
        spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon_basis,
                       kernel, sve_logistic, max_size, &status);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    assert(basis != NULL);

    // Get basis size
    int32_t n_basis;
    status = spir_basis_get_size(basis, &n_basis);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_basis: %d\n", n_basis);

    // Get imaginary-time sampling points
    int32_t n_tau;
    status = spir_basis_get_n_default_taus(basis, &n_tau);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_tau: %d\n", n_tau);

    double *tau_points = (double *)malloc(n_tau * sizeof(double));
    status = spir_basis_get_default_taus(basis, tau_points);
    assert(status == SPIR_COMPUTATION_SUCCESS);

    // Create sampling object for imaginary-time domain
    spir_sampling *tau_sampling =
        spir_tau_sampling_new(basis, n_tau, tau_points, &status);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    assert(tau_sampling != NULL);

    // Get Matsubara frequency indices
    int32_t n_matsubara;
    status =
        spir_basis_get_n_default_matsus(basis, positive_only, &n_matsubara);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    int64_t *matsubara_indices =
        (int64_t *)malloc(n_matsubara * sizeof(int64_t));
    status =
        spir_basis_get_default_matsus(basis, positive_only, matsubara_indices);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_matsubara: %d\n", n_matsubara);

    // Create sampling object for Matsubara domain
    spir_sampling *matsubara_sampling = spir_matsu_sampling_new(
        basis, positive_only, n_matsubara, matsubara_indices, &status);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    assert(matsubara_sampling != NULL);

    // Create Green's function with a pole at 0.5*omega_max
    status = spir_sampling_get_npoints(matsubara_sampling, &n_matsubara);
    assert(status == SPIR_COMPUTATION_SUCCESS);

    // [n_matsubara, extra_size]
    c_complex *g_matsu_z = (c_complex *)malloc(n_matsubara * extra_size * sizeof(c_complex));

    c_complex *g_tau_z = (c_complex *)malloc(n_tau * extra_size * sizeof(c_complex));

    // [n_basis, extra_size]
    double *g_basis_d = (double *)malloc(n_basis * extra_size * sizeof(double));
    c_complex *g_basis_z = (c_complex *)malloc(n_basis * extra_size * sizeof(c_complex));

    // target dimension for fit
    int32_t target_dim = 0;

    // Test: matsubara, fit_zz
    int32_t dims[2] = {n_matsubara, extra_size};
    status = spir_sampling_fit_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  ndim, dims, target_dim, g_matsu_z,
                                  g_basis_z); // First run to warm up the cache
    benchmark_start(&bench, "fit_zz (Matsubara)");
    for (int i = 0; i < nrun; ++i) {
        status =
            spir_sampling_fit_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                 dims, target_dim, g_matsu_z, g_basis_z);
        assert(status == SPIR_COMPUTATION_SUCCESS);
    }
    benchmark_end(&bench);


    // Test: matsubara, eval_zz
    dims[0] = n_basis;
    dims[1] = extra_size;
    benchmark_start(&bench, "eval_zz (Matsubara)");
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_z, g_matsu_z);
        assert(status == SPIR_COMPUTATION_SUCCESS);
    }
    benchmark_end(&bench);

    // Test: matsubara, eval_dz
    benchmark_start(&bench, "eval_dz (Matsubara)");
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_dz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                       ndim, dims, target_dim, g_basis_d, g_matsu_z);
        assert(status == SPIR_COMPUTATION_SUCCESS);
    }
    benchmark_end(&bench);

    benchmark_start(&bench, "fit_zz (Tau)");
    dims[0] = n_tau;
    dims[1] = extra_size;
    status = spir_sampling_fit_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  ndim, dims, target_dim, g_tau_z, g_basis_z); // First run to warm up the cache
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_fit_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                      dims, target_dim, g_tau_z, g_basis_z);
        assert(status == SPIR_COMPUTATION_SUCCESS);
    }
    assert(status == SPIR_COMPUTATION_SUCCESS);
    benchmark_end(&bench);

    dims[0] = n_basis;
    dims[1] = extra_size;
    benchmark_start(&bench, "eval_zz (Tau)");
    for (int i = 0; i < nrun; ++i) {
        status = spir_sampling_eval_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                       dims, target_dim, g_basis_z, g_tau_z);
        assert(status == SPIR_COMPUTATION_SUCCESS);
    }
    benchmark_end(&bench);
    
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

    int extra_size = 1000; // dimension of the extra space

    int nrun = 10000; // Number of runs to average over
    
    printf("Benchmark (positive only = false)\n");
    benchmark(beta, omega_max, epsilon, extra_size, nrun, false);
    printf("\n");

    printf("Benchmark (positive only = true)\n");
    benchmark(beta, omega_max, epsilon, extra_size, nrun, true);
    printf("\n");

    return 0;
}


int main()
{
    //printf("Benchmark (beta = 1e+3, epsilon = 1e-6)\n");
    //benchmark_internal(1e+3, 1e-6);
    //printf("\n");

    printf("Benchmark (beta = 1e+5, epsilon = 1e-10)\n");
    benchmark_internal(1e+5, 1e-10);

    return 0;
}

