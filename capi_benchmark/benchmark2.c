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

int benchmark_sve_only(double beta, double omega_max, double epsilon)
{
    Benchmark bench;

    int32_t status;

    printf("beta: %f\n", beta);
    printf("omega_max: %f\n", omega_max);
    printf("epsilon: %f\n", epsilon);

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

    // Get SVE result size
    int32_t n_svals;
    status = spir_sve_result_get_size(sve_logistic, &n_svals);
    assert(status == SPIR_COMPUTATION_SUCCESS);
    printf("n_svals: %d\n", n_svals);

    // Clean up
    spir_sve_result_release(sve_logistic);
    spir_kernel_release(kernel);

    return 0;
}

int main()
{
    printf("=== SVE Computation Benchmark ===\n\n");

    // Run multiple times for profiling
    int n_runs = 100;
    printf("Running %d iterations for profiling...\n", n_runs);
    for (int i = 0; i < n_runs; i++) {
        benchmark_sve_only(1e+3, 1.0, 1e-6);
    }
    printf("\nCompleted %d iterations\n", n_runs);

    return 0;
}

