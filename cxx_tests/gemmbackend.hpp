#ifndef GEMMBACKEND_HPP
#define GEMMBACKEND_HPP

#include <sparseir/sparseir.h>
#include <iostream>

#ifdef SPARSEIR_USE_BLAS
// Fortran BLAS function declarations (with trailing underscore)
extern "C" {
#ifdef SPARSEIR_USE_BLAS_ILP64
    // ILP64 BLAS (64-bit integers) - function names are the same, only argument types differ
    void dgemm_(const char* transa, const char* transb,
                const int64_t* m, const int64_t* n, const int64_t* k,
                const double* alpha, const double* a, const int64_t* lda,
                const double* b, const int64_t* ldb,
                const double* beta, double* c, const int64_t* ldc);

    void zgemm_(const char* transa, const char* transb,
                const int64_t* m, const int64_t* n, const int64_t* k,
                const void* alpha, const void* a, const int64_t* lda,
                const void* b, const int64_t* ldb,
                const void* beta, void* c, const int64_t* ldc);
#else
    // LP64 BLAS (32-bit integers)
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);

    void zgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const void* alpha, const void* a, const int* lda,
                const void* b, const int* ldb,
                const void* beta, void* c, const int* ldc);
#endif
}

// Global BLAS backend (initialized once)
static spir_gemm_backend* g_blas_backend = nullptr;

// Initialize BLAS backend if SPARSEIR_USE_BLAS is enabled
static void init_blas_backend() {
    if (g_blas_backend == nullptr) {
#ifdef SPARSEIR_USE_BLAS_ILP64
        // Use ILP64 BLAS (64-bit integers)
        std::cout << "Initializing BLAS backend: ILP64 (64-bit integers)" << std::endl;
        g_blas_backend = spir_gemm_backend_new_from_fblas_ilp64(
            reinterpret_cast<const void*>(dgemm_),
            reinterpret_cast<const void*>(zgemm_)
        );
#else
        // Use LP64 BLAS (32-bit integers, default)
        std::cout << "Initializing BLAS backend: LP64 (32-bit integers)" << std::endl;
        g_blas_backend = spir_gemm_backend_new_from_fblas_lp64(
            reinterpret_cast<const void*>(dgemm_),
            reinterpret_cast<const void*>(zgemm_)
        );
#endif
        if (g_blas_backend == nullptr) {
            std::cerr << "Warning: Failed to create BLAS backend, using default backend" << std::endl;
        } else {
            std::cout << "BLAS backend initialized successfully" << std::endl;
        }
    }
}
#endif

// Get backend pointer (returns BLAS backend if available, NULL otherwise)
static inline spir_gemm_backend* get_backend() {
#ifdef SPARSEIR_USE_BLAS
    init_blas_backend();
    return g_blas_backend;
#else
    return nullptr;
#endif
}

#endif // GEMMBACKEND_HPP

