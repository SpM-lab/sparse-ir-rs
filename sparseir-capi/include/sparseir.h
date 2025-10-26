/**
 * @file sparseir.h
 * @brief C API for SparseIR library
 *
 * This header provides C-compatible interface for the SparseIR library.
 * Compatible with libsparseir C API.
 * 
 * ============================================================================
 * MAINTENANCE INSTRUCTIONS
 * ============================================================================
 * 
 * This header is manually maintained based on cbindgen-generated output.
 * 
 * To update this header when Rust API changes:
 * 
 * 1. Ensure cbindgen.toml exists and is properly configured
 * 2. Run: cargo build --features shared-lib
 *    This generates sparseir_capi.h via build.rs
 * 3. Copy function declarations from sparseir_capi.h to this file:
 *    - Keep the header structure (includes, defines, forward declarations)
 *    - Replace function declarations section with updated ones
 *    - Maintain compatibility with libsparseir naming conventions
 * 
 * Key differences from sparseir_capi.h:
 * - File name: sparseir.h (not sparseir_capi.h)
 * - Header guards: SPARSEIR_H (not SPARSEIR_CAPI_H)
 * - Focus on public API functions only
 * - Maintain opaque type forward declarations
 * 
 * Last updated: $(date)
 * Generated from: sparseir_capi.h (via cbindgen)
 */

#ifndef SPARSEIR_H
#define SPARSEIR_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define SPIR_ORDER_ROW_MAJOR 0
#define SPIR_ORDER_COLUMN_MAJOR 1
#define SPIR_STATISTICS_BOSONIC 0
#define SPIR_STATISTICS_FERMIONIC 1

#define SPIR_COMPUTATION_SUCCESS 0
#define SPIR_GET_IMPL_FAILED -1
#define SPIR_INVALID_DIMENSION -2
#define SPIR_INPUT_DIMENSION_MISMATCH -3
#define SPIR_OUTPUT_DIMENSION_MISMATCH -4
#define SPIR_NOT_SUPPORTED -5
#define SPIR_INVALID_ARGUMENT -6
#define SPIR_INTERNAL_ERROR -7

/**
 * Error codes for C API (compatible with libsparseir)
 */
typedef int StatusCode;

/**
 * Complex number type for C API (compatible with C's double complex)
 *
 * This type is compatible with C99's `double complex` and C++'s `std::complex<double>`.
 * Layout: `{double re; double im;}` with standard alignment.
 */
typedef struct Complex64 {
    double re;
    double im;
} Complex64;

// Forward declarations for opaque types
typedef struct spir_kernel spir_kernel;
typedef struct spir_basis spir_basis;
typedef struct spir_funcs spir_funcs;
typedef struct spir_sampling spir_sampling;
typedef struct spir_sve_result spir_sve_result;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Kernel functions
// ============================================================================

/**
 * @brief Creates a new logistic kernel
 * @param lambda Lambda parameter
 * @param status Pointer to store status code
 * @return Pointer to new kernel object, or NULL on failure
 */
spir_kernel* spir_logistic_kernel_new(double lambda, StatusCode* status);

/**
 * @brief Gets the lambda parameter of a kernel
 * @param kernel Pointer to kernel object
 * @return Lambda value
 */
double spir_kernel_lambda(const spir_kernel* kernel);

/**
 * @brief Computes kernel value at given tau
 * @param kernel Pointer to kernel object
 * @param tau Tau value
 * @return Computed kernel value
 */
double spir_kernel_compute(const spir_kernel* kernel, double tau);

/**
 * @brief Releases a kernel object
 * @param kernel Pointer to kernel object
 */
void spir_kernel_release(spir_kernel* kernel);

/**
 * @brief Clones a kernel object
 * @param src Source kernel object
 * @return Cloned kernel object, or NULL on failure
 */
spir_kernel* spir_kernel_clone(const spir_kernel* src);

/**
 * @brief Checks if a kernel object is assigned
 * @param kernel Pointer to kernel object
 * @return 1 if assigned, 0 otherwise
 */
int spir_kernel_is_assigned(const spir_kernel* kernel);

// ============================================================================
// Basis functions
// ============================================================================

/**
 * Create a new basis object
 */
spir_basis *spir_basis_new(int statistics, double lambda, double eps, StatusCode *status);

/**
 * Release a basis object
 */
void spir_basis_release(spir_basis *basis);

/**
 * Clone a basis object
 */
spir_basis *spir_basis_clone(const spir_basis *src);

/**
 * Checks if a basis object is assigned
 */
int32_t spir_basis_is_assigned(const spir_basis *obj);

/**
 * Get basis size
 */
StatusCode spir_basis_get_size(const spir_basis *b, int *size);

/**
 * Get basis statistics
 */
StatusCode spir_basis_get_stats(const spir_basis *b, int *statistics);

/**
 * Get singular values
 */
StatusCode spir_basis_get_singular_values(const spir_basis *b, double *svals);

/**
 * Get singular values (alias)
 */
StatusCode spir_basis_get_svals(const spir_basis *b, double *svals);

/**
 * Get number of default tau sampling points
 */
StatusCode spir_basis_get_n_default_taus(const spir_basis *b, int *num_points);

/**
 * Get default tau sampling points
 */
StatusCode spir_basis_get_default_taus(const spir_basis *b, double *points);

/**
 * Get number of default Matsubara sampling points
 */
StatusCode spir_basis_get_n_default_matsus(const spir_basis *b, bool positive_only, int *num_points);

/**
 * Get default Matsubara sampling points
 */
StatusCode spir_basis_get_default_matsus(const spir_basis *b, bool positive_only, int64_t *points);

/**
 * Get number of default w sampling points
 */
StatusCode spir_basis_get_n_default_ws(const spir_basis *b, int *num_points);

/**
 * Get default w sampling points
 */
StatusCode spir_basis_get_default_ws(const spir_basis *b, double *points);

/**
 * Get U functions
 */
spir_funcs *spir_basis_get_u(const spir_basis *b, StatusCode *status);

/**
 * Get V functions
 */
spir_funcs *spir_basis_get_v(const spir_basis *b, StatusCode *status);

/**
 * Get Uhat functions
 */
spir_funcs *spir_basis_get_uhat(const spir_basis *b, StatusCode *status);

// ============================================================================
// Functions
// ============================================================================

/**
 * Release a functions object
 */
void spir_funcs_release(spir_funcs *funcs);

/**
 * Clone a functions object
 */
spir_funcs *spir_funcs_clone(const spir_funcs *src);

/**
 * Checks if a functions object is assigned
 */
int32_t spir_funcs_is_assigned(const spir_funcs *funcs);

/**
 * Get functions size
 */
StatusCode spir_funcs_get_size(const spir_funcs *funcs, int *size);

/**
 * Evaluates functions at a single point
 */
StatusCode spir_funcs_eval(const spir_funcs *funcs, double x, double *out);

/**
 * Evaluates functions at a Matsubara frequency
 */
StatusCode spir_funcs_eval_matsu(const spir_funcs *funcs, int64_t n, Complex64 *out);

/**
 * Evaluates functions at multiple points
 */
StatusCode spir_funcs_batch_eval(const spir_funcs *funcs, int order, int num_points, const double *xs, double *out);

/**
 * Evaluates functions at multiple Matsubara frequencies
 */
StatusCode spir_funcs_batch_eval_matsu(const spir_funcs *funcs, int order, int num_freqs, const int64_t *matsubara_freq_indices, Complex64 *out);

/**
 * Get a slice of functions
 */
spir_funcs *spir_funcs_get_slice(const spir_funcs *funcs, int start, int end, StatusCode *status);

// ============================================================================
// Sampling functions
// ============================================================================

/**
 * Release a sampling object
 */
void spir_sampling_release(spir_sampling *sampling);

/**
 * Clone a sampling object
 */
spir_sampling *spir_sampling_clone(const spir_sampling *src);

/**
 * Checks if a sampling object is assigned
 */
int32_t spir_sampling_is_assigned(const spir_sampling *sampling);

/**
 * Create a new tau sampling object
 */
spir_sampling *spir_tau_sampling_new(const spir_basis *b, StatusCode *status);

/**
 * Create a new Matsubara sampling object
 */
spir_sampling *spir_matsu_sampling_new(const spir_basis *b, StatusCode *status);

/**
 * Evaluates basis coefficients at sampling points (double → double)
 */
StatusCode spir_sampling_eval_dd(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const double *input, double *out);

/**
 * Evaluates basis coefficients at sampling points (double → complex)
 */
StatusCode spir_sampling_eval_dz(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const double *input, Complex64 *out);

/**
 * Evaluates basis coefficients at sampling points (complex → complex)
 */
StatusCode spir_sampling_eval_zz(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const Complex64 *input, Complex64 *out);

/**
 * Fits values at sampling points to basis coefficients (double → double)
 */
StatusCode spir_sampling_fit_dd(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const double *input, double *out);

/**
 * Fits values at sampling points to basis coefficients (complex → double)
 */
StatusCode spir_sampling_fit_zd(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const Complex64 *input, double *out);

/**
 * Fits values at sampling points to basis coefficients (complex → complex)
 */
StatusCode spir_sampling_fit_zz(const spir_sampling *s, int order, int ndim, const int *input_dims, int target_dim, const Complex64 *input, Complex64 *out);

// ============================================================================
// SVE functions
// ============================================================================

/**
 * Release an SVE result object
 */
void spir_sve_result_release(spir_sve_result *sve);

/**
 * Clone an SVE result object
 */
spir_sve_result *spir_sve_result_clone(const spir_sve_result *src);

/**
 * Checks if an SVE result object is assigned
 */
int32_t spir_sve_result_is_assigned(const spir_sve_result *obj);

/**
 * Create a new SVE result object
 */
spir_sve_result *spir_sve_result_new(const spir_kernel *k, StatusCode *status);

/**
 * Get SVE result size
 */
StatusCode spir_sve_result_get_size(const spir_sve_result *sve, int *size);

/**
 * Get singular values from SVE result
 */
StatusCode spir_sve_result_get_svals(const spir_sve_result *sve, double *svals);

/**
 * Truncate SVE result
 */
spir_sve_result *spir_sve_result_truncate(const spir_sve_result *sve, int rank, StatusCode *status);

// ============================================================================
// DLR functions
// ============================================================================

/**
 * Create a new DLR basis object
 */
spir_basis *spir_dlr_new(int statistics, double lambda, double eps, StatusCode *status);

/**
 * Create a new DLR basis object with custom poles
 */
spir_basis *spir_dlr_new_with_poles(const spir_basis *b, const double *poles, int npoles, StatusCode *status);

/**
 * Get number of poles in DLR basis
 */
StatusCode spir_dlr_get_npoles(const spir_basis *dlr, int *npoles);

/**
 * Get poles from DLR basis
 */
StatusCode spir_dlr_get_poles(const spir_basis *dlr, double *poles);

/**
 * Convert IR to DLR (double → double)
 */
StatusCode spir_ir2dlr_dd(const spir_basis *dlr, int order, int ndim, const int *input_dims, int target_dim, const double *input, double *out);

/**
 * Convert IR to DLR (complex → complex)
 */
StatusCode spir_ir2dlr_zz(const spir_basis *dlr, int order, int ndim, const int *input_dims, int target_dim, const Complex64 *input, Complex64 *out);

/**
 * Convert DLR to IR (double → double)
 */
StatusCode spir_dlr2ir_dd(const spir_basis *dlr, int order, int ndim, const int *input_dims, int target_dim, const double *input, double *out);

/**
 * Convert DLR to IR (complex → complex)
 */
StatusCode spir_dlr2ir_zz(const spir_basis *dlr, int order, int ndim, const int *input_dims, int target_dim, const Complex64 *input, Complex64 *out);

// ============================================================================
// BLAS functions
// ============================================================================

/**
 * Register BLAS functions
 */
StatusCode spir_register_blas_functions(const void *cblas_dgemm, const void *cblas_zgemm);

/**
 * Register ILP64 BLAS functions
 */
StatusCode spir_register_ilp64_functions(const void *cblas_dgemm64, const void *cblas_zgemm64);

#ifdef __cplusplus
}
#endif

#endif  // SPARSEIR_H