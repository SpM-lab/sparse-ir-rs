/**
 * @file sparseir_capi.h
 * @brief C API for SparseIR library (Manually generated)
 *
 * This header provides a C-compatible interface to the SparseIR Rust library.
 * Internal implementation details are hidden using opaque types.
 */

#ifndef SPARSEIR_CAPI_H
#define SPARSEIR_CAPI_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes
#define SPIR_COMPUTATION_SUCCESS 0
#define SPIR_INVALID_ARGUMENT -1
#define SPIR_INTERNAL_ERROR -2
#define SPIR_NOT_SUPPORTED -3

// Statistics constants
#define SPIR_STATISTICS_FERMIONIC 1
#define SPIR_STATISTICS_BOSONIC 0

// Order constants
#define SPIR_ORDER_ROW_MAJOR 0
#define SPIR_ORDER_COLUMN_MAJOR 1

// Complex number type for C compatibility
typedef struct {
    double real;
    double imag;
} Complex64;

// Opaque type declarations (incomplete types)
typedef struct spir_kernel spir_kernel;
typedef struct spir_basis spir_basis;
typedef struct spir_funcs spir_funcs;
typedef struct spir_sampling spir_sampling;
typedef struct spir_sve_result spir_sve_result;

// Status code type
typedef int StatusCode;

// ============================================================================
// Kernel functions
// ============================================================================

/**
 * @brief Creates a new logistic kernel
 * @param lambda Cutoff parameter Λ
 * @param status Pointer to store status code
 * @return Pointer to kernel object, or NULL on failure
 */
spir_kernel* spir_logistic_kernel_new(double lambda, int* status);

/**
 * @brief Gets the lambda parameter of a kernel
 * @param kernel Pointer to kernel object
 * @param lambda Pointer to store lambda value
 * @return Status code
 */
StatusCode spir_kernel_lambda(const spir_kernel* kernel, double* lambda);

/**
 * @brief Computes kernel value at given coordinates
 * @param kernel Pointer to kernel object
 * @param x X coordinate
 * @param y Y coordinate
 * @param result Pointer to store result
 * @return Status code
 */
StatusCode spir_kernel_compute(const spir_kernel* kernel, double x, double y, double* result);

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
// SVE Result functions
// ============================================================================

/**
 * @brief Releases an SVE result object
 * @param sve Pointer to SVE result object
 */
void spir_sve_result_release(spir_sve_result* sve);

/**
 * @brief Clones an SVE result object
 * @param src Source SVE result object
 * @return Cloned SVE result object, or NULL on failure
 */
spir_sve_result* spir_sve_result_clone(const spir_sve_result* src);

/**
 * @brief Checks if an SVE result object is assigned
 * @param sve Pointer to SVE result object
 * @return 1 if assigned, 0 otherwise
 */
int spir_sve_result_is_assigned(const spir_sve_result* sve);

// ============================================================================
// Basis functions
// ============================================================================

/**
 * @brief Releases a basis object
 * @param basis Pointer to basis object
 */
void spir_basis_release(spir_basis* basis);

/**
 * @brief Clones a basis object
 * @param src Source basis object
 * @return Cloned basis object, or NULL on failure
 */
spir_basis* spir_basis_clone(const spir_basis* src);

/**
 * @brief Checks if a basis object is assigned
 * @param basis Pointer to basis object
 * @return 1 if assigned, 0 otherwise
 */
int spir_basis_is_assigned(const spir_basis* basis);

/**
 * @brief Gets the size of a basis
 * @param b Pointer to basis object
 * @param size Pointer to store size
 * @return Status code
 */
StatusCode spir_basis_get_size(const spir_basis* b, int* size);

/**
 * @brief Gets the statistics type of a basis
 * @param b Pointer to basis object
 * @param statistics Pointer to store statistics type
 * @return Status code
 */
StatusCode spir_basis_get_stats(const spir_basis* b, int* statistics);

/**
 * @brief Gets the singular values of a basis
 * @param b Pointer to basis object
 * @param svals Pointer to store singular values
 * @return Status code
 */
StatusCode spir_basis_get_svals(const spir_basis* b, double* svals);

/**
 * @brief Gets the number of default tau sampling points
 * @param b Pointer to basis object
 * @param num_points Pointer to store number of points
 * @return Status code
 */
StatusCode spir_basis_get_n_default_taus(const spir_basis* b, int* num_points);

/**
 * @brief Gets the default tau sampling points
 * @param b Pointer to basis object
 * @param points Pre-allocated array to store tau points
 * @return Status code
 */
StatusCode spir_basis_get_default_taus(const spir_basis* b, double* points);

/**
 * @brief Gets the number of default omega sampling points
 * @param b Pointer to basis object
 * @param num_points Pointer to store number of points
 * @return Status code
 */
StatusCode spir_basis_get_n_default_ws(const spir_basis* b, int* num_points);

/**
 * @brief Gets the default omega sampling points
 * @param b Pointer to basis object
 * @param points Pre-allocated array to store omega points
 * @return Status code
 */
StatusCode spir_basis_get_default_ws(const spir_basis* b, double* points);

// ============================================================================
// Functions object functions
// ============================================================================

/**
 * @brief Releases a functions object
 * @param funcs Pointer to functions object
 */
void spir_funcs_release(spir_funcs* funcs);

/**
 * @brief Clones a functions object
 * @param src Source functions object
 * @return Cloned functions object, or NULL on failure
 */
spir_funcs* spir_funcs_clone(const spir_funcs* src);

/**
 * @brief Checks if a functions object is assigned
 * @param funcs Pointer to functions object
 * @return 1 if assigned, 0 otherwise
 */
int spir_funcs_is_assigned(const spir_funcs* funcs);

/**
 * @brief Evaluates functions at a single point
 * @param funcs Pointer to functions object
 * @param x Point at which to evaluate
 * @param out Pre-allocated array to store results
 * @return Status code
 */
StatusCode spir_funcs_eval(const spir_funcs* funcs, double x, double* out);

/**
 * @brief Evaluates functions at a Matsubara frequency
 * @param funcs Pointer to functions object
 * @param n Matsubara frequency index
 * @param out Pre-allocated array to store complex results
 * @return Status code
 */
StatusCode spir_funcs_eval_matsu(const spir_funcs* funcs, int64_t n, Complex64* out);

/**
 * @brief Evaluates functions at multiple points
 * @param funcs Pointer to functions object
 * @param order Memory layout order
 * @param num_points Number of points
 * @param xs Array of points
 * @param out Pre-allocated array to store results
 * @return Status code
 */
StatusCode spir_funcs_batch_eval(const spir_funcs* funcs, int order, int num_points, const double* xs, double* out);

/**
 * @brief Evaluates functions at multiple Matsubara frequencies
 * @param funcs Pointer to functions object
 * @param order Memory layout order
 * @param num_freqs Number of frequencies
 * @param matsubara_freq_indices Array of frequency indices
 * @param out Pre-allocated array to store complex results
 * @return Status code
 */
StatusCode spir_funcs_batch_eval_matsu(const spir_funcs* funcs, int order, int num_freqs, const int64_t* matsubara_freq_indices, Complex64* out);

// ============================================================================
// Sampling functions
// ============================================================================

/**
 * @brief Releases a sampling object
 * @param sampling Pointer to sampling object
 */
void spir_sampling_release(spir_sampling* sampling);

/**
 * @brief Clones a sampling object
 * @param src Source sampling object
 * @return Cloned sampling object, or NULL on failure
 */
spir_sampling* spir_sampling_clone(const spir_sampling* src);

/**
 * @brief Checks if a sampling object is assigned
 * @param sampling Pointer to sampling object
 * @return 1 if assigned, 0 otherwise
 */
int spir_sampling_is_assigned(const spir_sampling* sampling);

/**
 * @brief Evaluates basis coefficients at sampling points (double → double)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of basis coefficients
 * @param out Output array for evaluated values
 * @return Status code
 */
StatusCode spir_sampling_eval_dd(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const double* input, double* out);

/**
 * @brief Evaluates basis coefficients at sampling points (double → complex)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of basis coefficients
 * @param out Output array for evaluated values
 * @return Status code
 */
StatusCode spir_sampling_eval_dz(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const double* input, Complex64* out);

/**
 * @brief Evaluates basis coefficients at sampling points (complex → complex)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of basis coefficients
 * @param out Output array for evaluated values
 * @return Status code
 */
StatusCode spir_sampling_eval_zz(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const Complex64* input, Complex64* out);

/**
 * @brief Fits values at sampling points to basis coefficients (double → double)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of values at sampling points
 * @param out Output array for fitted basis coefficients
 * @return Status code
 */
StatusCode spir_sampling_fit_dd(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const double* input, double* out);

/**
 * @brief Fits values at sampling points to basis coefficients (complex → complex)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of values at sampling points
 * @param out Output array for fitted basis coefficients
 * @return Status code
 */
StatusCode spir_sampling_fit_zz(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const Complex64* input, Complex64* out);

/**
 * @brief Fits values at sampling points to basis coefficients (complex → double)
 * @param s Pointer to sampling object
 * @param order Memory layout order
 * @param ndim Number of dimensions
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for transformation
 * @param input Input array of values at sampling points
 * @param out Output array for fitted basis coefficients
 * @return Status code
 */
StatusCode spir_sampling_fit_zd(const spir_sampling* s, int order, int ndim, const int* input_dims, int target_dim, const Complex64* input, double* out);

#ifdef __cplusplus
}
#endif

#endif // SPARSEIR_CAPI_H
