/**
 * @file sparseir.h
 * @brief C API for SparseIR library
 *
 * This header provides C-compatible interface for the SparseIR library.
 * Compatible with libsparseir C API.
 *
 * This header is automatically generated from Rust source code using cbindgen.
 * Do not edit manually - changes will be overwritten on next build.
 */

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_MSC_VER) || defined(__cplusplus)
// MSVC doesn't support C99 complex types by default
// For C++ compilation, use std::complex to avoid C99 extension warnings
#include <complex>
typedef std::complex<double> c_complex;
#else
#include <complex.h>
// Define a C-compatible type alias for the C99 complex number.
typedef double _Complex c_complex;
#endif

#define SPIR_ORDER_ROW_MAJOR 0

#define SPIR_ORDER_COLUMN_MAJOR 1

#define SPIR_STATISTICS_BOSONIC 0

#define SPIR_STATISTICS_FERMIONIC 1

#define SPIR_TWORK_FLOAT64 0

#define SPIR_TWORK_FLOAT64X2 1

#define SPIR_TWORK_AUTO -1

#define SPIR_SVDSTRAT_FAST 0

#define SPIR_SVDSTRAT_ACCURATE 1

#define SPIR_SVDSTRAT_AUTO -1

/**
 * Opaque basis type for C API (compatible with libsparseir)
 *
 * Represents a finite temperature basis (IR or DLR).
 *
 * Note: Named `spir_basis` to match libsparseir C++ API exactly.
 * The internal structure is hidden using a void pointer to prevent exposing BasisType to C.
 */
typedef struct spir_basis {
  void *_private;
} spir_basis;

/**
 * Opaque kernel type for C API (compatible with libsparseir)
 *
 * This is a tagged union that can hold either LogisticKernel or RegularizedBoseKernel.
 * The actual type is determined by which constructor was used.
 *
 * Note: Named `spir_kernel` to match libsparseir C++ API exactly.
 * The internal structure is hidden using a void pointer to prevent exposing KernelType to C.
 */
typedef struct spir_kernel {
  void *_private;
} spir_kernel;

/**
 * Opaque SVE result type for C API (compatible with libsparseir)
 *
 * Contains singular values and singular functions from SVE computation.
 *
 * Note: Named `spir_sve_result` to match libsparseir C++ API exactly.
 * The internal structure is hidden using a void pointer to prevent exposing Arc<SVEResult> to C.
 */
typedef struct spir_sve_result {
  void *_private;
} spir_sve_result;

/**
 * Error codes for C API (compatible with libsparseir)
 */


/**
 * Opaque funcs type for C API (compatible with libsparseir)
 *
 * Wraps piecewise Legendre polynomial representations:
 * - PiecewiseLegendrePolyVector for u and v
 * - PiecewiseLegendreFTVector for uhat
 *
 * Note: Named `spir_funcs` to match libsparseir C++ API exactly.
 * The internal FuncsType is hidden using a void pointer, but beta is kept as a public field.
 */
typedef struct spir_funcs {
  void *_private;
  double beta;
} spir_funcs;



/**
 * Sampling type for C API (unified type for all domains)
 *
 * This wraps different sampling implementations:
 * - TauSampling (for tau-domain)
 * - MatsubaraSampling (for Matsubara frequencies, full range or positive-only)
 * The internal structure is hidden using a void pointer to prevent exposing SamplingType to C.
 */
typedef struct spir_sampling {
  void *_private;
} spir_sampling;

#define SPIR_COMPUTATION_SUCCESS 0

#define SPIR_GET_IMPL_FAILED -1

#define SPIR_INVALID_DIMENSION -2

#define SPIR_INPUT_DIMENSION_MISMATCH -3

#define SPIR_OUTPUT_DIMENSION_MISMATCH -4

#define SPIR_NOT_SUPPORTED -5

#define SPIR_INVALID_ARGUMENT -6

#define SPIR_INTERNAL_ERROR -7

/**
 * Manual release function (replaces macro-generated one)
 */
#ifdef __cplusplus
extern "C" {
#endif

 void spir_basis_release(struct spir_basis *basis);

/**
 * Manual clone function (replaces macro-generated one)
 */
 struct spir_basis *spir_basis_clone(const struct spir_basis *src);

/**
 * Manual is_assigned function (replaces macro-generated one)
 */
 int32_t spir_basis_is_assigned(const struct spir_basis *obj);

/**
 * Create a finite temperature basis (libsparseir compatible)
 *
 * # Arguments
 * * `statistics` - 0 for Bosonic, 1 for Fermionic
 * * `beta` - Inverse temperature (must be > 0)
 * * `omega_max` - Frequency cutoff (must be > 0)
 * * `epsilon` - Accuracy target (must be > 0)
 * * `k` - Kernel object (can be NULL if sve is provided)
 * * `sve` - Pre-computed SVE result (can be NULL, will compute if needed)
 * * `max_size` - Maximum basis size (-1 for no limit)
 * * `status` - Pointer to store status code
 *
 * # Returns
 * * Pointer to basis object, or NULL on failure
 *
 * # Safety
 * The caller must ensure `status` is a valid pointer.
 */

struct spir_basis *spir_basis_new(int statistics,
                                  double beta,
                                  double omega_max,
                                  double epsilon,
                                  const struct spir_kernel *k,
                                  const struct spir_sve_result *sve,
                                  int max_size,
                                  int *status);

/**
 * Create a finite temperature basis from SVE result and custom inv_weight function
 *
 * This function creates a basis from a pre-computed SVE result and a custom
 * inverse weight function. The inv_weight function is used to scale the basis
 * functions in the frequency domain.
 *
 * # Arguments
 * * `statistics` - 0 for Bosonic, 1 for Fermionic
 * * `beta` - Inverse temperature (must be > 0)
 * * `omega_max` - Frequency cutoff (must be > 0)
 * * `epsilon` - Accuracy target (must be > 0)
 * * `lambda` - Kernel parameter Λ = β * ωmax (must be > 0)
 * * `ypower` - Power of y in kernel (typically 0 or 1)
 * * `conv_radius` - Convergence radius for Fourier transform
 * * `sve` - Pre-computed SVE result (must not be NULL)
 * * `inv_weight_funcs` - Custom inv_weight function (must not be NULL)
 * * `max_size` - Maximum basis size (-1 for no limit)
 * * `status` - Pointer to store status code
 *
 * # Returns
 * * Pointer to basis object, or NULL on failure
 *
 * # Note
 * Currently, the inv_weight function is evaluated but the custom weight is not
 * fully integrated into the basis construction. The basis is created using
 * the standard from_sve_result method with the kernel's default inv_weight.
 * This is a limitation of the current Rust implementation compared to the C++ version.
 *
 * # Safety
 * The caller must ensure `status` is a valid pointer.
 */

struct spir_basis *spir_basis_new_from_sve_and_inv_weight(int statistics,
                                                          double beta,
                                                          double omega_max,
                                                          double epsilon,
                                                          double lambda,
                                                          int _ypower,
                                                          double _conv_radius,
                                                          const struct spir_sve_result *sve,
                                                          const struct spir_funcs *inv_weight_funcs,
                                                          int max_size,
                                                          int *status);

/**
 * Get the number of basis functions
 *
 * # Arguments
 * * `b` - Basis object
 * * `size` - Pointer to store the size
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or size is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_basis_get_size(const struct spir_basis *b, int *size);

/**
 * Get singular values from a basis
 *
 * # Arguments
 * * `b` - Basis object
 * * `svals` - Pre-allocated array to store singular values (size must be >= basis size)
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or svals is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_basis_get_svals(const struct spir_basis *b, double *svals);

/**
 * Get statistics type (Fermionic or Bosonic) of a basis
 *
 * # Arguments
 * * `b` - Basis object
 * * `statistics` - Pointer to store statistics (0 = Bosonic, 1 = Fermionic)
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or statistics is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_basis_get_stats(const struct spir_basis *b, int *statistics);

/**
 * Get singular values (alias for spir_basis_get_svals for libsparseir compatibility)
 */
 int spir_basis_get_singular_values(const struct spir_basis *b, double *svals);

/**
 * Get the number of default tau sampling points
 *
 * # Arguments
 * * `b` - Basis object
 * * `num_points` - Pointer to store the number of points
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_basis_get_n_default_taus(const struct spir_basis *b, int *num_points);

/**
 * Get default tau sampling points
 *
 * # Arguments
 * * `b` - Basis object
 * * `points` - Pre-allocated array to store tau points
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_basis_get_default_taus(const struct spir_basis *b, double *points);

/**
 * Get the number of default Matsubara sampling points
 *
 * # Arguments
 * * `b` - Basis object
 * * `positive_only` - If true, return only positive frequencies
 * * `num_points` - Pointer to store the number of points
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */

int spir_basis_get_n_default_matsus(const struct spir_basis *b,
                                           bool positive_only,
                                           int *num_points);

/**
 * Get default Matsubara sampling points
 *
 * # Arguments
 * * `b` - Basis object
 * * `positive_only` - If true, return only positive frequencies
 * * `points` - Pre-allocated array to store Matsubara indices
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */

int spir_basis_get_default_matsus(const struct spir_basis *b,
                                         bool positive_only,
                                         int64_t *points);

/**
 * Gets the basis functions in imaginary time (τ) domain
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
 *
 * # Safety
 * The caller must ensure that `b` is a valid pointer, and must call
 * `spir_funcs_release()` on the returned pointer when done.
 */
 struct spir_funcs *spir_basis_get_u(const struct spir_basis *b, int *status);

/**
 * Gets the basis functions in real frequency (ω) domain
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
 *
 * # Safety
 * The caller must ensure that `b` is a valid pointer, and must call
 * `spir_funcs_release()` on the returned pointer when done.
 */
 struct spir_funcs *spir_basis_get_v(const struct spir_basis *b, int *status);

/**
 * Gets the number of default omega (real frequency) sampling points
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object
 * * `num_points` - Pointer to store the number of sampling points
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success)
 *
 * # Safety
 * The caller must ensure that `b` and `num_points` are valid pointers
 */
 int spir_basis_get_n_default_ws(const struct spir_basis *b, int *num_points);

/**
 * Gets the default omega (real frequency) sampling points
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object
 * * `points` - Pre-allocated array to store the omega sampling points
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success)
 *
 * # Safety
 * The caller must ensure that `points` has size >= `spir_basis_get_n_default_ws(b)`
 */
 int spir_basis_get_default_ws(const struct spir_basis *b, double *points);

/**
 * Gets the basis functions in Matsubara frequency domain
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
 *
 * # Safety
 * The caller must ensure that `b` is a valid pointer, and must call
 * `spir_funcs_release()` on the returned pointer when done.
 */
 struct spir_funcs *spir_basis_get_uhat(const struct spir_basis *b, int *status);

/**
 * Gets the full (untruncated) Matsubara-frequency basis functions
 *
 * This function returns an object representing all basis functions
 * in the Matsubara-frequency domain, including those beyond the truncation
 * threshold. Unlike `spir_basis_get_uhat`, which returns only the truncated
 * basis functions (up to `basis.size()`), this function returns all basis
 * functions from the SVE result (up to `sve_result.s.size()`).
 *
 * # Arguments
 * * `b` - Pointer to the finite temperature basis object (must be an IR basis)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the basis functions object, or NULL if creation fails
 *
 * # Note
 * The returned object must be freed using `spir_funcs_release`
 * when no longer needed
 * This function is only available for IR basis objects (not DLR)
 * uhat_full.size() >= uhat.size() is always true
 * The first uhat.size() functions in uhat_full are identical to uhat
 *
 * # Safety
 * The caller must ensure that `b` is a valid pointer, and must call
 * `spir_funcs_release()` on the returned pointer when done.
 */
 struct spir_funcs *spir_basis_get_uhat_full(const struct spir_basis *b, int *status);

/**
 * Get default tau sampling points with custom limit (extended version)
 *
 * # Arguments
 * * `b` - Basis object
 * * `n_points` - Maximum number of points requested
 * * `points` - Pre-allocated array to store tau points (size >= n_points)
 * * `n_points_returned` - Pointer to store actual number of points returned
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or n_points < 0
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 *
 * # Note
 * Returns min(n_points, actual_default_points) sampling points
 */

int spir_basis_get_default_taus_ext(const struct spir_basis *b,
                                           int n_points,
                                           double *points,
                                           int *n_points_returned);

/**
 * Get number of default Matsubara sampling points with custom limit (extended version)
 *
 * # Arguments
 * * `b` - Basis object
 * * `positive_only` - If true, return only positive frequencies
 * * `L` - Requested number of sampling points
 * * `num_points_returned` - Pointer to store actual number of points
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or L < 0
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 *
 * # Note
 * Returns min(L, actual_default_points) sampling points
 */

int spir_basis_get_n_default_matsus_ext(const struct spir_basis *b,
                                               bool positive_only,
                                               int L,
                                               int *num_points_returned);

/**
 * Get default Matsubara sampling points with custom limit (extended version)
 *
 * # Arguments
 * * `b` - Basis object
 * * `positive_only` - If true, return only positive frequencies
 * * `mitigate` - If true, enable mitigation (fencing) to improve conditioning
 * * `n_points` - Maximum number of points requested
 * * `points` - Pre-allocated array to store Matsubara indices (size >= n_points)
 * * `n_points_returned` - Pointer to store actual number of points returned
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or n_points < 0
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 *
 * # Note
 * Returns min(n_points, actual_default_points) sampling points
 * When mitigate is true, may return more points than requested due to fencing
 */

int spir_basis_get_default_matsus_ext(const struct spir_basis *b,
                                             bool positive_only,
                                             bool mitigate,
                                             int n_points,
                                             int64_t *points,
                                             int *n_points_returned);

/**
 * Creates a new DLR from an IR basis with default poles
 *
 * # Arguments
 * * `b` - Pointer to a finite temperature basis object
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created DLR basis object, or NULL if creation fails
 *
 * # Safety
 * Caller must ensure `b` is a valid IR basis pointer
 */
 struct spir_basis *spir_dlr_new(const struct spir_basis *b, int *status);

/**
 * Creates a new DLR with custom poles
 *
 * # Arguments
 * * `b` - Pointer to a finite temperature basis object
 * * `npoles` - Number of poles to use
 * * `poles` - Array of pole locations on the real-frequency axis
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created DLR basis object, or NULL if creation fails
 *
 * # Safety
 * Caller must ensure `b` is valid and `poles` has `npoles` elements
 */

struct spir_basis *spir_dlr_new_with_poles(const struct spir_basis *b,
                                           int npoles,
                                           const double *poles,
                                           int *status);

/**
 * Gets the number of poles in a DLR
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `num_poles` - Pointer to store the number of poles
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure `dlr` is a valid DLR basis pointer
 */
 int spir_dlr_get_npoles(const struct spir_basis *dlr, int *num_poles);

/**
 * Gets the pole locations in a DLR
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `poles` - Pre-allocated array to store pole locations
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure `dlr` is valid and `poles` has sufficient size
 */
 int spir_dlr_get_poles(const struct spir_basis *dlr, double *poles);

/**
 * Convert IR coefficients to DLR (real-valued)
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `order` - Memory layout order
 * * `ndim` - Number of dimensions
 * * `input_dims` - Array of input dimensions
 * * `target_dim` - Dimension to transform
 * * `input` - IR coefficients
 * * `out` - Output DLR coefficients
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure pointers are valid and arrays have correct sizes
 */

int spir_ir2dlr_dd(const struct spir_basis *dlr,
                          int order,
                          int ndim,
                          const int *input_dims,
                          int target_dim,
                          const double *input,
                          double *out);

/**
 * Convert IR coefficients to DLR (complex-valued)
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `order` - Memory layout order
 * * `ndim` - Number of dimensions
 * * `input_dims` - Array of input dimensions
 * * `target_dim` - Dimension to transform
 * * `input` - Complex IR coefficients
 * * `out` - Output complex DLR coefficients
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure pointers are valid and arrays have correct sizes
 */

int spir_ir2dlr_zz(const struct spir_basis *dlr,
                          int order,
                          int ndim,
                          const int *input_dims,
                          int target_dim,
                          const c_complex *input,
                          c_complex *out);

/**
 * Convert DLR coefficients to IR (real-valued)
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `order` - Memory layout order
 * * `ndim` - Number of dimensions
 * * `input_dims` - Array of input dimensions
 * * `target_dim` - Dimension to transform
 * * `input` - DLR coefficients
 * * `out` - Output IR coefficients
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure pointers are valid and arrays have correct sizes
 */

int spir_dlr2ir_dd(const struct spir_basis *dlr,
                          int order,
                          int ndim,
                          const int *input_dims,
                          int target_dim,
                          const double *input,
                          double *out);

/**
 * Convert DLR coefficients to IR (complex-valued)
 *
 * # Arguments
 * * `dlr` - Pointer to a DLR basis object
 * * `order` - Memory layout order
 * * `ndim` - Number of dimensions
 * * `input_dims` - Array of input dimensions
 * * `target_dim` - Dimension to transform
 * * `input` - Complex DLR coefficients
 * * `out` - Output complex IR coefficients
 *
 * # Returns
 * Status code
 *
 * # Safety
 * Caller must ensure pointers are valid and arrays have correct sizes
 */

int spir_dlr2ir_zz(const struct spir_basis *dlr,
                          int order,
                          int ndim,
                          const int *input_dims,
                          int target_dim,
                          const c_complex *input,
                          c_complex *out);

/**
 * Manual release function (replaces macro-generated one)
 */
 void spir_funcs_release(struct spir_funcs *funcs);

/**
 * Manual clone function (replaces macro-generated one)
 */
 struct spir_funcs *spir_funcs_clone(const struct spir_funcs *src);

/**
 * Manual is_assigned function (replaces macro-generated one)
 */
 int32_t spir_funcs_is_assigned(const struct spir_funcs *obj);

/**
 * Create a spir_funcs object from piecewise Legendre polynomial coefficients
 *
 * Constructs a continuous function object from segments and Legendre polynomial
 * expansion coefficients. The coefficients are organized per segment, with each
 * segment containing nfuncs coefficients (degrees 0 to nfuncs-1).
 *
 * # Arguments
 * * `segments` - Array of segment boundaries (n_segments+1 elements). Must be monotonically increasing.
 * * `n_segments` - Number of segments (must be >= 1)
 * * `coeffs` - Array of Legendre coefficients. Layout: contiguous per segment,
 *              coefficients for segment i are stored at indices [i*nfuncs, (i+1)*nfuncs).
 *              Each segment has nfuncs coefficients for Legendre degrees 0 to nfuncs-1.
 * * `nfuncs` - Number of basis functions per segment (Legendre polynomial degrees 0 to nfuncs-1)
 * * `order` - Order parameter (currently unused, reserved for future use)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created funcs object, or NULL if creation fails
 *
 * # Note
 * The function creates a single piecewise Legendre polynomial function.
 * To create multiple functions, call this function multiple times.
 */

struct spir_funcs *spir_funcs_from_piecewise_legendre(const double *segments,
                                                      int n_segments,
                                                      const double *coeffs,
                                                      int nfuncs,
                                                      int _order,
                                                      int *status);

/**
 * Extract a subset of functions by indices
 *
 * # Arguments
 * * `funcs` - Pointer to the source funcs object
 * * `nslice` - Number of functions to select (length of indices array)
 * * `indices` - Array of indices specifying which functions to include
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to a new funcs object containing only the selected functions, or null on error
 *
 * # Safety
 * The caller must ensure that `funcs` and `indices` are valid pointers.
 * The returned pointer must be freed with `spir_funcs_release()`.
 */

struct spir_funcs *spir_funcs_get_slice(const struct spir_funcs *funcs,
                                        int32_t nslice,
                                        const int32_t *indices,
                                        int *status);

/**
 * Gets the number of basis functions
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `size` - Pointer to store the number of functions
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success)
 */
 int spir_funcs_get_size(const struct spir_funcs *funcs, int *size);

/**
 * Gets the number of knots for continuous functions
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `n_knots` - Pointer to store the number of knots
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
 */
 int spir_funcs_get_n_knots(const struct spir_funcs *funcs, int *n_knots);

/**
 * Gets the knot positions for continuous functions
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `knots` - Pre-allocated array to store knot positions
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
 *
 * # Safety
 * The caller must ensure that `knots` has size >= `spir_funcs_get_n_knots(funcs)`
 */
 int spir_funcs_get_knots(const struct spir_funcs *funcs, double *knots);

/**
 * Evaluate functions at a single point (continuous functions only)
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `x` - Point to evaluate at (tau coordinate in [-1, 1])
 * * `out` - Pre-allocated array to store function values
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
 *
 * # Safety
 * The caller must ensure that `out` has size >= `spir_funcs_get_size(funcs)`
 */
 int spir_funcs_eval(const struct spir_funcs *funcs, double x, double *out);

/**
 * Evaluate functions at a single Matsubara frequency
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `n` - Matsubara frequency index
 * * `out` - Pre-allocated array to store complex function values
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not Matsubara type)
 *
 * # Safety
 * The caller must ensure that `out` has size >= `spir_funcs_get_size(funcs)`
 * Complex numbers are laid out as [real, imag] pairs
 */
 int spir_funcs_eval_matsu(const struct spir_funcs *funcs, int64_t n, c_complex *out);

/**
 * Batch evaluate functions at multiple points (continuous functions only)
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `order` - Memory layout: 0 for row-major, 1 for column-major
 * * `num_points` - Number of evaluation points
 * * `xs` - Array of points to evaluate at
 * * `out` - Pre-allocated array to store results
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
 *
 * # Safety
 * - `xs` must have size >= `num_points`
 * - `out` must have size >= `num_points * spir_funcs_get_size(funcs)`
 * - Layout: row-major = out[point][func], column-major = out[func][point]
 */

int spir_funcs_batch_eval(const struct spir_funcs *funcs,
                                 int order,
                                 int num_points,
                                 const double *xs,
                                 double *out);

/**
 * Batch evaluate functions at multiple Matsubara frequencies
 *
 * # Arguments
 * * `funcs` - Pointer to the funcs object
 * * `order` - Memory layout: 0 for row-major, 1 for column-major
 * * `num_freqs` - Number of Matsubara frequencies
 * * `ns` - Array of Matsubara frequency indices
 * * `out` - Pre-allocated array to store complex results
 *
 * # Returns
 * Status code (SPIR_COMPUTATION_SUCCESS on success, SPIR_NOT_SUPPORTED if not Matsubara type)
 *
 * # Safety
 * - `ns` must have size >= `num_freqs`
 * - `out` must have size >= `num_freqs * spir_funcs_get_size(funcs)`
 * - Complex numbers are laid out as [real, imag] pairs
 * - Layout: row-major = out[freq][func], column-major = out[func][freq]
 */

int spir_funcs_batch_eval_matsu(const struct spir_funcs *funcs,
                                       int order,
                                       int num_freqs,
                                       const int64_t *ns,
                                       c_complex *out);

/**
 * Get default Matsubara sampling points from a Matsubara-space spir_funcs
 *
 * This function computes default sampling points in Matsubara frequencies (iωn) from
 * a spir_funcs object that represents Matsubara-space basis functions (e.g., uhat or uhat_full).
 * The statistics type (Fermionic/Bosonic) is automatically detected from the spir_funcs object type.
 *
 * This extracts the PiecewiseLegendreFTVector from spir_funcs and calls
 * `FiniteTempBasis::default_matsubara_sampling_points_impl` from `basis.rs` (lines 332-387)
 * to compute default sampling points.
 *
 * The implementation uses the same algorithm as defined in `sparseir-rust/src/basis.rs`,
 * which selects sampling points based on sign changes or extrema of the Matsubara basis functions.
 *
 * # Arguments
 * * `uhat` - Pointer to a spir_funcs object representing Matsubara-space basis functions
 * * `l` - Number of requested sampling points
 * * `positive_only` - If true, only positive frequencies are used
 * * `mitigate` - If true, enable mitigation (fencing) to improve conditioning by adding oversampling points
 * * `points` - Pre-allocated array to store the sampling points. The size of the array must be sufficient for the returned points (may exceed L if mitigate is true).
 * * `n_points_returned` - Pointer to store the number of sampling points returned (may exceed L if mitigate is true, or approximately L/2 when positive_only=true).
 *
 * # Returns
 * Status code:
 * - SPIR_COMPUTATION_SUCCESS (0) on success
 * - SPIR_INVALID_ARGUMENT if uhat, points, or n_points_returned is null
 * - SPIR_NOT_SUPPORTED if uhat is not a Matsubara-space function
 *
 * # Note
 * This function is only available for spir_funcs objects representing Matsubara-space basis functions
 * The statistics type is automatically detected from the spir_funcs object type
 * The default sampling points are chosen to provide near-optimal conditioning
 */

int spir_uhat_get_default_matsus(const struct spir_funcs *uhat,
                                        int l,
                                        bool positive_only,
                                        bool mitigate,
                                        int64_t *points,
                                        int *n_points_returned);

/**
 * Register custom BLAS functions (LP64: 32-bit integers)
 *
 * This function allows you to inject external BLAS implementations (OpenBLAS, MKL, Accelerate, etc.)
 * for matrix multiplication operations. The registered functions will be used for all subsequent
 * GEMM operations in the library.
 *
 * # Arguments
 * * `cblas_dgemm` - Function pointer to CBLAS dgemm (double precision)
 * * `cblas_zgemm` - Function pointer to CBLAS zgemm (complex double precision)
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
 *
 * # Safety
 * The provided function pointers must:
 * - Be valid CBLAS function pointers following the standard CBLAS interface
 * - Use 32-bit integers for all dimension parameters (LP64 interface)
 * - Be thread-safe (will be called from multiple threads)
 * - Remain valid for the entire lifetime of the program
 *
 * # Example (from C)
 * ```c
 * #include <cblas.h>
 *
 * // Register OpenBLAS
 * int status = spir_register_dgemm_zgemm_lp64(
 *     (void*)cblas_dgemm,
 *     (void*)cblas_zgemm
 * );
 *
 * if (status != SPIR_COMPUTATION_SUCCESS) {
 *     fprintf(stderr, "Failed to register BLAS functions\n");
 * }
 * ```
 *
 * # CBLAS Interface
 * The function pointers must match these signatures:
 * ```c
 * void cblas_dgemm(
 *     CblasOrder order,       // 102 for ColMajor
 *     CblasTranspose transa,  // 111 for NoTrans
 *     CblasTranspose transb,  // 111 for NoTrans
 *     int m, int n, int k,
 *     double alpha,
 *     const double *a, int lda,
 *     const double *b, int ldb,
 *     double beta,
 *     double *c, int ldc
 * );
 *
 * void cblas_zgemm(
 *     CblasOrder order,
 *     CblasTranspose transa,
 *     CblasTranspose transb,
 *     int m, int n, int k,
 *     const void *alpha,      // complex<double>*
 *     const void *a, int lda,
 *     const void *b, int ldb,
 *     const void *beta,       // complex<double>*
 *     void *c, int ldc
 * );
 * ```
 */

int spir_register_dgemm_zgemm_lp64(const void *cblas_dgemm,
                                          const void *cblas_zgemm);

/**
 * Register ILP64 BLAS functions (64-bit integers)
 *
 * This function allows you to inject ILP64 BLAS implementations (MKL ILP64, OpenBLAS with ILP64, etc.)
 * for matrix multiplication operations. ILP64 uses 64-bit integers for all dimension parameters,
 * enabling support for very large matrices (> 2^31 elements).
 *
 * # Arguments
 * * `cblas_dgemm64` - Function pointer to ILP64 CBLAS dgemm (double precision)
 * * `cblas_zgemm64` - Function pointer to ILP64 CBLAS zgemm (complex double precision)
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
 *
 * # Safety
 * The provided function pointers must:
 * - Be valid CBLAS function pointers following the standard CBLAS interface with ILP64
 * - Use 64-bit integers for all dimension parameters (ILP64 interface)
 * - Be thread-safe (will be called from multiple threads)
 * - Remain valid for the entire lifetime of the program
 *
 * # Example (from C with MKL ILP64)
 * ```c
 * #define MKL_ILP64
 * #include <mkl.h>
 *
 * // Register MKL ILP64
 * int status = spir_register_dgemm_zgemm_ilp64(
 *     (void*)cblas_dgemm,  // MKL's ILP64 version
 *     (void*)cblas_zgemm   // MKL's ILP64 version
 * );
 *
 * if (status != SPIR_COMPUTATION_SUCCESS) {
 *     fprintf(stderr, "Failed to register ILP64 BLAS functions\n");
 * }
 * ```
 *
 * # CBLAS ILP64 Interface
 * The function pointers must match these signatures (note: long long = 64-bit int):
 * ```c
 * void cblas_dgemm(
 *     CblasOrder order,
 *     CblasTranspose transa,
 *     CblasTranspose transb,
 *     long long m, long long n, long long k,
 *     double alpha,
 *     const double *a, long long lda,
 *     const double *b, long long ldb,
 *     double beta,
 *     double *c, long long ldc
 * );
 *
 * void cblas_zgemm(
 *     CblasOrder order,
 *     CblasTranspose transa,
 *     CblasTranspose transb,
 *     long long m, long long n, long long k,
 *     const void *alpha,
 *     const void *a, long long lda,
 *     const void *b, long long ldb,
 *     const void *beta,
 *     void *c, long long ldc
 * );
 * ```
 */

int spir_register_dgemm_zgemm_ilp64(const void *cblas_dgemm64,
                                           const void *cblas_zgemm64);

/**
 * Create a new Logistic kernel
 *
 * # Arguments
 * * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * * Pointer to the newly created kernel object, or NULL if creation fails
 *
 * # Safety
 * The caller must ensure `status` is a valid pointer.
 *
 * # Example (C)
 * ```c
 * int status;
 * spir_kernel* kernel = spir_logistic_kernel_new(10.0, &status);
 * if (kernel != NULL) {
 *     // Use kernel...
 *     spir_kernel_release(kernel);
 * }
 * ```
 */
 struct spir_kernel *spir_logistic_kernel_new(double lambda, int *status);

/**
 * Create a new RegularizedBose kernel
 *
 * # Arguments
 * * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * * Pointer to the newly created kernel object, or NULL if creation fails
 */
 struct spir_kernel *spir_reg_bose_kernel_new(double lambda, int *status);

/**
 * Get the lambda parameter of a kernel
 *
 * # Arguments
 * * `kernel` - Kernel object
 * * `lambda_out` - Pointer to store the lambda value
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if kernel or lambda_out is null
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */
 int spir_kernel_get_lambda(const struct spir_kernel *kernel, double *lambda_out);

/**
 * Compute kernel value K(x, y)
 *
 * # Arguments
 * * `kernel` - Kernel object
 * * `x` - First argument (typically in [-1, 1])
 * * `y` - Second argument (typically in [-1, 1])
 * * `out` - Pointer to store the result
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if kernel or out is null
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */
 int spir_kernel_compute(const struct spir_kernel *kernel, double x, double y, double *out);

/**
 * Manual release function (replaces macro-generated one)
 *
 * # Safety
 * This function drops the kernel. The inner KernelType data is automatically freed
 * by the Drop implementation when the spir_kernel structure is dropped.
 */
 void spir_kernel_release(struct spir_kernel *kernel);

/**
 * Manual clone function (replaces macro-generated one)
 */
 struct spir_kernel *spir_kernel_clone(const struct spir_kernel *src);

/**
 * Manual is_assigned function (replaces macro-generated one)
 */
 int32_t spir_kernel_is_assigned(const struct spir_kernel *obj);

/**
 * Get kernel domain boundaries
 *
 * # Arguments
 * * `k` - Kernel object
 * * `xmin` - Pointer to store minimum x value
 * * `xmax` - Pointer to store maximum x value
 * * `ymin` - Pointer to store minimum y value
 * * `ymax` - Pointer to store maximum y value
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if any pointer is null
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */

int spir_kernel_get_domain(const struct spir_kernel *k,
                                  double *xmin,
                                  double *xmax,
                                  double *ymin,
                                  double *ymax);

/**
 * Get x-segments for SVE discretization hints from a kernel
 *
 * This function should be called twice:
 * 1. First call with segments=NULL: set n_segments to the required array size
 * 2. Second call with segments allocated: fill segments[0..n_segments-1] with values
 *
 * # Arguments
 * * `k` - Kernel object
 * * `epsilon` - Accuracy target for the basis
 * * `segments` - Pointer to store segments array (NULL for first call)
 * * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */

int spir_kernel_get_sve_hints_segments_x(const struct spir_kernel *k,
                                                double epsilon,
                                                double *segments,
                                                int *n_segments);

/**
 * Get y-segments for SVE discretization hints from a kernel
 *
 * This function should be called twice:
 * 1. First call with segments=NULL: set n_segments to the required array size
 * 2. Second call with segments allocated: fill segments[0..n_segments-1] with values
 *
 * # Arguments
 * * `k` - Kernel object
 * * `epsilon` - Accuracy target for the basis
 * * `segments` - Pointer to store segments array (NULL for first call)
 * * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */

int spir_kernel_get_sve_hints_segments_y(const struct spir_kernel *k,
                                                double epsilon,
                                                double *segments,
                                                int *n_segments);

/**
 * Get the number of singular values hint from a kernel
 *
 * # Arguments
 * * `k` - Kernel object
 * * `epsilon` - Accuracy target for the basis
 * * `nsvals` - Pointer to store the number of singular values
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if k or nsvals is null
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */

int spir_kernel_get_sve_hints_nsvals(const struct spir_kernel *k,
                                            double epsilon,
                                            int *nsvals);

/**
 * Get the number of Gauss points hint from a kernel
 *
 * # Arguments
 * * `k` - Kernel object
 * * `epsilon` - Accuracy target for the basis
 * * `ngauss` - Pointer to store the number of Gauss points
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` on success
 * * `SPIR_INVALID_ARGUMENT` if k or ngauss is null
 * * `SPIR_INTERNAL_ERROR` if internal panic occurs
 */

int spir_kernel_get_sve_hints_ngauss(const struct spir_kernel *k,
                                            double epsilon,
                                            int *ngauss);

/**
 * Manual release function (replaces macro-generated one)
 */
 void spir_sampling_release(struct spir_sampling *sampling);

/**
 * Manual clone function (replaces macro-generated one)
 */
 struct spir_sampling *spir_sampling_clone(const struct spir_sampling *src);

/**
 * Manual is_assigned function (replaces macro-generated one)
 */
 int32_t spir_sampling_is_assigned(const struct spir_sampling *obj);

/**
 * Creates a new tau sampling object for sparse sampling in imaginary time
 *
 * # Arguments
 * * `b` - Pointer to a finite temperature basis object
 * * `num_points` - Number of sampling points
 * * `points` - Array of sampling points in imaginary time (τ)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created sampling object, or NULL if creation fails
 *
 * # Safety
 * Caller must ensure `b` is valid and `points` has `num_points` elements
 */

struct spir_sampling *spir_tau_sampling_new(const struct spir_basis *b,
                                            int num_points,
                                            const double *points,
                                            int *status);

/**
 * Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies
 *
 * # Arguments
 * * `b` - Pointer to a finite temperature basis object
 * * `positive_only` - If true, only positive frequencies are used
 * * `num_points` - Number of sampling points
 * * `points` - Array of Matsubara frequency indices (n)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created sampling object, or NULL if creation fails
 */

struct spir_sampling *spir_matsu_sampling_new(const struct spir_basis *b,
                                              bool positive_only,
                                              int num_points,
                                              const int64_t *points,
                                              int *status);

/**
 * Creates a new tau sampling object with custom sampling points and pre-computed matrix
 *
 * # Arguments
 * * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
 * * `basis_size` - Basis size
 * * `num_points` - Number of sampling points
 * * `points` - Array of sampling points in imaginary time (τ)
 * * `matrix` - Pre-computed matrix for the sampling points (num_points x basis_size)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created sampling object, or NULL if creation fails
 *
 * # Safety
 * Caller must ensure `points` and `matrix` have correct sizes
 */

struct spir_sampling *spir_tau_sampling_new_with_matrix(int order,
                                                        int statistics,
                                                        int basis_size,
                                                        int num_points,
                                                        const double *points,
                                                        const double *matrix,
                                                        int *status);

/**
 * Creates a new Matsubara sampling object with custom sampling points and pre-computed matrix
 *
 * # Arguments
 * * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
 * * `basis_size` - Basis size
 * * `positive_only` - If true, only positive frequencies are used
 * * `num_points` - Number of sampling points
 * * `points` - Array of Matsubara frequency indices (n)
 * * `matrix` - Pre-computed complex matrix (num_points x basis_size)
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Pointer to the newly created sampling object, or NULL if creation fails
 *
 * # Safety
 * Caller must ensure `points` and `matrix` have correct sizes
 */

struct spir_sampling *spir_matsu_sampling_new_with_matrix(int order,
                                                          int statistics,
                                                          int basis_size,
                                                          bool positive_only,
                                                          int num_points,
                                                          const int64_t *points,
                                                          const c_complex *matrix,
                                                          int *status);

/**
 * Gets the number of sampling points in a sampling object
 */
 int spir_sampling_get_npoints(const struct spir_sampling *s, int *num_points);

/**
 * Gets the imaginary time sampling points
 */
 int spir_sampling_get_taus(const struct spir_sampling *s, double *points);

/**
 * Gets the Matsubara frequency sampling points
 */
 int spir_sampling_get_matsus(const struct spir_sampling *s, int64_t *points);

/**
 * Gets the condition number of the sampling matrix
 *
 * Note: Currently returns a placeholder value.
 * TODO: Implement proper condition number calculation from SVD
 */
 int spir_sampling_get_cond_num(const struct spir_sampling *s, double *cond_num);

/**
 * Evaluate basis coefficients at sampling points (double → double)
 *
 * Transforms IR basis coefficients to values at sampling points.
 *
 * # Note
 * Currently only supports column-major order (SPIR_ORDER_COLUMN_MAJOR = 1).
 * Row-major support will be added in a future update.
 */

int spir_sampling_eval_dd(const struct spir_sampling *s,
                                 int order,
                                 int ndim,
                                 const int *input_dims,
                                 int target_dim,
                                 const double *input,
                                 double *out);

/**
 * Evaluate basis coefficients at sampling points (double → complex)
 *
 * For Matsubara sampling: transforms real IR coefficients to complex values.
 */

int spir_sampling_eval_dz(const struct spir_sampling *s,
                                 int order,
                                 int ndim,
                                 const int *input_dims,
                                 int target_dim,
                                 const double *input,
                                 c_complex *out);

/**
 * Evaluate basis coefficients at sampling points (complex → complex)
 *
 * For Matsubara sampling: transforms complex coefficients to complex values.
 */

int spir_sampling_eval_zz(const struct spir_sampling *s,
                                 int order,
                                 int ndim,
                                 const int *input_dims,
                                 int target_dim,
                                 const c_complex *input,
                                 c_complex *out);

/**
 * Fit basis coefficients from sampling point values (double → double)
 */

int spir_sampling_fit_dd(const struct spir_sampling *s,
                                int order,
                                int ndim,
                                const int *input_dims,
                                int target_dim,
                                const double *input,
                                double *out);

/**
 * Fit basis coefficients from sampling point values (complex → complex)
 */

int spir_sampling_fit_zz(const struct spir_sampling *s,
                                int order,
                                int ndim,
                                const int *input_dims,
                                int target_dim,
                                const c_complex *input,
                                c_complex *out);

/**
 * Fit basis coefficients from Matsubara sampling (complex → double, positive only)
 */

int spir_sampling_fit_zd(const struct spir_sampling *s,
                                int order,
                                int ndim,
                                const int *input_dims,
                                int target_dim,
                                const c_complex *input,
                                double *out);

/**
 * Manual release function (replaces macro-generated one)
 */
 void spir_sve_result_release(struct spir_sve_result *sve);

/**
 * Manual clone function (replaces macro-generated one)
 */
 struct spir_sve_result *spir_sve_result_clone(const struct spir_sve_result *src);

/**
 * Manual is_assigned function (replaces macro-generated one)
 */
 int32_t spir_sve_result_is_assigned(const struct spir_sve_result *obj);

/**
 * Compute Singular Value Expansion (SVE) of a kernel (libsparseir compatible)
 *
 * # Arguments
 * * `k` - Kernel object
 * * `epsilon` - Accuracy target for the basis
 * * `lmax` - Maximum number of Legendre polynomials (currently ignored, auto-determined)
 * * `n_gauss` - Number of Gauss points for integration (currently ignored, auto-determined)
 * * `Twork` - Working precision: 0=Float64, 1=Float64x2, -1=Auto
 * * `status` - Pointer to store status code
 *
 * # Returns
 * * Pointer to SVE result, or NULL on failure
 *
 * # Safety
 * The caller must ensure `status` is a valid pointer.
 *
 * # Note
 * Parameters `lmax` and `n_gauss` are accepted for libsparseir compatibility but
 * currently ignored. The Rust implementation automatically determines optimal values.
 * The cutoff is automatically set to 2*sqrt(machine_epsilon) internally.
 */

struct spir_sve_result *spir_sve_result_new(const struct spir_kernel *k,
                                            double epsilon,
                                            int _lmax,
                                            int _n_gauss,
                                            int twork,
                                            int *status);

/**
 * Get the number of singular values in an SVE result
 *
 * # Arguments
 * * `sve` - SVE result object
 * * `size` - Pointer to store the size
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if sve or size is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_sve_result_get_size(const struct spir_sve_result *sve, int *size);

/**
 * Truncate an SVE result based on epsilon and max_size
 *
 * This function creates a new SVE result containing only the singular values
 * that are larger than `epsilon * s[0]`, where `s[0]` is the largest singular value.
 * The result can also be limited to a maximum size.
 *
 * # Arguments
 * * `sve` - Source SVE result object
 * * `epsilon` - Relative threshold for truncation (singular values < epsilon * s[0] are removed)
 * * `max_size` - Maximum number of singular values to keep (-1 for no limit)
 * * `status` - Pointer to store status code
 *
 * # Returns
 * * Pointer to new truncated SVE result, or NULL on failure
 * * Status code:
 *   - `SPIR_COMPUTATION_SUCCESS` (0) on success
 *   - `SPIR_INVALID_ARGUMENT` (-6) if sve or status is null, or epsilon is invalid
 *   - `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 *
 * # Safety
 * The caller must ensure `status` is a valid pointer.
 * The returned pointer must be freed with `spir_sve_result_release()`.
 *
 * # Example (C)
 * ```c
 * spir_sve_result* sve = spir_sve_result_new(kernel, 1e-10, 0, 0, -1, &status);
 *
 * // Truncate to keep only singular values > 1e-8 * s[0], max 50 values
 * spir_sve_result* sve_truncated = spir_sve_result_truncate(sve, 1e-8, 50, &status);
 *
 * // Use truncated result...
 *
 * spir_sve_result_release(sve_truncated);
 * spir_sve_result_release(sve);
 * ```
 */

struct spir_sve_result *spir_sve_result_truncate(const struct spir_sve_result *sve,
                                                 double epsilon,
                                                 int max_size,
                                                 int *status);

/**
 * Get singular values from an SVE result
 *
 * # Arguments
 * * `sve` - SVE result object
 * * `svals` - Pre-allocated array to store singular values (size must be >= result size)
 *
 * # Returns
 * * `SPIR_COMPUTATION_SUCCESS` (0) on success
 * * `SPIR_INVALID_ARGUMENT` (-6) if sve or svals is null
 * * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
 */
 int spir_sve_result_get_svals(const struct spir_sve_result *sve, double *svals);

/**
 * Create a SVE result from a discretized kernel matrix
 *
 * This function performs singular value expansion (SVE) on a discretized kernel
 * matrix K. The matrix K should already be in the appropriate form (no weight
 * application needed). The function supports both double and DDouble precision
 * based on whether K_low is provided.
 *
 * # Arguments
 * * `K_high` - High part of the kernel matrix (required, size: nx * ny)
 * * `K_low` - Low part of the kernel matrix (optional, nullptr for double precision)
 * * `nx` - Number of rows in the matrix
 * * `ny` - Number of columns in the matrix
 * * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * * `segments_x` - X-direction segments (size: n_segments_x + 1)
 * * `n_segments_x` - Number of segments in x direction
 * * `segments_y` - Y-direction segments (size: n_segments_y + 1)
 * * `n_segments_y` - Number of segments in y direction
 * * `n_gauss` - Number of Gauss points per segment
 * * `epsilon` - Target accuracy
 * * `status` - Pointer to store status code
 *
 * # Returns
 * Pointer to SVE result on success, nullptr on failure
 */

struct spir_sve_result *spir_sve_result_from_matrix(const double *K_high,
                                                    const double *K_low,
                                                    int nx,
                                                    int ny,
                                                    int order,
                                                    const double *segments_x,
                                                    int n_segments_x,
                                                    const double *segments_y,
                                                    int n_segments_y,
                                                    int n_gauss,
                                                    double epsilon,
                                                    int *status);

/**
 * Create a SVE result from centrosymmetric discretized kernel matrices
 *
 * This function performs singular value expansion (SVE) on centrosymmetric
 * discretized kernel matrices using even/odd symmetry decomposition. The matrices
 * K_even and K_odd should already be in the appropriate form (no weight
 * application needed). The function supports both double and DDouble precision
 * based on whether K_low is provided.
 *
 * # Arguments
 * * `K_even_high` - High part of the even-symmetry kernel matrix (required, size: nx * ny)
 * * `K_even_low` - Low part of the even-symmetry kernel matrix (optional, nullptr for double precision)
 * * `K_odd_high` - High part of the odd-symmetry kernel matrix (required, size: nx * ny)
 * * `K_odd_low` - Low part of the odd-symmetry kernel matrix (optional, nullptr for double precision)
 * * `nx` - Number of rows in the matrix
 * * `ny` - Number of columns in the matrix
 * * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * * `segments_x` - X-direction segments (size: n_segments_x + 1)
 * * `n_segments_x` - Number of segments in x direction
 * * `segments_y` - Y-direction segments (size: n_segments_y + 1)
 * * `n_segments_y` - Number of segments in y direction
 * * `n_gauss` - Number of Gauss points per segment
 * * `epsilon` - Target accuracy
 * * `status` - Pointer to store status code
 *
 * # Returns
 * Pointer to SVE result on success, nullptr on failure
 */

struct spir_sve_result *spir_sve_result_from_matrix_centrosymmetric(const double *K_even_high,
                                                                    const double *K_even_low,
                                                                    const double *K_odd_high,
                                                                    const double *K_odd_low,
                                                                    int nx,
                                                                    int ny,
                                                                    int order,
                                                                    const double *segments_x,
                                                                    int n_segments_x,
                                                                    const double *segments_y,
                                                                    int n_segments_y,
                                                                    int n_gauss,
                                                                    double epsilon,
                                                                    int *status);

/**
 * Choose the working type (Twork) based on epsilon value
 *
 * This function determines the appropriate working precision type based on the
 * target accuracy epsilon. It follows the same logic as SPIR_TWORK_AUTO:
 * - Returns SPIR_TWORK_FLOAT64X2 if epsilon < 1e-8 or epsilon is NaN
 * - Returns SPIR_TWORK_FLOAT64 otherwise
 *
 * # Arguments
 * * `epsilon` - Target accuracy (must be non-negative, or NaN for auto-selection)
 *
 * # Returns
 * Working type constant:
 * - SPIR_TWORK_FLOAT64 (0): Use double precision (64-bit)
 * - SPIR_TWORK_FLOAT64X2 (1): Use extended precision (128-bit)
 */
 int spir_choose_working_type(double epsilon);

/**
 * Compute piecewise Gauss-Legendre quadrature rule (double precision)
 *
 * Generates a piecewise Gauss-Legendre quadrature rule with n points per segment.
 * The rule is concatenated across all segments, with points and weights properly
 * scaled for each segment interval.
 *
 * # Arguments
 * * `n` - Number of Gauss points per segment (must be >= 1)
 * * `segments` - Array of segment boundaries (n_segments + 1 elements).
 *                Must be monotonically increasing.
 * * `n_segments` - Number of segments (must be >= 1)
 * * `x` - Output array for Gauss points (size n * n_segments). Must be pre-allocated.
 * * `w` - Output array for Gauss weights (size n * n_segments). Must be pre-allocated.
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Status code:
 * - SPIR_COMPUTATION_SUCCESS (0) on success
 * - Non-zero error code on failure
 */

int spir_gauss_legendre_rule_piecewise_double(int n,
                                                     const double *segments,
                                                     int n_segments,
                                                     double *x,
                                                     double *w,
                                                     int *status);

/**
 * Compute piecewise Gauss-Legendre quadrature rule (DDouble precision)
 *
 * Generates a piecewise Gauss-Legendre quadrature rule with n points per segment,
 * computed using extended precision (DDouble). Returns high and low parts separately
 * for maximum precision.
 *
 * # Arguments
 * * `n` - Number of Gauss points per segment (must be >= 1)
 * * `segments` - Array of segment boundaries (n_segments + 1 elements).
 *                Must be monotonically increasing.
 * * `n_segments` - Number of segments (must be >= 1)
 * * `x_high` - Output array for high part of Gauss points (size n * n_segments).
 *              Must be pre-allocated.
 * * `x_low` - Output array for low part of Gauss points (size n * n_segments).
 *             Must be pre-allocated.
 * * `w_high` - Output array for high part of Gauss weights (size n * n_segments).
 *              Must be pre-allocated.
 * * `w_low` - Output array for low part of Gauss weights (size n * n_segments).
 *            Must be pre-allocated.
 * * `status` - Pointer to store the status code
 *
 * # Returns
 * Status code:
 * - SPIR_COMPUTATION_SUCCESS (0) on success
 * - Non-zero error code on failure
 */

int spir_gauss_legendre_rule_piecewise_ddouble(int n,
                                                      const double *segments,
                                                      int n_segments,
                                                      double *x_high,
                                                      double *x_low,
                                                      double *w_high,
                                                      double *w_low,
                                                      int *status);

#ifdef __cplusplus
}
#endif
