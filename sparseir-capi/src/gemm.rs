//! GEMM (Matrix Multiplication) C-API
//!
//! This module provides C-API functions for registering external BLAS implementations.
//! These functions allow users to inject their own BLAS libraries (OpenBLAS, MKL, Accelerate, etc.)
//! at runtime without recompiling.
//!
//! # API Functions
//! - `spir_register_dgemm_zgemm_lp64`: Register LP64 BLAS (32-bit integers)
//! - `spir_register_ilp64_functions`: Register ILP64 BLAS (64-bit integers)
//!
//! # Example (C)
//! ```c
//! #include <cblas.h>
//!
//! // Register OpenBLAS
//! spir_register_dgemm_zgemm_lp64(
//!     (void*)cblas_dgemm,
//!     (void*)cblas_zgemm
//! );
//!
//! // Now all matrix operations use OpenBLAS
//! ```

use crate::StatusCode;
use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT};

/// Register custom BLAS functions (LP64: 32-bit integers)
///
/// This function allows you to inject external BLAS implementations (OpenBLAS, MKL, Accelerate, etc.)
/// for matrix multiplication operations. The registered functions will be used for all subsequent
/// GEMM operations in the library.
///
/// # Arguments
/// * `cblas_dgemm` - Function pointer to CBLAS dgemm (double precision)
/// * `cblas_zgemm` - Function pointer to CBLAS zgemm (complex double precision)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid CBLAS function pointers following the standard CBLAS interface
/// - Use 32-bit integers for all dimension parameters (LP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the program
///
/// # Example (from C)
/// ```c
/// #include <cblas.h>
///
/// // Register OpenBLAS
/// int status = spir_register_dgemm_zgemm_lp64(
///     (void*)cblas_dgemm,
///     (void*)cblas_zgemm
/// );
///
/// if (status != SPIR_COMPUTATION_SUCCESS) {
///     fprintf(stderr, "Failed to register BLAS functions\n");
/// }
/// ```
///
/// # CBLAS Interface
/// The function pointers must match these signatures:
/// ```c
/// void cblas_dgemm(
///     CblasOrder order,       // 102 for ColMajor
///     CblasTranspose transa,  // 111 for NoTrans
///     CblasTranspose transb,  // 111 for NoTrans
///     int m, int n, int k,
///     double alpha,
///     const double *a, int lda,
///     const double *b, int ldb,
///     double beta,
///     double *c, int ldc
/// );
///
/// void cblas_zgemm(
///     CblasOrder order,
///     CblasTranspose transa,
///     CblasTranspose transb,
///     int m, int n, int k,
///     const void *alpha,      // complex<double>*
///     const void *a, int lda,
///     const void *b, int ldb,
///     const void *beta,       // complex<double>*
///     void *c, int ldc
/// );
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_register_dgemm_zgemm_lp64(
    cblas_dgemm: *const libc::c_void,
    cblas_zgemm: *const libc::c_void,
) -> StatusCode {
    // Validate input
    if cblas_dgemm.is_null() || cblas_zgemm.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    // Cast to function pointers
    let dgemm_fn: sparseir_rust::gemm::DgemmFnPtr = unsafe { std::mem::transmute(cblas_dgemm) };
    let zgemm_fn: sparseir_rust::gemm::ZgemmFnPtr = unsafe { std::mem::transmute(cblas_zgemm) };

    // Register with the Rust backend
    unsafe { sparseir_rust::gemm::set_blas_backend(dgemm_fn, zgemm_fn); }

    SPIR_COMPUTATION_SUCCESS
}

/// Register ILP64 BLAS functions (64-bit integers)
///
/// This function allows you to inject ILP64 BLAS implementations (MKL ILP64, OpenBLAS with ILP64, etc.)
/// for matrix multiplication operations. ILP64 uses 64-bit integers for all dimension parameters,
/// enabling support for very large matrices (> 2^31 elements).
///
/// # Arguments
/// * `cblas_dgemm64` - Function pointer to ILP64 CBLAS dgemm (double precision)
/// * `cblas_zgemm64` - Function pointer to ILP64 CBLAS zgemm (complex double precision)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid CBLAS function pointers following the standard CBLAS interface with ILP64
/// - Use 64-bit integers for all dimension parameters (ILP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the program
///
/// # Example (from C with MKL ILP64)
/// ```c
/// #define MKL_ILP64
/// #include <mkl.h>
///
/// // Register MKL ILP64
/// int status = spir_register_ilp64_functions(
///     (void*)cblas_dgemm,  // MKL's ILP64 version
///     (void*)cblas_zgemm   // MKL's ILP64 version
/// );
///
/// if (status != SPIR_COMPUTATION_SUCCESS) {
///     fprintf(stderr, "Failed to register ILP64 BLAS functions\n");
/// }
/// ```
///
/// # CBLAS ILP64 Interface
/// The function pointers must match these signatures (note: long long = 64-bit int):
/// ```c
/// void cblas_dgemm(
///     CblasOrder order,
///     CblasTranspose transa,
///     CblasTranspose transb,
///     long long m, long long n, long long k,
///     double alpha,
///     const double *a, long long lda,
///     const double *b, long long ldb,
///     double beta,
///     double *c, long long ldc
/// );
///
/// void cblas_zgemm(
///     CblasOrder order,
///     CblasTranspose transa,
///     CblasTranspose transb,
///     long long m, long long n, long long k,
///     const void *alpha,
///     const void *a, long long lda,
///     const void *b, long long ldb,
///     const void *beta,
///     void *c, long long ldc
/// );
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_register_ilp64_functions(
    cblas_dgemm64: *const libc::c_void,
    cblas_zgemm64: *const libc::c_void,
) -> StatusCode {
    // Validate input
    if cblas_dgemm64.is_null() || cblas_zgemm64.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    // Cast to function pointers
    let dgemm64_fn: sparseir_rust::gemm::Dgemm64FnPtr = unsafe { std::mem::transmute(cblas_dgemm64) };
    let zgemm64_fn: sparseir_rust::gemm::Zgemm64FnPtr = unsafe { std::mem::transmute(cblas_zgemm64) };

    // Register with the Rust backend
    unsafe { sparseir_rust::gemm::set_ilp64_backend(dgemm64_fn, zgemm64_fn); }

    SPIR_COMPUTATION_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock BLAS functions for testing
    unsafe extern "C" fn mock_dgemm(
        _order: libc::c_int,
        _transa: libc::c_int,
        _transb: libc::c_int,
        _m: libc::c_int,
        _n: libc::c_int,
        _k: libc::c_int,
        _alpha: libc::c_double,
        _a: *const libc::c_double,
        _lda: libc::c_int,
        _b: *const libc::c_double,
        _ldb: libc::c_int,
        _beta: libc::c_double,
        _c: *mut libc::c_double,
        _ldc: libc::c_int,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_zgemm(
        _order: libc::c_int,
        _transa: libc::c_int,
        _transb: libc::c_int,
        _m: libc::c_int,
        _n: libc::c_int,
        _k: libc::c_int,
        _alpha: *const num_complex::Complex<f64>,
        _a: *const num_complex::Complex<f64>,
        _lda: libc::c_int,
        _b: *const num_complex::Complex<f64>,
        _ldb: libc::c_int,
        _beta: *const num_complex::Complex<f64>,
        _c: *mut num_complex::Complex<f64>,
        _ldc: libc::c_int,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_dgemm64(
        _order: libc::c_int,
        _transa: libc::c_int,
        _transb: libc::c_int,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: libc::c_double,
        _a: *const libc::c_double,
        _lda: i64,
        _b: *const libc::c_double,
        _ldb: i64,
        _beta: libc::c_double,
        _c: *mut libc::c_double,
        _ldc: i64,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_zgemm64(
        _order: libc::c_int,
        _transa: libc::c_int,
        _transb: libc::c_int,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: *const num_complex::Complex<f64>,
        _a: *const num_complex::Complex<f64>,
        _lda: i64,
        _b: *const num_complex::Complex<f64>,
        _ldb: i64,
        _beta: *const num_complex::Complex<f64>,
        _c: *mut num_complex::Complex<f64>,
        _ldc: i64,
    ) {
        // Mock implementation - does nothing
    }

    #[test]
    fn test_register_blas_functions_success() {
        unsafe {
            let status =
                spir_register_dgemm_zgemm_lp64(mock_dgemm as *const _, mock_zgemm as *const _);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify backend was registered
            let (name, is_external, is_ilp64) = sparseir_rust::gemm::get_backend_info();
            assert!(is_external, "Backend should be external after registration");
            assert!(!is_ilp64, "Backend should not be ILP64");
            assert_eq!(name, "External BLAS (LP64)");

            // Clean up: reset to default
            sparseir_rust::gemm::clear_blas_backend();
        }
    }

    #[test]
    fn test_register_ilp64_functions_success() {
        unsafe {
            let status =
                spir_register_ilp64_functions(mock_dgemm64 as *const _, mock_zgemm64 as *const _);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify ILP64 backend was registered
            let (name, is_external, is_ilp64) = sparseir_rust::gemm::get_backend_info();
            assert!(is_external, "Backend should be external after registration");
            assert!(is_ilp64, "Backend should be ILP64");
            assert_eq!(name, "External BLAS (ILP64)");

            // Clean up: reset to default
            sparseir_rust::gemm::clear_blas_backend();
        }
    }

    #[test]
    fn test_register_blas_functions_null_dgemm() {
        unsafe {
            let status = spir_register_dgemm_zgemm_lp64(std::ptr::null(), mock_zgemm as *const _);
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_register_blas_functions_null_zgemm() {
        unsafe {
            let status = spir_register_dgemm_zgemm_lp64(mock_dgemm as *const _, std::ptr::null());
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_register_ilp64_functions_null_pointers() {
        unsafe {
            let status = spir_register_ilp64_functions(std::ptr::null(), std::ptr::null());
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
        }
    }
}
