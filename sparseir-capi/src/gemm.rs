//! GEMM (Matrix Multiplication) C-API
//!
//! This module provides C-API functions for registering external BLAS implementations.
//! These functions allow users to inject their own BLAS libraries (OpenBLAS, MKL, Accelerate, etc.)
//! at runtime without recompiling.
//!
//! # API Functions
//! - `spir_register_dgemm_zgemm_lp64`: Register LP64 BLAS (32-bit integers)
//! - `spir_register_dgemm_zgemm_ilp64`: Register ILP64 BLAS (64-bit integers)
//!
//! # Example (C)
//! ```c
//! // Link against BLAS library (e.g., OpenBLAS, MKL, Accelerate)
//! // Fortran BLAS functions typically have trailing underscore
//!
//! // Register Fortran BLAS
//! spir_register_dgemm_zgemm_lp64(
//!     (void*)dgemm_,
//!     (void*)zgemm_
//! );
//!
//! // Now all matrix operations use the registered BLAS
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
/// * `dgemm` - Function pointer to Fortran BLAS dgemm (double precision)
/// * `zgemm` - Function pointer to Fortran BLAS zgemm (complex double precision)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface
/// - Use 32-bit integers for all dimension parameters (LP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the program
///
/// # Example (from C)
/// ```c
/// // Link against BLAS library (e.g., -lblas or -lopenblas)
/// // Fortran BLAS functions typically have trailing underscore
///
/// // Register Fortran BLAS
/// int status = spir_register_dgemm_zgemm_lp64(
///     (void*)dgemm_,
///     (void*)zgemm_
/// );
///
/// if (status != SPIR_COMPUTATION_SUCCESS) {
///     fprintf(stderr, "Failed to register BLAS functions\n");
/// }
/// ```
///
/// # Fortran BLAS Interface
/// The function pointers must match these signatures:
/// ```c
/// void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
///             double *alpha, double *a, int *lda, double *b, int *ldb,
///             double *beta, double *c, int *ldc);
///
/// void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
///             void *alpha, void *a, int *lda, void *b, int *ldb,
///             void *beta, void *c, int *ldc);
/// ```
/// Note: All parameters are passed by reference (pointers).
/// Transpose options: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_register_dgemm_zgemm_lp64(
    dgemm: *const libc::c_void,  // Fortran BLAS dgemm_ function pointer
    zgemm: *const libc::c_void,  // Fortran BLAS zgemm_ function pointer
) -> StatusCode {
    // Validate input
    if dgemm.is_null() || zgemm.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    // Cast to Fortran BLAS function pointer types
    // Safe because the caller guarantees these are valid Fortran BLAS function pointers
    let dgemm_fn: sparseir_rust::gemm::DgemmFnPtr = unsafe { std::mem::transmute(dgemm) };
    let zgemm_fn: sparseir_rust::gemm::ZgemmFnPtr = unsafe { std::mem::transmute(zgemm) };

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
/// * `dgemm64` - Function pointer to ILP64 Fortran BLAS dgemm (double precision)
/// * `zgemm64` - Function pointer to ILP64 Fortran BLAS zgemm (complex double precision)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface with ILP64
/// - Use 64-bit integers for all dimension parameters (ILP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the program
///
/// # Example (from C with MKL ILP64)
/// ```c
/// #define MKL_ILP64
/// #include <mkl.h>
///
/// // Register MKL ILP64 Fortran BLAS
/// int status = spir_register_dgemm_zgemm_ilp64(
///     (void*)dgemm_,  // MKL's ILP64 Fortran BLAS version
///     (void*)zgemm_   // MKL's ILP64 Fortran BLAS version
/// );
///
/// if (status != SPIR_COMPUTATION_SUCCESS) {
///     fprintf(stderr, "Failed to register ILP64 BLAS functions\n");
/// }
/// ```
///
/// # Fortran BLAS ILP64 Interface
/// The function pointers must match these signatures (note: long long = 64-bit int):
/// ```c
/// void dgemm_(char *transa, char *transb, long long *m, long long *n, long long *k,
///             double *alpha, double *a, long long *lda, double *b, long long *ldb,
///             double *beta, double *c, long long *ldc);
///
/// void zgemm_(char *transa, char *transb, long long *m, long long *n, long long *k,
///             void *alpha, void *a, long long *lda, void *b, long long *ldb,
///             void *beta, void *c, long long *ldc);
/// ```
/// Note: All parameters are passed by reference (pointers).
/// Transpose options: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_register_dgemm_zgemm_ilp64(
    dgemm64: *const libc::c_void,  // Fortran BLAS dgemm_ function pointer (ILP64)
    zgemm64: *const libc::c_void,  // Fortran BLAS zgemm_ function pointer (ILP64)
) -> StatusCode {
    // Validate input
    if dgemm64.is_null() || zgemm64.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    // Cast to Fortran BLAS function pointer types (ILP64)
    // Safe because the caller guarantees these are valid Fortran BLAS function pointers
    let dgemm64_fn: sparseir_rust::gemm::Dgemm64FnPtr = unsafe { std::mem::transmute(dgemm64) };
    let zgemm64_fn: sparseir_rust::gemm::Zgemm64FnPtr = unsafe { std::mem::transmute(zgemm64) };

    // Register with the Rust backend
    unsafe { sparseir_rust::gemm::set_ilp64_backend(dgemm64_fn, zgemm64_fn); }

    SPIR_COMPUTATION_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock Fortran BLAS functions for testing
    unsafe extern "C" fn mock_dgemm(
        _transa: *const libc::c_char,
        _transb: *const libc::c_char,
        _m: *const libc::c_int,
        _n: *const libc::c_int,
        _k: *const libc::c_int,
        _alpha: *const libc::c_double,
        _a: *const libc::c_double,
        _lda: *const libc::c_int,
        _b: *const libc::c_double,
        _ldb: *const libc::c_int,
        _beta: *const libc::c_double,
        _c: *mut libc::c_double,
        _ldc: *const libc::c_int,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_zgemm(
        _transa: *const libc::c_char,
        _transb: *const libc::c_char,
        _m: *const libc::c_int,
        _n: *const libc::c_int,
        _k: *const libc::c_int,
        _alpha: *const num_complex::Complex<f64>,
        _a: *const num_complex::Complex<f64>,
        _lda: *const libc::c_int,
        _b: *const num_complex::Complex<f64>,
        _ldb: *const libc::c_int,
        _beta: *const num_complex::Complex<f64>,
        _c: *mut num_complex::Complex<f64>,
        _ldc: *const libc::c_int,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_dgemm64(
        _transa: *const libc::c_char,
        _transb: *const libc::c_char,
        _m: *const i64,
        _n: *const i64,
        _k: *const i64,
        _alpha: *const libc::c_double,
        _a: *const libc::c_double,
        _lda: *const i64,
        _b: *const libc::c_double,
        _ldb: *const i64,
        _beta: *const libc::c_double,
        _c: *mut libc::c_double,
        _ldc: *const i64,
    ) {
        // Mock implementation - does nothing
    }

    unsafe extern "C" fn mock_zgemm64(
        _transa: *const libc::c_char,
        _transb: *const libc::c_char,
        _m: *const i64,
        _n: *const i64,
        _k: *const i64,
        _alpha: *const num_complex::Complex<f64>,
        _a: *const num_complex::Complex<f64>,
        _lda: *const i64,
        _b: *const num_complex::Complex<f64>,
        _ldb: *const i64,
        _beta: *const num_complex::Complex<f64>,
        _c: *mut num_complex::Complex<f64>,
        _ldc: *const i64,
    ) {
        // Mock implementation - does nothing
    }

    #[test]
    fn test_register_blas_functions_success() {
        // Ensure clean state before test
        sparseir_rust::gemm::clear_blas_backend();
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
        // Ensure clean state before test
        sparseir_rust::gemm::clear_blas_backend();
        unsafe {
            let status =
                spir_register_dgemm_zgemm_ilp64(mock_dgemm64 as *const _, mock_zgemm64 as *const _);
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
            let status = spir_register_dgemm_zgemm_ilp64(std::ptr::null(), std::ptr::null());
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
        }
    }

    // System BLAS integration tests (always enabled during tests)
    #[cfg(test)]
    mod system_blas_tests {
        use super::*;
        use sparseir_rust::gemm::matmul_par;
        use mdarray::tensor;
        use blas_sys::{dgemm_, zgemm_, c_double_complex};

        // Helper function to convert Complex<f64> to c_double_complex
        fn complex_to_c_double_complex(c: num_complex::Complex<f64>) -> c_double_complex {
            [c.re, c.im]
        }

        // Helper function to convert c_double_complex to Complex<f64>
        fn c_double_complex_to_complex(c: c_double_complex) -> num_complex::Complex<f64> {
            num_complex::Complex::new(c[0], c[1])
        }

        #[test]
        fn test_system_blas_registration() {
            // Ensure clean state before test
            sparseir_rust::gemm::clear_blas_backend();
            unsafe {
                // Register system BLAS functions from blas-sys
                // Note: zgemm_ uses c_double_complex, but num_complex::Complex<f64> has the same memory layout
                let status = spir_register_dgemm_zgemm_lp64(
                    dgemm_ as *const _,
                    // Cast zgemm_ to match our function pointer type (memory layout is compatible)
                    std::mem::transmute::<
                        unsafe extern "C" fn(
                            *const libc::c_char,
                            *const libc::c_char,
                            *const libc::c_int,
                            *const libc::c_int,
                            *const libc::c_int,
                            *const blas_sys::c_double_complex,
                            *const blas_sys::c_double_complex,
                            *const libc::c_int,
                            *const blas_sys::c_double_complex,
                            *const libc::c_int,
                            *const blas_sys::c_double_complex,
                            *mut blas_sys::c_double_complex,
                            *const libc::c_int,
                        ),
                        sparseir_rust::gemm::ZgemmFnPtr,
                    >(zgemm_) as *const _,
                );
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

        // Helper to register blas-sys functions
        unsafe fn register_blas_sys() -> StatusCode {
            spir_register_dgemm_zgemm_lp64(
                dgemm_ as *const _,
                // Cast zgemm_ to match our function pointer type (memory layout is compatible)
                std::mem::transmute::<
                    unsafe extern "C" fn(
                        *const libc::c_char,
                        *const libc::c_char,
                        *const libc::c_int,
                        *const libc::c_int,
                        *const libc::c_int,
                        *const blas_sys::c_double_complex,
                        *const blas_sys::c_double_complex,
                        *const libc::c_int,
                        *const blas_sys::c_double_complex,
                        *const libc::c_int,
                        *const blas_sys::c_double_complex,
                        *mut blas_sys::c_double_complex,
                        *const libc::c_int,
                    ),
                    sparseir_rust::gemm::ZgemmFnPtr,
                >(zgemm_) as *const _,
            )
        }

        #[test]
        fn test_system_blas_matrix_multiplication_f64() {
            // Ensure clean state before test
            sparseir_rust::gemm::clear_blas_backend();
            unsafe {
                // Register system BLAS
                let status = register_blas_sys();
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

                // Test matrix multiplication: C = A * B
                // A = [[1.0, 2.0], [3.0, 4.0]]
                // B = [[5.0, 6.0], [7.0, 8.0]]
                // Expected: C = [[19.0, 22.0], [43.0, 50.0]]
                let a: mdarray::DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
                let b: mdarray::DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
                let c = matmul_par(&a, &b);

                // Debug: print actual values
                eprintln!("Actual result: c = [[{}, {}], [{}, {}]]", c[[0, 0]], c[[0, 1]], c[[1, 0]], c[[1, 1]]);
                eprintln!("Expected: c = [[19.0, 22.0], [43.0, 50.0]]");

                // Verify results
                assert!((c[[0, 0]] - 19.0).abs() < 1e-10, "c[0,0] should be 19.0, got {}", c[[0, 0]]);
                assert!((c[[0, 1]] - 22.0).abs() < 1e-10, "c[0,1] should be 22.0");
                assert!((c[[1, 0]] - 43.0).abs() < 1e-10, "c[1,0] should be 43.0");
                assert!((c[[1, 1]] - 50.0).abs() < 1e-10, "c[1,1] should be 50.0");

                // Clean up
                sparseir_rust::gemm::clear_blas_backend();
            }
        }

        #[test]
        fn test_system_blas_matrix_multiplication_complex() {
            // Ensure clean state before test
            sparseir_rust::gemm::clear_blas_backend();
            unsafe {
                // Register system BLAS
                let status = register_blas_sys();
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

                // Test complex matrix multiplication
                let a: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(1.0, 0.0), num_complex::Complex::new(2.0, 0.0)],
                    [num_complex::Complex::new(3.0, 0.0), num_complex::Complex::new(4.0, 0.0)]
                ];
                let b: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(5.0, 0.0), num_complex::Complex::new(6.0, 0.0)],
                    [num_complex::Complex::new(7.0, 0.0), num_complex::Complex::new(8.0, 0.0)]
                ];
                let c = matmul_par(&a, &b);

                // Verify results (same as real case)
                assert!((c[[0, 0]].re - 19.0).abs() < 1e-10);
                assert!((c[[0, 1]].re - 22.0).abs() < 1e-10);
                assert!((c[[1, 0]].re - 43.0).abs() < 1e-10);
                assert!((c[[1, 1]].re - 50.0).abs() < 1e-10);
                assert!(c[[0, 0]].im.abs() < 1e-10);

                // Clean up
                sparseir_rust::gemm::clear_blas_backend();
            }
        }

        #[test]
        fn test_system_blas_larger_matrix() {
            // Ensure clean state before test
            sparseir_rust::gemm::clear_blas_backend();
            unsafe {
                // Register system BLAS
                let status = register_blas_sys();
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

                // Test with larger matrices (3x2 * 2x4 = 3x4)
                let a: mdarray::DTensor<f64, 2> = tensor![
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0]
                ];
                let b: mdarray::DTensor<f64, 2> = tensor![
                    [7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0]
                ];
                let c = matmul_par(&a, &b);

                // Verify some results
                // First row: [1*7+2*11, 1*8+2*12, 1*9+2*13, 1*10+2*14] = [29, 32, 35, 38]
                assert!((c[[0, 0]] - 29.0).abs() < 1e-10);
                assert!((c[[0, 1]] - 32.0).abs() < 1e-10);
                assert!((c[[0, 2]] - 35.0).abs() < 1e-10);
                assert!((c[[0, 3]] - 38.0).abs() < 1e-10);

                // Clean up
                sparseir_rust::gemm::clear_blas_backend();
            }
        }
    }
}
