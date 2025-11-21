//! GEMM (Matrix Multiplication) C-API
//!
//! This module provides C-API functions for registering external BLAS implementations.
//! These functions allow users to inject their own BLAS libraries (OpenBLAS, MKL, Accelerate, etc.)
//! at runtime without recompiling.
//!
//! # API Functions
//! - `spir_gemm_backend_new_from_fblas_lp64`: Create backend from LP64 BLAS (32-bit integers)
//! - `spir_gemm_backend_new_from_fblas_ilp64`: Create backend from ILP64 BLAS (64-bit integers)
//! - `spir_gemm_backend_release`: Release backend handle
//!
//! # Example (C)
//! ```c
//! // Link against BLAS library (e.g., OpenBLAS, MKL, Accelerate)
//! // Fortran BLAS functions typically have trailing underscore
//!
//! // Create backend from Fortran BLAS (or pass NULL to use default backend)
//! spir_gemm_backend* backend = spir_gemm_backend_new_from_fblas_lp64(
//!     (void*)dgemm_,
//!     (void*)zgemm_
//! );
//!
//! // Use backend in evaluate/fit functions (pass NULL to use default backend)
//! // ...
//!
//! // Release backend when done
//! spir_gemm_backend_release(backend);
//! ```

use sparseir_rust::gemm::{GemmBackendHandle, ExternalBlasBackend, ExternalBlas64Backend, DgemmFnPtr, ZgemmFnPtr, Dgemm64FnPtr, Zgemm64FnPtr};

//==============================================================================
// Backend Handle C-API
//==============================================================================

/// Opaque pointer type for GEMM backend handle
///
/// This type wraps a `GemmBackendHandle` and provides a C-compatible interface.
/// The handle can be created, cloned, and passed to evaluate/fit functions.
///
/// Note: The internal structure is hidden using a void pointer to prevent exposing GemmBackendHandle to C.
#[repr(C)]
pub struct spir_gemm_backend {
    pub(crate) _private: *mut std::ffi::c_void,
}

impl spir_gemm_backend {
    /// Get a reference to the inner GemmBackendHandle
    pub(crate) fn inner(&self) -> &GemmBackendHandle {
        unsafe {
            &*(self._private as *const GemmBackendHandle)
        }
    }

    pub(crate) fn new(handle: GemmBackendHandle) -> Self {
        Self {
            _private: Box::into_raw(Box::new(handle)) as *mut std::ffi::c_void,
        }
    }
}

impl Drop for spir_gemm_backend {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut GemmBackendHandle);
            }
        }
    }
}

impl Clone for spir_gemm_backend {
    fn clone(&self) -> Self {
        // GemmBackendHandle is already Arc-based, so we can clone the inner handle
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

/// Create GEMM backend from Fortran BLAS function pointers (LP64)
///
/// Creates a new backend handle from Fortran BLAS function pointers.
///
/// # Arguments
/// * `dgemm` - Function pointer to Fortran BLAS dgemm (double precision)
/// * `zgemm` - Function pointer to Fortran BLAS zgemm (complex double precision)
///
/// # Returns
/// * Pointer to `spir_gemm_backend` on success
/// * `NULL` if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface
/// - Use 32-bit integers for all dimension parameters (LP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the backend handle
///
/// The returned pointer must be freed with `spir_gemm_backend_free` when no longer needed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_gemm_backend_new_from_fblas_lp64(
    dgemm: *const libc::c_void,
    zgemm: *const libc::c_void,
) -> *mut spir_gemm_backend {
    // Validate input
    if dgemm.is_null() || zgemm.is_null() {
        return std::ptr::null_mut();
    }

    // Cast to Fortran BLAS function pointer types
    let dgemm_fn: DgemmFnPtr = unsafe { std::mem::transmute(dgemm) };
    let zgemm_fn: ZgemmFnPtr = unsafe { std::mem::transmute(zgemm) };

    // Create backend
    let backend = ExternalBlasBackend::new(dgemm_fn, zgemm_fn);

    // Wrap in handle
    let handle = GemmBackendHandle::new(Box::new(backend));
    Box::into_raw(Box::new(spir_gemm_backend::new(handle)))
}

/// Create GEMM backend from Fortran BLAS function pointers (ILP64)
///
/// Creates a new backend handle from Fortran BLAS function pointers with 64-bit integers.
///
/// # Arguments
/// * `dgemm64` - Function pointer to Fortran BLAS dgemm (double precision, 64-bit integers)
/// * `zgemm64` - Function pointer to Fortran BLAS zgemm (complex double precision, 64-bit integers)
///
/// # Returns
/// * Pointer to `spir_gemm_backend` on success
/// * `NULL` if function pointers are null
///
/// # Safety
/// The provided function pointers must:
/// - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface
/// - Use 64-bit integers for all dimension parameters (ILP64 interface)
/// - Be thread-safe (will be called from multiple threads)
/// - Remain valid for the entire lifetime of the backend handle
///
/// The returned pointer must be freed with `spir_gemm_backend_free` when no longer needed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_gemm_backend_new_from_fblas_ilp64(
    dgemm64: *const libc::c_void,
    zgemm64: *const libc::c_void,
) -> *mut spir_gemm_backend {
    // Validate input
    if dgemm64.is_null() || zgemm64.is_null() {
        return std::ptr::null_mut();
    }

    // Cast to Fortran BLAS function pointer types
    let dgemm64_fn: Dgemm64FnPtr = unsafe { std::mem::transmute(dgemm64) };
    let zgemm64_fn: Zgemm64FnPtr = unsafe { std::mem::transmute(zgemm64) };

    // Create backend
    let backend = ExternalBlas64Backend::new(dgemm64_fn, zgemm64_fn);

    // Wrap in handle
    let handle = GemmBackendHandle::new(Box::new(backend));
    Box::into_raw(Box::new(spir_gemm_backend::new(handle)))
}

/// Release GEMM backend handle
///
/// Releases the memory associated with a backend handle.
///
/// # Arguments
/// * `backend` - Pointer to backend handle (can be NULL)
///
/// # Safety
/// The pointer must have been created by `spir_gemm_backend_new_from_fblas_lp64` or
/// `spir_gemm_backend_new_from_fblas_ilp64`.
/// After calling this function, the pointer must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_gemm_backend_release(backend: *mut spir_gemm_backend) {
    if !backend.is_null() {
        unsafe {
            let _ = Box::from_raw(backend);
        }
    }
}

/// Get backend handle from opaque pointer (internal use)
///
/// # Safety
/// The pointer must be valid and not null.
/// The returned reference is only valid while the backend pointer is valid.
/// The caller must ensure the backend pointer remains valid for the lifetime of the returned reference.
pub(crate) unsafe fn get_backend_handle<'a>(backend: *const spir_gemm_backend) -> Option<&'a GemmBackendHandle> {
    if backend.is_null() {
        None
    } else {
        unsafe {
            Some((*backend).inner())
        }
    }
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
    fn test_backend_new_from_fblas_lp64_success() {
        unsafe {
            let backend = spir_gemm_backend_new_from_fblas_lp64(
                mock_dgemm as *const _,
                mock_zgemm as *const _,
            );
            assert!(!backend.is_null(), "Backend should not be null");
            spir_gemm_backend_release(backend);
        }
    }

    #[test]
    fn test_backend_new_from_fblas_ilp64_success() {
        unsafe {
            let backend = spir_gemm_backend_new_from_fblas_ilp64(
                mock_dgemm64 as *const _,
                mock_zgemm64 as *const _,
            );
            assert!(!backend.is_null(), "Backend should not be null");
            spir_gemm_backend_release(backend);
        }
    }

    #[test]
    fn test_backend_new_from_fblas_lp64_null_dgemm() {
        unsafe {
            let backend = spir_gemm_backend_new_from_fblas_lp64(
                std::ptr::null(),
                mock_zgemm as *const _,
            );
            assert!(backend.is_null(), "Backend should be null when dgemm is null");
        }
    }

    #[test]
    fn test_backend_new_from_fblas_lp64_null_zgemm() {
        unsafe {
            let backend = spir_gemm_backend_new_from_fblas_lp64(
                mock_dgemm as *const _,
                std::ptr::null(),
            );
            assert!(backend.is_null(), "Backend should be null when zgemm is null");
        }
    }

    #[test]
    fn test_backend_new_from_fblas_ilp64_null_pointers() {
        unsafe {
            let backend = spir_gemm_backend_new_from_fblas_ilp64(
                std::ptr::null(),
                std::ptr::null(),
            );
            assert!(backend.is_null(), "Backend should be null when pointers are null");
        }
    }

    #[test]
    fn test_backend_release_null() {
        unsafe {
            // Should not panic when releasing null pointer
            spir_gemm_backend_release(std::ptr::null_mut());
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


        // Helper to create backend from blas-sys functions
        unsafe fn create_blas_backend() -> *mut spir_gemm_backend {
            unsafe {
                spir_gemm_backend_new_from_fblas_lp64(
                    dgemm_ as *const _,
                    // Cast zgemm_ to match our function pointer type (memory layout is compatible)
                    unsafe {
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
                        >(zgemm_)
                    } as *const _,
                )
            }
        }

        #[test]
        fn test_default_backend_matrix_multiplication_f64() {
            unsafe {
                // Use default backend (NULL means use default)
                let backend = std::ptr::null();

                // Test matrix multiplication: C = A * B
                // A = [[1.0, 2.0], [3.0, 4.0]]
                // B = [[5.0, 6.0], [7.0, 8.0]]
                // Expected: C = [[19.0, 22.0], [43.0, 50.0]]
                let a: mdarray::DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
                let b: mdarray::DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify results
                assert!((c[[0, 0]] - 19.0).abs() < 1e-10, "c[0,0] should be 19.0, got {}", c[[0, 0]]);
                assert!((c[[0, 1]] - 22.0).abs() < 1e-10, "c[0,1] should be 22.0");
                assert!((c[[1, 0]] - 43.0).abs() < 1e-10, "c[1,0] should be 43.0");
                assert!((c[[1, 1]] - 50.0).abs() < 1e-10, "c[1,1] should be 50.0");
            }
        }

        #[test]
        fn test_lp64_backend_matrix_multiplication_f64() {
            unsafe {
                // Create backend from system BLAS (LP64)
                let backend = create_blas_backend();
                assert!(!backend.is_null());

                // Test matrix multiplication: C = A * B
                // A = [[1.0, 2.0], [3.0, 4.0]]
                // B = [[5.0, 6.0], [7.0, 8.0]]
                // Expected: C = [[19.0, 22.0], [43.0, 50.0]]
                let a: mdarray::DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
                let b: mdarray::DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify results
                assert!((c[[0, 0]] - 19.0).abs() < 1e-10, "c[0,0] should be 19.0, got {}", c[[0, 0]]);
                assert!((c[[0, 1]] - 22.0).abs() < 1e-10, "c[0,1] should be 22.0");
                assert!((c[[1, 0]] - 43.0).abs() < 1e-10, "c[1,0] should be 43.0");
                assert!((c[[1, 1]] - 50.0).abs() < 1e-10, "c[1,1] should be 50.0");

                // Clean up
                spir_gemm_backend_release(backend);
            }
        }

        #[test]
        fn test_default_backend_matrix_multiplication_complex() {
            unsafe {
                // Use default backend (NULL means use default)
                let backend = std::ptr::null();

                // Test complex matrix multiplication
                let a: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(1.0, 0.0), num_complex::Complex::new(2.0, 0.0)],
                    [num_complex::Complex::new(3.0, 0.0), num_complex::Complex::new(4.0, 0.0)]
                ];
                let b: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(5.0, 0.0), num_complex::Complex::new(6.0, 0.0)],
                    [num_complex::Complex::new(7.0, 0.0), num_complex::Complex::new(8.0, 0.0)]
                ];
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify results (same as real case)
                assert!((c[[0, 0]].re - 19.0).abs() < 1e-10);
                assert!((c[[0, 1]].re - 22.0).abs() < 1e-10);
                assert!((c[[1, 0]].re - 43.0).abs() < 1e-10);
                assert!((c[[1, 1]].re - 50.0).abs() < 1e-10);
                assert!(c[[0, 0]].im.abs() < 1e-10);
            }
        }

        #[test]
        fn test_lp64_backend_matrix_multiplication_complex() {
            unsafe {
                // Create backend from system BLAS (LP64)
                let backend = create_blas_backend();
                assert!(!backend.is_null());

                // Test complex matrix multiplication
                let a: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(1.0, 0.0), num_complex::Complex::new(2.0, 0.0)],
                    [num_complex::Complex::new(3.0, 0.0), num_complex::Complex::new(4.0, 0.0)]
                ];
                let b: mdarray::DTensor<num_complex::Complex<f64>, 2> = tensor![
                    [num_complex::Complex::new(5.0, 0.0), num_complex::Complex::new(6.0, 0.0)],
                    [num_complex::Complex::new(7.0, 0.0), num_complex::Complex::new(8.0, 0.0)]
                ];
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify results (same as real case)
                assert!((c[[0, 0]].re - 19.0).abs() < 1e-10);
                assert!((c[[0, 1]].re - 22.0).abs() < 1e-10);
                assert!((c[[1, 0]].re - 43.0).abs() < 1e-10);
                assert!((c[[1, 1]].re - 50.0).abs() < 1e-10);
                assert!(c[[0, 0]].im.abs() < 1e-10);

                // Clean up
                spir_gemm_backend_release(backend);
            }
        }

        #[test]
        fn test_default_backend_larger_matrix() {
            unsafe {
                // Use default backend (NULL means use default)
                let backend = std::ptr::null();

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
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify some results
                // First row: [1*7+2*11, 1*8+2*12, 1*9+2*13, 1*10+2*14] = [29, 32, 35, 38]
                assert!((c[[0, 0]] - 29.0).abs() < 1e-10);
                assert!((c[[0, 1]] - 32.0).abs() < 1e-10);
                assert!((c[[0, 2]] - 35.0).abs() < 1e-10);
                assert!((c[[0, 3]] - 38.0).abs() < 1e-10);
            }
        }

        #[test]
        fn test_lp64_backend_larger_matrix() {
            unsafe {
                // Create backend from system BLAS (LP64)
                let backend = create_blas_backend();
                assert!(!backend.is_null());

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
                let backend_handle = get_backend_handle(backend);
                let c = matmul_par(&a, &b, backend_handle);

                // Verify some results
                // First row: [1*7+2*11, 1*8+2*12, 1*9+2*13, 1*10+2*14] = [29, 32, 35, 38]
                assert!((c[[0, 0]] - 29.0).abs() < 1e-10);
                assert!((c[[0, 1]] - 32.0).abs() < 1e-10);
                assert!((c[[0, 2]] - 35.0).abs() < 1e-10);
                assert!((c[[0, 3]] - 38.0).abs() < 1e-10);

                // Clean up
                spir_gemm_backend_release(backend);
            }
        }
    }
}
