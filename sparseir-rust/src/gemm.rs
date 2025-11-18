//! Matrix multiplication utilities with pluggable BLAS backend
//!
//! This module provides thin wrappers around matrix multiplication operations,
//! with support for runtime selection of BLAS implementations.
//!
//! # Design
//! - **Default**: Pure Rust Faer backend (no external dependencies)
//! - **Optional**: External BLAS via function pointer injection
//! - **Thread-safe**: Global dispatcher protected by RwLock
//!
//! # Example
//! ```ignore
//! use sparseir_rust::gemm::{matmul_par, set_blas_backend};
//!
//! // Use default Faer backend
//! let c = matmul_par(&a, &b);
//!
//! // Or inject custom BLAS (from C-API)
//! unsafe {
//!     set_blas_backend(my_dgemm_ptr, my_zgemm_ptr);
//! }
//! let c = matmul_par(&a, &b);  // Now uses custom BLAS
//! ```

use mdarray::{DSlice, DTensor, Layout};
use once_cell::sync::Lazy;
use std::sync::RwLock;

//==============================================================================
// BLAS Function Pointer Types
//==============================================================================

/// BLAS dgemm function pointer type (LP64: 32-bit integers)
///
/// Signature matches Fortran BLAS dgemm:
/// ```c
/// void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
///             double *alpha, double *a, int *lda, double *b, int *ldb,
///             double *beta, double *c, int *ldc);
/// ```
/// Note: All parameters are passed by reference (pointers).
/// Transpose options: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
pub type DgemmFnPtr = unsafe extern "C" fn(
    transa: *const libc::c_char,
    transb: *const libc::c_char,
    m: *const libc::c_int,
    n: *const libc::c_int,
    k: *const libc::c_int,
    alpha: *const libc::c_double,
    a: *const libc::c_double,
    lda: *const libc::c_int,
    b: *const libc::c_double,
    ldb: *const libc::c_int,
    beta: *const libc::c_double,
    c: *mut libc::c_double,
    ldc: *const libc::c_int,
);

/// BLAS zgemm function pointer type (LP64: 32-bit integers)
///
/// Signature matches Fortran BLAS zgemm:
/// ```c
/// void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
///             void *alpha, void *a, int *lda, void *b, int *ldb,
///             void *beta, void *c, int *ldc);
/// ```
/// Note: All parameters are passed by reference (pointers).
/// Complex numbers are passed as void* (typically complex<double>*).
/// Transpose options: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
pub type ZgemmFnPtr = unsafe extern "C" fn(
    transa: *const libc::c_char,
    transb: *const libc::c_char,
    m: *const libc::c_int,
    n: *const libc::c_int,
    k: *const libc::c_int,
    alpha: *const num_complex::Complex<f64>,
    a: *const num_complex::Complex<f64>,
    lda: *const libc::c_int,
    b: *const num_complex::Complex<f64>,
    ldb: *const libc::c_int,
    beta: *const num_complex::Complex<f64>,
    c: *mut num_complex::Complex<f64>,
    ldc: *const libc::c_int,
);

/// BLAS dgemm function pointer type (ILP64: 64-bit integers)
///
/// Signature matches Fortran BLAS dgemm (ILP64):
/// ```c
/// void dgemm_(char *transa, char *transb, long long *m, long long *n, long long *k,
///             double *alpha, double *a, long long *lda, double *b, long long *ldb,
///             double *beta, double *c, long long *ldc);
/// ```
pub type Dgemm64FnPtr = unsafe extern "C" fn(
    transa: *const libc::c_char,
    transb: *const libc::c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const libc::c_double,
    a: *const libc::c_double,
    lda: *const i64,
    b: *const libc::c_double,
    ldb: *const i64,
    beta: *const libc::c_double,
    c: *mut libc::c_double,
    ldc: *const i64,
);

/// BLAS zgemm function pointer type (ILP64: 64-bit integers)
///
/// Signature matches Fortran BLAS zgemm (ILP64):
/// ```c
/// void zgemm_(char *transa, char *transb, long long *m, long long *n, long long *k,
///             void *alpha, void *a, long long *lda, void *b, long long *ldb,
///             void *beta, void *c, long long *ldc);
/// ```
pub type Zgemm64FnPtr = unsafe extern "C" fn(
    transa: *const libc::c_char,
    transb: *const libc::c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const num_complex::Complex<f64>,
    a: *const num_complex::Complex<f64>,
    lda: *const i64,
    b: *const num_complex::Complex<f64>,
    ldb: *const i64,
    beta: *const num_complex::Complex<f64>,
    c: *mut num_complex::Complex<f64>,
    ldc: *const i64,
);

//==============================================================================
// Fortran BLAS Constants
//==============================================================================

// Fortran BLAS transpose characters
const FORTRAN_NO_TRANS: u8 = b'N';

//==============================================================================
// GemmBackend Trait
//==============================================================================

/// GEMM backend trait for runtime dispatch
pub trait GemmBackend: Send + Sync {
    /// Matrix multiplication: C = A * B (f64)
    /// 
    /// # Arguments
    /// * `m`, `n`, `k` - Matrix dimensions (M x K) * (K x N) = (M x N)
    /// * `a` - Pointer to matrix A (row-major, M x K)
    /// * `b` - Pointer to matrix B (row-major, K x N)
    /// * `c` - Pointer to output matrix C (row-major, M x N)
    /// * `ldc` - Leading dimension of C (typically M for row-major)
    unsafe fn dgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        ldc: usize,
    );

    /// Matrix multiplication: C = A * B (Complex<f64>)
    /// 
    /// # Arguments
    /// * `m`, `n`, `k` - Matrix dimensions (M x K) * (K x N) = (M x N)
    /// * `a` - Pointer to matrix A (row-major, M x K)
    /// * `b` - Pointer to matrix B (row-major, K x N)
    /// * `c` - Pointer to output matrix C (row-major, M x N)
    /// * `ldc` - Leading dimension of C (typically M for row-major)
    unsafe fn zgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const num_complex::Complex<f64>,
        b: *const num_complex::Complex<f64>,
        c: *mut num_complex::Complex<f64>,
        ldc: usize,
    );

    /// Returns true if this backend uses 64-bit integers (ILP64)
    fn is_ilp64(&self) -> bool {
        false
    }

    /// Returns backend name for debugging
    fn name(&self) -> &'static str;
}

//==============================================================================
// Faer Backend (Default, Pure Rust)
//==============================================================================

/// Default Faer backend (Pure Rust, no external dependencies)
struct FaerBackend;

impl GemmBackend for FaerBackend {
    unsafe fn dgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        ldc: usize,
    ) {
        use mdarray_linalg::prelude::MatMul;
        use mdarray_linalg::matmul::MatMulBuilder;
        use mdarray_linalg_faer::Faer;

        // Create tensors from pointers (row-major order)
        let a_slice = unsafe { std::slice::from_raw_parts(a, m * k) };
        let b_slice = unsafe { std::slice::from_raw_parts(b, k * n) };
        let a_tensor = DTensor::<f64, 2>::from_fn([m, k], |idx| a_slice[idx[0] * k + idx[1]]);
        let b_tensor = DTensor::<f64, 2>::from_fn([k, n], |idx| b_slice[idx[0] * n + idx[1]]);

        // Perform matrix multiplication
        let c_tensor = Faer.matmul(&*a_tensor, &*b_tensor).parallelize().eval();

        // Copy result back to output pointer (row-major order)
        let c_slice = unsafe { std::slice::from_raw_parts_mut(c, m * ldc) };
        for i in 0..m {
            for j in 0..n {
                c_slice[i * ldc + j] = c_tensor[[i, j]];
            }
        }
    }

    unsafe fn zgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const num_complex::Complex<f64>,
        b: *const num_complex::Complex<f64>,
        c: *mut num_complex::Complex<f64>,
        ldc: usize,
    ) {
        use mdarray_linalg::prelude::MatMul;
        use mdarray_linalg::matmul::MatMulBuilder;
        use mdarray_linalg_faer::Faer;

        // Create tensors from pointers (row-major order)
        let a_slice = unsafe { std::slice::from_raw_parts(a, m * k) };
        let b_slice = unsafe { std::slice::from_raw_parts(b, k * n) };
        let a_tensor =
            DTensor::<num_complex::Complex<f64>, 2>::from_fn([m, k], |idx| a_slice[idx[0] * k + idx[1]]);
        let b_tensor =
            DTensor::<num_complex::Complex<f64>, 2>::from_fn([k, n], |idx| b_slice[idx[0] * n + idx[1]]);

        // Perform matrix multiplication
        let c_tensor = Faer.matmul(&*a_tensor, &*b_tensor).parallelize().eval();

        // Copy result back to output pointer (row-major order)
        let c_slice = unsafe { std::slice::from_raw_parts_mut(c, m * ldc) };
        for i in 0..m {
            for j in 0..n {
                c_slice[i * ldc + j] = c_tensor[[i, j]];
            }
        }
    }

    fn name(&self) -> &'static str {
        "Faer (Pure Rust)"
    }
}

//==============================================================================
// External BLAS Backends (LP64 and ILP64)
//==============================================================================

/// External BLAS backend (LP64: 32-bit integers)
struct ExternalBlasBackend {
    dgemm: DgemmFnPtr,
    zgemm: ZgemmFnPtr,
}

impl GemmBackend for ExternalBlasBackend {
    unsafe fn dgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        ldc: usize,
    ) {
        // Validate dimensions fit in i32
        assert!(
            m <= i32::MAX as usize,
            "Matrix dimension m too large for LP64 BLAS"
        );
        assert!(
            n <= i32::MAX as usize,
            "Matrix dimension n too large for LP64 BLAS"
        );
        assert!(
            k <= i32::MAX as usize,
            "Matrix dimension k too large for LP64 BLAS"
        );
        assert!(
            ldc <= i32::MAX as usize,
            "Leading dimension ldc too large for LP64 BLAS"
        );

        // Fortran BLAS requires all parameters passed by reference
        let transa = FORTRAN_NO_TRANS as libc::c_char;
        let transb = FORTRAN_NO_TRANS as libc::c_char;
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let alpha = 1.0f64;
        let lda = m as i32;
        let ldb = k as i32;
        let beta = 0.0f64;
        let ldc_i32 = ldc as i32;

        unsafe {
            (self.dgemm)(
            &transa,
            &transb,
            &m_i32,
            &n_i32,
            &k_i32,
            &alpha,
            a,
            &lda,
            b,
            &ldb,
            &beta,
            c,
            &ldc_i32,
            );
        }
    }

    unsafe fn zgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const num_complex::Complex<f64>,
        b: *const num_complex::Complex<f64>,
        c: *mut num_complex::Complex<f64>,
        ldc: usize,
    ) {
        assert!(
            m <= i32::MAX as usize,
            "Matrix dimension m too large for LP64 BLAS"
        );
        assert!(
            n <= i32::MAX as usize,
            "Matrix dimension n too large for LP64 BLAS"
        );
        assert!(
            k <= i32::MAX as usize,
            "Matrix dimension k too large for LP64 BLAS"
        );
        assert!(
            ldc <= i32::MAX as usize,
            "Leading dimension ldc too large for LP64 BLAS"
        );

        // Fortran BLAS requires all parameters passed by reference
        let transa = FORTRAN_NO_TRANS as libc::c_char;
        let transb = FORTRAN_NO_TRANS as libc::c_char;
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let alpha = num_complex::Complex::new(1.0, 0.0);
        let lda = m as i32;
        let ldb = k as i32;
        let beta = num_complex::Complex::new(0.0, 0.0);
        let ldc_i32 = ldc as i32;

        unsafe {
            (self.zgemm)(
            &transa,
            &transb,
            &m_i32,
            &n_i32,
            &k_i32,
            &alpha,
            a as *const _,
            &lda,
            b as *const _,
            &ldb,
            &beta,
            c as *mut _,
            &ldc_i32,
            );
        }
    }

    fn name(&self) -> &'static str {
        "External BLAS (LP64)"
    }
}

/// External BLAS backend (ILP64: 64-bit integers)
struct ExternalBlas64Backend {
    dgemm64: Dgemm64FnPtr,
    zgemm64: Zgemm64FnPtr,
}

impl GemmBackend for ExternalBlas64Backend {
    unsafe fn dgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        ldc: usize,
    ) {
        // Fortran BLAS requires all parameters passed by reference
        let transa = FORTRAN_NO_TRANS as libc::c_char;
        let transb = FORTRAN_NO_TRANS as libc::c_char;
        let m_i64 = m as i64;
        let n_i64 = n as i64;
        let k_i64 = k as i64;
        let alpha = 1.0f64;
        let lda = m as i64;
        let ldb = k as i64;
        let beta = 0.0f64;
        let ldc_i64 = ldc as i64;

        unsafe {
            (self.dgemm64)(
            &transa,
            &transb,
            &m_i64,
            &n_i64,
            &k_i64,
            &alpha,
            a,
            &lda,
            b,
            &ldb,
            &beta,
            c,
            &ldc_i64,
            );
        }
    }

    unsafe fn zgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const num_complex::Complex<f64>,
        b: *const num_complex::Complex<f64>,
        c: *mut num_complex::Complex<f64>,
        ldc: usize,
    ) {
        // Fortran BLAS requires all parameters passed by reference
        let transa = FORTRAN_NO_TRANS as libc::c_char;
        let transb = FORTRAN_NO_TRANS as libc::c_char;
        let m_i64 = m as i64;
        let n_i64 = n as i64;
        let k_i64 = k as i64;
        let alpha = num_complex::Complex::new(1.0, 0.0);
        let lda = m as i64;
        let ldb = k as i64;
        let beta = num_complex::Complex::new(0.0, 0.0);
        let ldc_i64 = ldc as i64;

        unsafe {
            (self.zgemm64)(
            &transa,
            &transb,
            &m_i64,
            &n_i64,
            &k_i64,
            &alpha,
            a as *const _,
            &lda,
            b as *const _,
            &ldb,
            &beta,
            c as *mut _,
            &ldc_i64,
            );
        }
    }

    fn is_ilp64(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "External BLAS (ILP64)"
    }
}

//==============================================================================
// Global Dispatcher
//==============================================================================

/// Global BLAS dispatcher (thread-safe)
static BLAS_DISPATCHER: Lazy<RwLock<Box<dyn GemmBackend>>> =
    Lazy::new(|| RwLock::new(Box::new(FaerBackend)));

/// Set BLAS backend (LP64: 32-bit integers)
///
/// # Safety
/// - Function pointers must be valid and thread-safe
/// - Must remain valid for the lifetime of the program
/// - Must follow Fortran BLAS calling convention
///
/// # Example
/// ```ignore
/// unsafe {
///     set_blas_backend(dgemm_ as _, zgemm_ as _);
/// }
/// ```
pub unsafe fn set_blas_backend(dgemm: DgemmFnPtr, zgemm: ZgemmFnPtr) {
    let backend = ExternalBlasBackend { dgemm, zgemm };
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(backend);
}

/// Set ILP64 BLAS backend (64-bit integers)
///
/// # Safety
/// - Function pointers must be valid, thread-safe, and use 64-bit integers
/// - Must remain valid for the lifetime of the program
/// - Must follow Fortran BLAS calling convention with ILP64 interface
///
/// # Example
/// ```ignore
/// unsafe {
///     set_ilp64_backend(dgemm_ as _, zgemm_ as _);
/// }
/// ```
pub unsafe fn set_ilp64_backend(dgemm64: Dgemm64FnPtr, zgemm64: Zgemm64FnPtr) {
    let backend = ExternalBlas64Backend { dgemm64, zgemm64 };
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(backend);
}

/// Clear BLAS backend (reset to default Faer)
///
/// This function resets the GEMM dispatcher to use the default Pure Rust Faer backend.
pub fn clear_blas_backend() {
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(FaerBackend);
}

/// Get current BLAS backend information
///
/// Returns:
/// - `(backend_name, is_external, is_ilp64)`
pub fn get_backend_info() -> (&'static str, bool, bool) {
    let dispatcher = BLAS_DISPATCHER.read().unwrap();
    let name = dispatcher.name();
    let is_external = !name.contains("Faer");
    let is_ilp64 = dispatcher.is_ilp64();
    (name, is_external, is_ilp64)
}

//==============================================================================
// Public API
//==============================================================================

/// Parallel matrix multiplication: C = A * B
///
/// Dispatches to registered BLAS backend (external or Faer).
///
/// # Arguments
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
/// Result matrix (M x N)
///
/// # Panics
/// Panics if matrix dimensions are incompatible (A.cols != B.rows)
///
/// # Example
/// ```ignore
/// use mdarray::tensor;
/// use sparseir_rust::gemm::matmul_par;
///
/// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
/// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
/// let c = matmul_par(&a, &b);
/// // c = [[19.0, 22.0], [43.0, 50.0]]
/// ```
pub fn matmul_par<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + Copy + 'static,
{
    let (_m, k) = *a.shape();
    let (k2, _n) = *b.shape();

    // Validate dimensions
    assert_eq!(
        k, k2,
        "Matrix dimension mismatch: A.cols ({}) != B.rows ({})",
        k, k2
    );

    // Use Faer directly to avoid creating intermediate DTensors through backend
    // create _m x _n result tensor
    let mut result = DTensor::<T, 2>::from_elem([_m, _n], T::zero().into());
    matmul_par_overwrite(a, b, &mut result);
    result
}

/// Parallel matrix multiplication with overwrite: C = A * B (writes to existing buffer)
///
/// This function writes the result directly into the provided buffer `c`,
/// avoiding memory allocation. This is more memory-efficient for repeated operations.
///
/// # Arguments
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
/// * `c` - Output matrix (M x N) - will be overwritten with result
///
/// # Panics
/// Panics if matrix dimensions are incompatible (A.cols != B.rows or C.shape != [M, N])
pub fn matmul_par_overwrite<T, Lc: Layout>(
    a: &DTensor<T, 2>,
    b: &DTensor<T, 2>,
    c: &mut DSlice<T, 2, Lc>,
) where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + Copy + 'static,
{
    let (m, k) = *a.shape();
    let (k2, n) = *b.shape();
    let (mc, nc) = *c.shape();

    // Validate dimensions
    assert_eq!(
        k, k2,
        "Matrix dimension mismatch: A.cols ({}) != B.rows ({})",
        k, k2
    );
    assert_eq!(
        m, mc,
        "Output matrix dimension mismatch: C.rows ({}) != A.rows ({})",
        mc, m
    );
    assert_eq!(
        n, nc,
        "Output matrix dimension mismatch: C.cols ({}) != B.cols ({})",
        nc, n
    );

    // Get dispatcher
    let dispatcher = BLAS_DISPATCHER.read().unwrap();

    // Type dispatch: f64 or Complex<f64>
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // f64 case
        // Get pointers directly from DTensors (row-major order)
        let a_ptr = a.as_ptr() as *const f64;
        let b_ptr = b.as_ptr() as *const f64;
        let c_ptr = c.as_mut_ptr() as *mut f64;
        
        // Get leading dimension of c (for row-major, this is typically n, but we use the actual stride)
        // For DSlice, we need to determine the stride. For row-major, stride[1] = 1, stride[0] = n
        // We'll use n as ldc for row-major output
        let ldc = n;
        
        // Call backend directly with pointers (no temporary buffer needed)
        unsafe {
            dispatcher.dgemm(m, n, k, a_ptr, b_ptr, c_ptr, ldc);
        }
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<num_complex::Complex<f64>>() {
        // Complex<f64> case
        // Get pointers directly from DTensors (row-major order)
        let a_ptr = a.as_ptr() as *const num_complex::Complex<f64>;
        let b_ptr = b.as_ptr() as *const num_complex::Complex<f64>;
        let c_ptr = c.as_mut_ptr() as *mut num_complex::Complex<f64>;
        
        // Get leading dimension of c (for row-major, this is typically n)
        let ldc = n;
        
        // Call backend directly with pointers (no temporary buffer needed)
        unsafe {
            dispatcher.zgemm(m, n, k, a_ptr, b_ptr, c_ptr, ldc);
        }
    } else {
        // Fallback to Faer for unsupported types
        use mdarray_linalg::prelude::MatMul;
        use mdarray_linalg::matmul::MatMulBuilder;
        use mdarray_linalg_faer::Faer;

        Faer.matmul(a, b).parallelize().overwrite(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_is_faer() {
        let (name, is_external, is_ilp64) = get_backend_info();
        assert_eq!(name, "Faer (Pure Rust)");
        assert!(!is_external);
        assert!(!is_ilp64);
    }

    #[test]
    fn test_clear_backend() {
        // Should not panic
        clear_blas_backend();
        let (name, _, _) = get_backend_info();
        assert_eq!(name, "Faer (Pure Rust)");
    }

    #[test]
    fn test_matmul_f64() {
        let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let a = DTensor::<f64, 2>::from_fn([2, 3], |idx| a_data[idx[0] * 3 + idx[1]]);
        let b = DTensor::<f64, 2>::from_fn([3, 2], |idx| b_data[idx[0] * 2 + idx[1]]);
        let c = matmul_par(&a, &b);

        assert_eq!(*c.shape(), (2, 2));
        // First row: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Second row: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c[[0, 0]] - 58.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 64.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 139.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_par_basic() {
        use mdarray::tensor;
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
        let b: DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul_par(&a, &b);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_par_non_square() {
        use mdarray::tensor;
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let b: DTensor<f64, 2> = tensor![[7.0], [8.0], [9.0]]; // 3x1
        let c = matmul_par(&a, &b);

        // Expected: [[1*7+2*8+3*9], [4*7+5*8+6*9]]
        //         = [[50], [122]]
        assert!((c[[0, 0]] - 50.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 122.0).abs() < 1e-10);
    }
}
