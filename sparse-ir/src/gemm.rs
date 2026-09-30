//! Column-major GEMM with a pluggable BLAS backend.
//!
//! All matrices are column-major (Fortran/BLAS convention) and every call is
//! expressed in BLAS terms: `C <- alpha * op(A) * op(B) + beta * C` with
//! explicit leading dimensions, so strided sub-blocks are addressed without
//! copies.
//!
//! # Design
//! - **Default**: pure Rust faer backend (sequential), or system BLAS when
//!   the `system-blas` feature is enabled.
//! - **Optional**: external LP64/ILP64 BLAS injected through function
//!   pointers (used by the C API).
//! - **Thread-safe**: the process-wide default is protected by an `RwLock`;
//!   per-call backends are passed as [`GemmBackendHandle`].
//!
//! The safe entry point is [`gemm`], which validates buffer lengths and
//! leading dimensions before dispatching; [`matmul`] multiplies two
//! [`Matrix`](crate::Matrix) values.
//!
//! # Example
//! ```
//! use sparse_ir::Matrix;
//! use sparse_ir::gemm::{GemmBackendHandle, matmul};
//!
//! // Column-major data: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
//! let a = Matrix::<f64>::from_vec_col_major([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
//! let b = Matrix::<f64>::from_vec_col_major([2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap();
//!
//! // `None` uses the global dispatcher: the Faer backend by default (system
//! // BLAS with the `system-blas` feature, or a BLAS injected at runtime)
//! let c = matmul(None, &a, &b).unwrap();
//! assert_eq!(c.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
//!
//! // Or pass an explicit backend handle instead of relying on global state
//! let faer = GemmBackendHandle::default();
//! assert_eq!(
//!     matmul(Some(&faer), &a, &b).unwrap().host_data().unwrap(),
//!     c.host_data().unwrap()
//! );
//! ```
//!
//! A custom BLAS (e.g. from the C API) is injected at runtime with
//! [`set_blas_backend`] (LP64) or [`set_ilp64_backend`] (ILP64); see their
//! examples. [`clear_blas_backend`] restores the Faer backend.

use num_complex::Complex;
use num_traits::{One, Zero};
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

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

// When using system BLAS via `blas-sys`, we need a small wrapper to adapt
// `blas_sys::zgemm_` (which uses `c_double_complex = [f64; 2]`) to the
// `ZgemmFnPtr` signature that takes `num_complex::Complex<f64>`.
#[cfg(feature = "system-blas")]
unsafe extern "C" fn zgemm_wrapper(
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
) {
    // Safety: `blas_sys::c_double_complex` is defined as `[f64; 2]` and is
    // layout-compatible with `num_complex::Complex<f64>` in memory, so we can
    // cast between the two pointer types here.
    unsafe {
        blas_sys::zgemm_(
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const _ as *const blas_sys::c_double_complex,
            a as *const _ as *const blas_sys::c_double_complex,
            lda,
            b as *const _ as *const blas_sys::c_double_complex,
            ldb,
            beta as *const _ as *const blas_sys::c_double_complex,
            c as *mut _ as *mut blas_sys::c_double_complex,
            ldc,
        );
    }
}

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
// Operation descriptors and errors
//==============================================================================

/// Operation applied to a GEMM operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transpose {
    /// `op(X) = X`
    N,
    /// `op(X) = X^T`
    T,
    /// `op(X) = X^H`
    C,
}

impl Transpose {
    fn as_blas_char(self) -> libc::c_char {
        (match self {
            Transpose::N => b'N',
            Transpose::T => b'T',
            Transpose::C => b'C',
        }) as libc::c_char
    }
}

/// Errors reported by GEMM dispatch.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GemmError {
    /// A dimension or leading dimension does not fit the backend integer type.
    #[error("GEMM argument {name}={value} exceeds the {abi} integer range")]
    DimensionOverflow {
        name: &'static str,
        value: usize,
        abi: &'static str,
    },
    /// A buffer is too short or a leading dimension is too small.
    #[error("invalid GEMM argument: {0}")]
    InvalidArgument(String),
}

//==============================================================================
// GemmBackend Trait
//==============================================================================

/// GEMM backend trait for runtime dispatch.
///
/// Implementations compute `C <- alpha * op(A) * op(B) + beta * C` on
/// column-major operands, with BLAS semantics: `op(A)` is `m x k`, `op(B)`
/// is `k x n`, `C` is `m x n`, and `C` is not read when `beta == 0`.
pub trait GemmBackend: Send + Sync {
    /// Real double-precision GEMM.
    ///
    /// # Safety
    /// The pointers must address column-major matrices of the stated shapes
    /// and leading dimensions (`lda >= max(1, rows(A))`, and likewise for `B`
    /// and `C`); `c` must not alias `a` or `b`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) -> Result<(), GemmError>;

    /// Complex double-precision GEMM.
    ///
    /// # Safety
    /// Same contract as [`GemmBackend::dgemm`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn zgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        ldb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) -> Result<(), GemmError>;

    /// Returns true if this backend uses 64-bit integers (ILP64).
    fn is_ilp64(&self) -> bool {
        false
    }

    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;
}

//==============================================================================
// Faer Backend (Default, Pure Rust, Zero-Copy)
//==============================================================================

/// Pure Rust faer backend (sequential).
struct FaerBackend;

/// Shared faer implementation for `f64` and `Complex<f64>`.
///
/// # Safety
/// Same contract as [`GemmBackend::dgemm`].
#[allow(clippy::too_many_arguments)]
unsafe fn faer_gemm<T>(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    b: *const T,
    ldb: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) where
    T: faer_traits::ComplexField + Copy + PartialEq + Zero + One + std::ops::MulAssign,
{
    use faer::linalg::matmul::matmul_with_conj;
    use faer::mat::{MatMut, MatRef};
    use faer::{Accum, Conj, Par};

    if m == 0 || n == 0 {
        return;
    }
    // SAFETY: the caller guarantees `c` addresses an m x n column-major
    // matrix with leading dimension ldc >= m; `c` is non-null because m, n > 0.
    let mut dst = unsafe { MatMut::from_raw_parts_mut(c, m, n, 1isize, ldc as isize) };

    // faer accumulates with beta in {0, 1}; other values scale C first.
    let accum = if beta == T::zero() {
        Accum::Replace
    } else {
        if beta != T::one() {
            for j in 0..n {
                for i in 0..m {
                    // SAFETY: (i, j) is within the m x n destination.
                    unsafe { *dst.as_mut().get_mut_unchecked(i, j) *= beta };
                }
            }
        }
        Accum::Add
    };
    if k == 0 {
        if accum == Accum::Replace {
            dst.fill(T::zero());
        }
        return;
    }

    let operand = |ptr: *const T, op: Transpose, rows: usize, cols: usize, ld: usize| {
        // SAFETY: the caller guarantees `ptr` addresses the stored (untransposed)
        // `rows x cols` operand with leading dimension ld; rows, cols > 0.
        let stored = match op {
            Transpose::N => unsafe { MatRef::from_raw_parts(ptr, rows, cols, 1isize, ld as isize) },
            Transpose::T | Transpose::C => {
                unsafe { MatRef::from_raw_parts(ptr, cols, rows, 1isize, ld as isize) }.transpose()
            }
        };
        let conj = if op == Transpose::C {
            Conj::Yes
        } else {
            Conj::No
        };
        (stored, conj)
    };
    let (lhs, conj_lhs) = operand(a, transa, m, k, lda);
    let (rhs, conj_rhs) = operand(b, transb, k, n, ldb);
    matmul_with_conj(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, Par::Seq);
}

impl GemmBackend for FaerBackend {
    unsafe fn dgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) -> Result<(), GemmError> {
        // SAFETY: forwarded caller contract.
        unsafe { faer_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) };
        Ok(())
    }

    unsafe fn zgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex<f64>,
        a: *const Complex<f64>,
        lda: usize,
        b: *const Complex<f64>,
        ldb: usize,
        beta: Complex<f64>,
        c: *mut Complex<f64>,
        ldc: usize,
    ) -> Result<(), GemmError> {
        // SAFETY: forwarded caller contract.
        unsafe { faer_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) };
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Faer (Pure Rust)"
    }
}

//==============================================================================
// External BLAS Backends (LP64 and ILP64)
//==============================================================================

/// Checked conversion of the six BLAS integer arguments.
fn blas_ints<I: TryFrom<usize>>(
    abi: &'static str,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> Result<[I; 6], GemmError> {
    let conv = |name: &'static str, value: usize| {
        I::try_from(value).map_err(|_| GemmError::DimensionOverflow { name, value, abi })
    };
    Ok([
        conv("m", m)?,
        conv("n", n)?,
        conv("k", k)?,
        conv("lda", lda)?,
        conv("ldb", ldb)?,
        conv("ldc", ldc)?,
    ])
}

/// External BLAS backend (LP64: 32-bit integers).
pub struct ExternalBlasBackend {
    dgemm: DgemmFnPtr,
    zgemm: ZgemmFnPtr,
}

impl ExternalBlasBackend {
    /// Wrap LP64 Fortran `dgemm_`/`zgemm_` function pointers.
    pub fn new(dgemm: DgemmFnPtr, zgemm: ZgemmFnPtr) -> Self {
        Self { dgemm, zgemm }
    }
}

/// External BLAS backend (ILP64: 64-bit integers).
pub struct ExternalBlas64Backend {
    dgemm64: Dgemm64FnPtr,
    zgemm64: Zgemm64FnPtr,
}

impl ExternalBlas64Backend {
    /// Wrap ILP64 Fortran `dgemm_`/`zgemm_` function pointers.
    pub fn new(dgemm64: Dgemm64FnPtr, zgemm64: Zgemm64FnPtr) -> Self {
        Self { dgemm64, zgemm64 }
    }
}

macro_rules! impl_external_backend {
    ($ty:ty, $int:ty, $abi:literal, $dfield:ident, $zfield:ident, $ilp64:expr, $name:literal) => {
        impl GemmBackend for $ty {
            unsafe fn dgemm(
                &self,
                transa: Transpose,
                transb: Transpose,
                m: usize,
                n: usize,
                k: usize,
                alpha: f64,
                a: *const f64,
                lda: usize,
                b: *const f64,
                ldb: usize,
                beta: f64,
                c: *mut f64,
                ldc: usize,
            ) -> Result<(), GemmError> {
                if m == 0 || n == 0 {
                    return Ok(());
                }
                let [m, n, k, lda, ldb, ldc] = blas_ints::<$int>($abi, m, n, k, lda, ldb, ldc)?;
                let (ta, tb) = (transa.as_blas_char(), transb.as_blas_char());
                // SAFETY: arguments follow the Fortran BLAS reference-passing
                // convention; the caller guarantees the operand contract.
                unsafe {
                    (self.$dfield)(
                        &ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    )
                };
                Ok(())
            }

            unsafe fn zgemm(
                &self,
                transa: Transpose,
                transb: Transpose,
                m: usize,
                n: usize,
                k: usize,
                alpha: Complex<f64>,
                a: *const Complex<f64>,
                lda: usize,
                b: *const Complex<f64>,
                ldb: usize,
                beta: Complex<f64>,
                c: *mut Complex<f64>,
                ldc: usize,
            ) -> Result<(), GemmError> {
                if m == 0 || n == 0 {
                    return Ok(());
                }
                let [m, n, k, lda, ldb, ldc] = blas_ints::<$int>($abi, m, n, k, lda, ldb, ldc)?;
                let (ta, tb) = (transa.as_blas_char(), transb.as_blas_char());
                // SAFETY: as for dgemm; Complex<f64> is layout-compatible with
                // Fortran COMPLEX*16.
                unsafe {
                    (self.$zfield)(
                        &ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    )
                };
                Ok(())
            }

            fn is_ilp64(&self) -> bool {
                $ilp64
            }

            fn name(&self) -> &'static str {
                $name
            }
        }
    };
}

impl_external_backend!(
    ExternalBlasBackend,
    libc::c_int,
    "LP64",
    dgemm,
    zgemm,
    false,
    "External BLAS (LP64)"
);
impl_external_backend!(
    ExternalBlas64Backend,
    i64,
    "ILP64",
    dgemm64,
    zgemm64,
    true,
    "External BLAS (ILP64)"
);

//==============================================================================
// Backend Handle
//==============================================================================

/// Shared, cloneable GEMM backend selection.
///
/// # Example
/// ```
/// use sparse_ir::Matrix;
/// use sparse_ir::gemm::{GemmBackendHandle, matmul};
///
/// // Column-major data: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
/// let a = Matrix::<f64>::from_vec_col_major([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let b = Matrix::<f64>::from_vec_col_major([2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap();
/// let backend = GemmBackendHandle::default();
/// let result = matmul(Some(&backend), &a, &b).unwrap();
/// assert_eq!(result.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
/// ```
#[derive(Clone)]
pub struct GemmBackendHandle {
    inner: Arc<dyn GemmBackend>,
}

impl GemmBackendHandle {
    /// Wrap a backend.
    pub fn new(backend: Box<dyn GemmBackend>) -> Self {
        Self {
            inner: Arc::from(backend),
        }
    }

    /// Pure Rust faer backend.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: Arc::new(FaerBackend),
        }
    }

    pub(crate) fn as_ref(&self) -> &dyn GemmBackend {
        self.inner.as_ref()
    }
}

//==============================================================================
// Global Dispatcher
//==============================================================================

static BLAS_DISPATCHER: Lazy<RwLock<Box<dyn GemmBackend>>> = Lazy::new(|| {
    #[cfg(feature = "system-blas")]
    {
        // Use system BLAS (LP64) by default via `blas-sys`.
        let backend =
            ExternalBlasBackend::new(blas_sys::dgemm_ as DgemmFnPtr, zgemm_wrapper as ZgemmFnPtr);
        RwLock::new(Box::new(backend) as Box<dyn GemmBackend>)
    }
    #[cfg(not(feature = "system-blas"))]
    {
        RwLock::new(Box::new(FaerBackend) as Box<dyn GemmBackend>)
    }
});

/// Set the process-wide default to an LP64 BLAS.
///
/// # Safety
/// The function pointers must be valid Fortran-convention LP64 `dgemm_` and
/// `zgemm_` implementations that stay valid for the rest of the process.
///
/// # Example
/// ```
/// use num_complex::Complex;
/// use sparse_ir::Matrix;
/// use sparse_ir::gemm::{clear_blas_backend, get_backend_info, matmul, set_blas_backend};
/// # use std::ffi::{c_char, c_int};
/// # use std::ops::{Add, Mul};
/// #
/// # // Naive column-major C = alpha * A * B + beta * C for transa = transb = 'N'
/// # // (what sparse-ir passes), standing in for a BLAS library.
/// # unsafe fn gemm_nn<T>(mnk: [usize; 3], alpha: T, a: *const T, lda: usize,
/// #                    b: *const T, ldb: usize, beta: T, c: *mut T, ldc: usize)
/// # where
/// #     T: Copy + Default + PartialEq + Add<Output = T> + Mul<Output = T>,
/// # {
/// #     let [m, n, k] = mnk;
/// #     for j in 0..n {
/// #         for i in 0..m {
/// #             let mut acc = T::default();
/// #             for p in 0..k {
/// #                 acc = acc + unsafe { *a.add(i + p * lda) * *b.add(p + j * ldb) };
/// #             }
/// #             let cij = unsafe { &mut *c.add(i + j * ldc) };
/// #             *cij = if beta == T::default() { alpha * acc } else { alpha * acc + beta * *cij };
/// #         }
/// #     }
/// # }
/// # unsafe extern "C" fn my_dgemm(
/// #     _transa: *const c_char, _transb: *const c_char,
/// #     m: *const c_int, n: *const c_int, k: *const c_int,
/// #     alpha: *const f64, a: *const f64, lda: *const c_int,
/// #     b: *const f64, ldb: *const c_int,
/// #     beta: *const f64, c: *mut f64, ldc: *const c_int,
/// # ) {
/// #     unsafe {
/// #         let mnk = [*m as usize, *n as usize, *k as usize];
/// #         gemm_nn(mnk, *alpha, a, *lda as usize, b, *ldb as usize, *beta, c, *ldc as usize);
/// #     }
/// # }
/// # unsafe extern "C" fn my_zgemm(
/// #     _transa: *const c_char, _transb: *const c_char,
/// #     m: *const c_int, n: *const c_int, k: *const c_int,
/// #     alpha: *const Complex<f64>, a: *const Complex<f64>, lda: *const c_int,
/// #     b: *const Complex<f64>, ldb: *const c_int,
/// #     beta: *const Complex<f64>, c: *mut Complex<f64>, ldc: *const c_int,
/// # ) {
/// #     unsafe {
/// #         let mnk = [*m as usize, *n as usize, *k as usize];
/// #         gemm_nn(mnk, *alpha, a, *lda as usize, b, *ldb as usize, *beta, c, *ldc as usize);
/// #     }
/// # }
///
/// // `my_dgemm` and `my_zgemm` are the LP64 Fortran BLAS `dgemm_` and `zgemm_`
/// // to use, e.g. from OpenBLAS. (This example defines naive stand-ins in
/// // hidden lines so that it runs without a BLAS library.)
/// unsafe {
///     set_blas_backend(my_dgemm, my_zgemm);
/// }
/// assert_eq!(get_backend_info(), ("External BLAS (LP64)", true, false));
///
/// // Calls without an explicit backend handle now go through the injected BLAS
/// // Column-major data: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
/// let a = Matrix::<f64>::from_vec_col_major([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let b = Matrix::<f64>::from_vec_col_major([2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap();
/// let c = matmul(None, &a, &b).unwrap();
/// assert_eq!(c.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
///
/// // Complex matrices use `zgemm`: (i A) B = i (A B)
/// let to_complex = |m: &Matrix<f64>, imaginary: bool| {
///     let data = m.host_data().unwrap().iter().map(|&x| {
///         if imaginary { Complex::new(0.0, x) } else { Complex::new(x, 0.0) }
///     });
///     Matrix::from_vec_col_major([2, 2], data.collect()).unwrap()
/// };
/// let ic = matmul(None, &to_complex(&a, true), &to_complex(&b, false)).unwrap();
/// assert_eq!(ic.host_data().unwrap(), to_complex(&c, true).host_data().unwrap());
///
/// clear_blas_backend(); // back to the pure-Rust Faer backend
/// assert!(!get_backend_info().1);
/// ```
pub unsafe fn set_blas_backend(dgemm: DgemmFnPtr, zgemm: ZgemmFnPtr) {
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap_or_else(|e| e.into_inner());
    *dispatcher = Box::new(ExternalBlasBackend::new(dgemm, zgemm));
}

/// Set the process-wide default to an ILP64 BLAS.
///
/// # Safety
/// As for [`set_blas_backend`], with 64-bit integer arguments.
///
/// # Example
/// ```
/// use sparse_ir::Matrix;
/// use sparse_ir::gemm::{clear_blas_backend, get_backend_info, matmul, set_ilp64_backend};
/// # use num_complex::Complex;
/// # use std::ffi::c_char;
/// # use std::ops::{Add, Mul};
/// #
/// # // Naive column-major C = alpha * A * B + beta * C for transa = transb = 'N'
/// # // (what sparse-ir passes), standing in for a BLAS library.
/// # unsafe fn gemm_nn<T>(mnk: [usize; 3], alpha: T, a: *const T, lda: usize,
/// #                    b: *const T, ldb: usize, beta: T, c: *mut T, ldc: usize)
/// # where
/// #     T: Copy + Default + PartialEq + Add<Output = T> + Mul<Output = T>,
/// # {
/// #     let [m, n, k] = mnk;
/// #     for j in 0..n {
/// #         for i in 0..m {
/// #             let mut acc = T::default();
/// #             for p in 0..k {
/// #                 acc = acc + unsafe { *a.add(i + p * lda) * *b.add(p + j * ldb) };
/// #             }
/// #             let cij = unsafe { &mut *c.add(i + j * ldc) };
/// #             *cij = if beta == T::default() { alpha * acc } else { alpha * acc + beta * *cij };
/// #         }
/// #     }
/// # }
/// # unsafe extern "C" fn my_dgemm64(
/// #     _transa: *const c_char, _transb: *const c_char,
/// #     m: *const i64, n: *const i64, k: *const i64,
/// #     alpha: *const f64, a: *const f64, lda: *const i64,
/// #     b: *const f64, ldb: *const i64,
/// #     beta: *const f64, c: *mut f64, ldc: *const i64,
/// # ) {
/// #     unsafe {
/// #         let mnk = [*m as usize, *n as usize, *k as usize];
/// #         gemm_nn(mnk, *alpha, a, *lda as usize, b, *ldb as usize, *beta, c, *ldc as usize);
/// #     }
/// # }
/// # unsafe extern "C" fn my_zgemm64(
/// #     _transa: *const c_char, _transb: *const c_char,
/// #     m: *const i64, n: *const i64, k: *const i64,
/// #     alpha: *const Complex<f64>, a: *const Complex<f64>, lda: *const i64,
/// #     b: *const Complex<f64>, ldb: *const i64,
/// #     beta: *const Complex<f64>, c: *mut Complex<f64>, ldc: *const i64,
/// # ) {
/// #     unsafe {
/// #         let mnk = [*m as usize, *n as usize, *k as usize];
/// #         gemm_nn(mnk, *alpha, a, *lda as usize, b, *ldb as usize, *beta, c, *ldc as usize);
/// #     }
/// # }
///
/// // `my_dgemm64` and `my_zgemm64` are the ILP64 (64-bit integer) Fortran BLAS
/// // `dgemm_` and `zgemm_` to use. (This example defines naive stand-ins in
/// // hidden lines so that it runs without an ILP64 BLAS library.)
/// unsafe {
///     set_ilp64_backend(my_dgemm64, my_zgemm64);
/// }
/// assert_eq!(get_backend_info(), ("External BLAS (ILP64)", true, true));
///
/// // Calls without an explicit backend handle now go through the injected BLAS
/// // Column-major data: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
/// let a = Matrix::<f64>::from_vec_col_major([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let b = Matrix::<f64>::from_vec_col_major([2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap();
/// let c = matmul(None, &a, &b).unwrap();
/// assert_eq!(c.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
///
/// clear_blas_backend(); // back to the pure-Rust Faer backend
/// let (_, is_external, is_ilp64) = get_backend_info();
/// assert!(!is_external && !is_ilp64);
/// ```
pub unsafe fn set_ilp64_backend(dgemm64: Dgemm64FnPtr, zgemm64: Zgemm64FnPtr) {
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap_or_else(|e| e.into_inner());
    *dispatcher = Box::new(ExternalBlas64Backend::new(dgemm64, zgemm64));
}

/// Reset the process-wide default to the pure Rust faer backend.
pub fn clear_blas_backend() {
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap_or_else(|e| e.into_inner());
    *dispatcher = Box::new(FaerBackend);
}

/// Returns `(name, is_external, is_ilp64)` of the process-wide default.
pub fn get_backend_info() -> (&'static str, bool, bool) {
    let dispatcher = BLAS_DISPATCHER.read().unwrap_or_else(|e| e.into_inner());
    let name = dispatcher.name();
    (name, !name.contains("Faer"), dispatcher.is_ilp64())
}

/// Run `f` with the selected backend (explicit handle or process default).
fn with_backend<R>(
    backend: Option<&GemmBackendHandle>,
    f: impl FnOnce(&dyn GemmBackend) -> R,
) -> R {
    match backend {
        Some(handle) => f(handle.as_ref()),
        None => {
            let dispatcher = BLAS_DISPATCHER.read().unwrap_or_else(|e| e.into_inner());
            f(dispatcher.as_ref())
        }
    }
}

//==============================================================================
// Scalar dispatch and safe entry points
//==============================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Scalars supported by [`gemm`]: `f64` and `Complex<f64>`.
pub trait GemmScalar:
    sealed::Sealed
    + tenferro_tensor::TensorScalar
    + Copy
    + Send
    + Sync
    + PartialEq
    + Zero
    + One
    + std::ops::Mul<Output = Self>
    + 'static
{
    /// Dispatch to the backend routine for this scalar.
    ///
    /// # Safety
    /// Same contract as [`GemmBackend::dgemm`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_raw(
        backend: &dyn GemmBackend,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    ) -> Result<(), GemmError>;
}

impl GemmScalar for f64 {
    unsafe fn gemm_raw(
        backend: &dyn GemmBackend,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) -> Result<(), GemmError> {
        // SAFETY: forwarded caller contract.
        unsafe { backend.dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
    }
}

impl GemmScalar for Complex<f64> {
    unsafe fn gemm_raw(
        backend: &dyn GemmBackend,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: *const Self,
        lda: usize,
        b: *const Self,
        ldb: usize,
        beta: Self,
        c: *mut Self,
        ldc: usize,
    ) -> Result<(), GemmError> {
        // SAFETY: forwarded caller contract.
        unsafe { backend.zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
    }
}

/// Validate one column-major operand: `rows x cols` stored with leading
/// dimension `ld` in a buffer of `len` elements.
fn check_operand(
    name: &str,
    rows: usize,
    cols: usize,
    ld: usize,
    len: usize,
) -> Result<(), GemmError> {
    if ld < rows.max(1) {
        return Err(GemmError::InvalidArgument(format!(
            "leading dimension of {name} is {ld}, expected at least {}",
            rows.max(1)
        )));
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let required = ld
        .checked_mul(cols - 1)
        .and_then(|x| x.checked_add(rows))
        .ok_or_else(|| GemmError::InvalidArgument(format!("{name} extent overflows usize")))?;
    if len < required {
        return Err(GemmError::InvalidArgument(format!(
            "{name} buffer has {len} elements, {rows}x{cols} with ld={ld} needs {required}"
        )));
    }
    Ok(())
}

/// Safe column-major GEMM: `C <- alpha * op(A) * op(B) + beta * C`.
///
/// `op(A)` is `m x k`, `op(B)` is `k x n`, and `C` is `m x n`; the stored
/// operands are addressed through their leading dimensions.
///
/// # Errors
/// Returns [`GemmError::InvalidArgument`] when a buffer is too short or a
/// leading dimension is too small, and [`GemmError::DimensionOverflow`] when
/// an LP64 backend cannot represent a dimension.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: GemmScalar>(
    backend: Option<&GemmBackendHandle>,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<(), GemmError> {
    let (ar, ac) = if transa == Transpose::N {
        (m, k)
    } else {
        (k, m)
    };
    let (br, bc) = if transb == Transpose::N {
        (k, n)
    } else {
        (n, k)
    };
    check_operand("A", ar, ac, lda, a.len())?;
    check_operand("B", br, bc, ldb, b.len())?;
    check_operand("C", m, n, ldc, c.len())?;
    // SAFETY: extents validated above; `c` is a unique borrow so it cannot
    // alias `a` or `b`.
    with_backend(backend, |be| unsafe {
        T::gemm_raw(
            be,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        )
    })
}

/// Below this `pre` extent, a middle-axis contraction is packed into one
/// GEMM instead of looping over `post` slices.
const SLICE_LOOP_MIN_PRE: usize = 32;

/// Matrix-vector products up to this size bypass the default backend.
const SMALL_MATVEC_MAX_ELEMS: usize = 128 * 128;
const SMALL_MATVEC_MAX_POST: usize = 4;

/// Apply a contiguous column-major `m x n` matrix `a` along the middle axis
/// of a column-major `[pre, n, post]` array `x`, writing `[pre, m, post]`
/// into `y` (overwritten).
///
/// # Errors
/// Returns [`GemmError::InvalidArgument`] when a buffer length does not match
/// its extents, or a backend error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_along_axis<T: GemmScalar>(
    backend: Option<&GemmBackendHandle>,
    a: &[T],
    m: usize,
    n: usize,
    x: &[T],
    pre: usize,
    post: usize,
    y: &mut [T],
) -> Result<(), GemmError> {
    let len_ok = |len: usize, rows: usize| {
        pre.checked_mul(rows)
            .and_then(|v| v.checked_mul(post))
            .is_some_and(|v| v == len)
    };
    if a.len() != m * n || !len_ok(x.len(), n) || !len_ok(y.len(), m) {
        return Err(GemmError::InvalidArgument(format!(
            "apply_along_axis: A {}, X {}, Y {} elements for m={m}, n={n}, pre={pre}, post={post}",
            a.len(),
            x.len(),
            y.len()
        )));
    }
    if y.is_empty() {
        return Ok(());
    }
    let (one, zero) = (T::one(), T::zero());
    let lda = m.max(1);
    if pre == 1
        && post <= SMALL_MATVEC_MAX_POST
        && m * n <= SMALL_MATVEC_MAX_ELEMS
        && backend.is_none()
    {
        // Small matrix-vector products: the default backend's dispatch
        // overhead dominates, so accumulate columns of A directly.
        for q in 0..post {
            let yq = &mut y[q * m..(q + 1) * m];
            yq.fill(zero);
            for (j, col) in a.chunks_exact(m).enumerate() {
                let xj = x[q * n + j];
                for (yi, &aij) in yq.iter_mut().zip(col) {
                    *yi = *yi + aij * xj;
                }
            }
        }
        return Ok(());
    }
    if pre == 1 {
        // Y(m x post) = A(m x n) X(n x post)
        return gemm(
            backend,
            Transpose::N,
            Transpose::N,
            m,
            post,
            n,
            one,
            a,
            lda,
            x,
            n.max(1),
            zero,
            y,
            m.max(1),
        );
    }
    if post == 1 || pre >= SLICE_LOOP_MIN_PRE {
        // Y_q(pre x m) = X_q(pre x n) A^T for each post index q.
        let (xs, ys) = (pre * n, pre * m);
        for q in 0..post {
            gemm(
                backend,
                Transpose::N,
                Transpose::T,
                pre,
                m,
                n,
                one,
                &x[q * xs..(q + 1) * xs],
                pre,
                a,
                lda,
                zero,
                &mut y[q * ys..(q + 1) * ys],
                pre,
            )?;
        }
        return Ok(());
    }
    // Small pre, many post slices: pack X into [n, pre*post], one GEMM, unpack.
    let cols = pre * post;
    let mut xp = Vec::with_capacity(n * cols);
    for q in 0..post {
        for p in 0..pre {
            let base = p + pre * n * q;
            xp.extend((0..n).map(|j| x[base + pre * j]));
        }
    }
    let mut yp = vec![zero; m * cols];
    gemm(
        backend,
        Transpose::N,
        Transpose::N,
        m,
        cols,
        n,
        one,
        a,
        lda,
        &xp,
        n.max(1),
        zero,
        &mut yp,
        m.max(1),
    )?;
    for q in 0..post {
        for p in 0..pre {
            let src = &yp[m * (p + pre * q)..m * (p + pre * q + 1)];
            let base = p + pre * m * q;
            for (i, &v) in src.iter().enumerate() {
                y[base + pre * i] = v;
            }
        }
    }
    Ok(())
}

/// Matrix product `A * B` of two contiguous column-major matrices.
///
/// # Errors
/// Returns [`crate::Error::ShapeMismatch`] when the inner dimensions differ,
/// [`crate::Error::Tensor`] when an operand is not host-resident compact
/// column-major storage, or a backend error.
///
/// # Example
/// ```
/// use sparse_ir::Matrix;
/// use sparse_ir::gemm::{GemmBackendHandle, matmul};
///
/// // Column-major data: [[1, 2], [3, 4]] and [[5, 6], [7, 8]]
/// let a = Matrix::<f64>::from_vec_col_major([2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
/// let b = Matrix::<f64>::from_vec_col_major([2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap();
/// let backend = GemmBackendHandle::default();
/// let c = matmul(Some(&backend), &a, &b).unwrap();
/// assert_eq!(c.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]); // [[19, 22], [43, 50]]
///
/// // A 2x3 times 3x1 product
/// let a = Matrix::<f64>::from_vec_col_major([2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
/// let b = Matrix::<f64>::from_vec_col_major([3, 1], vec![7.0, 8.0, 9.0]).unwrap();
/// assert_eq!(matmul(None, &a, &b).unwrap().host_data().unwrap(), &[50.0, 122.0]);
/// ```
pub fn matmul<T: GemmScalar>(
    backend: Option<&GemmBackendHandle>,
    a: &crate::Matrix<T>,
    b: &crate::Matrix<T>,
) -> crate::Result<crate::Matrix<T>> {
    let (av, bv) = (a.host_col_major_view()?, b.host_col_major_view()?);
    let [m, k] = *av.shape();
    let [k2, n] = *bv.shape();
    if k != k2 {
        return Err(crate::Error::ShapeMismatch {
            which: crate::ArrayRole::Input,
            expected: vec![k, n],
            actual: vec![k2, n],
        });
    }
    let mut c = vec![T::zero(); m * n];
    gemm(
        backend,
        Transpose::N,
        Transpose::N,
        m,
        n,
        k,
        T::one(),
        av.as_slice(),
        m.max(1),
        bv.as_slice(),
        k.max(1),
        T::zero(),
        &mut c,
        m.max(1),
    )?;
    Ok(crate::Matrix::from_vec_col_major([m, n], c)?)
}

#[cfg(test)]
#[path = "gemm_tests.rs"]
mod tests;
