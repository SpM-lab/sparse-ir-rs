//! Common utilities for fitters
//!
//! This module contains shared helper functions and SVD structures
//! used by all fitter implementations.

use crate::error::{ArrayRole, Error};
use crate::fpu_check::FpuGuard;
use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DynRank, Shape, Slice, ViewMut};
use num_complex::Complex;

// ============================================================================
// InplaceFitter trait
// ============================================================================

/// Trait for inplace evaluation and fitting operations on N-dimensional arrays.
///
/// Uses BLAS-style naming convention for type suffixes:
/// - `d` = double (f64)
/// - `z` = double complex (Complex<f64>)
///
/// For example:
/// - `evaluate_nd_dd_to`: f64 input → f64 output
/// - `evaluate_nd_zz_to`: Complex<f64> input → Complex<f64> output
/// - `evaluate_nd_dz_to`: f64 input → Complex<f64> output
/// - `evaluate_nd_zd_to`: Complex<f64> input → f64 output
///
/// Each method returns `Ok(())` after writing the result to `out`. On an
/// error nothing is written to `out`:
/// - [`Error::NotSupported`] if the fitter does not support this pair of
///   types (the default implementations);
/// - [`Error::AxisOutOfRange`] if `dim` is not an axis of the input;
/// - [`Error::ShapeMismatch`] of the input if it does not have `basis_size`
///   (evaluate) or `n_points` (fit) along `dim`, and of the output if `out`
///   does not have the shape of the input with `n_points` (evaluate) or
///   `basis_size` (fit) along `dim`.
///
/// An empty input of the right shape (a batch axis of extent 0) gives
/// `Ok(())` without computing anything.
pub trait InplaceFitter {
    /// Number of sampling points
    fn n_points(&self) -> usize;

    /// Number of basis functions
    fn basis_size(&self) -> usize;

    /// Evaluate ND: f64 coeffs → f64 values
    fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, coeffs, dim, out);
        Err(not_supported(
            "evaluate_nd_dd_to (real coefficients to real values)",
        ))
    }

    /// Evaluate ND: f64 coeffs → Complex<f64> values
    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, coeffs, dim, out);
        Err(not_supported(
            "evaluate_nd_dz_to (real coefficients to complex values)",
        ))
    }

    /// Evaluate ND: Complex<f64> coeffs → f64 values
    fn evaluate_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, coeffs, dim, out);
        Err(not_supported(
            "evaluate_nd_zd_to (complex coefficients to real values)",
        ))
    }

    /// Evaluate ND: Complex<f64> coeffs → Complex<f64> values
    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, coeffs, dim, out);
        Err(not_supported(
            "evaluate_nd_zz_to (complex coefficients to complex values)",
        ))
    }

    /// Fit ND: f64 values → f64 coeffs
    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, values, dim, out);
        Err(not_supported(
            "fit_nd_dd_to (real values to real coefficients)",
        ))
    }

    /// Fit ND: f64 values → Complex<f64> coeffs
    fn fit_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, values, dim, out);
        Err(not_supported(
            "fit_nd_dz_to (real values to complex coefficients)",
        ))
    }

    /// Fit ND: Complex<f64> values → f64 coeffs
    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, values, dim, out);
        Err(not_supported(
            "fit_nd_zd_to (complex values to real coefficients)",
        ))
    }

    /// Fit ND: Complex<f64> values → Complex<f64> coeffs
    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        let _ = (backend, values, dim, out);
        Err(not_supported(
            "fit_nd_zz_to (complex values to complex coefficients)",
        ))
    }
}

/// The error of an [`InplaceFitter`] method that a fitter does not support
fn not_supported(operation: &str) -> Error {
    Error::NotSupported {
        what: format!("{operation} for this sampling"),
    }
}

// ============================================================================
// Permutation helpers
// ============================================================================

/// Generate permutation to move dimension `dim` to position 0
///
/// For example, with rank=4 and dim=2:
/// - Result: [2, 0, 1, 3]
pub(crate) fn make_perm_to_front(rank: usize, dim: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(dim);
    for i in 0..rank {
        if i != dim {
            perm.push(i);
        }
    }
    perm
}

// ============================================================================
// Shape validation
// ============================================================================

/// Check the input of an N-D evaluate or fit along axis `dim`
///
/// `Ok` if `dim` is an axis of the input and the input has extent `n_in`
/// along it. Call this before reading `input_dims[dim]` anywhere else, and
/// before allocating an output from the input shape.
///
/// # Errors
///
/// * [`Error::AxisOutOfRange`] if `dim` is not an axis of the input
/// * [`Error::ShapeMismatch`] of the input, with the shape it should have,
///   if its extent along `dim` is not `n_in`
pub(crate) fn check_input_shape(
    input_dims: &[usize],
    dim: usize,
    n_in: usize,
) -> Result<(), Error> {
    let rank = input_dims.len();
    if dim >= rank {
        return Err(Error::AxisOutOfRange { axis: dim, rank });
    }
    if input_dims[dim] != n_in {
        let mut expected = input_dims.to_vec();
        expected[dim] = n_in;
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected,
            actual: input_dims.to_vec(),
        });
    }
    Ok(())
}

/// Check the shapes of an N-D evaluate or fit along axis `dim` that writes
/// to `out`
///
/// The N-D methods of the fitters address the input and `out` as contiguous
/// matrices through unchecked views and pointer offsets computed from the
/// shape of the input. So the input must pass [`check_input_shape`], and
/// `out` must have the shape of the input with `n_out` along `dim` (in
/// particular its rank). Call this before any unchecked access or write.
///
/// # Errors
///
/// The errors of [`check_input_shape`], then [`Error::ShapeMismatch`] of the
/// output, with the shape it should have, if `out` has another shape
pub(crate) fn check_nd_shapes(
    input_dims: &[usize],
    dim: usize,
    n_in: usize,
    out_dims: &[usize],
    n_out: usize,
) -> Result<(), Error> {
    check_input_shape(input_dims, dim, n_in)?;
    let mut expected = input_dims.to_vec();
    expected[dim] = n_out;
    if out_dims != expected.as_slice() {
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Output,
            expected,
            actual: out_dims.to_vec(),
        });
    }
    Ok(())
}

// ============================================================================
// Strided copy helpers
// ============================================================================

/// Copy data from a contiguous slice to a strided view
///
/// This is useful for copying GEMM results back to a permuted output view.
///
/// # Arguments
/// * `src` - Source slice (contiguous)
/// * `dst` - Destination slice (may be strided)
pub(crate) fn copy_from_contiguous<T: Copy>(
    src: &[T],
    dst: &mut mdarray::Slice<T, mdarray::DynRank, mdarray::Strided>,
) {
    assert_eq!(src.len(), dst.len(), "Source size mismatch");

    // mdarray's iter_mut() returns elements in row-major order
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s;
    }
}

// ============================================================================
// Complex-Real reinterpretation helpers
// ============================================================================

/// Reinterpret a mutable Complex<f64> slice as a mutable f64 view with an extra dimension of size 2
///
/// Complex<f64> array `[d0, d1, ..., dN]` becomes f64 array `[d0, d1, ..., dN, 2]`
/// where the last dimension contains [re, im] pairs.
#[allow(dead_code)]
pub(crate) fn complex_slice_mut_as_real<'a>(
    out: &'a mut Slice<Complex<f64>, DynRank>,
) -> mdarray::ViewMut<'a, f64, DynRank, mdarray::Dense> {
    // Build new shape: [..., 2]
    let mut new_shape: Vec<usize> = Vec::with_capacity(out.rank() + 1);
    out.shape().with_dims(|dims| {
        for d in dims {
            new_shape.push(*d);
        }
    });
    new_shape.push(2);

    unsafe {
        let shape: DynRank = Shape::from_dims(&new_shape[..]);
        let mapping = mdarray::DenseMapping::new(shape);
        mdarray::ViewMut::new_unchecked(out.as_mut_ptr() as *mut f64, mapping)
    }
}

// ============================================================================
// SVD structures
// ============================================================================

/// Transpose of a matrix, as a new dense tensor
///
/// Zero-extent guard: mdarray 0.7.2 copies the transposed (strided) view of a
/// `[n, 0]` matrix out of bounds (https://github.com/fre-hu/mdarray/issues/21),
/// which the SVD of a matrix with a zero dimension produces. An empty matrix
/// has nothing to copy.
fn transposed<T: Clone + Default>(m: &DTensor<T, 2>) -> DTensor<T, 2> {
    let (rows, cols) = *m.shape();
    if m.is_empty() {
        return DTensor::<T, 2>::zeros([cols, rows]);
    }
    m.transpose().to_tensor()
}

/// SVD decomposition for real matrices
pub(crate) struct RealSVD {
    pub ut: DTensor<f64, 2>, // (min_dim, n_rows) - U^T
    pub s: Vec<f64>,         // (min_dim,)
    pub v: DTensor<f64, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl RealSVD {
    pub fn new(u: DTensor<f64, 2>, s: Vec<f64>, vt: DTensor<f64, 2>) -> Self {
        // Check dimensions
        let (_, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut and v from u and vt
        let ut = transposed(&u); // (min_dim, n_rows)
        let v = transposed(&vt); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

/// SVD decomposition for complex matrices
pub(crate) struct ComplexSVD {
    pub ut: DTensor<Complex<f64>, 2>, // (min_dim, n_rows) - U^H
    pub s: Vec<f64>,                  // (min_dim,) - singular values are real
    pub v: DTensor<Complex<f64>, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl ComplexSVD {
    pub fn new(u: DTensor<Complex<f64>, 2>, s: Vec<f64>, vt: DTensor<Complex<f64>, 2>) -> Self {
        // Check dimensions
        let (_, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut (U^H, conjugate transpose) and v from u and vt
        let ut = transposed(&u).map(|x| x.conj()); // conjugate transpose: U^H
        let v = transposed(&vt); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

/// Condition number `σ_max / σ_min` from the singular values of a fitting matrix
///
/// Conventions shared by the `condition_number` methods of all samplings and
/// by `spir_sampling_get_cond_num`:
/// - `f64::INFINITY` if `σ_min < 1e-15` (numerically singular matrix);
/// - `1.0` if there are no singular values (a matrix with a zero dimension);
/// - `NaN` if any singular value is NaN, so a failed decomposition never
///   yields a plausible finite value.
pub(crate) fn condition_number_from_singular_values(s: &[f64]) -> f64 {
    if s.is_empty() {
        return 1.0;
    }
    if s.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let s_max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s_min = s.iter().copied().fold(f64::INFINITY, f64::min);
    if s_min.abs() < 1e-15 {
        return f64::INFINITY;
    }
    s_max / s_min
}

// ============================================================================
// SVD computation functions
// ============================================================================

/// Compute SVD of a real matrix using mdarray-linalg
pub(crate) fn compute_real_svd(matrix: &DTensor<f64, 2>) -> RealSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    // Protect FPU state during SVD computation (required for Intel Fortran compatibility)
    let _guard = FpuGuard::new_protect_computation();

    let mut a = matrix.clone();
    let SVDDecomp { u, s, vt } = Faer.svd(&mut *a).expect("SVD computation failed");

    // Extract singular values from first row
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]]).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    RealSVD::new(u_trimmed, s_vec, vt_trimmed)
}

/// Compute SVD of a complex matrix directly
pub(crate) fn compute_complex_svd(matrix: &DTensor<Complex<f64>, 2>) -> ComplexSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    // Protect FPU state during SVD computation (required for Intel Fortran compatibility)
    let _guard = FpuGuard::new_protect_computation();

    // Use matrix directly (Complex<f64> is compatible with faer's c64)
    let mut matrix_c64 = matrix.clone();

    // Compute complex SVD directly
    let SVDDecomp { u, s, vt } = Faer
        .svd(&mut *matrix_c64)
        .expect("Complex SVD computation failed");

    // Extract singular values from first row (they are real even though stored as Complex)
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]].re).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    ComplexSVD::new(u_trimmed, s_vec, vt_trimmed)
}

// ============================================================================
// Complex-Real conversion helpers
// ============================================================================

/// Combine real and imaginary parts into complex tensor
pub(crate) fn combine_complex(
    re: &DTensor<f64, 2>,
    im: &DTensor<f64, 2>,
) -> DTensor<Complex<f64>, 2> {
    let (n_points, extra_size) = *re.shape();
    DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
        Complex::new(re[idx], im[idx])
    })
}

/// Extract real parts from complex tensor (for coefficients)
pub(crate) fn extract_real_parts_coeffs(coeffs_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (basis_size, extra_size) = *coeffs_2d.shape();
    DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].re)
}
