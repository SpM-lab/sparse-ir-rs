//! Fitter for a real matrix: `A ∈ R^{n×m}`.
//!
//! Evaluation computes `y = A x` and fitting the least-squares solution
//! `x = A^+ y` along one axis of a column-major tensor. Real matrices act on
//! complex data component-wise, so both `f64` and `Complex<f64>` tensors are
//! supported through the same real GEMM (a complex `[pre, n, post]` tensor is
//! a real `[2 pre, n, post]` tensor).

use super::common::{
    FitScalar, InplaceFitter, PinvFactors, compute_pinv, condition_number_from_singular_values,
    run_alloc, run_to,
};
use crate::Matrix;
use crate::error::Result;
use crate::gemm::{GemmBackendHandle, apply_along_axis};
use crate::matrix::Mat;
use num_complex::Complex;
use std::sync::OnceLock;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

/// Fitter for real matrix: A ∈ R^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all real
///
/// This type is thread-safe and can be shared across threads.
/// The SVD decomposition is computed lazily on first use.
///
/// # Example
///
/// This type is crate-private. [`TauSampling::from_matrix`](crate::TauSampling::from_matrix)
/// wraps a `RealMatrixFitter` around the given matrix, and its `evaluate` and
/// `fit` forward to the fitter, so the example goes through that public API.
///
/// ```
/// use sparse_ir::matrix::Mat;
/// use sparse_ir::{Fermionic, TauSampling};
/// use std::f64::consts::PI;
///
/// // A 10x5 matrix with orthogonal columns (DCT-II), so the fit is well conditioned
/// let (n, m) = (10, 5);
/// let matrix = Mat::<f64>::from_fn([n, m], |idx| {
///     (PI * (idx[0] as f64 + 0.5) * idx[1] as f64 / n as f64).cos()
/// });
/// // The τ points only label the rows here; the matrix is given explicitly
/// let tau: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
/// let sampling = TauSampling::<Fermionic>::from_matrix(tau, &matrix.into_typed()).unwrap();
///
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = sampling.evaluate(&coeffs).unwrap(); // values = A * coeffs
/// let fitted_coeffs = sampling.fit(&values).unwrap(); // least-squares solution
/// for (c, f) in coeffs.iter().zip(&fitted_coeffs) {
///     assert!((c - f).abs() < 1e-12);
/// }
/// ```
pub(crate) struct RealMatrixFitter {
    matrix: Matrix<f64>,
    /// Column-major copy of `matrix` for GEMM.
    a: Vec<f64>,
    n_points: usize,
    basis_size: usize,
    pinv: OnceLock<Result<PinvFactors<f64>>>,
}

impl RealMatrixFitter {
    /// Create a fitter for `matrix` (`n_points x basis_size`).
    ///
    /// The SVD is computed lazily on the first fit.
    pub fn new(matrix: Mat<f64>) -> Self {
        let (n_points, basis_size) = *matrix.shape();
        let a = matrix.as_slice().to_vec();
        Self {
            matrix: matrix.into_typed(),
            a,
            n_points,
            basis_size,
            pinv: OnceLock::new(),
        }
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// The sampling matrix.
    pub fn matrix(&self) -> &Matrix<f64> {
        &self.matrix
    }

    /// Condition number of the matrix that [`Self::fit`] solves
    ///
    /// Uses the SVD that fitting uses (computed on first use, then cached).
    /// See [`condition_number_from_singular_values`] for edge cases.
    ///
    /// # Errors
    ///
    /// [`Error::DecompositionFailed`] if the SVD fails
    pub fn condition_number(&self) -> Result<f64> {
        Ok(condition_number_from_singular_values(&self.pinv()?.s))
    }

    fn pinv(&self) -> Result<&PinvFactors<f64>> {
        self.pinv
            .get_or_init(|| compute_pinv(&self.a, self.n_points, self.basis_size))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Evaluate along the middle axis of a column-major
    /// `[pre, basis_size, post]` slab into `[pre, n_points, post]`.
    pub fn evaluate_slab<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[T],
        pre: usize,
        post: usize,
        out: &mut [T],
    ) -> Result<()> {
        apply_along_axis(
            backend,
            &self.a,
            self.n_points,
            self.basis_size,
            T::as_f64(coeffs),
            pre * T::REAL_WIDTH,
            post,
            T::as_f64_mut(out),
        )?;
        Ok(())
    }

    /// Fit along the middle axis of a column-major `[pre, n_points, post]`
    /// slab into `[pre, basis_size, post]`.
    pub fn fit_slab<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[T],
        pre: usize,
        post: usize,
        out: &mut [T],
    ) -> Result<()> {
        self.pinv()?.solve(
            backend,
            T::as_f64(values),
            pre * T::REAL_WIDTH,
            post,
            T::as_f64_mut(out),
        )
    }

    /// Evaluate a coefficient vector: `A x`.
    pub fn evaluate<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[T],
    ) -> Result<Vec<T>> {
        let mut out = vec![T::zero(); self.n_points];
        self.evaluate_to(backend, coeffs, &mut out)?;
        Ok(out)
    }

    /// Evaluate a coefficient vector into `out`.
    pub fn evaluate_to<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[T],
        out: &mut [T],
    ) -> Result<()> {
        check_vec_len("coeffs", coeffs.len(), self.basis_size)?;
        check_vec_len("out", out.len(), self.n_points)?;
        self.evaluate_slab(backend, coeffs, 1, 1, out)
    }

    /// Least-squares fit of a value vector: `A^+ y`.
    pub fn fit<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[T],
    ) -> Result<Vec<T>> {
        let mut out = vec![T::zero(); self.basis_size];
        self.fit_to(backend, values, &mut out)?;
        Ok(out)
    }

    /// Least-squares fit of a value vector into `out`.
    pub fn fit_to<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[T],
        out: &mut [T],
    ) -> Result<()> {
        check_vec_len("values", values.len(), self.n_points)?;
        check_vec_len("out", out.len(), self.basis_size)?;
        self.fit_slab(backend, values, 1, 1, out)
    }

    /// Evaluate along axis `dim` of an N-dimensional tensor.
    pub fn evaluate_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>> {
        run_alloc(coeffs, dim, self.basis_size, self.n_points, |x, s, y| {
            self.evaluate_slab(backend, x, s.pre, s.post, y)
        })
    }

    /// Fit along axis `dim` of an N-dimensional tensor.
    pub fn fit_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>> {
        run_alloc(values, dim, self.n_points, self.basis_size, |x, s, y| {
            self.fit_slab(backend, x, s.pre, s.post, y)
        })
    }

    /// Evaluate along axis `dim` into a compact column-major output view.
    pub fn evaluate_nd_to<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, T>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, T>,
    ) -> Result<()> {
        run_to(
            coeffs,
            dim,
            self.basis_size,
            self.n_points,
            out,
            |x, s, y| self.evaluate_slab(backend, x, s.pre, s.post, y),
        )
    }

    /// Fit along axis `dim` into a compact column-major output view.
    pub fn fit_nd_to<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, T>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, T>,
    ) -> Result<()> {
        run_to(
            values,
            dim,
            self.n_points,
            self.basis_size,
            out,
            |x, s, y| self.fit_slab(backend, x, s.pre, s.post, y),
        )
    }
}

/// `Ok` if a slice named `name` has `expected` elements; `name` is `"out"`
/// for the output and anything else for an input.
pub(crate) fn check_vec_len(name: &str, len: usize, expected: usize) -> Result<()> {
    super::common::check_len(super::complex::role_of(name), len, expected)
}

impl InplaceFitter for RealMatrixFitter {
    fn n_points(&self) -> usize {
        self.n_points
    }

    fn basis_size(&self) -> usize {
        self.basis_size
    }

    fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.evaluate_nd_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.evaluate_nd_to(backend, coeffs, dim, out)
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fit_nd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.fit_nd_to(backend, values, dim, out)
    }
}

#[cfg(test)]
#[path = "real_tests.rs"]
mod tests;
