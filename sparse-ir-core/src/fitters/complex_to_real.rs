//! Fitter for a complex matrix with real coefficients: `A ∈ C^{n×m}`,
//! `x ∈ R^m`.
//!
//! Used for positive-only Matsubara sampling. The least-squares problem
//! `min_x ||A x - y||` over real `x` is the real problem
//! `min_x ||[Re A; Im A] x - [Re y; Im y]||`, which is solved with a real SVD
//! of the stacked `2n x m` matrix.

use super::common::{
    InplaceFitter, PinvFactors, complex_to_stacked, compute_pinv_of,
    condition_number_from_singular_values, run_alloc, run_to,
};
use super::complex::{check_slab_len, evaluate_stacked, stack_complex_matrix};
use super::real::check_vec_len;
use crate::Matrix;
use crate::error::Result;
use crate::gemm::GemmBackendHandle;
use crate::matrix::Mat;
use num_complex::Complex;
use std::sync::OnceLock;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

type C64 = Complex<f64>;

/// Fitter for complex matrix with real coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A is complex, coeffs are real, values are complex
///
/// Strategy: Flatten complex to real problem
///   A_real ∈ R^{2n×m}: [Re(A[0,:]); Im(A[0,:]); Re(A[1,:]); Im(A[1,:]); ...]
///   values_flat ∈ R^{2n}: [Re(v[0]); Im(v[0]); Re(v[1]); Im(v[1]); ...]
///
/// # Example
///
/// This type is crate-private.
/// [`MatsubaraSamplingPositiveOnly::from_matrix`](crate::MatsubaraSamplingPositiveOnly::from_matrix)
/// wraps a `ComplexToRealFitter` around the given matrix, and its `evaluate`
/// and `fit` forward to the fitter, so the example goes through that public API.
///
/// ```
/// use num_complex::Complex;
/// use sparse_ir::matrix::Mat;
/// use sparse_ir::{Fermionic, FermionicFreq, MatsubaraSamplingPositiveOnly};
/// use std::f64::consts::PI;
///
/// // A 10x5 matrix whose flattened real form has orthonormal columns
/// let (n, m) = (10, 5);
/// let matrix = Mat::<Complex<f64>>::from_fn([n, m], |idx| {
///     let phase = 2.0 * PI * (idx[0] * idx[1]) as f64 / n as f64;
///     Complex::from_polar(1.0 / (n as f64).sqrt(), phase)
/// });
/// // The frequencies only label the rows here; the matrix is given explicitly
/// let freqs = (0..n as i64).map(|i| FermionicFreq::new(2 * i + 1).unwrap()).collect();
/// let sampling = MatsubaraSamplingPositiveOnly::<Fermionic>::from_matrix(freqs, &matrix.into_typed()).unwrap();
///
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = sampling.evaluate(&coeffs).unwrap(); // → Vec<Complex<f64>>
/// let fitted_coeffs = sampling.fit(&values).unwrap(); // ← Vec<Complex<f64>>, → Vec<f64>
/// for (c, f) in coeffs.iter().zip(&fitted_coeffs) {
///     assert!((c - f).abs() < 1e-12);
/// }
/// ```
pub(crate) struct ComplexToRealFitter {
    matrix: Matrix<C64>,
    /// `[Re A; Im A]`, column-major `2 n_points x basis_size`.
    a_stacked: Vec<f64>,
    n_points: usize,
    basis_size: usize,
    pinv: OnceLock<Result<PinvFactors<f64>>>,
}

impl ComplexToRealFitter {
    /// Create a fitter for `matrix` (`n_points x basis_size`).
    ///
    /// The SVD is computed lazily on the first fit.
    pub fn new(matrix: Mat<C64>) -> Self {
        let (n_points, basis_size) = *matrix.shape();
        let a_stacked = stack_complex_matrix(matrix.as_slice(), n_points, basis_size);
        Self {
            matrix: matrix.into_typed(),
            a_stacked,
            n_points,
            basis_size,
            pinv: OnceLock::new(),
        }
    }

    pub fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// The complex sampling matrix.
    pub fn matrix(&self) -> &Matrix<C64> {
        &self.matrix
    }

    /// Condition number of the matrix that [`Self::fit`] solves
    ///
    /// Computed from the singular values of the real 2n × m matrix stacked
    /// from `[Re A; Im A]`, not of the complex n × m `matrix`.
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
            .get_or_init(|| {
                compute_pinv_of(&self.a_stacked, 2 * self.n_points, self.basis_size, || {
                    format!(
                        "{} x {cols} real matrix stacked from the {} x {cols} \
                         complex sampling matrix",
                        2 * self.n_points,
                        self.n_points,
                        cols = self.basis_size,
                    )
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    // ------------------------------------------------------------------
    // Slab kernels on column-major [pre, *, post] buffers
    // ------------------------------------------------------------------

    /// Real coefficients to complex values.
    pub fn evaluate_slab_dz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        evaluate_stacked(
            backend,
            &self.a_stacked,
            self.n_points,
            self.basis_size,
            coeffs,
            pre,
            post,
            out,
        )
    }

    /// Complex coefficients to complex values.
    ///
    /// Coefficients of this fitter are real by construction, so only the
    /// real parts are used (matching the C API contract).
    pub fn evaluate_slab_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[C64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        let re: Vec<f64> = coeffs.iter().map(|z| z.re).collect();
        self.evaluate_slab_dz(backend, &re, pre, post, out)
    }

    /// Complex values to real coefficients (least squares).
    pub fn fit_slab_zd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [f64],
    ) -> Result<()> {
        check_slab_len("values", values.len(), pre, self.n_points, post)?;
        let stacked = complex_to_stacked(values, pre, self.n_points, post);
        self.pinv()?.solve(backend, &stacked, pre, post, out)
    }

    /// Complex values to complex coefficients with zero imaginary parts.
    pub fn fit_slab_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        check_slab_len("out", out.len(), pre, self.basis_size, post)?;
        let mut re = vec![0.0; out.len()];
        self.fit_slab_zd(backend, values, pre, post, &mut re)?;
        for (o, r) in out.iter_mut().zip(re) {
            *o = C64::new(r, 0.0);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Vector API
    // ------------------------------------------------------------------

    /// Evaluate real coefficients: `A x`.
    pub fn evaluate(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
    ) -> Result<Vec<C64>> {
        check_vec_len("coeffs", coeffs.len(), self.basis_size)?;
        let mut out = vec![C64::new(0.0, 0.0); self.n_points];
        self.evaluate_slab_dz(backend, coeffs, 1, 1, &mut out)?;
        Ok(out)
    }

    /// Least-squares fit to real coefficients.
    pub fn fit(&self, backend: Option<&GemmBackendHandle>, values: &[C64]) -> Result<Vec<f64>> {
        check_vec_len("values", values.len(), self.n_points)?;
        let mut out = vec![0.0; self.basis_size];
        self.fit_slab_zd(backend, values, 1, 1, &mut out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------
    // N-dimensional API (allocating)
    // ------------------------------------------------------------------

    pub fn evaluate_nd_dz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        run_alloc(coeffs, dim, self.basis_size, self.n_points, |x, s, y| {
            self.evaluate_slab_dz(backend, x, s.pre, s.post, y)
        })
    }

    pub fn fit_nd_zd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        run_alloc(values, dim, self.n_points, self.basis_size, |x, s, y| {
            self.fit_slab_zd(backend, x, s.pre, s.post, y)
        })
    }

    #[cfg(test)]
    pub fn evaluate_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        run_alloc(coeffs, dim, self.basis_size, self.n_points, |x, s, y| {
            self.evaluate_slab_zz(backend, x, s.pre, s.post, y)
        })
    }

    #[cfg(test)]
    pub fn fit_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        run_alloc(values, dim, self.n_points, self.basis_size, |x, s, y| {
            self.fit_slab_zz(backend, x, s.pre, s.post, y)
        })
    }
}

impl InplaceFitter for ComplexToRealFitter {
    fn n_points(&self) -> usize {
        self.n_points
    }

    fn basis_size(&self) -> usize {
        self.basis_size
    }

    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        run_to(
            coeffs,
            dim,
            self.basis_size,
            self.n_points,
            out,
            |x, s, y| self.evaluate_slab_dz(backend, x, s.pre, s.post, y),
        )
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        run_to(
            coeffs,
            dim,
            self.basis_size,
            self.n_points,
            out,
            |x, s, y| self.evaluate_slab_zz(backend, x, s.pre, s.post, y),
        )
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        run_to(
            values,
            dim,
            self.n_points,
            self.basis_size,
            out,
            |x, s, y| self.fit_slab_zd(backend, x, s.pre, s.post, y),
        )
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        run_to(
            values,
            dim,
            self.n_points,
            self.basis_size,
            out,
            |x, s, y| self.fit_slab_zz(backend, x, s.pre, s.post, y),
        )
    }
}

#[cfg(test)]
#[path = "complex_to_real_tests.rs"]
mod tests;
