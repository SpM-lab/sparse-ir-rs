//! Fitter for a complex matrix: `A ∈ C^{n×m}`.
//!
//! Complex coefficients map to complex values with a native complex GEMM.
//! Real coefficients (`dz`) use the stacked real matrix `[Re A; Im A]`, which
//! halves the arithmetic compared with promoting the coefficients to complex.
//! Fitting to real coefficients (`zd`) is the least-squares solution over
//! real coefficients, by the pseudo-inverse of the same stacked matrix.

use super::common::{
    AxisSplit, InplaceFitter, PinvFactors, complex_to_stacked, compute_pinv, compute_pinv_of,
    condition_number_from_singular_values, run_alloc, run_to, stacked_to_complex,
};
use super::real::check_vec_len;
use crate::Matrix;
use crate::error::{ArrayRole, Error, Result};
use crate::gemm::{GemmBackendHandle, apply_along_axis};
use crate::matrix::Mat;
use num_complex::Complex;
use std::sync::OnceLock;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

type C64 = Complex<f64>;

/// The real `2n x m` matrix `[Re A; Im A]` of a complex `n x m` matrix `A`,
/// which maps real coefficients to complex values
///
/// Least squares over real coefficients is the real problem
/// `min_x ||[Re A; Im A] x - [Re y; Im y]||`, solved with the pseudo-inverse
/// of the stacked matrix (computed on first use, then cached).
pub(crate) struct StackedMatrix {
    /// Column-major `2n x m`.
    a: Vec<f64>,
    n: usize,
    m: usize,
    pinv: OnceLock<Result<PinvFactors<f64>>>,
}

impl StackedMatrix {
    /// Stack the column-major complex `n x m` matrix `a`.
    pub fn new(a: &[C64], n: usize, m: usize) -> Self {
        let mut s = vec![0.0; 2 * n * m];
        for j in 0..m {
            for i in 0..n {
                let z = a[i + n * j];
                s[i + 2 * n * j] = z.re;
                s[n + i + 2 * n * j] = z.im;
            }
        }
        Self {
            a: s,
            n,
            m,
            pinv: OnceLock::new(),
        }
    }

    /// Pseudo-inverse of the stacked matrix
    pub fn pinv(&self) -> Result<&PinvFactors<f64>> {
        self.pinv
            .get_or_init(|| {
                compute_pinv_of(&self.a, 2 * self.n, self.m, || {
                    format!(
                        "{} x {cols} real matrix stacked from the {} x {cols} \
                         complex sampling matrix",
                        2 * self.n,
                        self.n,
                        cols = self.m,
                    )
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// `[pre, m, post]` real coefficients to `[pre, n, post]` complex values
    pub fn evaluate_slab(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        check_slab_len("out", out.len(), pre, self.n, post)?;
        let mut t = vec![0.0; 2 * out.len()];
        apply_along_axis(
            backend,
            &self.a,
            2 * self.n,
            self.m,
            coeffs,
            pre,
            post,
            &mut t,
        )?;
        stacked_to_complex(&t, pre, self.n, post, out);
        Ok(())
    }

    /// `[pre, n, post]` complex values to the `[pre, m, post]` real
    /// coefficients that fit them in the least-squares sense
    pub fn fit_slab(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [f64],
    ) -> Result<()> {
        check_slab_len("values", values.len(), pre, self.n, post)?;
        check_slab_len("out", out.len(), pre, self.m, post)?;
        let stacked = complex_to_stacked(values, pre, self.n, post);
        self.pinv()?.solve(backend, &stacked, pre, post, out)
    }
}

/// Check that a `[pre, rows, post]` slab has `len` elements; `name` is
/// `"out"` for the output and anything else for an input.
pub(crate) fn check_slab_len(
    name: &str,
    len: usize,
    pre: usize,
    rows: usize,
    post: usize,
) -> Result<()> {
    let fits = pre
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(post))
        .is_some_and(|expected| expected == len);
    if !fits {
        return Err(Error::ShapeMismatch {
            which: role_of(name),
            expected: vec![pre, rows, post],
            actual: vec![len],
        });
    }
    Ok(())
}

/// Which array a length check refers to, from its name
pub(crate) fn role_of(name: &str) -> ArrayRole {
    if name == "out" {
        ArrayRole::Output
    } else {
        ArrayRole::Input
    }
}

/// Fitter for complex matrix with complex coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all complex
///
/// # Example
///
/// This type is crate-private.
/// [`MatsubaraSampling::from_matrix`](crate::MatsubaraSampling::from_matrix)
/// wraps a `ComplexMatrixFitter` around the given matrix, and its `evaluate`
/// and `fit` forward to the fitter, so the example goes through that public API.
///
/// ```
/// use num_complex::Complex;
/// use sparse_ir::matrix::Mat;
/// use sparse_ir::{Fermionic, FermionicFreq, MatsubaraSampling};
/// use std::f64::consts::PI;
///
/// // A 10x5 matrix with orthonormal columns (DFT), so the fit is well conditioned
/// let (n, m) = (10, 5);
/// let matrix = Mat::<Complex<f64>>::from_fn([n, m], |idx| {
///     let phase = 2.0 * PI * (idx[0] * idx[1]) as f64 / n as f64;
///     Complex::from_polar(1.0 / (n as f64).sqrt(), phase)
/// });
/// // The frequencies only label the rows here; the matrix is given explicitly
/// let freqs = (0..n as i64).map(|i| FermionicFreq::new(2 * i - 9).unwrap()).collect();
/// let sampling = MatsubaraSampling::<Fermionic>::from_matrix(freqs, &matrix.into_typed()).unwrap();
///
/// let coeffs: Vec<Complex<f64>> = (0..m)
///     .map(|l| Complex::new(1.0 + l as f64, -0.5 * l as f64))
///     .collect();
/// let values = sampling.evaluate(&coeffs).unwrap(); // → Vec<Complex<f64>>
/// let fitted_coeffs = sampling.fit(&values).unwrap(); // ← Vec<Complex<f64>>, → Vec<Complex<f64>>
/// for (c, f) in coeffs.iter().zip(&fitted_coeffs) {
///     assert!((c - f).norm() < 1e-12);
/// }
/// ```
pub(crate) struct ComplexMatrixFitter {
    matrix: Matrix<C64>,
    /// Column-major copy of `matrix` for GEMM.
    a: Vec<C64>,
    /// `[Re A; Im A]`, for real coefficients
    stacked: StackedMatrix,
    n_points: usize,
    basis_size: usize,
    pinv: OnceLock<Result<PinvFactors<C64>>>,
}

impl ComplexMatrixFitter {
    /// Create a fitter for `matrix` (`n_points x basis_size`).
    ///
    /// The SVD is computed lazily on the first fit.
    pub fn new(matrix: Mat<C64>) -> Self {
        let (n_points, basis_size) = *matrix.shape();
        let a = matrix.as_slice().to_vec();
        let stacked = StackedMatrix::new(&a, n_points, basis_size);
        Self {
            matrix: matrix.into_typed(),
            a,
            stacked,
            n_points,
            basis_size,
            pinv: OnceLock::new(),
        }
    }

    pub fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// The sampling matrix.
    pub fn matrix(&self) -> &Matrix<C64> {
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

    fn pinv(&self) -> Result<&PinvFactors<C64>> {
        self.pinv
            .get_or_init(|| compute_pinv(&self.a, self.n_points, self.basis_size))
            .as_ref()
            .map_err(Clone::clone)
    }

    // ------------------------------------------------------------------
    // Slab kernels on column-major [pre, *, post] buffers
    // ------------------------------------------------------------------

    /// Complex coefficients to complex values.
    pub fn evaluate_slab_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[C64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        apply_along_axis(
            backend,
            &self.a,
            self.n_points,
            self.basis_size,
            coeffs,
            pre,
            post,
            out,
        )?;
        Ok(())
    }

    /// Real coefficients to complex values.
    pub fn evaluate_slab_dz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        self.stacked.evaluate_slab(backend, coeffs, pre, post, out)
    }

    /// Complex values to complex coefficients (least squares).
    pub fn fit_slab_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [C64],
    ) -> Result<()> {
        self.pinv()?.solve(backend, values, pre, post, out)
    }

    /// Complex values to real coefficients: the least-squares solution over
    /// real coefficients, `[Re A; Im A]⁺ [Re y; Im y]`
    ///
    /// For data that real coefficients reproduce exactly, this is the real
    /// part of the complex solution. Otherwise it differs: that real part is
    /// a left inverse of the stacked matrix too, and the pseudo-inverse is
    /// the one of least norm, so this fit amplifies noise less. It is one
    /// real GEMM over the `2n` stacked values.
    pub fn fit_slab_zd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [f64],
    ) -> Result<()> {
        self.stacked.fit_slab(backend, values, pre, post, out)
    }

    // ------------------------------------------------------------------
    // Vector API
    // ------------------------------------------------------------------

    /// Evaluate complex coefficients: `A x`.
    pub fn evaluate(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[C64],
    ) -> Result<Vec<C64>> {
        check_vec_len("coeffs", coeffs.len(), self.basis_size)?;
        let mut out = vec![C64::new(0.0, 0.0); self.n_points];
        self.evaluate_slab_zz(backend, coeffs, 1, 1, &mut out)?;
        Ok(out)
    }

    /// Evaluate real coefficients: `A x`.
    pub fn evaluate_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
    ) -> Result<Vec<C64>> {
        check_vec_len("coeffs", coeffs.len(), self.basis_size)?;
        let mut out = vec![C64::new(0.0, 0.0); self.n_points];
        self.evaluate_slab_dz(backend, coeffs, 1, 1, &mut out)?;
        Ok(out)
    }

    /// Least-squares fit to complex coefficients.
    pub fn fit(&self, backend: Option<&GemmBackendHandle>, values: &[C64]) -> Result<Vec<C64>> {
        check_vec_len("values", values.len(), self.n_points)?;
        let mut out = vec![C64::new(0.0, 0.0); self.basis_size];
        self.fit_slab_zz(backend, values, 1, 1, &mut out)?;
        Ok(out)
    }

    /// Least-squares fit to real coefficients.
    pub fn fit_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
    ) -> Result<Vec<f64>> {
        check_vec_len("values", values.len(), self.n_points)?;
        let mut out = vec![0.0; self.basis_size];
        self.fit_slab_zd(backend, values, 1, 1, &mut out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------
    // N-dimensional API (allocating)
    // ------------------------------------------------------------------

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
}

impl InplaceFitter for ComplexMatrixFitter {
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
            |x, s: AxisSplit, y| self.evaluate_slab_dz(backend, x, s.pre, s.post, y),
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
#[path = "complex_tests.rs"]
mod tests;
