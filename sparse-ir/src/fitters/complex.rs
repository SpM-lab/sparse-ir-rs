//! Fitter for a complex matrix: `A ∈ C^{n×m}`.
//!
//! Complex coefficients map to complex values with a native complex GEMM.
//! Real coefficients (`dz`) use the stacked real matrix `[Re A; Im A]`, which
//! halves the arithmetic compared with promoting the coefficients to complex.
//! Fitting to real coefficients (`zd`) takes the real part of the complex
//! least-squares solution.

use super::common::{
    AxisSplit, InplaceFitter, PinvFactors, compute_pinv, run_alloc, run_to, stacked_to_complex,
};
use super::real::check_vec_len;
use crate::Matrix;
use crate::error::{Error, Result};
use crate::gemm::{GemmBackendHandle, apply_along_axis};
use crate::matrix::Mat;
use num_complex::Complex;
use std::sync::OnceLock;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

type C64 = Complex<f64>;

/// Stack a column-major complex `n x m` matrix as the real `2n x m` matrix
/// `[Re A; Im A]` (column-major).
pub(crate) fn stack_complex_matrix(a: &[C64], n: usize, m: usize) -> Vec<f64> {
    let mut s = vec![0.0; 2 * n * m];
    for j in 0..m {
        for i in 0..n {
            let z = a[i + n * j];
            s[i + 2 * n * j] = z.re;
            s[n + i + 2 * n * j] = z.im;
        }
    }
    s
}

/// Check that a `[pre, rows, post]` slab has `len` elements.
pub(crate) fn check_slab_len(
    name: &str,
    len: usize,
    pre: usize,
    rows: usize,
    post: usize,
) -> Result<()> {
    let expected = pre
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(post))
        .ok_or_else(|| Error::ShapeMismatch(format!("{name}: slab size overflows usize")))?;
    if len != expected {
        return Err(Error::ShapeMismatch(format!(
            "{name} has {len} elements, expected {pre}x{rows}x{post}"
        )));
    }
    Ok(())
}

/// Fitter for a complex `n_points x basis_size` matrix.
pub(crate) struct ComplexMatrixFitter {
    matrix: Matrix<C64>,
    /// Column-major copy of `matrix` for GEMM.
    a: Vec<C64>,
    /// `[Re A; Im A]`, column-major `2 n_points x basis_size`.
    a_stacked: Vec<f64>,
    n_points: usize,
    basis_size: usize,
    pinv: OnceLock<std::result::Result<PinvFactors<C64>, String>>,
}

impl ComplexMatrixFitter {
    /// Create a fitter for `matrix` (`n_points x basis_size`).
    ///
    /// The SVD is computed lazily on the first fit.
    pub fn new(matrix: Mat<C64>) -> Self {
        let (n_points, basis_size) = *matrix.shape();
        let a = matrix.as_slice().to_vec();
        let a_stacked = stack_complex_matrix(&a, n_points, basis_size);
        Self {
            matrix: matrix.into_typed(),
            a,
            a_stacked,
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

    fn pinv(&self) -> Result<&PinvFactors<C64>> {
        self.pinv
            .get_or_init(|| {
                compute_pinv(&self.a, self.n_points, self.basis_size).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| Error::Numerical(e.clone()))
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

    /// Complex values to real coefficients: real part of the complex
    /// least-squares solution.
    pub fn fit_slab_zd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[C64],
        pre: usize,
        post: usize,
        out: &mut [f64],
    ) -> Result<()> {
        check_slab_len("out", out.len(), pre, self.basis_size, post)?;
        let mut tmp = vec![C64::new(0.0, 0.0); out.len()];
        self.fit_slab_zz(backend, values, pre, post, &mut tmp)?;
        for (o, z) in out.iter_mut().zip(&tmp) {
            *o = z.re;
        }
        Ok(())
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

/// `[pre, m, post]` real coefficients to `[pre, n, post]` complex values
/// through the stacked real matrix `[Re A; Im A]` (`2n x m`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_stacked(
    backend: Option<&GemmBackendHandle>,
    a_stacked: &[f64],
    n: usize,
    m: usize,
    coeffs: &[f64],
    pre: usize,
    post: usize,
    out: &mut [C64],
) -> Result<()> {
    check_slab_len("out", out.len(), pre, n, post)?;
    let mut t = vec![0.0; 2 * out.len()];
    apply_along_axis(backend, a_stacked, 2 * n, m, coeffs, pre, post, &mut t)?;
    stacked_to_complex(&t, pre, n, post, out);
    Ok(())
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
