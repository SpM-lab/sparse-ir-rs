//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::Matrix;
use crate::error::{Error, Result};
use crate::fitters::{ComplexMatrixFitter, ComplexToRealFitter, InplaceFitter};
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::sampling::mat_from_matrix;
use crate::traits::StatisticsType;
use num_complex::Complex;
use std::marker::PhantomData;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

type C64 = Complex<f64>;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Coefficient types that Matsubara sampling can evaluate (`f64` or
/// `Complex<f64>`).
///
/// This provides compile-time dispatch between the real-coefficient and
/// complex-coefficient kernels.
pub trait MatsubaraCoeffs: tenferro_tensor::TensorScalar + Copy + sealed::Sealed {
    /// Evaluate along axis `dim` into a new tensor.
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<Self>,
        dim: usize,
    ) -> Result<TypedTensor<C64>>;

    /// Evaluate along axis `dim` into an output view.
    fn evaluate_nd_to_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Self>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()>;
}

impl MatsubaraCoeffs for f64 {
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<Self>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        sampler.fitter.evaluate_nd_dz(backend, coeffs, dim)
    }

    fn evaluate_nd_to_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Self>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        sampler.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }
}

impl MatsubaraCoeffs for C64 {
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<Self>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        sampler.fitter.evaluate_nd_zz(backend, coeffs, dim)
    }

    fn evaluate_nd_to_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Self>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        sampler.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }
}

fn check_points<S: StatisticsType>(
    sampling_points: &[MatsubaraFreq<S>],
    matrix_rows: usize,
) -> Result<()> {
    if sampling_points.is_empty() {
        return Err(Error::InvalidArgument("no sampling points given".into()));
    }
    if matrix_rows != sampling_points.len() {
        return Err(Error::ShapeMismatch(format!(
            "matrix rows ({matrix_rows}) must match number of sampling points ({})",
            sampling_points.len()
        )));
    }
    if !sampling_points.windows(2).all(|w| w[0] <= w[1]) {
        return Err(Error::InvalidArgument(
            "sampling points must be sorted in ascending order".into(),
        ));
    }
    Ok(())
}

/// Matsubara sampling for full frequency range (positive and negative)
///
/// General complex problem without symmetry assumptions.
/// Supports both real and complex coefficients.
pub struct MatsubaraSampling<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexMatrixFitter,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> MatsubaraSampling<S> {
    /// Create Matsubara sampling with default sampling points
    ///
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(false);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom sampling points (sorted here)
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        sampling_points.sort();
        let matrix = basis.evaluate_matsubara(&sampling_points);
        let matrix =
            mat_from_matrix(&matrix).expect("Basis::evaluate_matsubara returns a host matrix");
        Self {
            sampling_points,
            fitter: ComplexMatrixFitter::new(matrix),
            _phantom: PhantomData,
        }
    }

    /// Create Matsubara sampling with sorted sampling points and a
    /// pre-computed matrix (`n_points x basis_size`).
    ///
    /// # Errors
    /// Returns an error if `sampling_points` is empty or unsorted, or if the
    /// matrix row count does not match.
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: &Matrix<C64>,
    ) -> Result<Self> {
        let matrix = mat_from_matrix(matrix)?;
        check_points(&sampling_points, matrix.nrows())?;
        Ok(Self {
            sampling_points,
            fitter: ComplexMatrixFitter::new(matrix),
            _phantom: PhantomData,
        })
    }

    /// Get sampling points
    pub fn sampling_points(&self) -> &[MatsubaraFreq<S>] {
        &self.sampling_points
    }

    /// Number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.sampling_points.len()
    }

    /// Basis size
    pub fn basis_size(&self) -> usize {
        self.fitter.basis_size()
    }

    /// Get the sampling matrix
    pub fn matrix(&self) -> &Matrix<C64> {
        self.fitter.matrix()
    }

    /// Evaluate complex basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[C64]) -> Result<Vec<C64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate real basis coefficients at sampling points
    pub fn evaluate_real(&self, coeffs: &[f64]) -> Result<Vec<C64>> {
        self.fitter.evaluate_real(None, coeffs)
    }

    /// Fit complex basis coefficients from values at sampling points
    pub fn fit(&self, values: &[C64]) -> Result<Vec<C64>> {
        self.fitter.fit(None, values)
    }

    /// Fit real basis coefficients (real part of the complex solution)
    pub fn fit_real(&self, values: &[C64]) -> Result<Vec<f64>> {
        self.fitter.fit_real(None, values)
    }

    /// Evaluate real or complex coefficients along axis `dim`
    pub fn evaluate_nd<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        T::evaluate_nd_with(self, backend, coeffs, dim)
    }

    /// Evaluate real coefficients along axis `dim`
    pub fn evaluate_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.evaluate_nd_dz(backend, coeffs, dim)
    }

    /// Fit complex values to complex coefficients along axis `dim`
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.fit_nd_zz(backend, values, dim)
    }

    /// Fit complex values to real coefficients along axis `dim`
    pub fn fit_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.fit_nd_zd(backend, values, dim)
    }

    /// Evaluate along axis `dim` into a column-major output view
    pub fn evaluate_nd_to<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, T>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        T::evaluate_nd_to_with(self, backend, coeffs, dim, out)
    }

    /// Fit complex values to complex coefficients into an output view
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

impl<S: StatisticsType> InplaceFitter for MatsubaraSampling<S> {
    fn n_points(&self) -> usize {
        self.n_sampling_points()
    }

    fn basis_size(&self) -> usize {
        self.basis_size()
    }

    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

/// Matsubara sampling for positive frequencies only
///
/// Exploits the symmetry `G(-iωn) = conj(G(iωn))` of physical Green's
/// functions to fit real coefficients from values at non-negative
/// frequencies. Supports: {0, 1, 2, 3, ...} (no negative frequencies)
pub struct MatsubaraSamplingPositiveOnly<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexToRealFitter,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> MatsubaraSamplingPositiveOnly<S> {
    /// Create Matsubara sampling with default positive-only sampling points
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(true);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom positive-only sampling points
    ///
    /// # Panics
    /// Panics if any sampling point is negative.
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        sampling_points.sort();
        assert!(
            sampling_points.iter().all(|f| f.n() >= 0),
            "All sampling points must be non-negative for positive-only Matsubara sampling"
        );
        let matrix = basis.evaluate_matsubara(&sampling_points);
        let matrix =
            mat_from_matrix(&matrix).expect("Basis::evaluate_matsubara returns a host matrix");
        Self {
            sampling_points,
            fitter: ComplexToRealFitter::new(matrix),
            _phantom: PhantomData,
        }
    }

    /// Create positive-only Matsubara sampling with sorted sampling points
    /// and a pre-computed matrix (`n_points x basis_size`).
    ///
    /// # Errors
    /// Returns an error if `sampling_points` is empty or unsorted, or if the
    /// matrix row count does not match.
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: &Matrix<C64>,
    ) -> Result<Self> {
        let matrix = mat_from_matrix(matrix)?;
        check_points(&sampling_points, matrix.nrows())?;
        Ok(Self {
            sampling_points,
            fitter: ComplexToRealFitter::new(matrix),
            _phantom: PhantomData,
        })
    }

    /// Get sampling points
    pub fn sampling_points(&self) -> &[MatsubaraFreq<S>] {
        &self.sampling_points
    }

    /// Number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.sampling_points.len()
    }

    /// Basis size
    pub fn basis_size(&self) -> usize {
        self.fitter.basis_size()
    }

    /// Get the original complex sampling matrix
    pub fn matrix(&self) -> &Matrix<C64> {
        self.fitter.matrix()
    }

    /// Evaluate basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[f64]) -> Result<Vec<C64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Fit basis coefficients from values at sampling points
    pub fn fit(&self, values: &[C64]) -> Result<Vec<f64>> {
        self.fitter.fit(None, values)
    }

    /// Evaluate real coefficients along axis `dim`
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.evaluate_nd_dz(backend, coeffs, dim)
    }

    /// Fit complex values to real coefficients along axis `dim`
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.fit_nd_zd(backend, values, dim)
    }

    /// Evaluate real coefficients along axis `dim` into an output view
    pub fn evaluate_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    /// Fit complex values to real coefficients into an output view
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zd_to(backend, values, dim, out)
    }
}

impl<S: StatisticsType> InplaceFitter for MatsubaraSamplingPositiveOnly<S> {
    fn n_points(&self) -> usize {
        self.n_sampling_points()
    }

    fn basis_size(&self) -> usize {
        self.basis_size()
    }

    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, C64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

#[cfg(test)]
#[path = "matsubara_sampling_tests.rs"]
mod tests;
