//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::Matrix;
use crate::error::{Error, Result};
use crate::fitters::{InplaceFitter, RealMatrixFitter};
use crate::gemm::GemmBackendHandle;
use crate::matrix::Mat;
use crate::traits::StatisticsType;
use num_complex::Complex;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

/// Copy a public matrix into the internal container.
///
/// Matrices produced by this crate and by `TypedTensor::from_vec_col_major`
/// are host-resident column-major, so this only fails for foreign storage.
pub(crate) fn mat_from_matrix<T: tenferro_tensor::TensorScalar + Copy>(
    m: &Matrix<T>,
) -> Result<Mat<T>> {
    Ok(Mat::from_typed(m)?)
}

/// Sparse sampling in imaginary time
///
/// Allows transformation between the IR basis and a set of sampling points
/// in imaginary time (τ).
///
/// Vector methods take and return slices; N-dimensional methods act on axis
/// `dim` of a column-major [`TypedTensor`]. All transforms return
/// [`Result`]: shape mismatches and layout errors are reported, not panics.
pub struct TauSampling<S>
where
    S: StatisticsType,
{
    /// Sampling points in imaginary time τ ∈ [-β, β]
    sampling_points: Vec<f64>,

    /// Real matrix fitter for least-squares fitting
    fitter: RealMatrixFitter,

    /// Marker for statistics type
    _phantom: std::marker::PhantomData<S>,
}

impl<S> TauSampling<S>
where
    S: StatisticsType,
{
    /// Create a new TauSampling with default sampling points
    ///
    /// The default sampling points are chosen as the extrema of the highest-order
    /// basis function, which gives near-optimal conditioning.
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_tau_sampling_points();
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create a new TauSampling with custom sampling points
    ///
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if any point is outside [-β, β]
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<f64>,
    ) -> Self
    where
        S: 'static,
    {
        assert!(!sampling_points.is_empty(), "No sampling points given");

        let beta = basis.beta();
        for &tau in &sampling_points {
            assert!(
                tau >= -beta && tau <= beta,
                "Sampling point τ={} is outside [-β, β]",
                tau
            );
        }

        let matrix = basis.evaluate_tau(&sampling_points);
        let matrix = mat_from_matrix(&matrix).expect("Basis::evaluate_tau returns a host matrix");
        Self {
            sampling_points,
            fitter: RealMatrixFitter::new(matrix),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new TauSampling with custom sampling points and pre-computed matrix
    ///
    /// # Errors
    /// Returns an error if `sampling_points` is empty, if the matrix row
    /// count does not match the number of points, or if the matrix is not
    /// host-resident.
    pub fn from_matrix(sampling_points: Vec<f64>, matrix: &Matrix<f64>) -> Result<Self> {
        if sampling_points.is_empty() {
            return Err(Error::InvalidArgument("no sampling points given".into()));
        }
        let matrix = mat_from_matrix(matrix)?;
        if matrix.nrows() != sampling_points.len() {
            return Err(Error::ShapeMismatch(format!(
                "matrix rows ({}) must match number of sampling points ({})",
                matrix.nrows(),
                sampling_points.len()
            )));
        }
        Ok(Self {
            sampling_points,
            fitter: RealMatrixFitter::new(matrix),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get sampling points
    pub fn sampling_points(&self) -> &[f64] {
        &self.sampling_points
    }

    /// Get the number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.fitter.n_points()
    }

    /// Get the basis size
    pub fn basis_size(&self) -> usize {
        self.fitter.basis_size()
    }

    /// Get the sampling matrix (`n_sampling_points x basis_size`)
    pub fn matrix(&self) -> &Matrix<f64> {
        self.fitter.matrix()
    }

    /// Evaluate basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[f64]) -> Result<Vec<f64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate basis coefficients at sampling points, writing to `out`
    pub fn evaluate_to(&self, coeffs: &[f64], out: &mut [f64]) -> Result<()> {
        self.fitter.evaluate_to(None, coeffs, out)
    }

    /// Fit values at sampling points to basis coefficients
    pub fn fit(&self, values: &[f64]) -> Result<Vec<f64>> {
        self.fitter.fit(None, values)
    }

    /// Fit values at sampling points to basis coefficients, writing to `out`
    pub fn fit_to(&self, values: &[f64], out: &mut [f64]) -> Result<()> {
        self.fitter.fit_to(None, values, out)
    }

    /// Evaluate complex basis coefficients at sampling points
    pub fn evaluate_zz(&self, coeffs: &[Complex<f64>]) -> Result<Vec<Complex<f64>>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate complex basis coefficients at sampling points, writing to `out`
    pub fn evaluate_zz_to(&self, coeffs: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<()> {
        self.fitter.evaluate_to(None, coeffs, out)
    }

    /// Fit complex values at sampling points to complex basis coefficients
    pub fn fit_zz(&self, values: &[Complex<f64>]) -> Result<Vec<Complex<f64>>> {
        self.fitter.fit(None, values)
    }

    /// Fit complex values to complex basis coefficients, writing to `out`
    pub fn fit_zz_to(&self, values: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<()> {
        self.fitter.fit_to(None, values, out)
    }

    /// Evaluate along axis `dim` of an N-dimensional coefficient tensor
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.evaluate_nd(backend, coeffs, dim)
    }

    /// Evaluate along axis `dim`, writing to a column-major output view
    pub fn evaluate_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_to(backend, coeffs, dim, out)
    }

    /// Fit along axis `dim` of an N-dimensional value tensor
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.fit_nd(backend, values, dim)
    }

    /// Fit along axis `dim`, writing to a column-major output view
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.fit_nd_to(backend, values, dim, out)
    }

    /// Evaluate complex coefficients along axis `dim`
    pub fn evaluate_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<Complex<f64>>,
        dim: usize,
    ) -> Result<TypedTensor<Complex<f64>>> {
        self.fitter.evaluate_nd(backend, coeffs, dim)
    }

    /// Evaluate complex coefficients along axis `dim` into an output view
    pub fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_to(backend, coeffs, dim, out)
    }

    /// Fit complex values along axis `dim`
    pub fn fit_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<Complex<f64>>,
        dim: usize,
    ) -> Result<TypedTensor<Complex<f64>>> {
        self.fitter.fit_nd(backend, values, dim)
    }

    /// Fit complex values along axis `dim` into an output view
    pub fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.fitter.fit_nd_to(backend, values, dim, out)
    }
}

impl<S: StatisticsType> InplaceFitter for TauSampling<S> {
    fn n_points(&self) -> usize {
        self.n_sampling_points()
    }

    fn basis_size(&self) -> usize {
        self.basis_size()
    }

    fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_to(backend, coeffs, dim, out)
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        self.fitter.fit_nd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        self.fitter.fit_nd_to(backend, values, dim, out)
    }
}

#[cfg(test)]
#[path = "tau_sampling_tests.rs"]
mod tests;
