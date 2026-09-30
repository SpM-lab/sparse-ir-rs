//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::Matrix;
use crate::error::{Error, Result};
use crate::fitters::{ComplexMatrixFitter, ComplexToRealFitter, InplaceFitter};
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::sampling::{check_finite_matrix, check_sampling_matrix_shape, mat_from_matrix};
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
    /// Evaluate coefficients using the given sampler
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`
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

/// `Ok` if no point is negative, as positive-only samplings require (#247)
///
/// # Errors
///
/// [`Error::InvalidMatsubaraIndex`] for the first negative point
fn check_non_negative<S: StatisticsType>(points: &[MatsubaraFreq<S>]) -> Result<(), Error> {
    match points.iter().find(|f| f.n() < 0) {
        Some(freq) => Err(Error::InvalidMatsubaraIndex {
            n: freq.n(),
            statistics: S::STATISTICS,
        }),
        None => Ok(()),
    }
}

/// Some part of `z` that is not finite: the real part if it is not, else
/// the imaginary part
fn non_finite_part(z: Complex<f64>) -> Option<f64> {
    if !z.re.is_finite() {
        Some(z.re)
    } else if !z.im.is_finite() {
        Some(z.im)
    } else {
        None
    }
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
    /// Uses the default sampling points of the basis (symmetric: positive and
    /// negative frequencies).
    ///
    /// # Errors
    ///
    /// The errors of [`Basis::default_matsubara_sampling_points`](crate::basis_trait::Basis::default_matsubara_sampling_points)
    /// (NotSupported for a DLR or for basis functions without a definite
    /// parity, #183)
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Result<Self>
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(false)?;
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom sampling points
    ///
    /// The points may be in any order, and are kept in the given order:
    /// [`Self::sampling_points`] returns them unchanged, and index i along the
    /// sampling-point axis of `evaluate` and `fit` refers to
    /// `sampling_points[i]`.
    ///
    /// Duplicate points are accepted; they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty
    /// * The errors of [`Basis::evaluate_matsubara`](crate::basis_trait::Basis::evaluate_matsubara)
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Result<Self>
    where
        S: 'static,
    {
        if sampling_points.is_empty() {
            return Err(Error::EmptyInput {
                name: "sampling_points",
            });
        }
        let matrix = mat_from_matrix(&basis.evaluate_matsubara(&sampling_points)?)?;
        Ok(Self {
            sampling_points,
            fitter: ComplexMatrixFitter::new(matrix),
            _phantom: PhantomData,
        })
    }

    /// Create Matsubara sampling with custom sampling points and pre-computed matrix
    ///
    /// This constructor is useful when the sampling matrix is already computed
    /// (e.g., from external sources or for testing).
    ///
    /// # Arguments
    /// * `sampling_points` - Matsubara frequency sampling points, in any order
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size); row i
    ///   belongs to `sampling_points[i]`
    ///
    /// The points are kept in the given order: [`Self::sampling_points`]
    /// returns them unchanged, and index i along the sampling-point axis of
    /// `evaluate` and `fit` refers to `sampling_points[i]`.
    ///
    /// Duplicate points are accepted; they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty, or `matrix`
    ///   has no columns
    /// * [`Error::ShapeMismatch`] of the input if `matrix` does not have one
    ///   row per point
    /// * [`Error::NonFiniteInput`] for the first entry of `matrix` with a NaN
    ///   or infinite part
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: &Matrix<C64>,
    ) -> Result<Self> {
        let matrix = mat_from_matrix(matrix)?;
        check_sampling_matrix_shape(sampling_points.len(), *matrix.shape())?;
        check_finite_matrix(&matrix, non_finite_part)?;
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

    /// Condition number of the sampling matrix, which fitting solves with
    ///
    /// Returns `σ_max / σ_min`, the ratio of the largest to the smallest of the
    /// `min(n_sampling_points, basis_size)` singular values of the complex
    /// `n_sampling_points × basis_size` matrix [`Self::matrix`]. It bounds how
    /// much [`Self::fit`] can amplify relative errors in the values.
    ///
    /// Returns `f64::INFINITY` if the smallest singular value is below `1e-15`
    /// (numerically singular matrix). The singular value decomposition is the
    /// one fitting uses: it is computed by the first call to this method or to
    /// a fit, then cached.
    ///
    /// # Errors
    ///
    /// [`Error::DecompositionFailed`] if the singular value decomposition
    /// fails, which a matrix of finite entries does not cause in practice
    /// (the constructors reject non-finite entries)
    pub fn condition_number(&self) -> Result<f64> {
        self.fitter.condition_number()
    }

    /// Evaluate complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Complex basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at Matsubara frequencies (length = n_sampling_points)
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    pub fn evaluate(&self, coeffs: &[C64]) -> Result<Vec<C64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate real basis coefficients at sampling points
    pub fn evaluate_real(&self, coeffs: &[f64]) -> Result<Vec<C64>> {
        self.fitter.evaluate_real(None, coeffs)
    }

    /// Fit complex basis coefficients from values at sampling points
    ///
    /// # Arguments
    /// * `values` - Complex values at Matsubara frequencies (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted complex basis coefficients (length = basis_size)
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit(&self, values: &[C64]) -> Result<Vec<C64>> {
        self.fitter.fit(None, values)
    }

    /// Fit real basis coefficients (real part of the complex solution)
    pub fn fit_real(&self, values: &[C64]) -> Result<Vec<f64>> {
        self.fitter.fit_real(None, values)
    }

    /// Evaluate N-dimensional coefficients at Matsubara sampling points
    ///
    /// Supports both real (`f64`) and complex (`Complex<f64>`) coefficients and
    /// always returns complex values at the Matsubara frequencies. The
    /// implementation is selected at compile time through the `MatsubaraCoeffs`
    /// trait.
    ///
    /// # Type Parameter
    /// * `T` - Must implement `MatsubaraCoeffs` (currently `f64` or `Complex<f64>`)
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (`None` uses the global dispatcher)
    /// * `coeffs` - N-dimensional tensor of basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies, with
    /// dimension `dim` of size n_sampling_points
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`
    ///
    /// # Example
    /// ```
    /// use num_complex::Complex;
    /// use sparse_ir::{FermionicBasis, LogisticKernel, MatsubaraSampling, TypedTensor};
    ///
    /// let beta = 10.0;
    /// let wmax = 1.0;
    /// let basis = FermionicBasis::new(LogisticKernel::new(beta * wmax).unwrap(), beta, Some(1e-6), None).unwrap();
    /// let sampling = MatsubaraSampling::new(&basis).unwrap();
    /// let (size, n_points) = (sampling.basis_size(), sampling.n_sampling_points());
    ///
    /// // Real coefficients: two sets stacked along axis 1 (column-major), evaluated along axis 0
    /// let real_data: Vec<f64> = (0..2 * size)
    ///     .map(|k| 1.0 / (1.0 + (k % size + k / size) as f64))
    ///     .collect();
    /// let coeffs_real = TypedTensor::from_vec_col_major(vec![size, 2], real_data.clone()).unwrap();
    /// let values = sampling.evaluate_nd::<f64>(None, &coeffs_real, 0).unwrap();
    /// assert_eq!(values.shape(), &[n_points, 2]);
    ///
    /// // Complex coefficients
    /// let complex_data: Vec<Complex<f64>> =
    ///     real_data.iter().map(|&x| Complex::new(x, -0.5 * x)).collect();
    /// let coeffs_complex = TypedTensor::from_vec_col_major(vec![size, 2], complex_data.clone()).unwrap();
    /// let values_z = sampling.evaluate_nd::<Complex<f64>>(None, &coeffs_complex, 0).unwrap();
    ///
    /// // Each column matches the 1-D `evaluate` of the corresponding coefficient set
    /// let (values, values_z) = (values.host_data().unwrap(), values_z.host_data().unwrap());
    /// for j in 0..2 {
    ///     let real: Vec<Complex<f64>> = real_data[j * size..(j + 1) * size].iter().map(|&x| x.into()).collect();
    ///     let complex = &complex_data[j * size..(j + 1) * size];
    ///     let (expected, expected_z) = (sampling.evaluate(&real).unwrap(), sampling.evaluate(complex).unwrap());
    ///     for i in 0..n_points {
    ///         assert!((values[i + n_points * j] - expected[i]).norm() < 1e-12);
    ///         assert!((values_z[i + n_points * j] - expected_z[i]).norm() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn evaluate_nd<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        T::evaluate_nd_with(self, backend, coeffs, dim)
    }

    /// Evaluate real basis coefficients at Matsubara sampling points (N-dimensional)
    ///
    /// This method takes real coefficients and produces complex values, useful when
    /// working with symmetry-exploiting representations or real-valued IR coefficients.
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (None uses default)
    /// * `coeffs` - N-dimensional tensor of real basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`
    pub fn evaluate_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.evaluate_nd_dz(backend, coeffs, dim)
    }

    /// Fit N-dimensional array of complex values to complex basis coefficients
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (None uses default)
    /// * `values` - N-dimensional tensor of complex values at Matsubara frequencies
    /// * `dim` - Dimension along which to fit (must have size = n_sampling_points)
    ///
    /// # Returns
    /// N-dimensional tensor of complex basis coefficients
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.fit_nd_zz(backend, values, dim)
    }

    /// Fit N-dimensional array of complex values to real basis coefficients
    ///
    /// This method fits complex Matsubara values to real IR coefficients.
    /// Takes the real part of the least-squares solution.
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (None uses default)
    /// * `values` - N-dimensional tensor of complex values at Matsubara frequencies
    /// * `dim` - Dimension along which to fit (must have size = n_sampling_points)
    ///
    /// # Returns
    /// N-dimensional tensor of real basis coefficients
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.fit_nd_zd(backend, values, dim)
    }

    /// Evaluate basis coefficients at Matsubara sampling points (N-dimensional) with in-place output
    ///
    /// # Type Parameters
    /// * `T` - Coefficient type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points` (Complex<f64>)
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`, and of the output if `out` does not have
    ///   the shape of `coeffs` with `n_sampling_points` along `dim`
    ///
    /// Nothing is written to `out` then.
    pub fn evaluate_nd_to<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, T>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        T::evaluate_nd_to_with(self, backend, coeffs, dim, out)
    }

    /// Fit N-dimensional complex values to complex coefficients with in-place output
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == basis_size` (Complex<f64>)
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`, and of the output if `out` does not have
    ///   the shape of `values` with `basis_size` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    ///
    /// Nothing is written to `out` then.
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
    ///
    /// Uses the default sampling points of the basis (non-negative frequencies only).
    /// Exploits symmetry to reconstruct real coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`Basis::default_matsubara_sampling_points`](crate::basis_trait::Basis::default_matsubara_sampling_points)
    /// (NotSupported for a DLR or for basis functions without a definite
    /// parity, #183)
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Result<Self>
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(true)?;
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom positive-only sampling points
    ///
    /// The points may be in any order, and are kept in the given order:
    /// [`Self::sampling_points`] returns them unchanged, and index i along the
    /// sampling-point axis of `evaluate` and `fit` refers to
    /// `sampling_points[i]`.
    ///
    /// Duplicate points are accepted; they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty
    /// * [`Error::InvalidMatsubaraIndex`] if a point is negative
    /// * The errors of [`Basis::evaluate_matsubara`](crate::basis_trait::Basis::evaluate_matsubara)
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Result<Self>
    where
        S: 'static,
    {
        if sampling_points.is_empty() {
            return Err(Error::EmptyInput {
                name: "sampling_points",
            });
        }
        // Positive-only sampling uses non-negative frequencies only (#247).
        check_non_negative(&sampling_points)?;
        let matrix = mat_from_matrix(&basis.evaluate_matsubara(&sampling_points)?)?;
        Ok(Self {
            sampling_points,
            fitter: ComplexToRealFitter::new(matrix),
            _phantom: PhantomData,
        })
    }

    /// Create Matsubara sampling (positive-only) with custom sampling points and pre-computed matrix
    ///
    /// This constructor is useful when the sampling matrix is already computed.
    /// Uses symmetry to fit real coefficients from complex values at non-negative frequencies.
    ///
    /// # Arguments
    /// * `sampling_points` - Matsubara frequency sampling points (must be
    ///   non-negative), in any order
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size); row i
    ///   belongs to `sampling_points[i]`
    ///
    /// The points are kept in the given order: [`Self::sampling_points`]
    /// returns them unchanged, and index i along the sampling-point axis of
    /// `evaluate` and `fit` refers to `sampling_points[i]`.
    ///
    /// Duplicate points are accepted; they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty, or `matrix`
    ///   has no columns
    /// * [`Error::ShapeMismatch`] of the input if `matrix` does not have one
    ///   row per point
    /// * [`Error::InvalidMatsubaraIndex`] for the first negative point
    /// * [`Error::NonFiniteInput`] for the first entry of `matrix` with a NaN
    ///   or infinite part
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: &Matrix<C64>,
    ) -> Result<Self> {
        let matrix = mat_from_matrix(matrix)?;
        check_sampling_matrix_shape(sampling_points.len(), *matrix.shape())?;
        check_non_negative(&sampling_points)?;
        check_finite_matrix(&matrix, non_finite_part)?;
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

    /// Condition number of the real least-squares problem that fitting solves
    ///
    /// Fitting real coefficients `x` to complex values `g` at non-negative
    /// frequencies solves `[Re A; Im A] x = [Re g; Im g]`, where `A` is the
    /// complex `n_sampling_points × basis_size` matrix [`Self::matrix`]. This
    /// returns `σ_max / σ_min`, the ratio of the largest to the smallest of the
    /// `min(2 n_sampling_points, basis_size)` singular values of that real
    /// `2 n_sampling_points × basis_size` matrix; it bounds how much
    /// [`Self::fit`] can amplify relative errors in the values. It is not the
    /// condition number of `A`: with `n_sampling_points ≈ basis_size / 2`, `A`
    /// is wide, and its condition number can understate that amplification by
    /// orders of magnitude.
    ///
    /// Returns `f64::INFINITY` if the smallest singular value is below `1e-15`
    /// (numerically singular matrix). The singular value decomposition is the
    /// one fitting uses: it is computed by the first call to this method or to
    /// a fit, then cached.
    ///
    /// # Errors
    ///
    /// [`Error::DecompositionFailed`] if the singular value decomposition
    /// fails, which a matrix of finite entries does not cause in practice
    /// (the constructors reject non-finite entries)
    pub fn condition_number(&self) -> Result<f64> {
        self.fitter.condition_number()
    }

    /// Evaluate basis coefficients at sampling points
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    pub fn evaluate(&self, coeffs: &[f64]) -> Result<Vec<C64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Fit basis coefficients from values at sampling points
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit(&self, values: &[C64]) -> Result<Vec<f64>> {
        self.fitter.fit(None, values)
    }

    /// Evaluate N-dimensional array of real basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of real basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<C64>> {
        self.fitter.evaluate_nd_dz(backend, coeffs, dim)
    }

    /// Fit N-dimensional array of complex values to real basis coefficients
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (None uses default)
    /// * `values` - N-dimensional tensor of complex values at Matsubara frequencies
    /// * `dim` - Dimension along which to fit (must have size = n_sampling_points)
    ///
    /// # Returns
    /// N-dimensional tensor of real basis coefficients
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<C64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>> {
        self.fitter.fit_nd_zd(backend, values, dim)
    }

    /// Evaluate real basis coefficients at Matsubara sampling points (N-dimensional) with in-place output
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of real coefficients with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points` (Complex<f64>)
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`, and of the output if `out` does not have
    ///   the shape of `coeffs` with `n_sampling_points` along `dim`
    ///
    /// Nothing is written to `out` then.
    pub fn evaluate_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, C64>,
    ) -> Result<()> {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    /// Fit N-dimensional complex values to real coefficients with in-place output
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == basis_size` (f64)
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`, and of the output if `out` does not have
    ///   the shape of `values` with `basis_size` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    ///
    /// Nothing is written to `out` then.
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
