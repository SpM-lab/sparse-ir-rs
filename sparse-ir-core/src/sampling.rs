//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::Matrix;
use crate::error::Error;
use crate::fitters::InplaceFitter;
use crate::gemm::GemmBackendHandle;
use crate::matrix::Mat;
use crate::traits::StatisticsType;
use num_complex::Complex;
use tenferro_tensor::{TensorScalar, TypedTensor, TypedTensorView, TypedTensorViewMut};

/// Copy a host matrix into the internal column-major container
#[doc(hidden)]
pub fn mat_from_matrix<T: TensorScalar + Copy>(m: &Matrix<T>) -> Result<Mat<T>, Error> {
    Ok(Mat::from_typed(m)?)
}

/// Move axis from position `src` to position `dst`
///
/// This is equivalent to numpy.moveaxis or libsparseir's movedim. The other
/// axes keep their order.
///
/// # Arguments
/// * `arr` - Input tensor
/// * `src` - Source axis position
/// * `dst` - Destination axis position
///
/// # Returns
/// A new tensor with the axes permuted
///
/// # Panics
///
/// Panics if `src` or `dst` is not an axis of `arr`.
///
/// # Example
/// ```
/// use sparse_ir::TypedTensor;
/// use sparse_ir::sampling::movedim;
///
/// // A 4D tensor with shape (2, 3, 4, 5) and entries 1000 i + 100 j + 10 k + l
/// let shape = [2usize, 3, 4, 5];
/// let data: Vec<f64> = (0..120)
///     .map(|lin| {
///         let (i, j, k, l) = (lin % 2, lin / 2 % 3, lin / 6 % 4, lin / 24);
///         (1000 * i + 100 * j + 10 * k + l) as f64
///     })
///     .collect();
/// let arr = TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap();
///
/// // movedim(arr, 0, 2) moves axis 0 to position 2
/// let moved = movedim(&arr, 0, 2);
///
/// // Result shape: (3, 4, 2, 5) with axes permuted as [1, 2, 0, 3]
/// assert_eq!(moved.shape(), &[3, 4, 2, 5]);
/// // Element [2, 3, 1, 4] of the result is element [1, 2, 3, 4] of arr
/// let at = |t: &TypedTensor<f64>, idx: [usize; 4]| {
///     let s = t.shape();
///     t.host_data().unwrap()[idx[0] + s[0] * (idx[1] + s[1] * (idx[2] + s[2] * idx[3]))]
/// };
/// assert_eq!(at(&moved, [2, 3, 1, 4]), at(&arr, [1, 2, 3, 4]));
/// ```
pub fn movedim<T: TensorScalar + Copy>(
    arr: &TypedTensor<T>,
    src: usize,
    dst: usize,
) -> TypedTensor<T> {
    let shape = arr.shape().to_vec();
    let rank = shape.len();
    assert!(
        src < rank,
        "src axis {} out of bounds for rank {}",
        src,
        rank
    );
    assert!(
        dst < rank,
        "dst axis {} out of bounds for rank {}",
        dst,
        rank
    );
    // Output axis k reads input axis perm[k].
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(src);
    perm.insert(dst, src);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let data = arr
        .host_data()
        .expect("an owned tensor is compact host storage");
    let len = data.len();
    let mut out = Vec::with_capacity(len);
    let mut out_idx = vec![0usize; rank];
    for _ in 0..len {
        // Column-major offset of the input element
        let mut offset = 0;
        let mut stride = 1;
        for axis in 0..rank {
            let k = perm.iter().position(|&p| p == axis).unwrap();
            offset += out_idx[k] * stride;
            stride *= shape[axis];
        }
        out.push(data[offset]);
        for (k, i) in out_idx.iter_mut().enumerate() {
            *i += 1;
            if *i < out_shape[k] {
                break;
            }
            *i = 0;
        }
    }
    TypedTensor::from_vec_col_major(out_shape, out).expect("the shape matches the data")
}

/// Check the shape of a given sampling matrix against its points: some
/// points, one row per point and at least one column
///
/// # Errors
///
/// * [`Error::EmptyInput`] named `sampling_points` if there are no points
/// * [`Error::ShapeMismatch`] of the input if the matrix does not have one
///   row per point
/// * [`Error::EmptyInput`] named `matrix` if it has no columns: it describes
///   no basis function
pub(crate) fn check_sampling_matrix_shape(
    n_points: usize,
    (rows, cols): (usize, usize),
) -> Result<(), Error> {
    if n_points == 0 {
        return Err(Error::EmptyInput {
            name: "sampling_points",
        });
    }
    if rows != n_points {
        return Err(Error::ShapeMismatch {
            which: crate::error::ArrayRole::Input,
            expected: vec![n_points, cols],
            actual: vec![rows, cols],
        });
    }
    if cols == 0 {
        return Err(Error::EmptyInput { name: "matrix" });
    }
    Ok(())
}

/// `Ok` if every entry of a given sampling matrix is finite (the fitter
/// factorizes it)
///
/// # Errors
///
/// [`Error::NonFiniteInput`] named `matrix` at the first NaN or infinite
/// entry in row-major order; for a complex entry, `value` is its real part
/// if that is not finite, and its imaginary part otherwise
pub(crate) fn check_finite_matrix<T: Copy>(
    matrix: &Mat<T>,
    non_finite_part: impl Fn(T) -> Option<f64>,
) -> Result<(), Error> {
    let (rows, cols) = *matrix.shape();
    for i in 0..rows {
        for j in 0..cols {
            if let Some(value) = non_finite_part(matrix[[i, j]]) {
                return Err(Error::NonFiniteInput {
                    name: "matrix",
                    index: vec![i, j],
                    value,
                });
            }
        }
    }
    Ok(())
}

/// Sparse sampling in imaginary time
///
/// Allows transformation between the IR basis and a set of sampling points
/// in imaginary time (τ).
pub struct TauSampling<S>
where
    S: StatisticsType,
{
    /// Sampling points in imaginary time, in the order given (τ ∈ [-β, β]
    /// unless given with a matrix)
    sampling_points: Vec<f64>,

    /// Real matrix fitter for least-squares fitting
    fitter: crate::fitters::RealMatrixFitter,

    /// Marker for statistics type
    _phantom: std::marker::PhantomData<S>,
}

impl<S> TauSampling<S>
where
    S: StatisticsType,
{
    /// Create a new TauSampling with default sampling points
    ///
    /// The default sampling points are the roots of the first discarded basis
    /// function u_L (the extrema of u_{L-1} when u_L is not available), which
    /// gives near-optimal conditioning.
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `basis` - Any basis implementing the `Basis` trait
    ///
    /// # Returns
    /// A new TauSampling object
    ///
    /// # Errors
    ///
    /// The errors of [`Basis::default_tau_sampling_points`](crate::basis_trait::Basis::default_tau_sampling_points)
    /// (e.g. NotSupported for a DLR, whose IR basis has the default points)
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Result<Self, Error>
    where
        S: 'static,
    {
        let sampling_points = basis.default_tau_sampling_points()?;
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create a new TauSampling with custom sampling points
    ///
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `basis` - Any basis implementing the `Basis` trait
    /// * `sampling_points` - Custom sampling points in τ ∈ [-β, β]
    ///
    /// # Returns
    /// A new TauSampling object
    ///
    /// The points are kept in the given order, and duplicates are accepted;
    /// they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty
    /// * [`Error::OutOfDomain`] if a point is outside [-β, β] or NaN (from
    ///   [`Basis::evaluate_tau`](crate::basis_trait::Basis::evaluate_tau))
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<f64>,
    ) -> Result<Self, Error>
    where
        S: 'static,
    {
        // With no points the sampling matrix would have no rows.
        if sampling_points.is_empty() {
            return Err(Error::EmptyInput {
                name: "sampling_points",
            });
        }

        // Compute sampling matrix: A[i, l] = u_l(τ_i); evaluate_tau checks
        // that every τ is in [-β, β].
        let matrix = mat_from_matrix(&basis.evaluate_tau(&sampling_points)?)?;
        let fitter = crate::fitters::RealMatrixFitter::new(matrix);

        Ok(Self {
            sampling_points,
            fitter,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a new TauSampling with custom sampling points and pre-computed matrix
    ///
    /// This constructor is useful when the sampling matrix is already computed
    /// (e.g., from external sources or for testing).
    ///
    /// # Arguments
    /// * `sampling_points` - Imaginary times τ that label the rows of
    ///   `matrix`, in any order. There is no β to check them against, so
    ///   any finite value is accepted and kept as given.
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size); row i
    ///   belongs to `sampling_points[i]`
    ///
    /// Duplicate points are accepted; they only raise the condition number.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `sampling_points` is empty, or `matrix`
    ///   has no columns
    /// * [`Error::ShapeMismatch`] of the input if `matrix` does not have one
    ///   row per point
    /// * [`Error::NonFiniteInput`] for the first NaN or infinite point, then
    ///   for the first NaN or infinite entry of `matrix`
    pub fn from_matrix(sampling_points: Vec<f64>, matrix: &Matrix<f64>) -> Result<Self, Error> {
        let matrix = mat_from_matrix(matrix)?;
        check_sampling_matrix_shape(sampling_points.len(), *matrix.shape())?;
        if let Some((i, &tau)) = sampling_points
            .iter()
            .enumerate()
            .find(|(_, tau)| !tau.is_finite())
        {
            return Err(Error::NonFiniteInput {
                name: "sampling_points",
                index: vec![i],
                value: tau,
            });
        }
        check_finite_matrix(&matrix, |x: f64| (!x.is_finite()).then_some(x))?;

        let fitter = crate::fitters::RealMatrixFitter::new(matrix);

        Ok(Self {
            sampling_points,
            fitter,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the sampling points
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

    /// Get the sampling matrix
    pub fn matrix(&self) -> &Matrix<f64> {
        self.fitter.matrix()
    }

    /// Condition number of the sampling matrix, which fitting solves with
    ///
    /// Returns `σ_max / σ_min`, the ratio of the largest to the smallest of the
    /// `min(n_sampling_points, basis_size)` singular values of the real
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
    pub fn condition_number(&self) -> Result<f64, Error> {
        self.fitter.condition_number()
    }

    // ========================================================================
    // 1D functions (real and complex)
    // ========================================================================

    /// Evaluate basis coefficients at sampling points
    ///
    /// Computes g(τ_i) = Σ_l a_l * u_l(τ_i) for all sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Values at sampling points (length = n_sampling_points)
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    pub fn evaluate(&self, coeffs: &[f64]) -> Result<Vec<f64>, Error> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate basis coefficients at sampling points, writing to output slice
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    /// * [`Error::ShapeMismatch`] of the output if `out` does not have length
    ///   `n_sampling_points`
    ///
    /// Nothing is written to `out` on an error.
    pub fn evaluate_to(&self, coeffs: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.fitter.evaluate_to(None, coeffs, out)
    }

    /// Fit values at sampling points to basis coefficients
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit(&self, values: &[f64]) -> Result<Vec<f64>, Error> {
        self.fitter.fit(None, values)
    }

    /// Fit values at sampling points to basis coefficients, writing to output slice
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::ShapeMismatch`] of the output if `out` does not have length
    ///   `basis_size`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    ///
    /// Nothing is written to `out` on an error.
    pub fn fit_to(&self, values: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.fitter.fit_to(None, values, out)
    }

    /// Evaluate complex basis coefficients at sampling points
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    pub fn evaluate_zz(&self, coeffs: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, Error> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate complex basis coefficients, writing to output slice
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have length
    ///   `basis_size`
    /// * [`Error::ShapeMismatch`] of the output if `out` does not have length
    ///   `n_sampling_points`
    ///
    /// Nothing is written to `out` on an error.
    pub fn evaluate_zz_to(
        &self,
        coeffs: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), Error> {
        self.fitter.evaluate_to(None, coeffs, out)
    }

    /// Fit complex values at sampling points to basis coefficients
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit_zz(&self, values: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, Error> {
        self.fitter.fit(None, values)
    }

    /// Fit complex values, writing to output slice
    ///
    /// # Errors
    ///
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have length
    ///   `n_sampling_points`
    /// * [`Error::ShapeMismatch`] of the output if `out` does not have length
    ///   `basis_size`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    ///
    /// Nothing is written to `out` on an error.
    pub fn fit_zz_to(
        &self,
        values: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), Error> {
        self.fitter.fit_to(None, values, out)
    }

    // ========================================================================
    // N-D functions (real)
    // ========================================================================

    /// Evaluate N-D real coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == n_sampling_points`
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
    ) -> Result<TypedTensor<f64>, Error> {
        self.fitter.evaluate_nd(backend, coeffs, dim)
    }

    /// Evaluate N-D real coefficients, writing to a mutable view
    ///
    /// `out` must have the shape of `coeffs` with `n_sampling_points` along
    /// `dim`.
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
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), Error> {
        InplaceFitter::evaluate_nd_dd_to(self, backend, coeffs, dim, out)
    }

    /// Fit N-D real values at sampling points to basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == basis_size`
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
        values: &TypedTensor<f64>,
        dim: usize,
    ) -> Result<TypedTensor<f64>, Error> {
        self.fitter.fit_nd(backend, values, dim)
    }

    /// Fit N-D real values, writing to a mutable view
    ///
    /// `out` must have the shape of `values` with `basis_size` along `dim`.
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
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), Error> {
        InplaceFitter::fit_nd_dd_to(self, backend, values, dim, out)
    }

    // ========================================================================
    // N-D functions (complex)
    // ========================================================================

    /// Evaluate N-D complex coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional complex array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    ///
    /// # Returns
    /// N-dimensional complex array with `result.shape().dim(dim) == n_sampling_points`
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`
    pub fn evaluate_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensor<Complex<f64>>,
        dim: usize,
    ) -> Result<TypedTensor<Complex<f64>>, Error> {
        self.fitter.evaluate_nd(backend, coeffs, dim)
    }

    /// Evaluate N-D complex coefficients, writing to a mutable view
    ///
    /// `out` must have the shape of `coeffs` with `n_sampling_points` along
    /// `dim`.
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `coeffs`
    /// * [`Error::ShapeMismatch`] of the input if `coeffs` does not have
    ///   `basis_size` along `dim`, and of the output if `out` does not have
    ///   the shape of `coeffs` with `n_sampling_points` along `dim`
    ///
    /// Nothing is written to `out` then.
    pub fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), Error> {
        InplaceFitter::evaluate_nd_zz_to(self, backend, coeffs, dim, out)
    }

    /// Fit N-D complex values at sampling points to basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional complex array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional complex array with `result.shape().dim(dim) == basis_size`
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `values`
    /// * [`Error::ShapeMismatch`] of the input if `values` does not have
    ///   `n_sampling_points` along `dim`
    /// * [`Error::DecompositionFailed`] if the singular value decomposition
    ///   fails
    pub fn fit_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensor<Complex<f64>>,
        dim: usize,
    ) -> Result<TypedTensor<Complex<f64>>, Error> {
        self.fitter.fit_nd(backend, values, dim)
    }

    /// Fit N-D complex values, writing to a mutable view
    ///
    /// `out` must have the shape of `values` with `basis_size` along `dim`.
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
    pub fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), Error> {
        InplaceFitter::fit_nd_zz_to(self, backend, values, dim, out)
    }
}

/// InplaceFitter implementation for TauSampling
///
/// Delegates to RealMatrixFitter which supports dd and zz operations.
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
    ) -> Result<(), Error> {
        self.fitter.evaluate_nd_dd_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), Error> {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), Error> {
        self.fitter.fit_nd_dd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), Error> {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}
