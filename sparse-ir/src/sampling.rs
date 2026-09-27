//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::error::Error;
use crate::fitters::InplaceFitter;
use crate::fitters::common::check_input_shape;
use crate::gemm::GemmBackendHandle;
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Slice, Tensor, ViewMut};
use num_complex::Complex;

/// Build output shape by replacing dimension `dim` with `new_size`
pub(crate) fn build_output_shape<S: Shape>(
    input_shape: &S,
    dim: usize,
    new_size: usize,
) -> Vec<usize> {
    let mut out_shape: Vec<usize> = Vec::with_capacity(input_shape.rank());
    input_shape.with_dims(|dims| {
        for (i, d) in dims.iter().enumerate() {
            if i == dim {
                out_shape.push(new_size);
            } else {
                out_shape.push(*d);
            }
        }
    });
    out_shape
}

/// Move axis from position `src` to position `dst`
///
/// This is equivalent to numpy.moveaxis or libsparseir's movedim.
/// It creates a permutation array that moves the specified axis.
///
/// # Arguments
/// * `arr` - Input array slice (Tensor or View)
/// * `src` - Source axis position
/// * `dst` - Destination axis position
///
/// # Returns
/// Tensor with axes permuted
///
/// # Panics
///
/// Panics if `src` or `dst` is not an axis of `arr`, also when they are equal.
///
/// # Example
/// ```
/// use sparse_ir::sampling::movedim;
/// use sparse_ir::{DynRank, Tensor};
///
/// // For a 4D tensor with shape (2, 3, 4, 5)
/// let arr = Tensor::<usize, DynRank>::from_fn(&[2, 3, 4, 5][..], |idx| {
///     1000 * idx[0] + 100 * idx[1] + 10 * idx[2] + idx[3]
/// });
///
/// // movedim(arr, 0, 2) moves axis 0 to position 2
/// let moved = movedim(&arr, 0, 2);
///
/// // Result shape: (3, 4, 2, 5) with axes permuted as [1, 2, 0, 3]
/// assert_eq!(moved.shape().dims(), &[3, 4, 2, 5]);
/// assert_eq!(moved[&[2, 3, 1, 4][..]], arr[&[1, 2, 3, 4][..]]);
/// ```
pub fn movedim<T: Clone>(arr: &Slice<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    let rank = arr.rank();
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
    if src == dst {
        return arr.to_tensor();
    }

    // Generate permutation: move src to dst position
    let mut perm = Vec::with_capacity(rank);
    let mut pos = 0;
    for i in 0..rank {
        if i == dst {
            perm.push(src);
        } else {
            // Skip src position
            if pos == src {
                pos += 1;
            }
            perm.push(pos);
            pos += 1;
        }
    }

    if arr.is_empty() {
        // Zero-extent guard: mdarray 0.7.2 (and 0.8.0) copies a permuted,
        // strided view out of bounds when an extent other than the last is
        // zero (https://github.com/fre-hu/mdarray/issues/21). An empty array
        // has no elements to move, so reshape it (a dense view) instead.
        let dims: Vec<usize> = perm.iter().map(|&axis| arr.shape().dim(axis)).collect();
        return arr.reshape(&dims[..]).to_tensor();
    }

    arr.permute(&perm[..]).to_tensor()
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
///   no basis function, and the fitter would transpose `[n, 0]` arrays,
///   which mdarray 0.7.2 does out of bounds
///   (<https://github.com/fre-hu/mdarray/issues/21>)
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
    matrix: &DTensor<T, 2>,
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
        // With no points the sampling matrix would have no rows, and the
        // fitter's transposes would go through the zero-extent paths of
        // mdarray 0.7.2 (https://github.com/fre-hu/mdarray/issues/21).
        if sampling_points.is_empty() {
            return Err(Error::EmptyInput {
                name: "sampling_points",
            });
        }

        // Compute sampling matrix: A[i, l] = u_l(τ_i); evaluate_tau checks
        // that every τ is in [-β, β].
        let matrix = basis.evaluate_tau(&sampling_points)?;
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
    pub fn from_matrix(sampling_points: Vec<f64>, matrix: DTensor<f64, 2>) -> Result<Self, Error> {
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
    pub fn matrix(&self) -> &DTensor<f64, 2> {
        &self.fitter.matrix
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
        self.fitter.evaluate_zz(None, coeffs)
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
        self.fitter.evaluate_zz_to(None, coeffs, out)
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
        self.fitter.fit_zz(None, values)
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
        self.fitter.fit_zz_to(None, values, out)
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
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
    ) -> Result<Tensor<f64, DynRank>, Error> {
        check_input_shape(coeffs.shape().dims(), dim, self.basis_size())?;
        let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
        let mut out = Tensor::<f64, DynRank>::zeros(&out_shape[..]);
        self.evaluate_nd_to(backend, coeffs, dim, &mut out.expr_mut())?;
        Ok(out)
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
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
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
        values: &Slice<f64, DynRank>,
        dim: usize,
    ) -> Result<Tensor<f64, DynRank>, Error> {
        check_input_shape(values.shape().dims(), dim, self.n_sampling_points())?;
        let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
        let mut out = Tensor::<f64, DynRank>::zeros(&out_shape[..]);
        self.fit_nd_to(backend, values, dim, &mut out.expr_mut())?;
        Ok(out)
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
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
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
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Result<Tensor<Complex<f64>, DynRank>, Error> {
        check_input_shape(coeffs.shape().dims(), dim, self.basis_size())?;
        let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&out_shape[..]);
        self.evaluate_nd_zz_to(backend, coeffs, dim, &mut out.expr_mut())?;
        Ok(out)
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
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
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
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Result<Tensor<Complex<f64>, DynRank>, Error> {
        check_input_shape(values.shape().dims(), dim, self.n_sampling_points())?;
        let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&out_shape[..]);
        self.fit_nd_zz_to(backend, values, dim, &mut out.expr_mut())?;
        Ok(out)
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
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
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
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        self.fitter.evaluate_nd_dd_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> Result<(), Error> {
        self.fitter.fit_nd_dd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> Result<(), Error> {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

#[cfg(test)]
#[path = "tau_sampling_tests.rs"]
mod tests;
