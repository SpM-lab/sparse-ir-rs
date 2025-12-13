//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::fpu_check::FpuGuard;
use crate::gemm::GemmBackendHandle;
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Slice, Tensor, ViewMut};
use num_complex::Complex;

/// Build output shape by replacing dimension `dim` with `new_size`
fn build_output_shape<S: Shape>(input_shape: &S, dim: usize, new_size: usize) -> Vec<usize> {
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
/// # Example
/// ```ignore
/// // For a 4D tensor with shape (2, 3, 4, 5)
/// // movedim(arr, 0, 2) moves axis 0 to position 2
/// // Result shape: (3, 4, 2, 5) with axes permuted as [1, 2, 0, 3]
/// ```
pub fn movedim<T: Clone>(arr: &Slice<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    if src == dst {
        return arr.to_tensor();
    }

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

    arr.permute(&perm[..]).to_tensor()
}

/// Sparse sampling in imaginary time
///
/// Allows transformation between the IR basis and a set of sampling points
/// in imaginary time (τ).
pub struct TauSampling<S>
where
    S: StatisticsType,
{
    /// Sampling points in imaginary time τ ∈ [-β/2, β/2]
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
    /// The default sampling points are chosen as the extrema of the highest-order
    /// basis function, which gives near-optimal conditioning.
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `basis` - Any basis implementing the `Basis` trait
    ///
    /// # Returns
    /// A new TauSampling object
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
    /// # Arguments
    /// * `basis` - Any basis implementing the `Basis` trait
    /// * `sampling_points` - Custom sampling points in τ ∈ [-β, β]
    ///
    /// # Returns
    /// A new TauSampling object
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
        assert!(
            basis.size() <= sampling_points.len(),
            "The number of sampling points must be greater than or equal to the basis size"
        );

        let beta = basis.beta();
        for &tau in &sampling_points {
            assert!(
                tau >= -beta && tau <= beta,
                "Sampling point τ={} is outside [-β, β]",
                tau
            );
        }

        // Compute sampling matrix: A[i, l] = u_l(τ_i)
        // Use Basis trait's evaluate_tau method
        let matrix = basis.evaluate_tau(&sampling_points);

        // Create fitter
        let fitter = crate::fitters::RealMatrixFitter::new(matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new TauSampling with custom sampling points and pre-computed matrix
    ///
    /// This constructor is useful when the sampling matrix is already computed
    /// (e.g., from external sources or for testing).
    ///
    /// # Arguments
    /// * `sampling_points` - Sampling points in τ ∈ [-β, β]
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size)
    ///
    /// # Returns
    /// A new TauSampling object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if matrix dimensions don't match
    pub fn from_matrix(sampling_points: Vec<f64>, matrix: DTensor<f64, 2>) -> Self {
        assert!(!sampling_points.is_empty(), "No sampling points given");
        assert_eq!(
            matrix.shape().0,
            sampling_points.len(),
            "Matrix rows ({}) must match number of sampling points ({})",
            matrix.shape().0,
            sampling_points.len()
        );

        let fitter = crate::fitters::RealMatrixFitter::new(matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: std::marker::PhantomData,
        }
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
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<f64> {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate(None, coeffs)
    }

    /// Evaluate basis coefficients at sampling points, writing to output slice
    pub fn evaluate_to(&self, coeffs: &[f64], out: &mut [f64]) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate_to(None, coeffs, out)
    }

    /// Fit values at sampling points to basis coefficients
    pub fn fit(&self, values: &[f64]) -> Vec<f64> {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit(None, values)
    }

    /// Fit values at sampling points to basis coefficients, writing to output slice
    pub fn fit_to(&self, values: &[f64], out: &mut [f64]) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit_to(None, values, out)
    }

    /// Evaluate complex basis coefficients at sampling points
    pub fn evaluate_zz(&self, coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate_zz(None, coeffs)
    }

    /// Evaluate complex basis coefficients, writing to output slice
    pub fn evaluate_zz_to(&self, coeffs: &[Complex<f64>], out: &mut [Complex<f64>]) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate_zz_to(None, coeffs, out)
    }

    /// Fit complex values at sampling points to basis coefficients
    pub fn fit_zz(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit_zz(None, values)
    }

    /// Fit complex values, writing to output slice
    pub fn fit_zz_to(&self, values: &[Complex<f64>], out: &mut [Complex<f64>]) {
        let _guard = FpuGuard::new_protect_computation();
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
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let _guard = FpuGuard::new_protect_computation();
        let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
        let mut out = Tensor::<f64, DynRank>::zeros(&out_shape[..]);
        self.evaluate_nd_to(backend, coeffs, dim, &mut out.expr_mut());
        out
    }

    /// Evaluate N-D real coefficients, writing to a mutable view
    pub fn evaluate_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate_nd_dd_to(backend, coeffs, dim, out);
    }

    /// Fit N-D real values at sampling points to basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == basis_size`
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let _guard = FpuGuard::new_protect_computation();
        let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
        let mut out = Tensor::<f64, DynRank>::zeros(&out_shape[..]);
        self.fit_nd_to(backend, values, dim, &mut out.expr_mut());
        out
    }

    /// Fit N-D real values, writing to a mutable view
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit_nd_dd_to(backend, values, dim, out);
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
    pub fn evaluate_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let _guard = FpuGuard::new_protect_computation();
        let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&out_shape[..]);
        self.evaluate_nd_zz_to(backend, coeffs, dim, &mut out.expr_mut());
        out
    }

    /// Evaluate N-D complex coefficients, writing to a mutable view
    pub fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out);
    }

    /// Fit N-D complex values at sampling points to basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional complex array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional complex array with `result.shape().dim(dim) == basis_size`
    pub fn fit_nd_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let _guard = FpuGuard::new_protect_computation();
        let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&out_shape[..]);
        self.fit_nd_zz_to(backend, values, dim, &mut out.expr_mut());
        out
    }

    /// Fit N-D complex values, writing to a mutable view
    pub fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit_nd_zz_to(backend, values, dim, out);
    }
}


#[cfg(test)]
#[path = "tau_sampling_tests.rs"]
mod tests;
