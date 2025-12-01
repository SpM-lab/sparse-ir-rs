//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::gemm::{GemmBackendHandle, matmul_par};
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Slice, Tensor};
use num_complex::Complex;

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
    fitter: crate::fitter::RealMatrixFitter,

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
        let fitter = crate::fitter::RealMatrixFitter::new(matrix);

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

        let fitter = crate::fitter::RealMatrixFitter::new(matrix);

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
    /// # Panics
    /// Panics if `coeffs.len() != basis_size`
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<f64> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Internal generic evaluate_nd implementation
    fn evaluate_nd_impl<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
    ) -> Tensor<T, DynRank>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + 'static + From<f64> + Copy,
    {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);

        // Check that the target dimension matches basis_size
        assert_eq!(
            target_dim_size, basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim, target_dim_size, basis_size
        );

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        // Convert DynRank to fixed Rank<2> for matmul_par
        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();
        let coeffs_2d = DTensor::<T, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // 3. Matrix multiply: result = A * coeffs
        //    A is real, convert to type T
        let n_points = self.n_sampling_points();
        let matrix_t = DTensor::<T, 2>::from_fn(*self.fitter.matrix.shape(), |idx| {
            self.fitter.matrix[idx].into()
        });
        let result_2d = matmul_par(&matrix_t, &coeffs_2d, backend);

        // 4. Reshape back to N-D with n_points at position 0
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });

        // Convert DTensor<T, 2> to DynRank using into_dyn()
        let result_2d_dyn = result_2d.into_dyn();
        let result_dim0 = result_2d_dyn.reshape(&result_shape[..]).to_tensor();

        // 5. Move dimension back to original position
        movedim(&result_dim0, 0, dim)
    }

    /// Evaluate basis coefficients at sampling points (N-dimensional)
    ///
    /// Evaluates along the specified dimension, keeping other dimensions intact.
    /// Supports both real (`f64`) and complex (`Complex<f64>`) coefficients.
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == n_sampling_points`
    ///
    /// # Panics
    /// Panics if `coeffs.shape().dim(dim) != basis_size` or if `dim >= rank`
    ///
    /// # Example
    /// ```ignore
    /// use num_complex::Complex;
    /// use mdarray::tensor;
    ///
    /// // Real coefficients
    /// let values_real = sampling.evaluate_nd::<f64>(&coeffs_real, 0);
    ///
    /// // Complex coefficients
    /// let values_complex = sampling.evaluate_nd::<Complex<f64>>(&coeffs_complex, 0);
    /// ```
    pub fn evaluate_nd<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
    ) -> Tensor<T, DynRank>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + 'static + From<f64> + Copy,
    {
        self.evaluate_nd_impl(backend, coeffs, dim)
    }

    /// Fit N-D values for the real case `T = f64`
    fn fit_nd_impl_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let n_points = self.n_sampling_points();
        let basis_size = self.basis_size();
        let target_dim_size = values.shape().dim(dim);

        assert_eq!(
            target_dim_size, n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, target_dim_size, n_points
        );

        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);

        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        let values_2d = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();

        // 3. Fit using real 2D fitter directly on a view
        // Convert Tensor<f64, DynRank> to DView<f64, 2> by creating DTensor and taking view
        let values_2d_dtensor = DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| {
            values_2d[&[idx[0], idx[1]][..]]
        });
        let values_2d_view_2d = values_2d_dtensor.view(.., ..);
        let coeffs_2d = self.fitter.fit_2d(backend, &values_2d_view_2d);

        // 4. Reshape back to N-D with basis_size at position 0
        let mut coeffs_shape = vec![basis_size];
        values_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                coeffs_shape.push(dims[i]);
            }
        });

        let coeffs_dim0 = coeffs_2d.into_dyn().reshape(&coeffs_shape[..]).to_tensor();

        // 5. Move dimension 0 back to original position dim
        movedim(&coeffs_dim0, 0, dim)
    }

    /// Fit N-D values for the complex case `T = Complex<f64>`
    fn fit_nd_impl_complex(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<num_complex::Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<num_complex::Complex<f64>, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let n_points = self.n_sampling_points();
        let basis_size = self.basis_size();
        let target_dim_size = values.shape().dim(dim);

        assert_eq!(
            target_dim_size, n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, target_dim_size, n_points
        );

        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);

        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        let values_2d = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();

        // 3. Fit using complex 2D fitter directly on a view
        // Convert Tensor<Complex<f64>, DynRank> to DView<Complex<f64>, 2> by creating DTensor and taking view
        let values_2d_dtensor =
            DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
                values_2d[&[idx[0], idx[1]][..]]
            });
        let values_2d_view_2d = values_2d_dtensor.view(.., ..);
        let coeffs_2d = self.fitter.fit_complex_2d(backend, &values_2d_view_2d);

        // 4. Reshape back to N-D with basis_size at position 0
        let mut coeffs_shape = vec![basis_size];
        values_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                coeffs_shape.push(dims[i]);
            }
        });

        let coeffs_dim0 = coeffs_2d.into_dyn().reshape(&coeffs_shape[..]).to_tensor();

        // 5. Move dimension 0 back to original position dim
        movedim(&coeffs_dim0, 0, dim)
    }

    /// Fit basis coefficients from values at sampling points (N-dimensional)
    ///
    /// Fits along the specified dimension, keeping other dimensions intact.
    /// Supports both real (`f64`) and complex (`Complex<f64>`) values.
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == basis_size`
    ///
    /// # Panics
    /// Panics if `values.shape().dim(dim) != n_sampling_points`, if `dim >= rank`, or if SVD not computed
    ///
    /// # Example
    /// ```ignore
    /// use num_complex::Complex;
    /// use mdarray::tensor;
    ///
    /// // Real values
    /// let coeffs_real = sampling.fit_nd::<f64>(&values_real, 0);
    ///
    /// // Complex values
    /// let coeffs_complex = sampling.fit_nd::<Complex<f64>>(&values_complex, 0);
    /// ```
    pub fn fit_nd<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<T, DynRank>,
        dim: usize,
    ) -> Tensor<T, DynRank>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + 'static
            + From<f64>
            + Copy
            + Default,
    {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Real case: reinterpret as f64 tensor, call real implementation, then cast back to T
            let values_f64 =
                unsafe { &*(values as *const Tensor<T, DynRank> as *const Tensor<f64, DynRank>) };
            let coeffs_f64 = self.fit_nd_impl_real(backend, values_f64, dim);
            unsafe { std::mem::transmute::<Tensor<f64, DynRank>, Tensor<T, DynRank>>(coeffs_f64) }
        } else if TypeId::of::<T>() == TypeId::of::<num_complex::Complex<f64>>() {
            // Complex case: reinterpret as Complex<f64> tensor, call complex implementation, then cast back
            let values_c64 = unsafe {
                &*(values as *const Tensor<T, DynRank>
                    as *const Tensor<num_complex::Complex<f64>, DynRank>)
            };
            let coeffs_c64 = self.fit_nd_impl_complex(backend, values_c64, dim);
            unsafe {
                std::mem::transmute::<Tensor<num_complex::Complex<f64>, DynRank>, Tensor<T, DynRank>>(
                    coeffs_c64,
                )
            }
        } else {
            panic!("Unsupported type for fit_nd: must be f64 or Complex<f64>");
        }
    }
}

#[cfg(test)]
#[path = "tau_sampling_tests.rs"]
mod tests;
