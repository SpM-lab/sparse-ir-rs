//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::fitters::{ComplexMatrixFitter, ComplexToRealFitter, InplaceFitter};
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::sampling::{build_output_shape, movedim};
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Slice, Tensor, ViewMut};
use num_complex::Complex;
use std::marker::PhantomData;

/// Trait for coefficient types that can be evaluated by Matsubara sampling
///
/// This provides compile-time dispatch for different coefficient types,
/// avoiding runtime TypeId checks and unsafe pointer casts.
pub trait MatsubaraCoeffs: Copy + 'static {
    /// Evaluate coefficients using the given sampler
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Self, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank>;
}

impl MatsubaraCoeffs for f64 {
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Self, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        sampler.evaluate_nd_impl_real(backend, coeffs, dim)
    }
}

impl MatsubaraCoeffs for Complex<f64> {
    fn evaluate_nd_with<S: StatisticsType>(
        sampler: &MatsubaraSampling<S>,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Self, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        sampler.evaluate_nd_impl_complex(backend, coeffs, dim)
    }
}

/// Matsubara sampling for full frequency range (positive and negative)
///
/// General complex problem without symmetry → complex coefficients
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
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(false);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom sampling points
    ///
    /// The points may be in any order, and are kept in the given order:
    /// [`Self::sampling_points`] returns them unchanged, and index i along the
    /// sampling-point axis of `evaluate` and `fit` refers to
    /// `sampling_points[i]`.
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        // With no points the sampling matrix would have no rows; building it
        // and the fitter's transposes would go through the zero-extent paths
        // of mdarray 0.7.2 (https://github.com/fre-hu/mdarray/issues/21).
        assert!(!sampling_points.is_empty(), "No sampling points given");

        // Evaluate matrix at sampling points
        // Use Basis trait's evaluate_matsubara method
        let matrix = basis.evaluate_matsubara(&sampling_points);

        // Create fitter (complex → complex, no symmetry)
        let fitter = ComplexMatrixFitter::new(matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
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
    /// # Returns
    /// A new MatsubaraSampling object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty, if the number of matrix rows
    /// differs from the number of sampling points, or if the matrix has no
    /// columns (no basis functions)
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: DTensor<Complex<f64>, 2>,
    ) -> Self {
        assert!(!sampling_points.is_empty(), "No sampling points given");
        assert_eq!(
            matrix.shape().0,
            sampling_points.len(),
            "Matrix rows ({}) must match number of sampling points ({})",
            matrix.shape().0,
            sampling_points.len()
        );
        // A matrix without columns would make the fitter transpose [n, 0]
        // arrays, which mdarray 0.7.2 does out of bounds (mdarray#21,
        // https://github.com/fre-hu/mdarray/issues/21); there is nothing to fit.
        assert!(
            matrix.shape().1 > 0,
            "Matrix must have at least one column (basis function), got shape {:?}",
            matrix.shape()
        );

        let fitter = ComplexMatrixFitter::new(matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
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
    pub fn matrix(&self) -> &DTensor<Complex<f64>, 2> {
        &self.fitter.matrix
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
    pub fn condition_number(&self) -> f64 {
        self.fitter.condition_number()
    }

    /// Evaluate complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Complex basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at Matsubara frequencies (length = n_sampling_points)
    pub fn evaluate(&self, coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Fit complex basis coefficients from values at sampling points
    ///
    /// # Arguments
    /// * `values` - Complex values at Matsubara frequencies (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted complex basis coefficients (length = basis_size)
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fitter.fit(None, values)
    }

    /// Evaluate N-D coefficients for the real case `T = f64`
    fn evaluate_nd_impl_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);

        assert_eq!(
            target_dim_size, basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim, target_dim_size, basis_size
        );

        if coeffs.is_empty() {
            // Zero-extent guard: an empty batch has nothing to evaluate.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();

        // 3. Convert to DTensor and evaluate using evaluate_2d_real
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });
        let coeffs_2d_view = coeffs_2d.view(.., ..);
        let result_2d = self.fitter.evaluate_2d_real(backend, &coeffs_2d_view);

        // 4. Reshape back to N-D with n_points at position 0
        let n_points = self.n_sampling_points();
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });

        let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();

        // 5. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
    }

    /// Evaluate N-D coefficients for the complex case `T = Complex<f64>`
    fn evaluate_nd_impl_complex(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);

        assert_eq!(
            target_dim_size, basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim, target_dim_size, basis_size
        );

        if coeffs.is_empty() {
            // Zero-extent guard: an empty batch has nothing to evaluate.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();

        // 3. Convert to DTensor and evaluate using evaluate_2d
        let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });
        let coeffs_2d_view = coeffs_2d.view(.., ..);
        let result_2d = self.fitter.evaluate_2d(backend, &coeffs_2d_view);

        // 4. Reshape back to N-D with n_points at position 0
        let n_points = self.n_sampling_points();
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });

        let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();

        // 5. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
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
    /// # Example
    /// ```
    /// use num_complex::Complex;
    /// use sparse_ir::{DynRank, FermionicBasis, LogisticKernel, MatsubaraSampling, Tensor};
    ///
    /// let beta = 10.0;
    /// let wmax = 1.0;
    /// let basis = FermionicBasis::new(LogisticKernel::new(beta * wmax), beta, Some(1e-6), None);
    /// let sampling = MatsubaraSampling::new(&basis);
    /// let (size, n_points) = (sampling.basis_size(), sampling.n_sampling_points());
    ///
    /// // Real coefficients: two sets stacked along axis 1, evaluated along axis 0
    /// let coeffs_real = Tensor::<f64, DynRank>::from_fn(&[size, 2][..], |idx| {
    ///     1.0 / (1.0 + (idx[0] + idx[1]) as f64)
    /// });
    /// let values = sampling.evaluate_nd::<f64>(None, &coeffs_real, 0);
    /// assert_eq!(values.shape().dims(), &[n_points, 2]);
    ///
    /// // Complex coefficients
    /// let coeffs_complex =
    ///     Tensor::<Complex<f64>, DynRank>::from_fn(&[size, 2][..], |idx| {
    ///         Complex::new(coeffs_real[idx], -0.5 * coeffs_real[idx])
    ///     });
    /// let values_z = sampling.evaluate_nd::<Complex<f64>>(None, &coeffs_complex, 0);
    ///
    /// // Each column matches the 1-D `evaluate` of the corresponding coefficient set
    /// for j in 0..2 {
    ///     let real: Vec<Complex<f64>> = (0..size).map(|l| coeffs_real[&[l, j][..]].into()).collect();
    ///     let complex: Vec<Complex<f64>> = (0..size).map(|l| coeffs_complex[&[l, j][..]]).collect();
    ///     let (expected, expected_z) = (sampling.evaluate(&real), sampling.evaluate(&complex));
    ///     for i in 0..n_points {
    ///         assert!((values[&[i, j][..]] - expected[i]).norm() < 1e-12);
    ///         assert!((values_z[&[i, j][..]] - expected_z[i]).norm() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn evaluate_nd<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
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
    pub fn evaluate_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);

        assert_eq!(
            target_dim_size, basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim, target_dim_size, basis_size
        );

        if coeffs.is_empty() {
            // Zero-extent guard: an empty batch has nothing to evaluate.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();

        // 3. Convert to DTensor and evaluate using ComplexMatrixFitter
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // 4. Evaluate: values = A * coeffs (A is complex, coeffs is real)
        let coeffs_2d_view = coeffs_2d.view(.., ..);
        let values_2d = self.fitter.evaluate_2d_real(backend, &coeffs_2d_view);

        // 5. Reshape result back to N-D with first dimension = n_sampling_points
        let n_points = self.n_sampling_points();
        let mut result_shape = Vec::with_capacity(rank);
        result_shape.push(n_points);
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });

        let result_dim0 = values_2d.into_dyn().reshape(&result_shape[..]).to_tensor();

        // 6. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
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
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);

        assert_eq!(
            target_dim_size, n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, target_dim_size, n_points
        );

        if values.is_empty() {
            // Zero-extent guard: an empty batch has nothing to fit.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);

        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        let values_2d_dyn = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();

        // 3. Convert to DTensor and fit using GEMM
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
            values_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Use fitter's efficient 2D fit (GEMM-based)
        let values_2d_view = values_2d.view(.., ..);
        let coeffs_2d = self.fitter.fit_2d(backend, &values_2d_view);

        // 4. Reshape back to N-D with basis_size at position 0
        let basis_size = self.basis_size();
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
    pub fn fit_nd_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);

        assert_eq!(
            target_dim_size, n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, target_dim_size, n_points
        );

        if values.is_empty() {
            // Zero-extent guard: an empty batch has nothing to fit.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);

        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        let values_2d_dyn = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();

        // 3. Convert to DTensor and fit
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
            values_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Use fitter's fit_2d_real method
        let values_2d_view = values_2d.view(.., ..);
        let coeffs_2d = self.fitter.fit_2d_real(backend, &values_2d_view);

        // 4. Reshape back to N-D with basis_size at position 0
        let basis_size = self.basis_size();
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

    /// Evaluate basis coefficients at Matsubara sampling points (N-dimensional) with in-place output
    ///
    /// # Type Parameters
    /// * `T` - Coefficient type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points` (Complex<f64>)
    pub fn evaluate_nd_to<T: MatsubaraCoeffs>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
        out: &mut Tensor<Complex<f64>, DynRank>,
    ) {
        // Validate output shape
        let rank = coeffs.rank();
        assert_eq!(
            out.rank(),
            rank,
            "out.rank()={} must equal coeffs.rank()={}",
            out.rank(),
            rank
        );

        let n_points = self.n_sampling_points();
        let out_dim_size = out.shape().dim(dim);
        assert_eq!(
            out_dim_size, n_points,
            "out.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, out_dim_size, n_points
        );

        // Validate other dimensions match
        for d in 0..rank {
            if d != dim {
                let coeffs_d = coeffs.shape().dim(d);
                let out_d = out.shape().dim(d);
                assert_eq!(
                    coeffs_d, out_d,
                    "coeffs.shape().dim({}) = {} must equal out.shape().dim({}) = {}",
                    d, coeffs_d, d, out_d
                );
            }
        }

        // Compute result and copy to out
        let result = self.evaluate_nd(backend, coeffs, dim);

        // Copy result to out
        let total = out.len();
        for i in 0..total {
            let mut idx = vec![0usize; rank];
            let mut remaining = i;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            out[&idx[..]] = result[&idx[..]];
        }
    }

    /// Fit N-dimensional complex values to complex coefficients with in-place output
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == basis_size` (Complex<f64>)
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut Tensor<Complex<f64>, DynRank>,
    ) {
        // Validate output shape
        let rank = values.rank();
        assert_eq!(
            out.rank(),
            rank,
            "out.rank()={} must equal values.rank()={}",
            out.rank(),
            rank
        );

        let basis_size = self.basis_size();
        let out_dim_size = out.shape().dim(dim);
        assert_eq!(
            out_dim_size, basis_size,
            "out.shape().dim({}) = {} must equal basis_size = {}",
            dim, out_dim_size, basis_size
        );

        // Validate other dimensions match
        for d in 0..rank {
            if d != dim {
                let values_d = values.shape().dim(d);
                let out_d = out.shape().dim(d);
                assert_eq!(
                    values_d, out_d,
                    "values.shape().dim({}) = {} must equal out.shape().dim({}) = {}",
                    d, values_d, d, out_d
                );
            }
        }

        // Compute result and copy to out
        let result = self.fit_nd(backend, values, dim);

        // Copy result to out
        let total = out.len();
        for i in 0..total {
            let mut idx = vec![0usize; rank];
            let mut remaining = i;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            out[&idx[..]] = result[&idx[..]];
        }
    }
}

/// InplaceFitter implementation for MatsubaraSampling
///
/// Delegates to ComplexMatrixFitter which supports:
/// - zz: Complex input → Complex output (full support)
/// - dz: Real input → Complex output (evaluate only)
/// - zd: Complex input → Real output (fit only, takes real part)
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
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        self.fitter.fit_nd_zd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

/// Matsubara sampling for non-negative frequencies only
///
/// Exploits symmetry to reconstruct real coefficients from non-negative frequencies only.
/// Supports reduced frequencies n ≥ 0 (no negative frequencies)
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
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(true);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom positive-only sampling points
    ///
    /// The points may be in any order, and are kept in the given order:
    /// [`Self::sampling_points`] returns them unchanged, and index i along the
    /// sampling-point axis of `evaluate` and `fit` refers to
    /// `sampling_points[i]`.
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or a sampling point is negative
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        // With no points the sampling matrix would have no rows; building it
        // and the fitter's transposes would go through the zero-extent paths
        // of mdarray 0.7.2 (https://github.com/fre-hu/mdarray/issues/21).
        assert!(!sampling_points.is_empty(), "No sampling points given");

        // Validate that all points are non-negative
        assert!(
            sampling_points.iter().all(|f| f.n() >= 0),
            "All sampling points must be non-negative for positive-only Matsubara sampling"
        );

        // Evaluate matrix at sampling points
        // Use Basis trait's evaluate_matsubara method
        let matrix = basis.evaluate_matsubara(&sampling_points);

        // Create fitter (complex → real, exploits symmetry)
        let fitter = ComplexToRealFitter::new(&matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
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
    /// # Returns
    /// A new MatsubaraSamplingPositiveOnly object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty, if the number of matrix rows
    /// differs from the number of sampling points, or if the matrix has no
    /// columns (no basis functions)
    pub fn from_matrix(
        sampling_points: Vec<MatsubaraFreq<S>>,
        matrix: DTensor<Complex<f64>, 2>,
    ) -> Self {
        assert!(!sampling_points.is_empty(), "No sampling points given");
        assert_eq!(
            matrix.shape().0,
            sampling_points.len(),
            "Matrix rows ({}) must match number of sampling points ({})",
            matrix.shape().0,
            sampling_points.len()
        );
        // A matrix without columns would make the fitter transpose [n, 0]
        // arrays, which mdarray 0.7.2 does out of bounds (mdarray#21,
        // https://github.com/fre-hu/mdarray/issues/21); there is nothing to fit.
        assert!(
            matrix.shape().1 > 0,
            "Matrix must have at least one column (basis function), got shape {:?}",
            matrix.shape()
        );

        let fitter = ComplexToRealFitter::new(&matrix);

        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
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
    pub fn matrix(&self) -> &DTensor<Complex<f64>, 2> {
        &self.fitter.matrix
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
    pub fn condition_number(&self) -> f64 {
        self.fitter.condition_number()
    }

    /// Evaluate basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<Complex<f64>> {
        self.fitter.evaluate(None, coeffs)
    }

    /// Fit basis coefficients from values at sampling points
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<f64> {
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
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);

        assert_eq!(
            target_dim_size, basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim, target_dim_size, basis_size
        );

        if coeffs.is_empty() {
            // Zero-extent guard: an empty batch has nothing to evaluate.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(coeffs.shape(), dim, self.n_sampling_points());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();

        // 3. Convert to DTensor and evaluate using GEMM
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Use fitter's efficient 2D evaluate (GEMM-based)
        let coeffs_2d_view = coeffs_2d.view(.., ..);
        let result_2d = self.fitter.evaluate_2d(backend, &coeffs_2d_view);

        // 4. Reshape back to N-D with n_points at position 0
        let n_points = self.n_sampling_points();
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });

        let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();

        // 5. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
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
    pub fn fit_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);

        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);

        assert_eq!(
            target_dim_size, n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, target_dim_size, n_points
        );

        if values.is_empty() {
            // Zero-extent guard: an empty batch has nothing to fit.
            // Returning early keeps it away from the permuted copies that
            // mdarray 0.7.2 does out of bounds for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21) and from
            // zero-size GEMMs.
            let out_shape = build_output_shape(values.shape(), dim, self.basis_size());
            return Tensor::zeros(&out_shape[..]);
        }

        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);

        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        let values_2d_dyn = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();

        // 3. Convert to DTensor and fit using GEMM
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
            values_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Use fitter's efficient 2D fit (GEMM-based)
        let values_2d_view = values_2d.view(.., ..);
        let coeffs_2d = self.fitter.fit_2d(backend, &values_2d_view);

        // 4. Reshape back to N-D with basis_size at position 0
        let basis_size = self.basis_size();
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

    /// Evaluate real basis coefficients at Matsubara sampling points (N-dimensional) with in-place output
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of real coefficients with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points` (Complex<f64>)
    pub fn evaluate_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Tensor<f64, DynRank>,
        dim: usize,
        out: &mut Tensor<Complex<f64>, DynRank>,
    ) {
        // Validate output shape
        let rank = coeffs.rank();
        assert_eq!(
            out.rank(),
            rank,
            "out.rank()={} must equal coeffs.rank()={}",
            out.rank(),
            rank
        );

        let n_points = self.n_sampling_points();
        let out_dim_size = out.shape().dim(dim);
        assert_eq!(
            out_dim_size, n_points,
            "out.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim, out_dim_size, n_points
        );

        // Validate other dimensions match
        for d in 0..rank {
            if d != dim {
                let coeffs_d = coeffs.shape().dim(d);
                let out_d = out.shape().dim(d);
                assert_eq!(
                    coeffs_d, out_d,
                    "coeffs.shape().dim({}) = {} must equal out.shape().dim({}) = {}",
                    d, coeffs_d, d, out_d
                );
            }
        }

        // Compute result and copy to out
        let result = self.evaluate_nd(backend, coeffs, dim);

        // Copy result to out
        let total = out.len();
        for i in 0..total {
            let mut idx = vec![0usize; rank];
            let mut remaining = i;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            out[&idx[..]] = result[&idx[..]];
        }
    }

    /// Fit N-dimensional complex values to real coefficients with in-place output
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == basis_size` (f64)
    pub fn fit_nd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut Tensor<f64, DynRank>,
    ) {
        // Validate output shape
        let rank = values.rank();
        assert_eq!(
            out.rank(),
            rank,
            "out.rank()={} must equal values.rank()={}",
            out.rank(),
            rank
        );

        let basis_size = self.basis_size();
        let out_dim_size = out.shape().dim(dim);
        assert_eq!(
            out_dim_size, basis_size,
            "out.shape().dim({}) = {} must equal basis_size = {}",
            dim, out_dim_size, basis_size
        );

        // Validate other dimensions match
        for d in 0..rank {
            if d != dim {
                let values_d = values.shape().dim(d);
                let out_d = out.shape().dim(d);
                assert_eq!(
                    values_d, out_d,
                    "values.shape().dim({}) = {} must equal out.shape().dim({}) = {}",
                    d, values_d, d, out_d
                );
            }
        }

        // Compute result and copy to out
        let result = self.fit_nd(backend, values, dim);

        // Copy result to out
        let total = out.len();
        for i in 0..total {
            let mut idx = vec![0usize; rank];
            let mut remaining = i;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            out[&idx[..]] = result[&idx[..]];
        }
    }
}

/// InplaceFitter implementation for MatsubaraSamplingPositiveOnly
///
/// Delegates to ComplexToRealFitter which supports:
/// - dz: Real coefficients → Complex values (evaluate)
/// - zz: Complex coefficients → Complex values (evaluate, extracts real parts)
/// - zd: Complex values → Real coefficients (fit)
/// - zz: Complex values → Complex coefficients (fit, with zero imaginary parts)
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
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.evaluate_nd_dz_to(backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.evaluate_nd_zz_to(backend, coeffs, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        self.fitter.fit_nd_zd_to(backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        self.fitter.fit_nd_zz_to(backend, values, dim, out)
    }
}

#[cfg(test)]
#[path = "matsubara_sampling_tests.rs"]
mod tests;
