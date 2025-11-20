//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::fitter::{ComplexMatrixFitter, ComplexToRealFitter};
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Tensor};
use num_complex::Complex;
use std::marker::PhantomData;

/// Move axis from position src to position dst
fn movedim<T: Clone>(arr: &Tensor<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    if src == dst {
        return arr.clone();
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
            if pos == src {
                pos += 1;
            }
            if pos < rank {
                perm.push(pos);
                pos += 1;
            }
        }
    }

    arr.permute(&perm[..]).to_tensor()
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
    /// Uses extrema-based sampling point selection (symmetric: positive and negative frequencies).
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(false);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom sampling points
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        // Sort sampling points
        sampling_points.sort();

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
    /// * `sampling_points` - Matsubara frequency sampling points
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size)
    ///
    /// # Returns
    /// A new MatsubaraSampling object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if matrix dimensions don't match
    pub fn from_matrix(
        mut sampling_points: Vec<MatsubaraFreq<S>>,
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

        // Sort sampling points
        sampling_points.sort();

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

    /// Evaluate N-dimensional array of complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of complex basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    pub fn evaluate_nd(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Tensor<Complex<f64>, DynRank>,
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

        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);

        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;

        let coeffs_2d_dyn = coeffs_dim0
            .reshape(&[basis_size, extra_size][..])
            .to_tensor();

        // 3. Convert to DTensor and evaluate using GEMM
        let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Use fitter's efficient 2D evaluate (GEMM-based)
        let n_points = self.n_sampling_points();
        let result_2d = self.fitter.evaluate_2d(backend, &coeffs_2d);

        // 4. Reshape back to N-D with n_points at position 0
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
        let values_2d = self.fitter.evaluate_2d_real(backend, &coeffs_2d);

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
        let coeffs_2d = self.fitter.fit_2d(backend, &values_2d);

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
        let coeffs_2d = self.fitter.fit_2d_real(backend, &values_2d);

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
}

/// Matsubara sampling for positive frequencies only
///
/// Exploits symmetry to reconstruct real coefficients from positive frequencies only.
/// Supports: {0, 1, 2, 3, ...} (no negative frequencies)
pub struct MatsubaraSamplingPositiveOnly<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexToRealFitter,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> MatsubaraSamplingPositiveOnly<S> {
    /// Create Matsubara sampling with default positive-only sampling points
    ///
    /// Uses extrema-based sampling point selection (positive frequencies only).
    /// Exploits symmetry to reconstruct real coefficients.
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let sampling_points = basis.default_matsubara_sampling_points(true);
        Self::with_sampling_points(basis, sampling_points)
    }

    /// Create Matsubara sampling with custom positive-only sampling points
    pub fn with_sampling_points(
        basis: &impl crate::basis_trait::Basis<S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        S: 'static,
    {
        // Sort and validate (all n >= 0)
        sampling_points.sort();

        // TODO: Validate that all points are non-negative

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
    /// Uses symmetry to fit real coefficients from complex values at positive frequencies.
    ///
    /// # Arguments
    /// * `sampling_points` - Matsubara frequency sampling points (should be positive)
    /// * `matrix` - Pre-computed sampling matrix (n_points × basis_size)
    ///
    /// # Returns
    /// A new MatsubaraSamplingPositiveOnly object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if matrix dimensions don't match
    pub fn from_matrix(
        mut sampling_points: Vec<MatsubaraFreq<S>>,
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

        // Sort sampling points
        sampling_points.sort();

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
        let result_2d = self.fitter.evaluate_2d(backend, &coeffs_2d);

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
        let coeffs_2d = self.fitter.fit_2d(backend, &values_2d);

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
}

#[cfg(test)]
#[path = "matsubara_sampling_tests.rs"]
mod tests;
