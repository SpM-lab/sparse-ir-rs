//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::fitter::{ComplexMatrixFitter, ComplexToRealFitter};
use crate::fpu_check::FpuGuard;
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::traits::StatisticsType;
use mdarray::{DTensor, DynRank, Shape, Slice, Tensor};
use num_complex::Complex;
use std::marker::PhantomData;

/// Move axis from position src to position dst
fn movedim<T: Clone>(arr: &Slice<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
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

    /// Get the sampling matrix
    pub fn matrix(&self) -> &DTensor<Complex<f64>, 2> {
        &self.fitter.matrix
    }

    /// Evaluate complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Complex basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at Matsubara frequencies (length = n_sampling_points)
    pub fn evaluate(&self, coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.fit(None, values)
    }

    /// Evaluate N-dimensional array of basis coefficients at sampling points
    ///
    /// Supports both real (`f64`) and complex (`Complex<f64>`) coefficients.
    /// Always returns complex values at Matsubara frequencies.
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend handle (None uses default)
    /// * `coeffs` - N-dimensional tensor of basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    ///
    /// # Example
    /// ```ignore
    /// use num_complex::Complex;
    ///
    /// // Real coefficients
    /// let values = matsubara_sampling.evaluate_nd::<f64>(None, &coeffs_real, 0);
    ///
    /// // Complex coefficients
    /// let values = matsubara_sampling.evaluate_nd::<Complex<f64>>(None, &coeffs_complex, 0);
    /// ```
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

    pub fn evaluate_nd<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank>
    where
        T: Copy + 'static,
    {
        let _guard = FpuGuard::new_protect_computation();
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Safe: TypeId check ensures T == f64 at runtime
            // We need unsafe because Rust can't statically prove this
            let coeffs_f64 =
                unsafe { &*(coeffs as *const Slice<T, DynRank> as *const Slice<f64, DynRank>) };
            self.evaluate_nd_impl_real(backend, coeffs_f64, dim)
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            // Safe: TypeId check ensures T == Complex<f64> at runtime
            // We need unsafe because Rust can't statically prove this
            let coeffs_complex = unsafe {
                &*(coeffs as *const Slice<T, DynRank> as *const Slice<Complex<f64>, DynRank>)
            };
            self.evaluate_nd_impl_complex(backend, coeffs_complex, dim)
        } else {
            panic!("Unsupported type for evaluate_nd: must be f64 or Complex<f64>");
        }
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
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();
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
    pub fn evaluate_nd_to<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
        out: &mut Tensor<Complex<f64>, DynRank>,
    ) where
        T: Copy + 'static,
    {
        let _guard = FpuGuard::new_protect_computation();

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
        let _guard = FpuGuard::new_protect_computation();

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

    /// Get the original complex sampling matrix
    pub fn matrix(&self) -> &DTensor<Complex<f64>, 2> {
        &self.fitter.matrix
    }

    /// Evaluate basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<Complex<f64>> {
        let _guard = FpuGuard::new_protect_computation();
        self.fitter.evaluate(None, coeffs)
    }

    /// Fit basis coefficients from values at sampling points
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<f64> {
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();

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
        let _guard = FpuGuard::new_protect_computation();

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

#[cfg(test)]
#[path = "matsubara_sampling_tests.rs"]
mod tests;
