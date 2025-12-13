//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::fpu_check::FpuGuard;
use crate::gemm::{GemmBackendHandle, matmul_par};
use crate::traits::StatisticsType;
use crate::working_buffer::{copy_from_contiguous, copy_to_contiguous};
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

/// Generate permutation to move dimension `dim` to position 0
///
/// For example, with rank=4 and dim=2:
/// - Result: [2, 0, 1, 3]
fn make_perm_to_front(rank: usize, dim: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(dim);
    for i in 0..rank {
        if i != dim {
            perm.push(i);
        }
    }
    perm
}

/// Generate permutation to move dimension 0 back to position `dim`
///
/// This is the inverse of `make_perm_to_front`.
/// For example, with rank=4 and dim=2:
/// - Result: [1, 2, 0, 3]
fn make_perm_from_front(rank: usize, dim: usize) -> Vec<usize> {
    let mut perm = vec![0; rank];
    perm[dim] = 0;
    let mut pos = 0;
    for i in 0..rank {
        if i != dim {
            if pos == 0 {
                pos = 1;
            }
            perm[i] = pos;
            pos += 1;
        }
    }
    perm
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
        let _guard = FpuGuard::new_protect_computation();
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
        let _guard = FpuGuard::new_protect_computation();
        self.evaluate_nd_impl(backend, coeffs, dim)
    }

    /// Evaluate basis coefficients at sampling points (N-dimensional) with in-place output
    ///
    /// Evaluates along the specified dimension, keeping other dimensions intact.
    /// Writes the result directly to the output tensor.
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points`
    ///
    /// # Note
    /// This version writes directly to `out`, reducing final copy overhead.
    /// Internal allocations for dimension permutation are still present.
    pub fn evaluate_nd_to<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<T, DynRank>,
        dim: usize,
        out: &mut Tensor<T, DynRank>,
    ) where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + 'static + From<f64> + Copy,
    {
        let _guard = FpuGuard::new_protect_computation();

        // Validate output shape
        let rank = coeffs.rank();
        assert_eq!(out.rank(), rank, "out.rank()={} must equal coeffs.rank()={}", out.rank(), rank);

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
        let result = self.evaluate_nd_impl(backend, coeffs, dim);

        // Copy result to out (flat iteration)
        let total = out.len();
        for i in 0..total {
            // Convert flat index to multi-dim index
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

    /// Evaluate basis coefficients at sampling points with in-place output
    ///
    /// Optimized version that uses permuted views and temporary buffers
    /// to avoid unnecessary copies where possible.
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == n_sampling_points`
    ///
    /// # Performance
    /// - When `dim == 0`: Fast path, minimal overhead
    /// - Otherwise: Uses temporary buffers for dimension permutation
    pub fn evaluate_nd_inplace(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut Tensor<f64, DynRank>,
    ) {
        let _guard = FpuGuard::new_protect_computation();

        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_sampling_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        let total = coeffs.len();
        let extra_size = total / basis_size;

        // Step 1: Copy permuted input to contiguous buffer (uninit allocation)
        let perm = make_perm_to_front(rank, dim);
        let permuted_view = coeffs.permute(&perm[..]);

        // Allocate uninit buffer and copy
        let mut input_buffer: Vec<f64> = Vec::with_capacity(total);
        unsafe { input_buffer.set_len(total); }
        copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

        // Step 2: Create DTensor from buffer for GEMM
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            input_buffer[idx[0] * extra_size + idx[1]]
        });

        // Step 3: Matrix multiply
        let result_2d = matmul_par(&self.fitter.matrix, &coeffs_2d, backend);

        // Step 4: Copy result to output buffer (uninit allocation)
        let out_total = out.len();
        let mut output_buffer: Vec<f64> = Vec::with_capacity(out_total);
        unsafe { output_buffer.set_len(out_total); }
        for i in 0..n_points {
            for j in 0..extra_size {
                output_buffer[i * extra_size + j] = result_2d[[i, j]];
            }
        }

        // Step 5: Copy from contiguous buffer to strided output view (inverse permutation)
        let inv_perm = make_perm_from_front(rank, dim);
        let mut out_permuted = out.permute_mut(&inv_perm[..]);
        copy_from_contiguous(&output_buffer, &mut out_permuted.into_dyn());
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
        let _guard = FpuGuard::new_protect_computation();
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

    /// Fit basis coefficients from values at sampling points (N-dimensional) with in-place output
    ///
    /// Fits along the specified dimension, keeping other dimensions intact.
    /// Writes the result directly to the output tensor.
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output tensor with `out.shape().dim(dim) == basis_size`
    ///
    /// # Note
    /// This version writes directly to `out`, reducing final copy overhead.
    /// Internal allocations for dimension permutation are still present.
    pub fn fit_nd_to<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Tensor<T, DynRank>,
        dim: usize,
        out: &mut Tensor<T, DynRank>,
    ) where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + 'static
            + From<f64>
            + Copy
            + Default,
    {
        let _guard = FpuGuard::new_protect_computation();

        // Validate output shape
        let rank = values.rank();
        assert_eq!(out.rank(), rank, "out.rank()={} must equal values.rank()={}", out.rank(), rank);

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

        // Compute result using existing fit_nd and copy to out
        let result = self.fit_nd(backend, values, dim);

        // Copy result to out (flat iteration)
        let total = out.len();
        for i in 0..total {
            // Convert flat index to multi-dim index
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
#[path = "tau_sampling_tests.rs"]
mod tests;
