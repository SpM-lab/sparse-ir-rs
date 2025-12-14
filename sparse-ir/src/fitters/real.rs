//! Real matrix fitter: A ∈ R^{n×m}
//!
//! This module provides `RealMatrixFitter` for solving least-squares problems
//! where the matrix, coefficients, and values are all real.

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView, DynRank, Shape, Slice, ViewMut};
use num_complex::Complex;
use std::sync::OnceLock;

use super::common::{
    RealSVD, complex_slice_as_real, compute_real_svd, copy_from_contiguous, copy_to_contiguous,
    make_perm_to_front,
};

/// Fitter for real matrix: A ∈ R^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all real
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| ...);
/// let fitter = RealMatrixFitter::new(matrix);
///
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = fitter.evaluate(&coeffs);
/// let fitted_coeffs = fitter.fit(&values);
/// ```
/// Fitter for real matrix: A ∈ R^{n×m}
///
/// This type is thread-safe and can be shared across threads.
/// The SVD decomposition is computed lazily on first use.
pub(crate) struct RealMatrixFitter {
    pub matrix: DTensor<f64, 2>, // (n_points, basis_size)
    svd: OnceLock<RealSVD>,
}

impl RealMatrixFitter {
    /// Create a new fitter with the given real matrix
    pub fn new(matrix: DTensor<f64, 2>) -> Self {
        Self {
            matrix,
            svd: OnceLock::new(),
        }
    }

    /// Get SVD decomposition, computing it lazily if needed
    fn get_svd(&self) -> &RealSVD {
        self.svd.get_or_init(|| compute_real_svd(&self.matrix))
    }

    /// Number of data points
    pub fn n_points(&self) -> usize {
        self.matrix.shape().0
    }

    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix.shape().1
    }

    /// Evaluate: coeffs (real) → values (real)
    ///
    /// Computes: values = A * coeffs
    pub fn evaluate(&self, backend: Option<&GemmBackendHandle>, coeffs: &[f64]) -> Vec<f64> {
        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = vec![0.0; n_points];
        self.evaluate_to(backend, coeffs, &mut out);
        out
    }

    /// Evaluate: coeffs (real) → values (real), writing to output slice
    ///
    /// Computes: out = A * coeffs
    pub fn evaluate_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
        out: &mut [f64],
    ) {
        assert_eq!(coeffs.len(), self.basis_size());
        assert_eq!(out.len(), self.n_points());

        // Create views treating slices as column vectors [N, 1]
        let coeffs_view = unsafe {
            let mapping = mdarray::DenseMapping::new((coeffs.len(), 1));
            mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.evaluate_2d_dd_to(backend, &coeffs_view, &mut out_view);
    }

    /// Fit: values (real) → coeffs (real)
    ///
    /// Solves: min ||A * coeffs - values||^2 using SVD
    #[allow(dead_code)]
    pub fn fit(&self, backend: Option<&GemmBackendHandle>, values: &[f64]) -> Vec<f64> {
        assert_eq!(
            values.len(),
            self.n_points(),
            "values.len()={} must equal n_points={}",
            values.len(),
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = vec![0.0; basis_size];
        self.fit_to(backend, values, &mut out);
        out
    }

    /// Fit: values (real) → coeffs (real), writing to output slice
    ///
    /// Solves: min ||A * coeffs - values||^2 using SVD
    pub fn fit_to(&self, backend: Option<&GemmBackendHandle>, values: &[f64], out: &mut [f64]) {
        assert_eq!(values.len(), self.n_points());
        assert_eq!(out.len(), self.basis_size());

        // Create views treating slices as column vectors [N, 1]
        let values_view = unsafe {
            let mapping = mdarray::DenseMapping::new((values.len(), 1));
            mdarray::DView::<'_, f64, 2>::new_unchecked(values.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.fit_2d_dd_to(backend, &values_view, &mut out_view);
    }

    /// Fit complex values by fitting real and imaginary parts separately
    ///
    /// # Arguments
    /// * `values` - Complex values at sampling points (length = n_points)
    ///
    /// # Returns
    /// Complex coefficients (length = basis_size)
    #[allow(dead_code)]
    pub fn fit_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        assert_eq!(
            values.len(),
            self.n_points(),
            "values.len()={} must equal n_points={}",
            values.len(),
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = vec![Complex::new(0.0, 0.0); basis_size];
        self.fit_zz_to(backend, values, &mut out);
        out
    }

    /// Fit complex values, writing to output slice
    pub fn fit_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        assert_eq!(values.len(), self.n_points());
        assert_eq!(out.len(), self.basis_size());

        // Create views treating slices as column vectors [N, 1]
        let values_view = unsafe {
            let mapping = mdarray::DenseMapping::new((values.len(), 1));
            mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.fit_2d_zz_to(backend, &values_view, &mut out_view);
    }

    /// Evaluate complex coefficients
    ///
    /// # Arguments
    /// * `coeffs` - Complex coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at sampling points (length = n_points)
    #[allow(dead_code)]
    pub fn evaluate_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = vec![Complex::new(0.0, 0.0); n_points];
        self.evaluate_zz_to(backend, coeffs, &mut out);
        out
    }

    /// Evaluate complex coefficients, writing to output slice
    pub fn evaluate_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        assert_eq!(coeffs.len(), self.basis_size());
        assert_eq!(out.len(), self.n_points());

        // Create views treating slices as column vectors [N, 1]
        let coeffs_view = unsafe {
            let mapping = mdarray::DenseMapping::new((coeffs.len(), 1));
            mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(coeffs.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.evaluate_2d_zz_to(backend, &coeffs_view, &mut out_view);
    }

    /// Evaluate 2D real tensor (along dim=0) using matrix multiplication
    ///
    /// Computes: values_2d = matrix * coeffs_2d
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    ///
    /// # Note
    /// This is a wrapper around `evaluate_2d_to` that allocates the output tensor.
    pub fn evaluate_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
    ) -> mdarray::DTensor<f64, 2> {
        let (basis_size, extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = mdarray::DTensor::<f64, 2>::zeros([n_points, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.evaluate_2d_dd_to(backend, coeffs_2d, &mut out_view);
        out
    }

    /// Evaluate 2D real tensor (along dim=0) with in-place output to mutable view
    ///
    /// Computes: out = matrix * coeffs_2d
    ///
    /// This version writes directly to a mutable view, enabling zero-copy
    /// writes to pre-allocated buffers (e.g., C pointers via FFI).
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output mutable view, shape: [n_points, extra_size] (will be overwritten)
    ///
    /// # Safety
    /// The output view must have the correct shape and be contiguous.
    pub fn evaluate_2d_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
    ) {
        self.evaluate_2d_to_dim(backend, coeffs_2d, out, 0);
    }

    /// Evaluate 2D real tensor along specified dimension, writing to a mutable view
    ///
    /// # Arguments
    /// * `coeffs_2d` - Coefficient view
    /// * `out` - Output mutable view (will be overwritten)
    /// * `dim` - Target dimension (0 or 1)
    ///   - dim=0: coeffs[basis_size, extra], out[n_points, extra], computes out = matrix * coeffs
    ///   - dim=1: coeffs[extra, basis_size], out[extra, n_points], computes out = coeffs * matrix^T
    ///
    /// # Safety
    /// The output view must have the correct shape and be contiguous.
    fn evaluate_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
        dim: usize,
    ) {
        use crate::gemm::matmul_par_to_viewmut;

        let (coeffs_rows, coeffs_cols) = *coeffs_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        if dim == 0 {
            // out[n_points, extra] = matrix[n_points, basis_size] * coeffs[basis_size, extra]
            let basis_size = coeffs_rows;
            let extra_size = coeffs_cols;

            assert_eq!(
                basis_size,
                self.basis_size(),
                "coeffs_2d.shape().0={} must equal basis_size={}",
                basis_size,
                self.basis_size()
            );
            assert_eq!(
                out_rows,
                self.n_points(),
                "out.shape().0={} must equal n_points={}",
                out_rows,
                self.n_points()
            );
            assert_eq!(
                out_cols, extra_size,
                "out.shape().1={} must equal extra_size={}",
                out_cols, extra_size
            );

            let matrix_view = self.matrix.view(.., ..);
            matmul_par_to_viewmut(&matrix_view, coeffs_2d, out, backend);
        } else {
            // dim == 1
            // out[extra, n_points] = coeffs[extra, basis_size] * matrix^T[basis_size, n_points]
            let extra_size = coeffs_rows;
            let basis_size = coeffs_cols;

            assert_eq!(
                basis_size,
                self.basis_size(),
                "coeffs_2d.shape().1={} must equal basis_size={}",
                basis_size,
                self.basis_size()
            );
            assert_eq!(
                out_rows, extra_size,
                "out.shape().0={} must equal extra_size={}",
                out_rows, extra_size
            );
            assert_eq!(
                out_cols,
                self.n_points(),
                "out.shape().1={} must equal n_points={}",
                out_cols,
                self.n_points()
            );

            // Create transposed view of matrix
            let matrix_t = self.matrix.permute([1, 0]);
            // Need to convert to DTensor for matmul (permuted view is strided)
            let matrix_t_tensor = matrix_t.to_tensor();
            let matrix_t_view = matrix_t_tensor.view(.., ..);
            matmul_par_to_viewmut(coeffs_2d, &matrix_t_view, out, backend);
        }
    }

    /// Evaluate N-D real tensor along specified dimension, writing to a mutable view
    ///
    /// Optimized version that uses permuted views and temporary buffers
    /// to avoid unnecessary copies where possible.
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output mutable view with `out.shape().dim(dim) == n_points`
    ///
    /// # Performance
    /// - When `dim == 0`: Fast path, no temporary buffers needed
    /// - When `dim == rank - 1`: Fast path, no temporary buffers needed
    /// - Otherwise: Uses temporary buffers for dimension permutation
    pub fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        let total = coeffs.len();
        let extra_size = total / basis_size;

        if dim == 0 {
            // Fast path 1: dim == 0, no movedim needed
            // coeffs is [basis_size, ...], out is [n_points, ...]
            // Row-major contiguous, treat as 2D: coeffs[basis_size, extra], out[n_points, extra]
            // Compute: out = matrix * coeffs

            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1, no movedim needed
            // coeffs is [..., basis_size], out is [..., n_points]
            // Row-major contiguous, treat as 2D: coeffs[extra, basis_size], out[extra, n_points]
            // Compute: out = coeffs * matrix^T

            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 1);
        } else {
            // General path: batched GEMM approach
            // For shape [A, basis, B, ...] with dim=d, we batch over dimensions before dim
            // and treat dimensions after dim as the "extra" dimension for each GEMM
            self.evaluate_nd_dd_to_batched(backend, coeffs, dim, out);
        }
        true
    }

    /// Batched GEMM implementation for middle dimensions
    /// For shape [batch_dims..., basis, extra_dims...] with target dim in the middle,
    /// we iterate over batch_dims and call GEMM for each batch.
    fn evaluate_nd_dd_to_batched(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Calculate batch size (product of dims before target dim)
        // and extra size (product of dims after target dim)
        let mut batch_size = 1usize;
        let mut extra_size = 1usize;
        coeffs.shape().with_dims(|dims| {
            for i in 0..dim {
                batch_size *= dims[i];
            }
            for i in (dim + 1)..rank {
                extra_size *= dims[i];
            }
        });

        // Strides in the flattened array
        let coeffs_batch_stride = basis_size * extra_size;
        let out_batch_stride = n_points * extra_size;

        // For row-major layout:
        // coeffs[b, l, e] = coeffs_ptr[b * batch_stride + l * extra_size + e]
        // Each batch slice coeffs[b, :, :] is contiguous with shape [basis_size, extra_size]
        let coeffs_ptr = coeffs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for b in 0..batch_size {
            // Create view for this batch
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(
                    coeffs_ptr.add(b * coeffs_batch_stride),
                    mapping,
                )
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(
                    out_ptr.add(b * out_batch_stride),
                    mapping,
                )
            };

            // GEMM: out[p, e] = sum_l matrix[p, l] * coeffs[l, e]
            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);
        }
    }

    /// Fit 2D real tensor (along dim=0) using matrix multiplication
    ///
    /// Efficiently computes: coeffs_2d = V * S^{-1} * U^T * values_2d
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Coefficients tensor, shape: [basis_size, extra_size]
    ///
    /// # Note
    /// This is a wrapper around `fit_2d_to` that allocates the output tensor.
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, f64, 2>,
    ) -> mdarray::DTensor<f64, 2> {
        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = mdarray::DTensor::<f64, 2>::zeros([basis_size, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.fit_2d_dd_to(backend, values_2d, &mut out_view);
        out
    }

    /// Fit 2D real tensor along specified dimension, writing to a mutable view
    ///
    /// # Arguments
    /// * `values_2d` - Coefficient view
    /// * `out` - Output mutable view (will be overwritten)
    /// * `dim` - Target dimension (0 or 1)
    ///   - dim=0: values[n_points, extra], out[basis_size, extra], computes out = V * S^{-1} * U^T * values
    ///   - dim=1: values[extra, n_points], out[extra, basis_size], computes out = (V * S^{-1} * U^T * values^T)^T
    ///
    /// # Safety
    /// The output view must have the correct shape and be contiguous.
    fn fit_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
        dim: usize,
    ) {
        use crate::gemm::{matmul_par_to_viewmut, matmul_par_view};

        let (values_rows, values_cols) = *values_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        if dim == 0 {
            // out[basis_size, extra] = V * S^{-1} * U^T * values[n_points, extra]
            let n_points = values_rows;
            let extra_size = values_cols;

            assert_eq!(
                n_points,
                self.n_points(),
                "values_2d.shape().0={} must equal n_points={}",
                n_points,
                self.n_points()
            );
            assert_eq!(
                out_rows,
                self.basis_size(),
                "out.shape().0={} must equal basis_size={}",
                out_rows,
                self.basis_size()
            );
            assert_eq!(
                out_cols, extra_size,
                "out.shape().1={} must equal extra_size={}",
                out_cols, extra_size
            );

            // Get SVD (computed lazily on first access)
            let svd = self.get_svd();

            // out = V * S^{-1} * U^T * values_2d

            // 1. U^T * values_2d (intermediate allocation needed)
            let ut_view = svd.ut.view(.., ..);
            let mut ut_values = matmul_par_view(&ut_view, values_2d, backend); // [min_dim, extra_size]

            // 2. S^{-1} * (U^T * values_2d) - in-place division
            let min_dim = svd.s.len();
            for i in 0..min_dim {
                for j in 0..extra_size {
                    ut_values[[i, j]] /= svd.s[i];
                }
            }

            // 3. V * (S^{-1} * U^T * values_2d) → out
            let v_view = svd.v.view(.., ..);
            let ut_values_view = ut_values.view(.., ..);
            matmul_par_to_viewmut(&v_view, &ut_values_view, out, backend);
        } else {
            // dim == 1
            // out[extra, basis_size] = (V * S^{-1} * U^T * values^T[extra, n_points])^T
            let extra_size = values_rows;
            let n_points = values_cols;

            assert_eq!(
                n_points,
                self.n_points(),
                "values_2d.shape().1={} must equal n_points={}",
                n_points,
                self.n_points()
            );
            assert_eq!(
                out_rows, extra_size,
                "out.shape().0={} must equal extra_size={}",
                out_rows, extra_size
            );
            assert_eq!(
                out_cols,
                self.basis_size(),
                "out.shape().1={} must equal basis_size={}",
                out_cols,
                self.basis_size()
            );

            // Transpose values: values[extra, n_points] -> values_t[n_points, extra]
            let mut values_t = mdarray::DTensor::<f64, 2>::zeros([n_points, extra_size]);
            let mut values_t_view = values_t.view_mut(.., ..);
            for i in 0..n_points {
                for j in 0..extra_size {
                    values_t_view[[i, j]] = values_2d[[j, i]];
                }
            }

            // Fit transposed values: coeffs_t[basis_size, extra] = V * S^{-1} * U^T * values_t[n_points, extra]
            let mut coeffs_t = mdarray::DTensor::<f64, 2>::zeros([self.basis_size(), extra_size]);
            let mut coeffs_t_view = coeffs_t.view_mut(.., ..);
            self.fit_2d_to_dim(backend, &values_t.view(.., ..), &mut coeffs_t_view, 0);

            // Transpose result back: out[extra, basis_size] = coeffs_t^T[basis_size, extra]
            let basis_size = self.basis_size();
            for i in 0..basis_size {
                for j in 0..extra_size {
                    out[[j, i]] = coeffs_t[[i, j]];
                }
            }
        }
    }

    /// Fit 2D real tensor (along dim=0) with in-place output to mutable view
    ///
    /// Efficiently computes: out = V * S^{-1} * U^T * values_2d
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output mutable view, shape: [basis_size, extra_size] (will be overwritten)
    ///
    /// # Note
    /// An intermediate buffer of size [min_dim, extra_size] is still allocated internally.
    pub fn fit_2d_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
    ) {
        self.fit_2d_to_dim(backend, values_2d, out, 0);
    }

    /// Fit N-D real tensor along specified dimension, writing to a mutable view
    ///
    /// Optimized version that uses permuted views and temporary buffers
    /// to avoid unnecessary copies where possible.
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output mutable view with `out.shape().dim(dim) == basis_size`
    ///
    /// # Performance
    /// - When `dim == 0`: Fast path, no temporary buffers needed
    /// - When `dim == rank - 1`: Fast path, no temporary buffers needed
    /// - Otherwise: Uses temporary buffers for dimension permutation
    pub fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        let rank = values.rank();
        let n_points = self.n_points();
        let basis_size = self.basis_size();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(values.shape().dim(dim), n_points);
        assert_eq!(out.shape().dim(dim), basis_size);

        let total = values.len();
        let extra_size = total / n_points;

        if dim == 0 {
            // Fast path 1: dim == 0, no movedim needed
            // values is [n_points, ...], out is [basis_size, ...]
            // Row-major contiguous, treat as 2D: values[n_points, extra], out[basis_size, extra]
            // Compute: out = V * S^{-1} * U^T * values

            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_dd_to(backend, &values_2d, &mut out_2d);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1, no movedim needed
            // values is [..., n_points], out is [..., basis_size]
            // Row-major contiguous, treat as 2D: values[extra, n_points], out[extra, basis_size]
            // Use fit_2d_to_dim with dim=1

            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DView::<'_, f64, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 1);
        } else {
            // General path: batched GEMM approach
            // For shape [batch_dims..., n_points, extra_dims...] with target dim in the middle,
            // we iterate over batch_dims and call GEMM for each batch.
            self.fit_nd_dd_to_batched(backend, values, dim, out);
        }
        true
    }

    /// Batched GEMM implementation for fit with middle dimensions
    fn fit_nd_dd_to_batched(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
        let rank = values.rank();
        let n_points = self.n_points();
        let basis_size = self.basis_size();

        // Calculate batch size (product of dims before target dim)
        // and extra size (product of dims after target dim)
        let mut batch_size = 1usize;
        let mut extra_size = 1usize;
        values.shape().with_dims(|dims| {
            for i in 0..dim {
                batch_size *= dims[i];
            }
            for i in (dim + 1)..rank {
                extra_size *= dims[i];
            }
        });

        // Strides in the flattened array
        let values_batch_stride = n_points * extra_size;
        let out_batch_stride = basis_size * extra_size;

        let values_ptr = values.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for b in 0..batch_size {
            // Create view for this batch
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(
                    values_ptr.add(b * values_batch_stride),
                    mapping,
                )
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(
                    out_ptr.add(b * out_batch_stride),
                    mapping,
                )
            };

            // GEMM: out[l, e] = fit_matrix * values[:, e]
            self.fit_2d_dd_to(backend, &values_2d, &mut out_2d);
        }
    }

    /// Fit 2D complex tensor (along dim=0) using matrix multiplication
    ///
    /// Fits real and imaginary parts separately, then combines.
    /// Efficiently computes using GEMM operations.
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Complex coefficients tensor, shape: [basis_size, extra_size]
    pub fn fit_2d_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> mdarray::DTensor<Complex<f64>, 2> {
        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = mdarray::DTensor::<Complex<f64>, 2>::zeros([basis_size, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.fit_2d_zz_to(backend, values_2d, &mut out_view);
        out
    }

    /// Fit 2D complex values with in-place output to mutable view
    ///
    /// Uses the memory layout of Complex<f64> (re, im are contiguous) to delegate
    /// to the real version by treating complex array [n_points, extra_size] as
    /// real array [n_points, extra_size, 2].
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output mutable view, shape: [basis_size, extra_size] (will be overwritten)
    pub fn fit_2d_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        // Convert to DynRank and delegate to ND version
        let values_dyn = values_2d.into_dyn();
        let mut out_dyn = out.expr_mut().into_dyn();
        self.fit_nd_zz_to(backend, &values_dyn, 0, &mut out_dyn);
    }

    /// Fit N-D complex tensor along specified dimension, writing to a mutable view
    ///
    /// Uses the memory layout of Complex<f64> (re, im are contiguous) to delegate
    /// to the real version by treating complex array [..., dN] as real array [..., dN, 2].
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend
    /// * `values` - N-dimensional complex array with `values.shape().dim(dim) == n_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    /// * `out` - Output mutable view with `out.shape().dim(dim) == basis_size`
    ///
    /// # Performance
    /// For dim==0 with contiguous arrays, uses zero-copy reinterpretation.
    /// Otherwise, uses temporary buffers.
    pub fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let rank = values.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(values.shape().dim(dim), n_points);
        assert_eq!(out.shape().dim(dim), basis_size);

        let total = values.len();
        let extra_size = total / n_points;

        if dim == 0 {
            // Fast path: dim == 0, zero-copy reinterpretation
            // Complex<f64> has layout [re, im], so we can treat [n_points, extra_size] complex
            // as [n_points, extra_size * 2] real in 2D

            // Create view of input as real: [n_points, extra_size * 2]
            let values_2d_real = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size * 2));
                mdarray::DView::<'_, f64, 2>::new_unchecked(values.as_ptr() as *const f64, mapping)
            };

            // Create view of output as real: [basis_size, extra_size * 2]
            let mut out_2d_real = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size * 2));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(
                    out.as_mut_ptr() as *mut f64,
                    mapping,
                )
            };

            // Fit: coeffs = V * S^{-1} * U^T * values (real matrix, real data)
            self.fit_2d_dd_to(backend, &values_2d_real, &mut out_2d_real);
        } else if dim == rank - 1 {
            // Fast path: dim == N-1, use 2 separate GEMMs for re/im parts
            // Input: [extra_size, n_points] complex
            // Output: [extra_size, basis_size] complex
            // Operation: fit along last dimension

            // 1. Extract real and imaginary parts (de-interleave)
            let mut values_re: Vec<f64> = Vec::with_capacity(total);
            let mut values_im: Vec<f64> = Vec::with_capacity(total);
            for v in values.iter() {
                values_re.push(v.re);
                values_im.push(v.im);
            }

            // Create views: [extra_size, n_points]
            let values_re_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DView::<'_, f64, 2>::new_unchecked(values_re.as_ptr(), mapping)
            };
            let values_im_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DView::<'_, f64, 2>::new_unchecked(values_im.as_ptr(), mapping)
            };

            // 2. Allocate output buffers for real results
            let out_total = extra_size * basis_size;
            let mut out_re: Vec<f64> = vec![0.0; out_total];
            let mut out_im: Vec<f64> = vec![0.0; out_total];

            let mut out_re_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_re.as_mut_ptr(), mapping)
            };
            let mut out_im_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_im.as_mut_ptr(), mapping)
            };

            // 3. Do 2 fits
            self.fit_2d_to_dim(backend, &values_re_view, &mut out_re_view, 1);
            self.fit_2d_to_dim(backend, &values_im_view, &mut out_im_view, 1);

            // 4. Interleave into complex output
            let out_ptr = out.as_mut_ptr() as *mut f64;
            unsafe {
                for i in 0..out_total {
                    *out_ptr.add(2 * i) = out_re[i];
                    *out_ptr.add(2 * i + 1) = out_im[i];
                }
            }
        } else {
            // General path
            return self.fit_nd_zz_to_general(backend, values, dim, out);
        }
        true
    }

    /// General implementation for fit_nd_zz_to (non-fast-path cases)
    /// Uses 2-GEMM approach: separate re/im, do 2 fits, then interleave
    fn fit_nd_zz_to_general(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let total = values.len();
        let out_total = out.len();

        // 1. Extract real and imaginary parts (de-interleave)
        let mut values_re: Vec<f64> = Vec::with_capacity(total);
        let mut values_im: Vec<f64> = Vec::with_capacity(total);
        for v in values.iter() {
            values_re.push(v.re);
            values_im.push(v.im);
        }

        // 2. Get input shape
        let mut values_shape: Vec<usize> = Vec::with_capacity(values.rank());
        values.shape().with_dims(|dims| {
            for d in dims {
                values_shape.push(*d);
            }
        });
        let shape: mdarray::DynRank = mdarray::Shape::from_dims(&values_shape[..]);

        let values_re_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape.clone());
            mdarray::View::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                values_re.as_ptr(),
                mapping,
            )
        };
        let values_im_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape);
            mdarray::View::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                values_im.as_ptr(),
                mapping,
            )
        };

        // 3. Allocate output buffers
        let mut out_re: Vec<f64> = vec![0.0; out_total];
        let mut out_im: Vec<f64> = vec![0.0; out_total];

        // Get output shape
        let mut out_shape: Vec<usize> = Vec::with_capacity(out.rank());
        out.shape().with_dims(|dims| {
            for d in dims {
                out_shape.push(*d);
            }
        });
        let out_shape_dyn: mdarray::DynRank = mdarray::Shape::from_dims(&out_shape[..]);

        let mut out_re_view = unsafe {
            let mapping = mdarray::DenseMapping::new(out_shape_dyn.clone());
            mdarray::ViewMut::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                out_re.as_mut_ptr(),
                mapping,
            )
        };
        let mut out_im_view = unsafe {
            let mapping = mdarray::DenseMapping::new(out_shape_dyn);
            mdarray::ViewMut::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                out_im.as_mut_ptr(),
                mapping,
            )
        };

        // 4. Do 2 fits
        self.fit_nd_dd_to(backend, &values_re_view, dim, &mut out_re_view);
        self.fit_nd_dd_to(backend, &values_im_view, dim, &mut out_im_view);

        // 5. Interleave into complex output
        let out_ptr = out.as_mut_ptr() as *mut f64;
        unsafe {
            for i in 0..out_total {
                *out_ptr.add(2 * i) = out_re[i];
                *out_ptr.add(2 * i + 1) = out_im[i];
            }
        }
        true
    }

    /// Evaluate 2D complex coefficients to complex values using GEMM
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    pub fn evaluate_2d_zz(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
    ) -> mdarray::DTensor<Complex<f64>, 2> {
        let (basis_size, extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = mdarray::DTensor::<Complex<f64>, 2>::zeros([n_points, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.evaluate_2d_zz_to(backend, coeffs_2d, &mut out_view);
        out
    }

    /// Evaluate 2D complex coefficients with in-place output to mutable view
    ///
    /// Uses the memory layout of Complex<f64> (re, im are contiguous) to delegate
    /// to the real version by treating complex array [basis_size, extra_size] as
    /// real array [basis_size, extra_size, 2].
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output mutable view, shape: [n_points, extra_size] (will be overwritten)
    pub fn evaluate_2d_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        let (basis_size, extra_size) = *coeffs_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );
        assert_eq!(
            out_rows,
            self.n_points(),
            "out.shape().0={} must equal n_points={}",
            out_rows,
            self.n_points()
        );
        assert_eq!(
            out_cols, extra_size,
            "out.shape().1={} must equal extra_size={}",
            out_cols, extra_size
        );

        // Convert to DynRank and delegate to ND version
        let coeffs_dyn = coeffs_2d.into_dyn();
        let mut out_dyn = out.expr_mut().into_dyn();
        self.evaluate_nd_zz_to(backend, &coeffs_dyn, 0, &mut out_dyn);
    }

    /// Evaluate N-D complex tensor along specified dimension, writing to a mutable view
    ///
    /// Uses the memory layout of Complex<f64> (re, im are contiguous) to delegate
    /// to the real version by treating complex array [..., dN] as real array [..., dN, 2].
    ///
    /// # Arguments
    /// * `backend` - Optional GEMM backend
    /// * `coeffs` - N-dimensional complex array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    /// * `out` - Output mutable view with `out.shape().dim(dim) == n_points`
    ///
    /// # Performance
    /// For dim==0 or dim==rank-1 with contiguous arrays, uses zero-copy reinterpretation.
    /// Otherwise, uses temporary buffers.
    pub fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        let total = coeffs.len();
        let extra_size = total / basis_size;

        if dim == 0 {
            // Fast path: dim == 0, zero-copy reinterpretation
            // Complex<f64> has layout [re, im], so we can treat [basis_size, extra_size] complex
            // as [basis_size, extra_size, 2] real, which is [basis_size, extra_size * 2] in 2D

            // Create view of input as real: [basis_size, extra_size * 2]
            let coeffs_2d_real = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size * 2));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr() as *const f64, mapping)
            };

            // Create view of output as real: [n_points, extra_size * 2]
            let mut out_2d_real = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size * 2));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(
                    out.as_mut_ptr() as *mut f64,
                    mapping,
                )
            };

            // Evaluate: out = matrix * coeffs (real matrix, real data)
            self.evaluate_2d_dd_to(backend, &coeffs_2d_real, &mut out_2d_real);
        } else if dim == rank - 1 {
            // Fast path: dim == N-1, use 2 separate GEMMs for re/im parts
            // Input: [extra_size, basis_size] complex
            // Output: [extra_size, n_points] complex
            // Operation: out = coeffs @ matrix^T (apply matrix along last dimension)

            // 1. Extract real and imaginary parts (de-interleave)
            let mut coeffs_re: Vec<f64> = Vec::with_capacity(total);
            let mut coeffs_im: Vec<f64> = Vec::with_capacity(total);
            for c in coeffs.iter() {
                coeffs_re.push(c.re);
                coeffs_im.push(c.im);
            }

            // Create views: [extra_size, basis_size]
            let coeffs_re_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs_re.as_ptr(), mapping)
            };
            let coeffs_im_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs_im.as_ptr(), mapping)
            };

            // 2. Allocate output buffers for real results
            let out_total = extra_size * n_points;
            let mut out_re: Vec<f64> = vec![0.0; out_total];
            let mut out_im: Vec<f64> = vec![0.0; out_total];

            let mut out_re_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_re.as_mut_ptr(), mapping)
            };
            let mut out_im_view = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_im.as_mut_ptr(), mapping)
            };

            // 3. Do 2 GEMMs: out = coeffs @ matrix^T
            self.evaluate_2d_to_dim(backend, &coeffs_re_view, &mut out_re_view, 1);
            self.evaluate_2d_to_dim(backend, &coeffs_im_view, &mut out_im_view, 1);

            // 4. Interleave into complex output
            let out_ptr = out.as_mut_ptr() as *mut f64;
            unsafe {
                for i in 0..out_total {
                    *out_ptr.add(2 * i) = out_re[i];
                    *out_ptr.add(2 * i + 1) = out_im[i];
                }
            }
        } else {
            // General path
            return self.evaluate_nd_zz_to_general(backend, coeffs, dim, out);
        }
        true
    }

    /// General implementation for evaluate_nd_zz_to (non-fast-path cases)
    /// Uses 2-GEMM approach: separate re/im, do 2 GEMMs, then interleave
    fn evaluate_nd_zz_to_general(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let total = coeffs.len();
        let out_total = out.len();

        // 1. Extract real and imaginary parts (de-interleave)
        let mut coeffs_re: Vec<f64> = Vec::with_capacity(total);
        let mut coeffs_im: Vec<f64> = Vec::with_capacity(total);
        for c in coeffs.iter() {
            coeffs_re.push(c.re);
            coeffs_im.push(c.im);
        }

        // 2. Get input shape
        let mut coeffs_shape: Vec<usize> = Vec::with_capacity(coeffs.rank());
        coeffs.shape().with_dims(|dims| {
            for d in dims {
                coeffs_shape.push(*d);
            }
        });
        let shape: mdarray::DynRank = mdarray::Shape::from_dims(&coeffs_shape[..]);

        let coeffs_re_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape.clone());
            mdarray::View::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                coeffs_re.as_ptr(),
                mapping,
            )
        };
        let coeffs_im_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape);
            mdarray::View::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                coeffs_im.as_ptr(),
                mapping,
            )
        };

        // 3. Allocate output buffers
        let mut out_re: Vec<f64> = vec![0.0; out_total];
        let mut out_im: Vec<f64> = vec![0.0; out_total];

        // Get output shape
        let mut out_shape: Vec<usize> = Vec::with_capacity(out.rank());
        out.shape().with_dims(|dims| {
            for d in dims {
                out_shape.push(*d);
            }
        });
        let out_shape_dyn: mdarray::DynRank = mdarray::Shape::from_dims(&out_shape[..]);

        let mut out_re_view = unsafe {
            let mapping = mdarray::DenseMapping::new(out_shape_dyn.clone());
            mdarray::ViewMut::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                out_re.as_mut_ptr(),
                mapping,
            )
        };
        let mut out_im_view = unsafe {
            let mapping = mdarray::DenseMapping::new(out_shape_dyn);
            mdarray::ViewMut::<'_, f64, mdarray::DynRank, mdarray::Dense>::new_unchecked(
                out_im.as_mut_ptr(),
                mapping,
            )
        };

        // 4. Do 2 GEMMs
        self.evaluate_nd_dd_to(backend, &coeffs_re_view, dim, &mut out_re_view);
        self.evaluate_nd_dd_to(backend, &coeffs_im_view, dim, &mut out_im_view);

        // 5. Interleave into complex output
        let out_ptr = out.as_mut_ptr() as *mut f64;
        unsafe {
            for i in 0..out_total {
                *out_ptr.add(2 * i) = out_re[i];
                *out_ptr.add(2 * i + 1) = out_im[i];
            }
        }
        true
    }

    /// Generic 2D evaluate (works for both f64 and Complex<f64>)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    pub fn evaluate_2d_generic<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &mdarray::DTensor<T, 2>,
    ) -> mdarray::DTensor<T, 2>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + From<f64>
            + Copy
            + Default
            + 'static,
    {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let coeffs_f64 = unsafe {
                &*(coeffs_2d as *const mdarray::DTensor<T, 2> as *const mdarray::DTensor<f64, 2>)
            };
            let coeffs_view = coeffs_f64.view(.., ..);
            let result = self.evaluate_2d(backend, &coeffs_view);
            unsafe {
                std::mem::transmute::<mdarray::DTensor<f64, 2>, mdarray::DTensor<T, 2>>(result)
            }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            let coeffs_complex = unsafe {
                &*(coeffs_2d as *const mdarray::DTensor<T, 2>
                    as *const mdarray::DTensor<Complex<f64>, 2>)
            };
            let coeffs_view = coeffs_complex.view(.., ..);
            let result = self.evaluate_2d_zz(backend, &coeffs_view);
            unsafe {
                std::mem::transmute::<mdarray::DTensor<Complex<f64>, 2>, mdarray::DTensor<T, 2>>(
                    result,
                )
            }
        } else {
            panic!("Unsupported type for evaluate_2d_generic");
        }
    }

    /// Generic 2D fit (works for both f64 and Complex<f64>)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Coefficients tensor, shape: [basis_size, extra_size]
    pub fn fit_2d_generic<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &mdarray::DTensor<T, 2>,
    ) -> mdarray::DTensor<T, 2>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + From<f64>
            + Copy
            + Default
            + 'static,
    {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let values_f64 = unsafe {
                &*(values_2d as *const mdarray::DTensor<T, 2> as *const mdarray::DTensor<f64, 2>)
            };
            let values_view = values_f64.view(.., ..);
            let result = self.fit_2d(backend, &values_view);
            unsafe {
                std::mem::transmute::<mdarray::DTensor<f64, 2>, mdarray::DTensor<T, 2>>(result)
            }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            let values_complex = unsafe {
                &*(values_2d as *const mdarray::DTensor<T, 2>
                    as *const mdarray::DTensor<Complex<f64>, 2>)
            };
            let values_view = values_complex.view(.., ..);
            let result = self.fit_2d_zz(backend, &values_view);
            unsafe {
                std::mem::transmute::<mdarray::DTensor<Complex<f64>, 2>, mdarray::DTensor<T, 2>>(
                    result,
                )
            }
        } else {
            panic!("Unsupported type for fit_2d_generic");
        }
    }
}

// ============================================================================
// InplaceFitter trait implementation
// ============================================================================

impl super::common::InplaceFitter for RealMatrixFitter {
    fn n_points(&self) -> usize {
        self.n_points()
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
    ) -> bool {
        RealMatrixFitter::evaluate_nd_dd_to(self, backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        RealMatrixFitter::evaluate_nd_zz_to(self, backend, coeffs, dim, out)
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        RealMatrixFitter::fit_nd_dd_to(self, backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        RealMatrixFitter::fit_nd_zz_to(self, backend, values, dim, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::DTensor;
    use num_complex::Complex;

    #[test]
    fn test_roundtrip() {
        let n_points = 10;
        let basis_size = 5;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);
        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.5).collect();

        let values = fitter.evaluate(None, &coeffs);
        assert_eq!(values.len(), n_points);

        let fitted_coeffs = fitter.fit(None, &values);
        assert_eq!(fitted_coeffs.len(), basis_size);

        for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
            let error = (orig - fitted).abs();
            assert!(error < 1e-10, "Roundtrip error: {}", error);
        }
    }

    #[test]
    fn test_overdetermined() {
        let n_points = 20;
        let basis_size = 5;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64;
            let j = idx[1] as f64;
            ((i + 1.0) / (n_points as f64)).powi(j as i32)
        });

        let fitter = RealMatrixFitter::new(matrix);
        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64) * 0.3).collect();

        let values = fitter.evaluate(None, &coeffs);
        let fitted_coeffs = fitter.fit(None, &values);

        for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
            let error = (orig - fitted).abs();
            assert!(error < 1e-10, "Overdetermined roundtrip error: {}", error);
        }
    }

    #[test]
    #[should_panic(expected = "must equal basis_size")]
    fn test_wrong_coeffs_size() {
        let matrix = DTensor::<f64, 2>::from_fn([5, 3], |_| 1.0);
        let fitter = RealMatrixFitter::new(matrix);
        let wrong_coeffs = vec![1.0; 5];
        let _values = fitter.evaluate(None, &wrong_coeffs);
    }

    #[test]
    #[should_panic(expected = "must equal n_points")]
    fn test_wrong_values_size() {
        let matrix = DTensor::<f64, 2>::from_fn([5, 3], |_| 1.0);
        let fitter = RealMatrixFitter::new(matrix);
        let wrong_values = vec![1.0; 10];
        let _coeffs = fitter.fit(None, &wrong_values);
    }

    #[test]
    fn test_evaluate_2d_to_matches_evaluate_2d() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
        });
        let coeffs_view = coeffs_2d.view(.., ..);

        let expected = fitter.evaluate_2d(None, &coeffs_view);

        let mut actual = DTensor::<f64, 2>::from_elem([n_points, extra_size], 0.0);
        {
            let mut actual_view = actual.view_mut(.., ..);
            fitter.evaluate_2d_dd_to(None, &coeffs_view, &mut actual_view);
        }

        assert_eq!(actual.shape(), expected.shape());
        for i in 0..n_points {
            for j in 0..extra_size {
                let diff = (actual[[i, j]] - expected[[i, j]]).abs();
                assert!(diff < 1e-14, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_fit_2d_to_matches_fit_2d() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let values_2d = DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
        });
        let values_view = values_2d.view(.., ..);

        let expected = fitter.fit_2d(None, &values_view);

        let mut actual = DTensor::<f64, 2>::from_elem([basis_size, extra_size], 0.0);
        {
            let mut actual_view = actual.view_mut(.., ..);
            fitter.fit_2d_dd_to(None, &values_view, &mut actual_view);
        }

        assert_eq!(actual.shape(), expected.shape());
        for i in 0..basis_size {
            for j in 0..extra_size {
                let diff = (actual[[i, j]] - expected[[i, j]]).abs();
                assert!(diff < 1e-14, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_evaluate_fit_roundtrip_with_inplace() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            (idx[0] as f64 + 1.0) * 0.3 + (idx[1] as f64) * 0.1
        });
        let coeffs_view = coeffs.view(.., ..);

        let mut values = DTensor::<f64, 2>::from_elem([n_points, extra_size], 0.0);
        {
            let mut values_mut = values.view_mut(.., ..);
            fitter.evaluate_2d_dd_to(None, &coeffs_view, &mut values_mut);
        }

        let values_view = values.view(.., ..);
        let mut fitted_coeffs = DTensor::<f64, 2>::from_elem([basis_size, extra_size], 0.0);
        {
            let mut fitted_view = fitted_coeffs.view_mut(.., ..);
            fitter.fit_2d_dd_to(None, &values_view, &mut fitted_view);
        }

        for i in 0..basis_size {
            for j in 0..extra_size {
                let diff = (fitted_coeffs[[i, j]] - coeffs[[i, j]]).abs();
                assert!(diff < 1e-10, "Roundtrip mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_evaluate_2d_zz_to_matches_evaluate_2d_zz() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                (idx[0] as f64) * 0.3,
            )
        });
        let coeffs_view = coeffs_2d.view(.., ..);

        let expected = fitter.evaluate_2d_zz(None, &coeffs_view);

        let mut actual =
            DTensor::<Complex<f64>, 2>::from_elem([n_points, extra_size], Complex::new(0.0, 0.0));
        {
            let mut actual_view = actual.view_mut(.., ..);
            fitter.evaluate_2d_zz_to(None, &coeffs_view, &mut actual_view);
        }

        assert_eq!(actual.shape(), expected.shape());
        for i in 0..n_points {
            for j in 0..extra_size {
                let diff = (actual[[i, j]] - expected[[i, j]]).norm();
                assert!(diff < 1e-14, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_fit_2d_zz_to_matches_fit_2d_zz() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                (idx[0] as f64) * 0.2,
            )
        });
        let values_view = values_2d.view(.., ..);

        let expected = fitter.fit_2d_zz(None, &values_view);

        let mut actual =
            DTensor::<Complex<f64>, 2>::from_elem([basis_size, extra_size], Complex::new(0.0, 0.0));
        {
            let mut actual_view = actual.view_mut(.., ..);
            fitter.fit_2d_zz_to(None, &values_view, &mut actual_view);
        }

        assert_eq!(actual.shape(), expected.shape());
        for i in 0..basis_size {
            for j in 0..extra_size {
                let diff = (actual[[i, j]] - expected[[i, j]]).norm();
                assert!(diff < 1e-14, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_zz_roundtrip_with_inplace() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
            Complex::new((idx[0] as f64 + 1.0) * 0.3, (idx[1] as f64) * 0.1)
        });
        let coeffs_view = coeffs.view(.., ..);

        let mut values =
            DTensor::<Complex<f64>, 2>::from_elem([n_points, extra_size], Complex::new(0.0, 0.0));
        {
            let mut values_mut = values.view_mut(.., ..);
            fitter.evaluate_2d_zz_to(None, &coeffs_view, &mut values_mut);
        }

        let values_view = values.view(.., ..);
        let mut fitted_coeffs =
            DTensor::<Complex<f64>, 2>::from_elem([basis_size, extra_size], Complex::new(0.0, 0.0));
        {
            let mut fitted_view = fitted_coeffs.view_mut(.., ..);
            fitter.fit_2d_zz_to(None, &values_view, &mut fitted_view);
        }

        for i in 0..basis_size {
            for j in 0..extra_size {
                let diff = (fitted_coeffs[[i, j]] - coeffs[[i, j]]).norm();
                assert!(diff < 1e-10, "Roundtrip mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_evaluate_2d_to_dim0() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.3)
        });
        let coeffs_view = coeffs.view(.., ..);

        let expected = fitter.evaluate_2d(None, &coeffs_view);

        let mut out_data = vec![0.0f64; n_points * extra_size];
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((n_points, extra_size));
            mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_data.as_mut_ptr(), mapping)
        };
        fitter.evaluate_2d_to_dim(None, &coeffs_view, &mut out_view, 0);

        for i in 0..n_points {
            for j in 0..extra_size {
                let e = expected[[i, j]];
                let a = out_data[i * extra_size + j];
                let diff = (e - a).abs();
                assert!(diff < 1e-14, "Mismatch at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_evaluate_2d_to_dim1() {
        let n_points = 10;
        let basis_size = 5;
        let extra_size = 3;

        let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
        });

        let fitter = RealMatrixFitter::new(matrix);

        let coeffs = DTensor::<f64, 2>::from_fn([extra_size, basis_size], |idx| {
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.3)
        });
        let coeffs_view = coeffs.view(.., ..);

        let mut expected = DTensor::<f64, 2>::from_elem([extra_size, n_points], 0.0);
        for e in 0..extra_size {
            for p in 0..n_points {
                let mut sum = 0.0;
                for b in 0..basis_size {
                    sum += coeffs[[e, b]] * fitter.matrix[[p, b]];
                }
                expected[[e, p]] = sum;
            }
        }

        let mut out_data = vec![0.0f64; extra_size * n_points];
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((extra_size, n_points));
            mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_data.as_mut_ptr(), mapping)
        };
        fitter.evaluate_2d_to_dim(None, &coeffs_view, &mut out_view, 1);

        for e in 0..extra_size {
            for p in 0..n_points {
                let exp = expected[[e, p]];
                let act = out_data[e * n_points + p];
                let diff = (exp - act).abs();
                assert!(diff < 1e-12, "Mismatch at [{}, {}]", e, p);
            }
        }
    }
}
