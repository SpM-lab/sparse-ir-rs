//! Real matrix fitter: A ∈ R^{n×m}
//!
//! This module provides `RealMatrixFitter` for solving least-squares problems
//! where the matrix, coefficients, and values are all real.

use crate::gemm::GemmBackendHandle;
use crate::working_buffer::{copy_from_contiguous, copy_to_contiguous};
use mdarray::{DTensor, DView, DynRank, Shape, Slice, ViewMut};
use num_complex::Complex;
use std::cell::RefCell;

use super::common::{
    complex_slice_as_real, compute_real_svd, make_perm_to_front, RealSVD,
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
pub(crate) struct RealMatrixFitter {
    pub matrix: DTensor<f64, 2>, // (n_points, basis_size)
    svd: RefCell<Option<RealSVD>>,
}

impl RealMatrixFitter {
    /// Create a new fitter with the given real matrix
    pub fn new(matrix: DTensor<f64, 2>) -> Self {
        Self {
            matrix,
            svd: RefCell::new(None),
        }
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
        self.evaluate_2d_to(backend, &coeffs_view, &mut out_view);
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
    pub fn fit_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[f64],
        out: &mut [f64],
    ) {
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
        self.fit_2d_to(backend, &values_view, &mut out_view);
    }

    /// Fit complex values by fitting real and imaginary parts separately
    ///
    /// # Arguments
    /// * `values` - Complex values at sampling points (length = n_points)
    ///
    /// # Returns
    /// Complex coefficients (length = basis_size)
    #[allow(dead_code)]
    pub fn fit_complex(
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
        self.fit_complex_to(backend, values, &mut out);
        out
    }

    /// Fit complex values, writing to output slice
    pub fn fit_complex_to(
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
        self.fit_complex_2d_to(backend, &values_view, &mut out_view);
    }

    /// Evaluate complex coefficients
    ///
    /// # Arguments
    /// * `coeffs` - Complex coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at sampling points (length = n_points)
    #[allow(dead_code)]
    pub fn evaluate_complex(
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
        self.evaluate_complex_to(backend, coeffs, &mut out);
        out
    }

    /// Evaluate complex coefficients, writing to output slice
    pub fn evaluate_complex_to(
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
        self.evaluate_complex_2d_to(backend, &coeffs_view, &mut out_view);
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
        self.evaluate_2d_to(backend, coeffs_2d, &mut out_view);
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
    pub fn evaluate_2d_to(
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
    pub fn evaluate_2d_to_dim(
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
    pub fn evaluate_nd_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
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
            // General path: use temporary buffers for movedim
            // Move target dim to position 0, compute, then move back

            // Step 1: Copy permuted input to contiguous buffer
            let perm = make_perm_to_front(rank, dim);
            let permuted_view = coeffs.permute(&perm[..]);

            let mut input_buffer: Vec<f64> = Vec::with_capacity(total);
            unsafe { input_buffer.set_len(total); }
            copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

            // Step 2: Create DView from buffer for GEMM
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(input_buffer.as_ptr(), mapping)
            };

            // Step 3: Allocate output buffer and create DViewMut
            let out_total = out.len();
            let mut output_buffer: Vec<f64> = Vec::with_capacity(out_total);
            unsafe { output_buffer.set_len(out_total); }

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(output_buffer.as_mut_ptr(), mapping)
            };

            // Step 4: Matrix multiply directly into output buffer
            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);

            // Step 5: Copy from contiguous buffer to strided output view
            // Use same perm as input (not inverse) because buffer has shape [n_points, ...rest...]
            // which matches the permuted output view order
            let out_permuted = (&mut *out).permute_mut(&perm[..]);
            copy_from_contiguous(&output_buffer, &mut out_permuted.into_dyn());
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
        self.fit_2d_to(backend, values_2d, &mut out_view);
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
    pub fn fit_2d_to_dim(
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

            // Compute SVD lazily
            if self.svd.borrow().is_none() {
                let svd = compute_real_svd(&self.matrix);
                *self.svd.borrow_mut() = Some(svd);
            }

            let svd = self.svd.borrow();
            let svd = svd.as_ref().unwrap();

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
    pub fn fit_2d_to(
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
    pub fn fit_nd_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
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

            self.fit_2d_to(backend, &values_2d, &mut out_2d);
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
            // General path: use temporary buffers for movedim
            // Move target dim to position 0, compute, then move back

            // Step 1: Copy permuted input to contiguous buffer
            let perm = make_perm_to_front(rank, dim);
            let permuted_view = values.permute(&perm[..]);

            let mut input_buffer: Vec<f64> = Vec::with_capacity(total);
            unsafe { input_buffer.set_len(total); }
            copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

            // Step 2: Create DView from buffer for GEMM
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(input_buffer.as_ptr(), mapping)
            };

            // Step 3: Allocate output buffer and create DViewMut
            let out_total = out.len();
            let mut output_buffer: Vec<f64> = Vec::with_capacity(out_total);
            unsafe { output_buffer.set_len(out_total); }

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(output_buffer.as_mut_ptr(), mapping)
            };

            // Step 4: Fit directly into output buffer
            self.fit_2d_to(backend, &values_2d, &mut out_2d);

            // Step 5: Copy from contiguous buffer to strided output view
            // Use same perm as input (not inverse) because buffer has shape [basis_size, ...rest...]
            // which matches the permuted output view order
            let out_permuted = (&mut *out).permute_mut(&perm[..]);
            copy_from_contiguous(&output_buffer, &mut out_permuted.into_dyn());
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
    pub fn fit_complex_2d(
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
        self.fit_complex_2d_to(backend, values_2d, &mut out_view);
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
    pub fn fit_complex_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        // Convert to DynRank and delegate to ND version
        let values_dyn = values_2d.into_dyn();
        let mut out_dyn = out.expr_mut().into_dyn();
        self.fit_complex_nd_to_dim(backend, &values_dyn, 0, &mut out_dyn);
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
    /// Delegates to the real version by adding an extra dimension of size 2.
    /// For contiguous arrays, this is very efficient.
    pub fn fit_complex_nd_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) {
        let rank = values.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(values.shape().dim(dim), n_points);
        assert_eq!(out.shape().dim(dim), basis_size);

        // Copy to contiguous buffer first to ensure correct memory layout
        let values_tensor = values.to_tensor();
        let values_as_f64 = complex_slice_as_real(&values_tensor);

        // Create contiguous output buffer
        let mut out_shape_f64: Vec<usize> = Vec::with_capacity(rank + 1);
        out.shape().with_dims(|dims| {
            for d in dims {
                out_shape_f64.push(*d);
            }
        });
        out_shape_f64.push(2);
        let mut out_f64_buffer = mdarray::Tensor::<f64, DynRank>::zeros(&out_shape_f64[..]);

        // Fit to contiguous buffer
        {
            let mut out_f64_view = out_f64_buffer.expr_mut();
            self.fit_nd_to_dim(backend, &values_as_f64, dim, &mut out_f64_view);
        }

        // Copy back to output
        let out_total = out.len();
        let mut idx_vec = vec![0usize; rank];
        for flat in 0..out_total {
            let mut remaining = flat;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx_vec[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            let re = out_f64_buffer[&{
                let mut idx_f64 = idx_vec.clone();
                idx_f64.push(0);
                idx_f64
            }[..]];
            let im = out_f64_buffer[&{
                let mut idx_f64 = idx_vec.clone();
                idx_f64.push(1);
                idx_f64
            }[..]];
            out[&idx_vec[..]] = Complex::new(re, im);
        }
    }

    /// Evaluate 2D complex coefficients to complex values using GEMM
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    pub fn evaluate_complex_2d(
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
        self.evaluate_complex_2d_to(backend, coeffs_2d, &mut out_view);
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
    pub fn evaluate_complex_2d_to(
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
        self.evaluate_complex_nd_to_dim(backend, &coeffs_dyn, 0, &mut out_dyn);
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
    /// Delegates to the real version by adding an extra dimension of size 2.
    /// For contiguous arrays, this is very efficient.
    pub fn evaluate_complex_nd_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) {
        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        // Copy to contiguous buffer first to ensure correct memory layout
        let coeffs_tensor = coeffs.to_tensor();
        let coeffs_as_f64 = complex_slice_as_real(&coeffs_tensor);

        // Create contiguous output buffer
        let mut out_shape_f64: Vec<usize> = Vec::with_capacity(rank + 1);
        out.shape().with_dims(|dims| {
            for d in dims {
                out_shape_f64.push(*d);
            }
        });
        out_shape_f64.push(2);
        let mut out_f64_buffer = mdarray::Tensor::<f64, DynRank>::zeros(&out_shape_f64[..]);

        // Evaluate to contiguous buffer
        {
            let mut out_f64_view = out_f64_buffer.expr_mut();
            self.evaluate_nd_to_dim(backend, &coeffs_as_f64, dim, &mut out_f64_view);
        }

        // Copy back to output
        let out_total = out.len();
        let mut idx_vec = vec![0usize; rank];
        for flat in 0..out_total {
            let mut remaining = flat;
            for d in (0..rank).rev() {
                let dim_size = out.shape().dim(d);
                idx_vec[d] = remaining % dim_size;
                remaining /= dim_size;
            }
            let re = out_f64_buffer[&{
                let mut idx_f64 = idx_vec.clone();
                idx_f64.push(0);
                idx_f64
            }[..]];
            let im = out_f64_buffer[&{
                let mut idx_f64 = idx_vec.clone();
                idx_f64.push(1);
                idx_f64
            }[..]];
            out[&idx_vec[..]] = Complex::new(re, im);
        }
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
            let result = self.evaluate_complex_2d(backend, &coeffs_view);
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
            let result = self.fit_complex_2d(backend, &values_view);
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
