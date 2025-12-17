//! Complex-to-Real matrix fitter: A ∈ C^{n×m}
//!
//! This module provides `ComplexToRealFitter` for solving least-squares problems
//! where the matrix and values are complex, but the coefficients are real.
//!
//! Strategy: Flatten complex to real problem
//!   A_real ∈ R^{2n×m}: [Re(A[0,:]); Im(A[0,:]); Re(A[1,:]); Im(A[1,:]); ...]
//!   values_flat ∈ R^{2n}: [Re(v[0]); Im(v[0]); Re(v[1]); Im(v[1]); ...]

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView, DynRank, Shape, Slice, ViewMut};
use num_complex::Complex;
use std::cell::RefCell;

use super::common::{InplaceFitter, RealSVD, compute_real_svd};

// ============================================================================
// Helper functions for efficient interleave/deinterleave
// ============================================================================

/// Combine real and imaginary tensors into complex output (contiguous layout)
///
/// Assumes all tensors have the same shape and are contiguous in memory.
#[inline]
fn interleave_to_complex(
    values_re: &DTensor<f64, 2>,
    values_im: &DTensor<f64, 2>,
    out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
) {
    let total = values_re.len();
    debug_assert_eq!(values_im.len(), total);
    debug_assert_eq!(out.len(), total);

    let out_ptr = out.as_mut_ptr();
    let re_ptr = values_re.as_ptr();
    let im_ptr = values_im.as_ptr();

    unsafe {
        for k in 0..total {
            *out_ptr.add(k) = Complex::new(*re_ptr.add(k), *im_ptr.add(k));
        }
    }
}

/// Flatten complex tensor to interleaved real layout: [n, m] → [2n, m]
/// Row 2i contains real parts, row 2i+1 contains imaginary parts
#[inline]
fn flatten_complex_to_real_rows(values: &DView<'_, Complex<f64>, 2>, out: &mut DTensor<f64, 2>) {
    let (n_points, extra_size) = *values.shape();
    debug_assert_eq!(*out.shape(), (2 * n_points, extra_size));

    let src_ptr = values.as_ptr();
    let dst_ptr = out.as_mut_ptr();

    unsafe {
        for i in 0..n_points {
            for j in 0..extra_size {
                let val = *src_ptr.add(i * extra_size + j);
                *dst_ptr.add((2 * i) * extra_size + j) = val.re;
                *dst_ptr.add((2 * i + 1) * extra_size + j) = val.im;
            }
        }
    }
}

/// Flatten complex tensor to interleaved real layout: [n, m] → [n, 2m]
/// Column 2j contains real parts, column 2j+1 contains imaginary parts
#[inline]
fn flatten_complex_to_real_cols(values: &DView<'_, Complex<f64>, 2>, out: &mut DTensor<f64, 2>) {
    let (extra_size, n_points) = *values.shape();
    debug_assert_eq!(*out.shape(), (extra_size, 2 * n_points));

    let src_ptr = values.as_ptr();
    let dst_ptr = out.as_mut_ptr();

    unsafe {
        for i in 0..extra_size {
            for j in 0..n_points {
                let val = *src_ptr.add(i * n_points + j);
                *dst_ptr.add(i * (2 * n_points) + 2 * j) = val.re;
                *dst_ptr.add(i * (2 * n_points) + 2 * j + 1) = val.im;
            }
        }
    }
}

/// Fitter for complex matrix with real coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A is complex, coeffs are real, values are complex
///
/// Strategy: Flatten complex to real problem
///   A_real ∈ R^{2n×m}: [Re(A[0,:]); Im(A[0,:]); Re(A[1,:]); Im(A[1,:]); ...]
///   values_flat ∈ R^{2n}: [Re(v[0]); Im(v[0]); Re(v[1]); Im(v[1]); ...]
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| Complex::new(...));
/// let fitter = ComplexToRealFitter::new(&matrix);
///
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = fitter.evaluate(&coeffs);  // → Vec<Complex<f64>>
/// let fitted_coeffs = fitter.fit(&values);  // ← Vec<Complex<f64>>, → Vec<f64>
/// ```
pub(crate) struct ComplexToRealFitter {
    // A_real ∈ R^{2n×m}: flattened complex matrix (for fit)
    // A_real[2i,   :] = Re(A[i, :])
    // A_real[2i+1, :] = Im(A[i, :])
    matrix_real: DTensor<f64, 2>, // (2*n_points, basis_size)
    // Separate real/imag parts for evaluate (pre-computed)
    matrix_re: DTensor<f64, 2>, // (n_points, basis_size)
    matrix_im: DTensor<f64, 2>, // (n_points, basis_size)
    // Transposed versions for dim=1 operations (pre-computed)
    matrix_re_t: DTensor<f64, 2>,         // (basis_size, n_points)
    matrix_im_t: DTensor<f64, 2>,         // (basis_size, n_points)
    pub matrix: DTensor<Complex<f64>, 2>, // (n_points, basis_size) - original complex matrix
    svd: RefCell<Option<RealSVDExtended>>,
    n_points: usize, // Original complex point count
}

/// Extended SVD structure that includes pre-computed transposes for dim=1 operations
struct RealSVDExtended {
    /// Core SVD components
    svd: RealSVD,
    /// U matrix (transpose of ut): shape (2*n_points, min_dim)
    u: DTensor<f64, 2>,
    /// V^T matrix (transpose of v): shape (min_dim, basis_size)
    vt: DTensor<f64, 2>,
}

impl RealSVDExtended {
    /// Create extended SVD from matrix with pre-computed transposes
    fn from_matrix(matrix: &DTensor<f64, 2>) -> Self {
        let svd = compute_real_svd(matrix);
        let min_dim = svd.s.len();
        let (rows, cols) = *matrix.shape();

        // U = ut^T: shape (rows, min_dim)
        let u = DTensor::<f64, 2>::from_fn([rows, min_dim], |idx| svd.ut[[idx[1], idx[0]]]);

        // V^T = v^T: shape (min_dim, cols)
        let vt = DTensor::<f64, 2>::from_fn([min_dim, cols], |idx| svd.v[[idx[1], idx[0]]]);

        Self { svd, u, vt }
    }
}

impl ComplexToRealFitter {
    /// Create from complex matrix A ∈ C^{n×m}
    pub fn new(matrix_complex: &DTensor<Complex<f64>, 2>) -> Self {
        let (n_points, basis_size) = *matrix_complex.shape();

        // Flatten to real: (2*n_points, basis_size) - for fit
        // A_real[2i,   j] = Re(A[i, j])
        // A_real[2i+1, j] = Im(A[i, j])
        let matrix_real = DTensor::<f64, 2>::from_fn([2 * n_points, basis_size], |idx| {
            let i = idx[0] / 2;
            let j = idx[1];
            let val = matrix_complex[[i, j]];

            if idx[0] % 2 == 0 {
                val.re // Even row: real part
            } else {
                val.im // Odd row: imaginary part
            }
        });

        // Pre-compute separate real/imaginary parts for evaluate
        let matrix_re =
            DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| matrix_complex[idx].re);
        let matrix_im =
            DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| matrix_complex[idx].im);

        // Pre-compute transposed versions for dim=1 operations
        let matrix_re_t =
            DTensor::<f64, 2>::from_fn([basis_size, n_points], |idx| matrix_re[[idx[1], idx[0]]]);
        let matrix_im_t =
            DTensor::<f64, 2>::from_fn([basis_size, n_points], |idx| matrix_im[[idx[1], idx[0]]]);

        Self {
            matrix_real,
            matrix_re,
            matrix_im,
            matrix_re_t,
            matrix_im_t,
            matrix: matrix_complex.clone(),
            svd: RefCell::new(None),
            n_points,
        }
    }

    /// Number of complex data points
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix_real.shape().1
    }

    /// Evaluate: coeffs (real) → values (complex)
    ///
    /// Computes: values = A * coeffs where A is complex
    pub fn evaluate(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[f64],
    ) -> Vec<Complex<f64>> {
        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        // Convert coeffs to column vector and use evaluate_2d
        let basis_size = coeffs.len();
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, 1], |idx| coeffs[idx[0]]);
        let coeffs_view = coeffs_2d.view(.., ..);
        let values_2d = self.evaluate_2d(backend, &coeffs_view);

        // Extract as Vec
        let n_points = self.n_points();
        (0..n_points).map(|i| values_2d[[i, 0]]).collect()
    }

    /// Fit: values (complex) → coeffs (real)
    ///
    /// Solves: min ||A * coeffs - values||^2 using flattened real SVD
    pub fn fit(&self, backend: Option<&GemmBackendHandle>, values: &[Complex<f64>]) -> Vec<f64> {
        assert_eq!(
            values.len(),
            self.n_points(),
            "values.len()={} must equal n_points={}",
            values.len(),
            self.n_points()
        );

        // Convert complex values to 2D tensor (column vector) and use fit_2d
        let n = values.len();
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n, 1], |idx| values[idx[0]]);
        let values_view = values_2d.view(.., ..);
        let coeffs_2d = self.fit_2d(backend, &values_view);

        // Extract as Vec
        let basis_size = self.basis_size();
        (0..basis_size).map(|i| coeffs_2d[[i, 0]]).collect()
    }

    /// Evaluate 2D real tensor to complex (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Complex values tensor, shape: [n_points, extra_size]
    pub fn evaluate_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        let (basis_size, extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = DTensor::<Complex<f64>, 2>::zeros([n_points, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.evaluate_2d_to(backend, coeffs_2d, &mut out_view);
        out
    }

    /// Evaluate 2D real tensor to complex (along dim=0), writing to output
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output mutable view, shape: [n_points, extra_size]
    pub fn evaluate_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        use crate::gemm::matmul_par_view;

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

        // Compute real and imaginary parts separately using pre-computed matrices
        let matrix_re_view = self.matrix_re.view(.., ..);
        let matrix_im_view = self.matrix_im.view(.., ..);
        let values_re = matmul_par_view(&matrix_re_view, coeffs_2d, backend);
        let values_im = matmul_par_view(&matrix_im_view, coeffs_2d, backend);

        // Combine to complex output (interleave)
        interleave_to_complex(&values_re, &values_im, out);
    }

    /// Evaluate 2D real tensor to complex with configurable target dimension
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape depends on dim: [basis_size, extra] if dim=0, [extra, basis_size] if dim=1
    /// * `out` - Output mutable view
    /// * `dim` - Target dimension (0 or 1)
    fn evaluate_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
        dim: usize,
    ) {
        use crate::gemm::matmul_par_view;

        let (coeffs_rows, coeffs_cols) = *coeffs_2d.shape();
        let (out_rows, out_cols) = *out.shape();
        let n_points = self.n_points();

        if dim == 0 {
            // out[n_points, extra] = matrix[n_points, basis_size] * coeffs[basis_size, extra]
            let basis_size = coeffs_rows;
            let extra_size = coeffs_cols;

            assert_eq!(basis_size, self.basis_size());
            assert_eq!(out_rows, n_points);
            assert_eq!(out_cols, extra_size);

            let matrix_re_view = self.matrix_re.view(.., ..);
            let matrix_im_view = self.matrix_im.view(.., ..);
            let values_re = matmul_par_view(&matrix_re_view, coeffs_2d, backend);
            let values_im = matmul_par_view(&matrix_im_view, coeffs_2d, backend);

            interleave_to_complex(&values_re, &values_im, out);
        } else {
            // dim == 1
            // out[extra, n_points] = coeffs[extra, basis_size] * matrix^T[basis_size, n_points]
            let extra_size = coeffs_rows;
            let basis_size = coeffs_cols;

            assert_eq!(basis_size, self.basis_size());
            assert_eq!(out_rows, extra_size);
            assert_eq!(out_cols, n_points);

            // Use pre-computed transposed matrices
            let matrix_re_t_view = self.matrix_re_t.view(.., ..);
            let matrix_im_t_view = self.matrix_im_t.view(.., ..);
            let values_re = matmul_par_view(coeffs_2d, &matrix_re_t_view, backend);
            let values_im = matmul_par_view(coeffs_2d, &matrix_im_t_view, backend);

            interleave_to_complex(&values_re, &values_im, out);
        }
    }

    /// Evaluate ND real tensor to complex (along specified dim)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional real tensor with coeffs.shape().dim(dim) == basis_size
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view with out.shape().dim(dim) == n_points
    pub fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
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
            // Fast path 1: dim == 0
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 1);
        } else {
            // General path: batched GEMM approach
            self.evaluate_nd_dz_to_batched(backend, coeffs, dim, out);
        }
        true
    }

    /// Batched GEMM implementation for evaluate_nd_dz_to with middle dimensions
    fn evaluate_nd_dz_to_batched(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
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

        let coeffs_ptr = coeffs.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for b in 0..batch_size {
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(
                    coeffs_ptr.add(b * coeffs_batch_stride),
                    mapping,
                )
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(
                    out_ptr.add(b * out_batch_stride),
                    mapping,
                )
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);
        }
    }

    /// Fit 2D complex tensor to real coefficients (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Real coefficients tensor, shape: [basis_size, extra_size]
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<f64, 2> {
        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = DTensor::<f64, 2>::zeros([basis_size, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.fit_2d_to(backend, values_2d, &mut out_view);
        out
    }

    /// Fit 2D complex tensor to real coefficients (along dim=0), writing to output
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output mutable view, shape: [basis_size, extra_size]
    pub fn fit_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
    ) {
        use crate::gemm::{matmul_par_to_viewmut, matmul_par_view};

        let (n_points, extra_size) = *values_2d.shape();
        let (out_rows, out_cols) = *out.shape();

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
        self.ensure_svd();

        // Flatten complex values to real: [n_points, extra_size] → [2*n_points, extra_size]
        let mut values_flat = DTensor::<f64, 2>::zeros([2 * n_points, extra_size]);
        flatten_complex_to_real_rows(values_2d, &mut values_flat);

        let svd_ext = self.svd.borrow();
        let svd_ext = svd_ext.as_ref().unwrap();
        let svd = &svd_ext.svd;

        // coeffs = V * S^{-1} * U^T * values_flat

        // 1. U^T * values_flat
        let ut_view = svd.ut.view(.., ..);
        let values_flat_view = values_flat.view(.., ..);
        let mut ut_values = matmul_par_view(&ut_view, &values_flat_view, backend);

        // 2. S^{-1} * (U^T * values_flat) - in-place division
        let min_dim = svd.s.len();
        for i in 0..min_dim {
            for j in 0..extra_size {
                ut_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^T * values_flat) → out
        let v_view = svd.v.view(.., ..);
        let ut_values_view = ut_values.view(.., ..);
        matmul_par_to_viewmut(&v_view, &ut_values_view, out, backend);
    }

    /// Ensure SVD is computed (lazy initialization)
    fn ensure_svd(&self) {
        if self.svd.borrow().is_none() {
            let n_points = self.n_points();
            let basis_size = self.basis_size();
            // For positive-only mode, we have symmetry: 2*n_points effective points
            let effective_points = 2 * n_points;
            if effective_points < basis_size {
                eprintln!(
                    "Warning: Effective number of sampling points ({} × 2 = {}) is less than basis size ({}). \
                     Fitting may be ill-conditioned.",
                    n_points, effective_points, basis_size
                );
            }
            let svd_ext = RealSVDExtended::from_matrix(&self.matrix_real);
            *self.svd.borrow_mut() = Some(svd_ext);
        }
    }

    /// Fit 2D complex tensor to real coefficients with configurable target dimension
    ///
    /// # Arguments
    /// * `values_2d` - Shape depends on dim: [n_points, extra] if dim=0, [extra, n_points] if dim=1
    /// * `out` - Output mutable view
    /// * `dim` - Target dimension (0 or 1)
    fn fit_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, f64, 2>,
        dim: usize,
    ) {
        use crate::gemm::{matmul_par_to_viewmut, matmul_par_view};

        let (values_rows, values_cols) = *values_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        // Compute SVD lazily
        self.ensure_svd();

        let svd_ext = self.svd.borrow();
        let svd_ext = svd_ext.as_ref().unwrap();
        let svd = &svd_ext.svd;
        let min_dim = svd.s.len();
        let n_points = self.n_points();
        let basis_size = self.basis_size();

        if dim == 0 {
            // values[n_points, extra] → coeffs[basis_size, extra]
            let extra_size = values_cols;

            assert_eq!(values_rows, n_points);
            assert_eq!(out_rows, basis_size);
            assert_eq!(out_cols, extra_size);

            // Flatten complex values to real
            let mut values_flat = DTensor::<f64, 2>::zeros([2 * n_points, extra_size]);
            flatten_complex_to_real_rows(values_2d, &mut values_flat);

            // coeffs = V * S^{-1} * U^T * values_flat
            let ut_view = svd.ut.view(.., ..);
            let values_flat_view = values_flat.view(.., ..);
            let mut ut_values = matmul_par_view(&ut_view, &values_flat_view, backend);

            for i in 0..min_dim {
                for j in 0..extra_size {
                    ut_values[[i, j]] /= svd.s[i];
                }
            }

            let v_view = svd.v.view(.., ..);
            let ut_values_view = ut_values.view(.., ..);
            matmul_par_to_viewmut(&v_view, &ut_values_view, out, backend);
        } else {
            // dim == 1
            // values[extra, n_points] → coeffs[extra, basis_size]
            let extra_size = values_rows;

            assert_eq!(values_cols, n_points);
            assert_eq!(out_rows, extra_size);
            assert_eq!(out_cols, basis_size);

            // Flatten complex values to real: [extra, n_points] → [extra, 2*n_points]
            let mut values_flat = DTensor::<f64, 2>::zeros([extra_size, 2 * n_points]);
            flatten_complex_to_real_cols(values_2d, &mut values_flat);

            // coeffs = values_flat * U * S^{-1} * V^T
            // Use pre-computed U and V^T
            let u_view = svd_ext.u.view(.., ..);
            let values_flat_view = values_flat.view(.., ..);
            let mut values_u = matmul_par_view(&values_flat_view, &u_view, backend); // [extra, min_dim]

            // Apply S^{-1}
            for i in 0..extra_size {
                for j in 0..min_dim {
                    values_u[[i, j]] /= svd.s[j];
                }
            }

            // Multiply by V^T (pre-computed)
            let vt_view = svd_ext.vt.view(.., ..);
            let values_u_view = values_u.view(.., ..);
            matmul_par_to_viewmut(&values_u_view, &vt_view, out, backend);
        }
    }

    /// Fit ND complex tensor to real coefficients (along specified dim)
    ///
    /// # Arguments
    /// * `values` - N-dimensional complex tensor with values.shape().dim(dim) == n_points
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view with out.shape().dim(dim) == basis_size
    pub fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
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
            // Fast path 1: dim == 0
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 0);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 1);
        } else {
            // General path: batched GEMM approach
            self.fit_nd_zd_to_batched(backend, values, dim, out);
        }
        true
    }

    /// Batched GEMM implementation for fit_nd_zd_to with middle dimensions
    fn fit_nd_zd_to_batched(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) {
        let rank = values.rank();
        let n_points = self.n_points();
        let basis_size = self.basis_size();

        // Calculate batch size and extra size
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
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(
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

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 0);
        }
    }
}

/// InplaceFitter implementation for ComplexToRealFitter
///
/// Supports `dz` (real → complex) for evaluate and `zd` (complex → real) for fit.
/// Also supports `zz` variants by extracting real parts (for positive_only Matsubara).
impl InplaceFitter for ComplexToRealFitter {
    fn n_points(&self) -> usize {
        self.n_points()
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
        ComplexToRealFitter::evaluate_nd_dz_to(self, backend, coeffs, dim, out)
    }

    /// Evaluate ND: Complex<f64> coeffs → Complex<f64> values
    ///
    /// For ComplexToRealFitter (positive_only Matsubara), the input coefficients
    /// are expected to have zero imaginary parts (since IR coefficients are real
    /// for physical Green's functions). This extracts real parts and delegates
    /// to evaluate_nd_dz_to.
    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        use mdarray::Shape;

        // Get input shape for real temporary buffer
        let mut coeffs_shape: Vec<usize> = Vec::with_capacity(coeffs.rank());
        coeffs.shape().with_dims(|dims| {
            for d in dims {
                coeffs_shape.push(*d);
            }
        });

        // Extract real parts to temporary buffer
        let total = coeffs.len();
        let mut real_buffer: Vec<f64> = Vec::with_capacity(total);
        for c in coeffs.iter() {
            real_buffer.push(c.re);
        }

        // Create view into the real buffer
        let shape: DynRank = Shape::from_dims(&coeffs_shape[..]);
        let real_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape);
            mdarray::View::new_unchecked(real_buffer.as_ptr(), mapping)
        };

        // Delegate to dz
        ComplexToRealFitter::evaluate_nd_dz_to(self, backend, &real_view, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        ComplexToRealFitter::fit_nd_zd_to(self, backend, values, dim, out)
    }

    /// Fit ND: Complex<f64> values → Complex<f64> coeffs
    ///
    /// For ComplexToRealFitter, this performs fit_zd and converts real output to complex
    /// (with zero imaginary parts). This is valid because the underlying IR coefficients
    /// are guaranteed to be real for physical Green's functions.
    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        use mdarray::Shape;

        // Get output shape for real temporary buffer
        let mut out_shape: Vec<usize> = Vec::with_capacity(out.rank());
        out.shape().with_dims(|dims| {
            for d in dims {
                out_shape.push(*d);
            }
        });

        // Create temporary real buffer
        let total = out.len();
        let mut real_buffer: Vec<f64> = vec![0.0; total];

        // Create view into the real buffer
        let shape: DynRank = Shape::from_dims(&out_shape[..]);
        let mut real_view = unsafe {
            let mapping = mdarray::DenseMapping::new(shape);
            mdarray::ViewMut::new_unchecked(real_buffer.as_mut_ptr(), mapping)
        };

        // Fit to real coefficients
        if !ComplexToRealFitter::fit_nd_zd_to(self, backend, values, dim, &mut real_view) {
            return false;
        }

        // Copy real coefficients to complex output (with zero imaginary parts)
        for (c, r) in out.iter_mut().zip(real_buffer.iter()) {
            *c = Complex::new(*r, 0.0);
        }

        true
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

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let re = i.powi(j);
            let im = (i * (j as f64) * 0.1).sin();
            Complex::new(re, im)
        });

        let fitter = ComplexToRealFitter::new(&matrix);

        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.5).collect();

        let values = fitter.evaluate(None, &coeffs);
        assert_eq!(values.len(), n_points);
        assert!(values.iter().any(|z| z.im.abs() > 1e-10));

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

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64;
            let j = idx[1] as f64;
            let phase = 2.0 * std::f64::consts::PI * i * j / (n_points as f64);
            Complex::new(phase.cos(), phase.sin()) / (j + 1.0)
        });

        let fitter = ComplexToRealFitter::new(&matrix);

        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64) * 0.3).collect();

        let values = fitter.evaluate(None, &coeffs);
        let fitted_coeffs = fitter.fit(None, &values);

        for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
            let error = (orig - fitted).abs();
            assert!(error < 1e-10, "Overdetermined roundtrip error: {}", error);
        }
    }

    #[test]
    fn test_nd_roundtrip() {
        use mdarray::Tensor;

        let n_points = 8;
        let basis_size = 4;
        let extra1 = 3;
        let extra2 = 2;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let re = i.powi(j);
            let im = (i * (j as f64) * 0.1).sin();
            Complex::new(re, im)
        });

        let fitter = ComplexToRealFitter::new(&matrix);

        // Test dim=0
        {
            let coeffs =
                Tensor::<f64, mdarray::DynRank>::from_fn(&[basis_size, extra1][..], |idx| {
                    (idx[0] + idx[1]) as f64 * 0.3
                });

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[n_points, extra1][..]);
            let mut fitted = Tensor::<f64, mdarray::DynRank>::zeros(&[basis_size, extra1][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs.expr(), 0, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 0, &mut fitted.expr_mut());

            for i in 0..basis_size {
                for j in 0..extra1 {
                    let error = (coeffs[&[i, j][..]] - fitted[&[i, j][..]]).abs();
                    assert!(
                        error < 1e-8,
                        "dim=0 roundtrip error at [{}, {}]: {}",
                        i,
                        j,
                        error
                    );
                }
            }
        }

        // Test dim=1
        {
            let coeffs =
                Tensor::<f64, mdarray::DynRank>::from_fn(&[extra1, basis_size][..], |idx| {
                    (idx[0] + idx[1]) as f64 * 0.3
                });

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, n_points][..]);
            let mut fitted = Tensor::<f64, mdarray::DynRank>::zeros(&[extra1, basis_size][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 1, &mut fitted.expr_mut());

            for i in 0..extra1 {
                for j in 0..basis_size {
                    let error = (coeffs[&[i, j][..]] - fitted[&[i, j][..]]).abs();
                    assert!(
                        error < 1e-8,
                        "dim=1 roundtrip error at [{}, {}]: {}",
                        i,
                        j,
                        error
                    );
                }
            }
        }

        // Test dim=1 in 3D array (middle dimension)
        {
            let coeffs = Tensor::<f64, mdarray::DynRank>::from_fn(
                &[extra1, basis_size, extra2][..],
                |idx| (idx[0] + idx[1] + idx[2]) as f64 * 0.2,
            );

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, n_points, extra2][..]);
            let mut fitted =
                Tensor::<f64, mdarray::DynRank>::zeros(&[extra1, basis_size, extra2][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 1, &mut fitted.expr_mut());

            for i in 0..extra1 {
                for j in 0..basis_size {
                    for k in 0..extra2 {
                        let error = (coeffs[&[i, j, k][..]] - fitted[&[i, j, k][..]]).abs();
                        assert!(
                            error < 1e-8,
                            "dim=1 (3D) roundtrip error at [{}, {}, {}]: {}",
                            i,
                            j,
                            k,
                            error
                        );
                    }
                }
            }
        }
    }
}
