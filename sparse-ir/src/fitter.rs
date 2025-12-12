//! Fitters for least-squares problems with various matrix types
//!
//! This module provides fitters for solving min ||A * coeffs - values||^2
//! where the matrix A and value types can vary.

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView};
use num_complex::Complex;
use std::cell::RefCell;

/// SVD decomposition for real matrices
struct RealSVD {
    ut: DTensor<f64, 2>, // (min_dim, n_rows) - U^T
    s: Vec<f64>,         // (min_dim,)
    v: DTensor<f64, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl RealSVD {
    fn new(u: DTensor<f64, 2>, s: Vec<f64>, vt: DTensor<f64, 2>) -> Self {
        // Check dimensions
        let (_, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut and v from u and vt
        let ut = u.transpose().to_tensor(); // (min_dim, n_rows)
        let v = vt.transpose().to_tensor(); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

/// SVD decomposition for complex matrices
struct ComplexSVD {
    ut: DTensor<Complex<f64>, 2>, // (min_dim, n_rows) - U^H
    s: Vec<f64>,                  // (min_dim,) - singular values are real
    v: DTensor<Complex<f64>, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl ComplexSVD {
    fn new(u: DTensor<Complex<f64>, 2>, s: Vec<f64>, vt: DTensor<Complex<f64>, 2>) -> Self {
        // Check dimensions
        let (u_rows, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut (U^H, conjugate transpose) and v from u and vt
        let ut = DTensor::<Complex<f64>, 2>::from_fn([u_cols, u_rows], |idx| {
            u[[idx[1], idx[0]]].conj() // conjugate transpose: U^H
        });
        let v = vt.transpose().to_tensor(); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

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

        // Convert to column vector and use evaluate_2d
        let basis_size = coeffs.len();
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, 1], |idx| coeffs[idx[0]]);
        let coeffs_view = coeffs_2d.view(.., ..);
        let values_2d = self.evaluate_2d(backend, &coeffs_view);

        // Extract as Vec
        let n_points = self.n_points();
        (0..n_points).map(|i| values_2d[[i, 0]]).collect()
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

        // Convert values to column vector and use fit_2d
        let n = values.len();
        let values_2d = DTensor::<f64, 2>::from_fn([n, 1], |idx| values[idx[0]]);
        let values_view = values_2d.view(.., ..);
        let coeffs_2d = self.fit_2d(backend, &values_view);

        // Extract as Vec
        let basis_size = self.basis_size();
        (0..basis_size).map(|i| coeffs_2d[[i, 0]]).collect()
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
        use num_complex::Complex;

        assert_eq!(
            values.len(),
            self.n_points(),
            "values.len()={} must equal n_points={}",
            values.len(),
            self.n_points()
        );

        // Convert to 2D and use fit_complex_2d
        let n = values.len();
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n, 1], |idx| values[idx[0]]);
        let values_view = values_2d.view(.., ..);
        let coeffs_2d = self.fit_complex_2d(backend, &values_view);

        // Extract as Vec
        let basis_size = self.basis_size();
        (0..basis_size).map(|i| coeffs_2d[[i, 0]]).collect()
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
        use num_complex::Complex;

        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        // Convert to column vector and use evaluate_complex_2d
        let basis_size = coeffs.len();
        let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, 1], |idx| coeffs[idx[0]]);
        let coeffs_view = coeffs_2d.view(.., ..);
        let values_2d = self.evaluate_complex_2d(backend, &coeffs_view);

        // Extract as Vec
        let n_points = self.n_points();
        (0..n_points).map(|i| values_2d[[i, 0]]).collect()
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
    pub fn evaluate_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
    ) -> mdarray::DTensor<f64, 2> {
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // values_2d = matrix * coeffs_2d
        let matrix_view = self.matrix.view(.., ..);
        matmul_par_view(&matrix_view, coeffs_2d, backend)
    }

    /// Evaluate 2D real tensor (along dim=0) with in-place output
    ///
    /// Computes: out = matrix * coeffs_2d
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output tensor, shape: [n_points, extra_size] (will be overwritten)
    ///
    /// # Safety
    /// The output tensor must have the correct shape [n_points, extra_size].
    pub fn evaluate_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DTensor<f64, 2>,
    ) {
        use crate::gemm::matmul_par_overwrite_view;

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

        // out = matrix * coeffs_2d
        let matrix_view = self.matrix.view(.., ..);
        matmul_par_overwrite_view(&matrix_view, coeffs_2d, out, backend);
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
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, f64, 2>,
    ) -> mdarray::DTensor<f64, 2> {
        use crate::gemm::matmul_par_view;

        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_real_svd(&self.matrix);
            *self.svd.borrow_mut() = Some(svd);
        }

        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();

        // coeffs_2d = V * S^{-1} * U^T * values_2d

        // 1. U^T * values_2d
        let ut_view = svd.ut.view(.., ..);
        let mut ut_values = matmul_par_view(&ut_view, values_2d, backend); // [min_dim, extra_size]

        // 2. S^{-1} * (U^T * values_2d) - in-place division
        let min_dim = svd.s.len();
        // In-place division by singular values
        for i in 0..min_dim {
            for j in 0..extra_size {
                ut_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^T * values_2d)
        let v_view = svd.v.view(.., ..);
        let ut_values_view = ut_values.view(.., ..);
        matmul_par_view(&v_view, &ut_values_view, backend) // [basis_size, extra_size]
    }

    /// Fit 2D real tensor (along dim=0) with in-place output
    ///
    /// Efficiently computes: out = V * S^{-1} * U^T * values_2d
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output tensor, shape: [basis_size, extra_size] (will be overwritten)
    ///
    /// # Note
    /// An intermediate buffer of size [min_dim, extra_size] is still allocated internally.
    pub fn fit_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, f64, 2>,
        out: &mut mdarray::DTensor<f64, 2>,
    ) {
        use crate::gemm::{matmul_par_overwrite_view, matmul_par_view};

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
        matmul_par_overwrite_view(&v_view, &ut_values_view, out, backend);
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
        let (n_points, _extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        // Extract real and imaginary parts (need to convert to DTensor for extract functions)
        let values_tensor =
            DTensor::<Complex<f64>, 2>::from_fn(*values_2d.shape(), |idx| values_2d[idx]);
        let values_re = extract_real_parts(&values_tensor);
        let values_im = extract_imag_parts(&values_tensor);

        // Fit real and imaginary parts separately using matrix multiplication
        let values_re_view = values_re.view(.., ..);
        let values_im_view = values_im.view(.., ..);
        let coeffs_re = self.fit_2d(backend, &values_re_view);
        let coeffs_im = self.fit_2d(backend, &values_im_view);

        // Combine back to complex
        combine_complex_coeffs(&coeffs_re, &coeffs_im)
    }

    /// Fit 2D complex values with in-place output
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output tensor, shape: [basis_size, extra_size] (will be overwritten)
    ///
    /// # Note
    /// Internal buffers for real/imaginary parts are still allocated.
    pub fn fit_complex_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DTensor<Complex<f64>, 2>,
    ) {
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

        // Extract real and imaginary parts
        let values_re = DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| values_2d[idx].re);
        let values_im = DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| values_2d[idx].im);

        // Fit real and imaginary parts separately
        let values_re_view = values_re.view(.., ..);
        let values_im_view = values_im.view(.., ..);
        let mut coeffs_re = DTensor::<f64, 2>::from_elem([self.basis_size(), extra_size], 0.0);
        let mut coeffs_im = DTensor::<f64, 2>::from_elem([self.basis_size(), extra_size], 0.0);
        self.fit_2d_to(backend, &values_re_view, &mut coeffs_re);
        self.fit_2d_to(backend, &values_im_view, &mut coeffs_im);

        // Write directly to output
        for i in 0..out_rows {
            for j in 0..out_cols {
                out[[i, j]] = Complex::new(coeffs_re[[i, j]], coeffs_im[[i, j]]);
            }
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
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // Extract real and imaginary parts (need to convert to DTensor for extract functions)
        let coeffs_tensor =
            DTensor::<Complex<f64>, 2>::from_fn(*coeffs_2d.shape(), |idx| coeffs_2d[idx]);
        let coeffs_re = extract_real_parts_coeffs(&coeffs_tensor);
        let coeffs_im = extract_imag_parts_coeffs(&coeffs_tensor);

        // Evaluate real and imaginary parts separately: values = matrix * coeffs
        let matrix_view = self.matrix.view(.., ..);
        let coeffs_re_view = coeffs_re.view(.., ..);
        let coeffs_im_view = coeffs_im.view(.., ..);
        let values_re = matmul_par_view(&matrix_view, &coeffs_re_view, backend);
        let values_im = matmul_par_view(&matrix_view, &coeffs_im_view, backend);

        // Combine to complex
        combine_complex(&values_re, &values_im)
    }

    /// Evaluate 2D complex coefficients with in-place output
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output tensor, shape: [n_points, extra_size] (will be overwritten)
    ///
    /// # Note
    /// Internal buffers for real/imaginary parts are still allocated.
    pub fn evaluate_complex_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DTensor<Complex<f64>, 2>,
    ) {
        use crate::gemm::matmul_par_overwrite_view;

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

        // Extract real and imaginary parts
        let coeffs_re = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].re);
        let coeffs_im = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].im);

        // Evaluate real and imaginary parts separately
        let matrix_view = self.matrix.view(.., ..);
        let coeffs_re_view = coeffs_re.view(.., ..);
        let coeffs_im_view = coeffs_im.view(.., ..);

        let mut values_re = DTensor::<f64, 2>::from_elem([self.n_points(), extra_size], 0.0);
        let mut values_im = DTensor::<f64, 2>::from_elem([self.n_points(), extra_size], 0.0);
        matmul_par_overwrite_view(&matrix_view, &coeffs_re_view, &mut values_re, backend);
        matmul_par_overwrite_view(&matrix_view, &coeffs_im_view, &mut values_im, backend);

        // Write directly to output
        for i in 0..out_rows {
            for j in 0..out_cols {
                out[[i, j]] = Complex::new(values_re[[i, j]], values_im[[i, j]]);
            }
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
    // A_real ∈ R^{2n×m}: flattened complex matrix
    // A_real[2i,   :] = Re(A[i, :])
    // A_real[2i+1, :] = Im(A[i, :])
    matrix_real: DTensor<f64, 2>,         // (2*n_points, basis_size)
    pub matrix: DTensor<Complex<f64>, 2>, // (n_points, basis_size) - original complex matrix
    svd: RefCell<Option<RealSVD>>,
    n_points: usize, // Original complex point count
}

impl ComplexToRealFitter {
    /// Create from complex matrix A ∈ C^{n×m}
    pub fn new(matrix_complex: &DTensor<Complex<f64>, 2>) -> Self {
        let (n_points, basis_size) = *matrix_complex.shape();

        // Flatten to real: (2*n_points, basis_size)
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

        Self {
            matrix_real,
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
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // values_2d = A * coeffs_2d (complex matrix * real coeffs)
        // Split into real and imaginary parts for GEMM
        // Extract real and imaginary parts of matrix
        let matrix_re = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].re);
        let matrix_im = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].im);

        // Compute real and imaginary parts separately using GEMM
        let matrix_re_view = matrix_re.view(.., ..);
        let matrix_im_view = matrix_im.view(.., ..);
        let values_re = matmul_par_view(&matrix_re_view, coeffs_2d, backend);
        let values_im = matmul_par_view(&matrix_im_view, coeffs_2d, backend);

        // Combine to complex
        combine_complex(&values_re, &values_im)
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
        use crate::gemm::matmul_par_view;

        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_real_svd(&self.matrix_real);
            *self.svd.borrow_mut() = Some(svd);
        }

        // Flatten complex values to real: [n_points, extra_size] → [2*n_points, extra_size]
        let values_flat = DTensor::<f64, 2>::from_fn([2 * n_points, extra_size], |idx| {
            let i = idx[0] / 2;
            let j = idx[1];
            let val = values_2d[[i, j]];
            if idx[0] % 2 == 0 { val.re } else { val.im }
        });

        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();

        // coeffs_2d = V * S^{-1} * U^T * values_flat

        // 1. U^T * values_flat
        let ut_view = svd.ut.view(.., ..);
        let values_flat_view = values_flat.view(.., ..);
        let mut ut_values = matmul_par_view(&ut_view, &values_flat_view, backend);

        // 2. S^{-1} * (U^T * values_flat) - in-place division
        let min_dim = svd.s.len();
        if *ut_values.shape() != (min_dim, extra_size) {
            panic!(
                "ut_values.shape()={:?} must equal [min_dim, extra_size]=[{}, {}]",
                ut_values.shape(),
                min_dim,
                extra_size
            );
        }

        // In-place division by singular values
        for i in 0..min_dim {
            for j in 0..extra_size {
                ut_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^T * values_flat)
        let v_view = svd.v.view(.., ..);
        let ut_values_view = ut_values.view(.., ..);
        matmul_par_view(&v_view, &ut_values_view, backend)
    }
}

/// Fitter for complex matrix with complex coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all complex
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| Complex::new(...));
/// let fitter = ComplexMatrixFitter::new(matrix);
///
/// let coeffs: Vec<Complex<f64>> = vec![...];
/// let values = fitter.evaluate(&coeffs);  // → Vec<Complex<f64>>
/// let fitted_coeffs = fitter.fit(&values);  // ← Vec<Complex<f64>>, → Vec<Complex<f64>>
/// ```
pub(crate) struct ComplexMatrixFitter {
    pub matrix: DTensor<Complex<f64>, 2>, // (n_points, basis_size)
    svd: RefCell<Option<ComplexSVD>>,
}

impl ComplexMatrixFitter {
    /// Create a new fitter with the given complex matrix
    pub fn new(matrix: DTensor<Complex<f64>, 2>) -> Self {
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

    /// Evaluate: coeffs (complex) → values (complex)
    ///
    /// Computes: values = A * coeffs using GEMM
    pub fn evaluate(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        use crate::gemm::matmul_par;

        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Convert coeffs to column vector
        let coeffs_col = DTensor::<Complex<f64>, 2>::from_fn([basis_size, 1], |idx| coeffs[idx[0]]);

        // values = A * coeffs
        let values_col = matmul_par(&self.matrix, &coeffs_col, backend);

        // Extract as Vec
        (0..n_points).map(|i| values_col[[i, 0]]).collect()
    }

    /// Fit: values (complex) → coeffs (complex)
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    pub fn fit(
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

        // Convert values to column vector and use fit_2d
        let n = values.len();
        let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n, 1], |idx| values[idx[0]]);
        let values_view = values_2d.view(.., ..);
        let coeffs_2d = self.fit_2d(backend, &values_view);

        // Extract as Vec
        let basis_size = self.basis_size();
        (0..basis_size).map(|i| coeffs_2d[[i, 0]]).collect()
    }

    /// Evaluate 2D complex tensor (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    pub fn evaluate_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // values_2d = A * coeffs_2d
        let matrix_view = self.matrix.view(.., ..);
        matmul_par_view(&matrix_view, coeffs_2d, backend)
    }

    /// Evaluate 2D real tensor to complex values (along dim=0) using matrix multiplication
    ///
    /// Computes: values_2d = A * coeffs_2d where A is complex and coeffs_2d is real
    ///
    /// # Arguments
    /// * `coeffs_2d` - Real coefficients, shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Complex values, shape: [n_points, extra_size]
    pub fn evaluate_2d_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // Split matrix into real and imaginary parts
        let matrix_re = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].re);
        let matrix_im = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].im);

        // Compute real and imaginary parts separately
        let matrix_re_view = matrix_re.view(.., ..);
        let matrix_im_view = matrix_im.view(.., ..);
        let values_re = matmul_par_view(&matrix_re_view, coeffs_2d, backend);
        let values_im = matmul_par_view(&matrix_im_view, coeffs_2d, backend);

        // Combine to complex
        combine_complex(&values_re, &values_im)
    }

    /// Fit 2D complex tensor (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Complex coefficients tensor, shape: [basis_size, extra_size]
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        use crate::gemm::matmul_par_view;

        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_complex_svd(&self.matrix);
            *self.svd.borrow_mut() = Some(svd);
        }

        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();

        // coeffs_2d = V * S^{-1} * U^H * values_2d

        // 1. U^H * values_2d (ut is already U^H)
        let ut_view = svd.ut.view(.., ..);
        let mut uh_values = matmul_par_view(&ut_view, values_2d, backend); // [min_dim, extra_size]

        // 2. S^{-1} * (U^H * values_2d) - in-place division
        let min_dim = svd.s.len();
        // In-place division by singular values
        for i in 0..min_dim {
            for j in 0..extra_size {
                uh_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^H * values_2d)
        let v_view = svd.v.view(.., ..);
        let uh_values_view = uh_values.view(.., ..);
        matmul_par_view(&v_view, &uh_values_view, backend) // [basis_size, extra_size]
    }

    /// Fit 2D complex values to real coefficients (along dim=0)
    ///
    /// This method fits complex values at Matsubara frequencies to real IR coefficients.
    /// It takes the real part of the least-squares solution.
    ///
    /// # Arguments
    /// * `values_2d` - Complex values, shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Real coefficients, shape: [basis_size, extra_size]
    pub fn fit_2d_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<f64, 2> {
        // Fit as complex, then take real part
        let coeffs_complex = self.fit_2d(backend, values_2d);

        // Extract real part
        extract_real_parts_coeffs(&coeffs_complex)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract real parts from complex tensor
fn extract_real_parts(values_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (n_points, extra_size) = *values_2d.shape();
    DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| values_2d[idx].re)
}

/// Extract imaginary parts from complex tensor
fn extract_imag_parts(values_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (n_points, extra_size) = *values_2d.shape();
    DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| values_2d[idx].im)
}

/// Combine real and imaginary parts into complex tensor
fn combine_complex(re: &DTensor<f64, 2>, im: &DTensor<f64, 2>) -> DTensor<Complex<f64>, 2> {
    let (n_points, extra_size) = *re.shape();
    DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
        Complex::new(re[idx], im[idx])
    })
}

/// Extract real parts from complex tensor (for coefficients)
fn extract_real_parts_coeffs(coeffs_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (basis_size, extra_size) = *coeffs_2d.shape();
    DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].re)
}

/// Extract imaginary parts from complex tensor (for coefficients)
fn extract_imag_parts_coeffs(coeffs_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (basis_size, extra_size) = *coeffs_2d.shape();
    DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].im)
}

/// Combine real and imaginary parts into complex tensor (for coefficients)
fn combine_complex_coeffs(re: &DTensor<f64, 2>, im: &DTensor<f64, 2>) -> DTensor<Complex<f64>, 2> {
    let (basis_size, extra_size) = *re.shape();
    DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
        Complex::new(re[idx], im[idx])
    })
}

/// Compute SVD of a real matrix using mdarray-linalg
fn compute_real_svd(matrix: &DTensor<f64, 2>) -> RealSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    let mut a = matrix.clone();
    let SVDDecomp { u, s, vt } = Faer.svd(&mut *a).expect("SVD computation failed");

    // Extract singular values from first row
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]]).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    RealSVD::new(u_trimmed, s_vec, vt_trimmed)
}

/// Compute SVD of a complex matrix directly
fn compute_complex_svd(matrix: &DTensor<Complex<f64>, 2>) -> ComplexSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    // Use matrix directly (Complex<f64> is compatible with faer's c64)
    let mut matrix_c64 = matrix.clone();

    // Compute complex SVD directly
    let SVDDecomp { u, s, vt } = Faer
        .svd(&mut *matrix_c64)
        .expect("Complex SVD computation failed");

    // Extract singular values from first row (they are real even though stored as Complex)
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]].re).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    ComplexSVD::new(u_trimmed, s_vec, vt_trimmed)
}

#[cfg(test)]
#[path = "fitter_tests.rs"]
mod tests;
