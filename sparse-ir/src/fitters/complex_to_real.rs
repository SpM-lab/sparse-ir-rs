//! Complex-to-Real matrix fitter: A ∈ C^{n×m}
//!
//! This module provides `ComplexToRealFitter` for solving least-squares problems
//! where the matrix and values are complex, but the coefficients are real.
//!
//! Strategy: Flatten complex to real problem
//!   A_real ∈ R^{2n×m}: [Re(A[0,:]); Im(A[0,:]); Re(A[1,:]); Im(A[1,:]); ...]
//!   values_flat ∈ R^{2n}: [Re(v[0]); Im(v[0]); Re(v[1]); Im(v[1]); ...]

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView};
use num_complex::Complex;
use std::cell::RefCell;

use super::common::{combine_complex, compute_real_svd, RealSVD};

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
}
