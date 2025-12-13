//! Complex matrix fitter: A ∈ C^{n×m}
//!
//! This module provides `ComplexMatrixFitter` for solving least-squares problems
//! where the matrix, coefficients, and values are all complex.

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView};
use num_complex::Complex;
use std::cell::RefCell;

use super::common::{combine_complex, compute_complex_svd, extract_real_parts_coeffs, ComplexSVD};

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
        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = vec![Complex::new(0.0, 0.0); n_points];
        self.evaluate_to(backend, coeffs, &mut out);
        out
    }

    /// Evaluate: coeffs (complex) → values (complex), writing to output slice
    ///
    /// Computes: out = A * coeffs
    pub fn evaluate_to(
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
        self.evaluate_2d_to(backend, &coeffs_view, &mut out_view);
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

        let basis_size = self.basis_size();
        let mut out = vec![Complex::new(0.0, 0.0); basis_size];
        self.fit_to(backend, values, &mut out);
        out
    }

    /// Fit: values (complex) → coeffs (complex), writing to output slice
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    pub fn fit_to(
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
        self.fit_2d_to(backend, &values_view, &mut out_view);
    }

    /// Evaluate 2D complex tensor (along dim=0) using matrix multiplication
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
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
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

    /// Evaluate 2D complex tensor (along dim=0), writing to a mutable view
    ///
    /// Computes: out = matrix * coeffs_2d
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output mutable view, shape: [n_points, extra_size] (will be overwritten)
    pub fn evaluate_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        use crate::gemm::matmul_par_to_viewmut;

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

        let matrix_view = self.matrix.view(.., ..);
        matmul_par_to_viewmut(&matrix_view, coeffs_2d, out, backend);
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
    ///
    /// # Note
    /// This is a wrapper around `fit_2d_to` that allocates the output tensor.
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = DTensor::<Complex<f64>, 2>::zeros([basis_size, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.fit_2d_to(backend, values_2d, &mut out_view);
        out
    }

    /// Fit 2D complex tensor (along dim=0), writing to a mutable view
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
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
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
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
        for i in 0..min_dim {
            for j in 0..extra_size {
                uh_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^H * values_2d) → out
        let v_view = svd.v.view(.., ..);
        let uh_values_view = uh_values.view(.., ..);
        matmul_par_to_viewmut(&v_view, &uh_values_view, out, backend);
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

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::DTensor;
    use num_complex::Complex;

    trait ErrorNorm {
        fn error_norm(&self) -> f64;
    }

    impl ErrorNorm for Complex<f64> {
        fn error_norm(&self) -> f64 {
            self.norm()
        }
    }

    #[test]
    fn test_roundtrip() {
        let n_points = 10;
        let basis_size = 5;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let mag = i.powi(j);
            let phase = (j as f64) * 0.5;
            Complex::new(mag * phase.cos(), mag * phase.sin())
        });

        let fitter = ComplexMatrixFitter::new(matrix);

        let coeffs: Vec<Complex<f64>> = (0..basis_size)
            .map(|i| Complex::new((i as f64 + 1.0) * 0.5, (i as f64) * 0.3))
            .collect();

        let values = fitter.evaluate(None, &coeffs);
        assert_eq!(values.len(), n_points);

        let fitted_coeffs = fitter.fit(None, &values);
        assert_eq!(fitted_coeffs.len(), basis_size);

        for (i, (orig, fitted)) in coeffs.iter().zip(fitted_coeffs.iter()).enumerate() {
            let error = (orig - fitted).error_norm();
            assert!(error < 1e-8, "Complex matrix roundtrip error at {}: {}", i, error);
        }
    }

    #[test]
    fn test_vs_complex_to_real() {
        use crate::fitters::ComplexToRealFitter;

        let n_points = 8;
        let basis_size = 4;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            Complex::new(i.powi(j), (i * (j as f64) * 0.1).sin())
        });

        let fitter_c2r = ComplexToRealFitter::new(&matrix);
        let fitter_complex = ComplexMatrixFitter::new(matrix);

        let coeffs_real: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

        let values = fitter_c2r.evaluate(None, &coeffs_real);

        let fitted_complex = fitter_complex.fit(None, &values);

        for (i, &coeff_real) in coeffs_real.iter().enumerate() {
            let diff_re = (coeff_real - fitted_complex[i].re).abs();
            let im = fitted_complex[i].im.abs();

            assert!(diff_re < 1e-10, "Real part mismatch at {}: {}", i, diff_re);
            assert!(im < 1e-10, "Imaginary part should be small at {}: {}", i, im);
        }
    }
}
