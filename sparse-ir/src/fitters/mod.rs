//! Fitters for least-squares problems with various matrix types
//!
//! This module provides fitters for solving min ||A * coeffs - values||^2
//! where the matrix A and value types can vary:
//!
//! - [`RealMatrixFitter`]: Real matrix A ∈ R^{n×m}
//! - [`ComplexToRealFitter`]: Complex matrix A ∈ C^{n×m}, real coefficients
//! - [`ComplexMatrixFitter`]: Complex matrix A ∈ C^{n×m}, complex coefficients

pub mod common;
mod complex;
mod complex_to_real;
mod real;

pub use common::InplaceFitter;
pub(crate) use complex::ComplexMatrixFitter;
pub(crate) use complex_to_real::ComplexToRealFitter;
pub(crate) use real::RealMatrixFitter;

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::DTensor;
    use num_complex::Complex;

    #[test]
    fn test_fitter_dimensions() {
        let n_points = 8;
        let basis_size = 4;

        let matrix_real =
            DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| (idx[0] + idx[1]) as f64);
        let fitter_real = RealMatrixFitter::new(matrix_real);

        assert_eq!(fitter_real.n_points(), n_points);
        assert_eq!(fitter_real.basis_size(), basis_size);

        let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            Complex::new((idx[0] + idx[1]) as f64, 0.0)
        });
        let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

        assert_eq!(fitter_complex.n_points(), n_points);
        assert_eq!(fitter_complex.basis_size(), basis_size);
    }

    #[test]
    fn test_condition_number_from_singular_values_conventions() {
        use super::common::condition_number_from_singular_values as cond;

        assert_eq!(cond(&[4.0, 2.0, 0.5]), 8.0);
        // Independent of the order of the singular values
        assert_eq!(cond(&[0.5, 4.0, 2.0]), 8.0);
        assert_eq!(cond(&[3.0]), 1.0);
        // No singular values (a matrix with a zero dimension)
        assert_eq!(cond(&[]), 1.0);
        // Numerically singular
        assert_eq!(cond(&[1.0, 1e-16]), f64::INFINITY);
        assert_eq!(cond(&[1.0, 0.0]), f64::INFINITY);
        // A NaN singular value is never turned into a finite value
        assert!(cond(&[1.0, f64::NAN]).is_nan());
        assert!(cond(&[f64::NAN, 1.0]).is_nan());
    }

    #[test]
    fn test_complex_fitter_real_matrix_equivalence() {
        let n_points = 8;
        let basis_size = 4;

        let matrix_real = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            Complex::new(i.powi(j), 0.0)
        });

        let fitter_real = RealMatrixFitter::new(matrix_real);
        let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

        let coeffs: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

        let values_real = fitter_real.evaluate(None, &coeffs);
        let values_complex = fitter_complex.evaluate(None, &coeffs);

        for (v_real, v_complex) in values_real.iter().zip(values_complex.iter()) {
            assert!((v_real - v_complex.re).abs() < 1e-14, "Real part mismatch");
            assert!(v_complex.im.abs() < 1e-14, "Imaginary part should be ~0");
        }

        let values_complex_zero_im: Vec<Complex<f64>> =
            values_real.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let fitted_real = fitter_real.fit(None, &values_real);
        let fitted_complex = fitter_complex.fit(None, &values_complex_zero_im);

        for (real, complex) in fitted_real.iter().zip(fitted_complex.iter()) {
            assert!((real - complex).abs() < 1e-12, "Fitted coeffs mismatch");
        }
    }
}
