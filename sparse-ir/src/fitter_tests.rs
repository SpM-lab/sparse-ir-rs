use super::*;
use mdarray::DTensor;
use num_complex::Complex;

// Helper trait for error norm (from tests/common.rs)
trait ErrorNorm {
    fn error_norm(&self) -> f64;
}

impl ErrorNorm for Complex<f64> {
    fn error_norm(&self) -> f64 {
        self.norm()
    }
}

#[test]
fn test_real_matrix_fitter_roundtrip() {
    // Create a simple real matrix
    let n_points = 10;
    let basis_size = 5;

    // Create a full-rank matrix (Vandermonde-like)
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64); // Normalize to [0, 1)
        let j = idx[1] as i32;
        i.powi(j) // Vandermonde matrix
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create test coefficients
    let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.5).collect();

    // Evaluate
    let values = fitter.evaluate(None, &coeffs);
    assert_eq!(values.len(), n_points);

    // Fit back
    let fitted_coeffs = fitter.fit(None, &values);
    assert_eq!(fitted_coeffs.len(), basis_size);

    // Check roundtrip
    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let error = (orig - fitted).abs();
        assert!(error < 1e-10, "Roundtrip error: {}", error);
    }
}

#[test]
fn test_real_matrix_fitter_overdetermined() {
    // Overdetermined system: n_points > basis_size
    let n_points = 20;
    let basis_size = 5;

    // Create a matrix with known structure
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64;
        let j = idx[1] as f64;
        ((i + 1.0) / (n_points as f64)).powi(j as i32)
    });

    let fitter = RealMatrixFitter::new(matrix);

    let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64) * 0.3).collect();

    // Roundtrip
    let values = fitter.evaluate(None, &coeffs);
    let fitted_coeffs = fitter.fit(None, &values);

    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let error = (orig - fitted).abs();
        assert!(error < 1e-10, "Overdetermined roundtrip error: {}", error);
    }
}

#[test]
fn test_complex_to_real_fitter_roundtrip() {
    // Create a complex matrix
    let n_points = 10;
    let basis_size = 5;

    // Create a full-rank complex matrix (Vandermonde-like with phase)
    let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        let re = i.powi(j);
        let im = (i * (j as f64) * 0.1).sin(); // Add some imaginary part
        Complex::new(re, im)
    });

    let fitter = ComplexToRealFitter::new(&matrix);

    // Create real coefficients
    let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.5).collect();

    // Evaluate: real coeffs → complex values
    let values = fitter.evaluate(None, &coeffs);
    assert_eq!(values.len(), n_points);

    // All values should be complex (non-zero imaginary part expected)
    assert!(values.iter().any(|z| z.im.abs() > 1e-10));

    // Fit back: complex values → real coeffs
    let fitted_coeffs = fitter.fit(None, &values);
    assert_eq!(fitted_coeffs.len(), basis_size);

    // Check roundtrip
    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let error = (orig - fitted).abs();
        assert!(error < 1e-10, "Roundtrip error: {}", error);
    }
}

#[test]
fn test_complex_to_real_fitter_overdetermined() {
    // Overdetermined: n_points > basis_size
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

    // Roundtrip
    let values = fitter.evaluate(None, &coeffs);
    let fitted_coeffs = fitter.fit(None, &values);

    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let error = (orig - fitted).abs();
        assert!(error < 1e-10, "Overdetermined roundtrip error: {}", error);
    }
}

#[test]
fn test_fitter_dimensions() {
    let n_points = 8;
    let basis_size = 4;

    // Real fitter
    let matrix_real =
        DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| (idx[0] + idx[1]) as f64);
    let fitter_real = RealMatrixFitter::new(matrix_real);

    assert_eq!(fitter_real.n_points(), n_points);
    assert_eq!(fitter_real.basis_size(), basis_size);

    // Complex fitter
    let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        Complex::new((idx[0] + idx[1]) as f64, 0.0)
    });
    let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

    assert_eq!(fitter_complex.n_points(), n_points);
    assert_eq!(fitter_complex.basis_size(), basis_size);
}

#[test]
#[should_panic(expected = "must equal basis_size")]
fn test_real_fitter_wrong_coeffs_size() {
    let matrix = DTensor::<f64, 2>::from_fn([5, 3], |_| 1.0);
    let fitter = RealMatrixFitter::new(matrix);

    let wrong_coeffs = vec![1.0; 5]; // Should be 3
    let _values = fitter.evaluate(None, &wrong_coeffs);
}

#[test]
#[should_panic(expected = "must equal n_points")]
fn test_real_fitter_wrong_values_size() {
    let matrix = DTensor::<f64, 2>::from_fn([5, 3], |_| 1.0);
    let fitter = RealMatrixFitter::new(matrix);

    let wrong_values = vec![1.0; 10]; // Should be 5
    let _coeffs = fitter.fit(None, &wrong_values);
}

#[test]
fn test_complex_fitter_real_matrix_equivalence() {
    // When complex matrix has zero imaginary part,
    // ComplexToRealFitter should match RealMatrixFitter

    let n_points = 8;
    let basis_size = 4;

    // Use Vandermonde matrix for full rank
    let matrix_real = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        Complex::new(i.powi(j), 0.0) // Zero imaginary part
    });

    let fitter_real = RealMatrixFitter::new(matrix_real);
    let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

    let coeffs: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

    // Evaluate
    let values_real = fitter_real.evaluate(None, &coeffs);
    let values_complex = fitter_complex.evaluate(None, &coeffs);

    // Complex values should have negligible imaginary part
    for (v_real, v_complex) in values_real.iter().zip(values_complex.iter()) {
        assert!((v_real - v_complex.re).abs() < 1e-14, "Real part mismatch");
        assert!(v_complex.im.abs() < 1e-14, "Imaginary part should be ~0");
    }

    // Fit (use complex values with zero imaginary)
    let values_complex_zero_im: Vec<Complex<f64>> =
        values_real.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let fitted_real = fitter_real.fit(None, &values_real);
    let fitted_complex = fitter_complex.fit(None, &values_complex_zero_im);

    // Should give same coefficients
    for (real, complex) in fitted_real.iter().zip(fitted_complex.iter()) {
        assert!((real - complex).abs() < 1e-12, "Fitted coeffs mismatch");
    }
}

#[test]
fn test_complex_matrix_fitter_roundtrip() {
    // Create a complex matrix with both real and imaginary parts
    let n_points = 10;
    let basis_size = 5;

    // Vandermonde-like matrix with phase
    let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        let mag = i.powi(j);
        let phase = (j as f64) * 0.5; // Simpler phase
        Complex::new(mag * phase.cos(), mag * phase.sin())
    });

    let fitter = ComplexMatrixFitter::new(matrix);

    // Create complex coefficients
    let coeffs: Vec<Complex<f64>> = (0..basis_size)
        .map(|i| Complex::new((i as f64 + 1.0) * 0.5, (i as f64) * 0.3))
        .collect();

    // Evaluate
    let values = fitter.evaluate(None, &coeffs);
    assert_eq!(values.len(), n_points);

    // Fit back
    let fitted_coeffs = fitter.fit(None, &values);
    assert_eq!(fitted_coeffs.len(), basis_size);

    // Check roundtrip
    for (i, (orig, fitted)) in coeffs.iter().zip(fitted_coeffs.iter()).enumerate() {
        let error = (orig - fitted).error_norm();
        assert!(
            error < 1e-8,
            "Complex matrix roundtrip error at {}: orig={}, fitted={}, error={}",
            i,
            orig,
            fitted,
            error
        );
    }
}

#[test]
fn test_complex_matrix_fitter_vs_complex_to_real() {
    // When coefficients are real, ComplexMatrixFitter should give
    // complex coefficients with negligible imaginary part

    let n_points = 8;
    let basis_size = 4;

    let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        Complex::new(i.powi(j), (i * (j as f64) * 0.1).sin())
    });

    let fitter_c2r = ComplexToRealFitter::new(&matrix);
    let fitter_complex = ComplexMatrixFitter::new(matrix);

    // Real coefficients
    let coeffs_real: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

    // Evaluate with ComplexToRealFitter
    let values = fitter_c2r.evaluate(None, &coeffs_real);

    // Fit with ComplexMatrixFitter (should give complex coeffs with small imaginary)
    let fitted_complex = fitter_complex.fit(None, &values);

    // Real parts should match, imaginary parts should be small
    for (i, &coeff_real) in coeffs_real.iter().enumerate() {
        let diff_re = (coeff_real - fitted_complex[i].re).abs();
        let im = fitted_complex[i].im.abs();

        assert!(diff_re < 1e-10, "Real part mismatch at {}: {}", i, diff_re);
        assert!(
            im < 1e-10,
            "Imaginary part should be small at {}: {}",
            i,
            im
        );
    }
}

#[test]
fn test_evaluate_2d_to_matches_evaluate_2d_basic() {
    // Test that evaluate_2d_to produces identical results to evaluate_2d
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create 2D coefficients
    let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
    });
    let coeffs_view = coeffs_2d.view(.., ..);

    // Use existing evaluate_2d
    let expected = fitter.evaluate_2d(None, &coeffs_view);

    // Use evaluate_2d_to
    let mut actual = DTensor::<f64, 2>::from_elem([n_points, extra_size], 0.0);
    {
        let mut actual_view = actual.view_mut(.., ..);
        fitter.evaluate_2d_to(None, &coeffs_view, &mut actual_view);
    }

    // Compare results
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..n_points {
        for j in 0..extra_size {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: actual={}, expected={}, diff={}",
                i, j, actual[[i, j]], expected[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_evaluate_2d_to_matches_evaluate_2d() {
    // Test that evaluate_2d_to produces identical results to evaluate_2d
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create 2D coefficients
    let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
    });
    let coeffs_view = coeffs_2d.view(.., ..);

    // Use existing evaluate_2d
    let expected = fitter.evaluate_2d(None, &coeffs_view);

    // Use new evaluate_2d_to with a mutable view
    let mut actual = DTensor::<f64, 2>::from_elem([n_points, extra_size], 0.0);
    {
        let mut actual_view = actual.view_mut(.., ..);
        fitter.evaluate_2d_to(None, &coeffs_view, &mut actual_view);
    }

    // Compare results
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..n_points {
        for j in 0..extra_size {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: actual={}, expected={}, diff={}",
                i, j, actual[[i, j]], expected[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_fit_2d_to_matches_fit_2d() {
    // Test that fit_2d_to produces identical results to fit_2d
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create 2D values (sampling points)
    let values_2d = DTensor::<f64, 2>::from_fn([n_points, extra_size], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
    });
    let values_view = values_2d.view(.., ..);

    // Use existing fit_2d
    let expected = fitter.fit_2d(None, &values_view);

    // Use fit_2d_to
    let mut actual = DTensor::<f64, 2>::from_elem([basis_size, extra_size], 0.0);
    {
        let mut actual_view = actual.view_mut(.., ..);
        fitter.fit_2d_to(None, &values_view, &mut actual_view);
    }

    // Compare results
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..basis_size {
        for j in 0..extra_size {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: actual={}, expected={}, diff={}",
                i, j, actual[[i, j]], expected[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_evaluate_fit_roundtrip_with_inplace() {
    // Test roundtrip: coeffs → evaluate_2d_to → values → fit_2d_to → coeffs
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a well-conditioned Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Original coefficients
    let coeffs = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
        (idx[0] as f64 + 1.0) * 0.3 + (idx[1] as f64) * 0.1
    });
    let coeffs_view = coeffs.view(.., ..);

    // evaluate_2d_to
    let mut values = DTensor::<f64, 2>::from_elem([n_points, extra_size], 0.0);
    {
        let mut values_view_mut = values.view_mut(.., ..);
        fitter.evaluate_2d_to(None, &coeffs_view, &mut values_view_mut);
    }

    // fit_2d_to
    let values_view = values.view(.., ..);
    let mut fitted_coeffs = DTensor::<f64, 2>::from_elem([basis_size, extra_size], 0.0);
    {
        let mut fitted_view = fitted_coeffs.view_mut(.., ..);
        fitter.fit_2d_to(None, &values_view, &mut fitted_view);
    }

    // Compare
    for i in 0..basis_size {
        for j in 0..extra_size {
            let diff = (fitted_coeffs[[i, j]] - coeffs[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Roundtrip mismatch at [{}, {}]: orig={}, fitted={}, diff={}",
                i, j, coeffs[[i, j]], fitted_coeffs[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_evaluate_complex_2d_to_matches_evaluate_complex_2d() {
    // Test that evaluate_complex_2d_to produces identical results
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create 2D complex coefficients
    let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
        Complex::new(
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
            (idx[0] as f64) * 0.3,
        )
    });
    let coeffs_view = coeffs_2d.view(.., ..);

    // Use existing evaluate_complex_2d
    let expected = fitter.evaluate_complex_2d(None, &coeffs_view);

    // Use evaluate_complex_2d_to
    let mut actual = DTensor::<Complex<f64>, 2>::from_elem([n_points, extra_size], Complex::new(0.0, 0.0));
    {
        let mut actual_view = actual.view_mut(.., ..);
        fitter.evaluate_complex_2d_to(None, &coeffs_view, &mut actual_view);
    }

    // Compare results
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..n_points {
        for j in 0..extra_size {
            let diff = (actual[[i, j]] - expected[[i, j]]).norm();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: actual={}, expected={}, diff={}",
                i, j, actual[[i, j]], expected[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_fit_complex_2d_to_matches_fit_complex_2d() {
    // Test that fit_complex_2d_to produces identical results
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    // Create a Vandermonde-like matrix
    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Create 2D complex values
    let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
        Complex::new(
            (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
            (idx[0] as f64) * 0.2,
        )
    });
    let values_view = values_2d.view(.., ..);

    // Use existing fit_complex_2d
    let expected = fitter.fit_complex_2d(None, &values_view);

    // Use fit_complex_2d_to
    let mut actual = DTensor::<Complex<f64>, 2>::from_elem([basis_size, extra_size], Complex::new(0.0, 0.0));
    {
        let mut actual_view = actual.view_mut(.., ..);
        fitter.fit_complex_2d_to(None, &values_view, &mut actual_view);
    }

    // Compare results
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..basis_size {
        for j in 0..extra_size {
            let diff = (actual[[i, j]] - expected[[i, j]]).norm();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: actual={}, expected={}, diff={}",
                i, j, actual[[i, j]], expected[[i, j]], diff
            );
        }
    }
}

#[test]
fn test_complex_roundtrip_with_inplace() {
    // Test roundtrip for complex: coeffs → evaluate_complex_2d_to → values → fit_complex_2d_to → coeffs
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // Original complex coefficients
    let coeffs = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
        Complex::new((idx[0] as f64 + 1.0) * 0.3, (idx[1] as f64) * 0.1)
    });
    let coeffs_view = coeffs.view(.., ..);

    // evaluate_complex_2d_to
    let mut values = DTensor::<Complex<f64>, 2>::from_elem([n_points, extra_size], Complex::new(0.0, 0.0));
    {
        let mut values_view_mut = values.view_mut(.., ..);
        fitter.evaluate_complex_2d_to(None, &coeffs_view, &mut values_view_mut);
    }

    // fit_complex_2d_to
    let values_view = values.view(.., ..);
    let mut fitted_coeffs = DTensor::<Complex<f64>, 2>::from_elem([basis_size, extra_size], Complex::new(0.0, 0.0));
    {
        let mut fitted_view = fitted_coeffs.view_mut(.., ..);
        fitter.fit_complex_2d_to(None, &values_view, &mut fitted_view);
    }

    // Compare
    for i in 0..basis_size {
        for j in 0..extra_size {
            let diff = (fitted_coeffs[[i, j]] - coeffs[[i, j]]).norm();
            assert!(
                diff < 1e-10,
                "Roundtrip mismatch at [{}, {}]: orig={}, fitted={}, diff={}",
                i, j, coeffs[[i, j]], fitted_coeffs[[i, j]], diff
            );
        }
    }
}

// ============================================================================
// Tests for evaluate_2d_to_dim
// ============================================================================

#[test]
fn test_evaluate_2d_to_dim0() {
    // Test dim=0: out[n_points, extra] = matrix[n_points, basis] * coeffs[basis, extra]
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

    // Expected using evaluate_2d
    let expected = fitter.evaluate_2d(None, &coeffs_view);

    // Actual using evaluate_2d_to_dim
    let mut out_data = vec![0.0f64; n_points * extra_size];
    let mut out_view = unsafe {
        let mapping = mdarray::DenseMapping::new((n_points, extra_size));
        mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_data.as_mut_ptr(), mapping)
    };
    fitter.evaluate_2d_to_dim(None, &coeffs_view, &mut out_view, 0);

    // Compare
    for i in 0..n_points {
        for j in 0..extra_size {
            let e = expected[[i, j]];
            let a = out_data[i * extra_size + j];
            let diff = (e - a).abs();
            assert!(
                diff < 1e-14,
                "Mismatch at [{}, {}]: expected={}, actual={}, diff={}",
                i, j, e, a, diff
            );
        }
    }
}

#[test]
fn test_evaluate_2d_to_dim1() {
    // Test dim=1: out[extra, n_points] = coeffs[extra, basis] * matrix^T[basis, n_points]
    let n_points = 10;
    let basis_size = 5;
    let extra_size = 3;

    let matrix = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5)
    });

    let fitter = RealMatrixFitter::new(matrix);

    // coeffs shape: [extra, basis]
    let coeffs = DTensor::<f64, 2>::from_fn([extra_size, basis_size], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.3)
    });
    let coeffs_view = coeffs.view(.., ..);

    // Compute expected: out[extra, n_points] = coeffs[extra, basis] * matrix^T[basis, n_points]
    // This is equivalent to: for each row of coeffs, multiply by matrix^T
    let mut expected = DTensor::<f64, 2>::from_elem([extra_size, n_points], 0.0);
    for e in 0..extra_size {
        for p in 0..n_points {
            let mut sum = 0.0;
            for b in 0..basis_size {
                sum += coeffs[[e, b]] * fitter.matrix[[p, b]]; // matrix^T[b,p] = matrix[p,b]
            }
            expected[[e, p]] = sum;
        }
    }

    // Actual using evaluate_2d_to_dim
    let mut out_data = vec![0.0f64; extra_size * n_points];
    let mut out_view = unsafe {
        let mapping = mdarray::DenseMapping::new((extra_size, n_points));
        mdarray::DViewMut::<'_, f64, 2>::new_unchecked(out_data.as_mut_ptr(), mapping)
    };
    fitter.evaluate_2d_to_dim(None, &coeffs_view, &mut out_view, 1);

    // Compare
    for e in 0..extra_size {
        for p in 0..n_points {
            let exp = expected[[e, p]];
            let act = out_data[e * n_points + p];
            let diff = (exp - act).abs();
            assert!(
                diff < 1e-12,
                "Mismatch at [{}, {}]: expected={}, actual={}, diff={}",
                e, p, exp, act, diff
            );
        }
    }
}
