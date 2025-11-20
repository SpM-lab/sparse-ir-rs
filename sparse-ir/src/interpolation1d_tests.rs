use crate::gauss::{legendre_generic, legendre_vandermonde};
use crate::interpolation1d::{
    evaluate_interpolated_polynomial, interpolate_1d_legendre, legendre_collocation_matrix,
};
use crate::{CustomNumeric, Interpolate1D, Df64};
use mdarray::DTensor;

/// Test that the collocation matrix is approximately the inverse of the Vandermonde matrix
#[test]
fn test_legendre_collocation_matrix_inverse() {
    // Test with different sizes
    for n in [2, 3, 5, 10] {
        let gauss_rule = legendre_generic::<f64>(n).reseat(-1.0, 1.0);

        // Create Vandermonde matrix
        let vandermonde = legendre_vandermonde(&gauss_rule.x.to_vec(), n - 1);

        // Create collocation matrix
        let collocation = legendre_collocation_matrix(&gauss_rule);

        // Compute V * C and check if it's approximately the identity matrix
        let mut product = DTensor::<f64, 2>::from_elem([n, n], 0.0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    product[[i, j]] += vandermonde[[i, k]] * collocation[[k, j]];
                }
            }
        }

        // Check that V * C ≈ I
        let mut error = 0.0;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                error += (product[[i, j]] - expected).abs();
            }
        }
        error /= (n * n) as f64;

        println!("n={}, error={}", n, error);
        assert!(
            error < 1e-10,
            "Collocation matrix is not inverse of Vandermonde matrix for n={}: error={}",
            n,
            error
        );
    }
}

/// Test the interpolation method with various functions
#[test]
fn test_interpolate_1d_legendre_functions() {
    // Test with different sizes and functions
    for n in [2, 3, 5] {
        let gauss_rule = legendre_generic::<f64>(n).reseat(-1.0, 1.0);

        // Test different functions
        let test_functions = vec![
            |x: f64| x,         // Linear
            |x: f64| x * x,     // Quadratic
            |x: f64| x * x * x, // Cubic
            |x: f64| x.sin(),   // Sine
        ];

        for (func_idx, func) in test_functions.iter().enumerate() {
            // Sample function at Gauss points
            let values: Vec<f64> = gauss_rule.x.iter().map(|&x| func(x)).collect();

            // Get coefficients using the fast method
            let coeffs = interpolate_1d_legendre(&values, &gauss_rule);

            // Test interpolation at grid points (should be exact)
            for &x_grid in gauss_rule.x.iter() {
                let expected = func(x_grid);
                let interpolated = evaluate_interpolated_polynomial(x_grid, &coeffs);
                let error = (interpolated - expected).abs();

                assert!(
                    error < 1e-12,
                    "Interpolation failed for n={}, func={}, point={}: expected {}, got {}, error={}",
                    n,
                    func_idx,
                    x_grid,
                    expected,
                    interpolated,
                    error
                );
            }

            println!("n={}, func={}: interpolation successful", n, func_idx);
        }
    }
}

/// Test Interpolate1D struct functionality
#[test]
fn test_interpolate1d_struct() {
    // Test with f64
    test_interpolate1d_struct_generic::<f64>();

    // Test with Df64
    test_interpolate1d_struct_generic::<Df64>();
}

/// Generic test for Interpolate1D struct
fn test_interpolate1d_struct_generic<T: CustomNumeric + 'static>() {
    let n = 10;
    let rule = legendre_generic::<T>(n).reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));

    // Create test function values (sin(x))
    let values: Vec<T> = rule.x.iter().map(|&x| x.sin()).collect();

    // Create interpolator
    let interp = Interpolate1D::new(&values, &rule);

    // Test domain
    let (x_min, x_max) = interp.domain();
    assert_eq!(x_min, T::from_f64_unchecked(-1.0));
    assert_eq!(x_max, T::from_f64_unchecked(1.0));
    assert_eq!(interp.n_points(), n);

    // Test evaluation at Gauss points (should be exact)
    for i in 0..n {
        let x = rule.x[i];
        let expected = values[i];
        let computed = interp.evaluate(x);
        let error = (computed - expected).abs_as_same_type();

        // Should be very close at Gauss points
        assert!(
            error < T::from_f64_unchecked(1e-14),
            "Interpolation error at Gauss point {}: {} > 1e-14",
            i,
            error
        );
    }

    // Test evaluation at intermediate points
    let test_points = vec![
        T::from_f64_unchecked(-0.5),
        T::from_f64_unchecked(0.0),
        T::from_f64_unchecked(0.3),
        T::from_f64_unchecked(0.7),
    ];

    for &x in &test_points {
        let expected = x.sin();
        let computed = interp.evaluate(x);
        let error = (computed - expected).abs_as_same_type();

        // Should have reasonable accuracy
        assert!(
            error < T::from_f64_unchecked(1e-10),
            "Interpolation error at {}: {} > 1e-10",
            x,
            error
        );
    }
}

/// Test Interpolate1D with sin(x) function for high precision
#[test]
fn test_interpolate1d_sin_precision() {
    // Test f64 precision
    test_interpolate1d_sin_generic::<f64>(
        100,   // n_points
        1e-12, // tolerance
    );

    // Test Df64 precision
    // Note: Precision limited by Df64 sin() function (~15 digits)
    test_interpolate1d_sin_generic::<Df64>(
        200,   // n_points
        1e-14, // tolerance (limited by Df64 trigonometric precision)
    );
}

/// Generic test for Interpolate1D with sin(x) function
fn test_interpolate1d_sin_generic<T: CustomNumeric + 'static>(n_points: usize, tolerance: f64) {
    let rule = legendre_generic::<T>(n_points).reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));

    // Create sin(x) values
    let values: Vec<T> = rule.x.iter().map(|&x| x.sin()).collect();

    // Create interpolator
    let interp = Interpolate1D::new(&values, &rule);

    // Test points
    let test_points = vec![
        T::from_f64_unchecked(-0.8),
        T::from_f64_unchecked(-0.5),
        T::from_f64_unchecked(-0.2),
        T::from_f64_unchecked(0.1),
        T::from_f64_unchecked(0.4),
        T::from_f64_unchecked(0.7),
        T::from_f64_unchecked(0.9),
    ];

    for &x in &test_points {
        let expected = x.sin();
        let computed = interp.evaluate(x);
        let error = (computed - expected).abs_as_same_type();

        assert!(
            error < T::from_f64_unchecked(tolerance),
            "High-precision interpolation failed at point {}: expected {}, got {}, error={} > tolerance={}",
            x,
            expected,
            computed,
            error,
            tolerance
        );
    }
}
