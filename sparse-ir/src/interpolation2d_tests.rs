use crate::gauss::legendre_generic;
use crate::interpolation2d::{evaluate_2d_legendre_polynomial, interpolate_2d_legendre};
use crate::{CustomNumeric, Interpolate2D, Df64};
use mdarray::DTensor;
use simba::scalar::ComplexField;

#[test]
fn test_interpolate_2d_legendre_basic() {
    // Test with a simple 2D function: f(x,y) = x + y
    let gauss_x = legendre_generic::<f64>(2).reseat(-1.0, 1.0);
    let gauss_y = legendre_generic::<f64>(2).reseat(-1.0, 1.0);

    // Create test values
    let mut values = DTensor::<f64, 2>::from_elem([2, 2], 0.0);
    for i in 0..2 {
        for j in 0..2 {
            values[[i, j]] = gauss_x.x[i] + gauss_y.x[j];
        }
    }

    let coeffs = interpolate_2d_legendre(&values, &gauss_x, &gauss_y);

    // Test interpolation at grid points (should be exact)
    for i in 0..2 {
        for j in 0..2 {
            let expected = gauss_x.x[i] + gauss_y.x[j];
            let interpolated = evaluate_2d_legendre_polynomial(
                gauss_x.x[i],
                gauss_y.x[j],
                &coeffs,
                &gauss_x,
                &gauss_y,
            );
            assert!(
                (interpolated - expected).abs() < 1e-12,
                "Interpolation failed at ({}, {}): expected {}, got {}",
                gauss_x.x[i],
                gauss_y.x[j],
                expected,
                interpolated
            );
        }
    }
}

#[test]
fn test_interpolate_2d_object() {
    let gauss_x = legendre_generic::<f64>(2).reseat(0.0, 1.0);
    let gauss_y = legendre_generic::<f64>(2).reseat(0.0, 2.0);

    let values = DTensor::<f64, 2>::from_elem([2, 2], 1.0);
    let interp = Interpolate2D::new(&values, &gauss_x, &gauss_y);

    // Test interpolation at center of cell
    let result = interp.interpolate(0.5, 1.0);
    assert!(result.abs() < 10.0); // Just check it doesn't panic and returns reasonable value
}

#[test]
fn test_interpolate_2d_quadratic_polynomial() {
    // Test with 2D quadratic function: f(x,y) = x^2 + y^2 + x*y
    // degree 3 (4 Gauss points) should exactly interpolate quadratic polynomials
    let gauss_x =
        legendre_generic::<Df64>(4).reseat(Df64::from_f64_unchecked(-1.0), Df64::from_f64_unchecked(1.0));
    let gauss_y =
        legendre_generic::<Df64>(4).reseat(Df64::from_f64_unchecked(-1.0), Df64::from_f64_unchecked(1.0));

    // Create test values for f(x,y) = x^2 + y^2 + x*y
    let mut values = DTensor::<Df64, 2>::from_elem([4, 4], num_traits::Zero::zero());
    for i in 0..4 {
        for j in 0..4 {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            values[[i, j]] = x * x + y * y + x * y;
        }
    }

    let coeffs = interpolate_2d_legendre(&values, &gauss_x, &gauss_y);

    // Test interpolation at grid points (should be exact)
    for i in 0..4 {
        for j in 0..4 {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            let expected = x * x + y * y + x * y;
            let interpolated = evaluate_2d_legendre_polynomial(x, y, &coeffs, &gauss_x, &gauss_y);
            assert!(
                (interpolated - expected).abs() < Df64::from_f64_unchecked(1e-12),
                "Interpolation failed at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                interpolated
            );
        }
    }

    // Test interpolation at intermediate points (should also be exact for quadratic)
    let test_points = vec![
        (Df64::from_f64_unchecked(-0.5), Df64::from_f64_unchecked(-0.3)),
        (Df64::from_f64_unchecked(0.0), Df64::from_f64_unchecked(0.0)),
        (Df64::from_f64_unchecked(0.3), Df64::from_f64_unchecked(0.7)),
        (Df64::from_f64_unchecked(0.8), Df64::from_f64_unchecked(-0.4)),
    ];

    println!("Df64 2D interpolation error analysis:");
    for (x, y) in test_points {
        let expected = x * x + y * y + x * y;
        let interpolated = evaluate_2d_legendre_polynomial(x, y, &coeffs, &gauss_x, &gauss_y);
        let error = (interpolated - expected).abs();
        println!("Point ({}, {}): error = {}", x, y, error);

        assert!(
            error < Df64::from_f64_unchecked(1e-14),
            "Interpolation failed at ({}, {}): expected {}, got {}, error = {}",
            x,
            y,
            expected,
            interpolated,
            error
        );
    }
}

#[test]
fn test_interpolate2d_struct() {
    // Test Interpolate2D struct with f64
    test_interpolate2d_struct_generic::<f64>();

    // Test Interpolate2D struct with Df64
    test_interpolate2d_struct_generic::<Df64>();
}

/// Generic test for Interpolate2D struct
fn test_interpolate2d_struct_generic<T: CustomNumeric + 'static>() {
    let n = 3;
    let gauss_x = legendre_generic::<T>(n).reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));
    let gauss_y = legendre_generic::<T>(n).reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));

    // Create test function values: f(x,y) = x^2 + y^2
    let mut values = DTensor::<T, 2>::from_elem([n, n], T::zero());
    for i in 0..n {
        for j in 0..n {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            values[[i, j]] = x * x + y * y;
        }
    }

    // Create interpolator
    let interp = Interpolate2D::new(&values, &gauss_x, &gauss_y);

    // Test domain
    let (x_min, x_max, y_min, y_max) = interp.domain();
    assert_eq!(x_min, T::from_f64_unchecked(-1.0));
    assert_eq!(x_max, T::from_f64_unchecked(1.0));
    assert_eq!(y_min, T::from_f64_unchecked(-1.0));
    assert_eq!(y_max, T::from_f64_unchecked(1.0));
    assert_eq!(interp.n_points_x(), n);
    assert_eq!(interp.n_points_y(), n);

    // Test evaluation at Gauss points (should be exact)
    for i in 0..n {
        for j in 0..n {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            let expected = x * x + y * y;
            let computed = interp.evaluate(x, y);
            let error = (computed - expected).abs_as_same_type();

            // Should be very close at Gauss points
            assert!(
                error < T::from_f64_unchecked(1e-12),
                "Interpolation error at Gauss point ({}, {}): {} > 1e-12",
                i,
                j,
                error
            );
        }
    }

    // Test evaluation at intermediate points
    let test_points = vec![
        (T::from_f64_unchecked(-0.5), T::from_f64_unchecked(-0.3)),
        (T::from_f64_unchecked(0.0), T::from_f64_unchecked(0.0)),
        (T::from_f64_unchecked(0.3), T::from_f64_unchecked(0.7)),
        (T::from_f64_unchecked(0.8), T::from_f64_unchecked(-0.4)),
    ];

    for (x, y) in test_points {
        let expected = x * x + y * y;
        let computed = interp.evaluate(x, y);
        let error = (computed - expected).abs_as_same_type();

        // Should have reasonable accuracy
        assert!(
            error < T::from_f64_unchecked(1e-8),
            "Interpolation error at ({}, {}): {} > 1e-8",
            x,
            y,
            error
        );
    }
}
