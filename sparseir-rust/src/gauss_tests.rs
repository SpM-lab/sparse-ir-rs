use super::*;
use crate::interpolation1d::{
    evaluate_interpolated_polynomial, interpolate_1d_legendre, legendre_collocation_matrix,
};
use crate::numeric::CustomNumeric;
use mdarray::DTensor;
use crate::Df64;

#[test]
fn test_rule_constructor() {
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule = Rule::new(x.clone(), w.clone(), -1.0, 1.0);
    assert_eq!(rule.x, x);
    assert_eq!(rule.w, w);
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
}

#[test]
fn test_rule_from_vectors() {
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule = Rule::from_vectors(x.clone(), w.clone(), -1.0, 1.0);
    assert_eq!(rule.x, x);
    assert_eq!(rule.w, w);
}

#[test]
fn test_rule_empty() {
    let rule = Rule::<f64>::empty();
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
}

#[test]
fn test_rule_validation() {
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule = Rule::new(x, w, -1.0, 1.0);
    assert!(rule.validate());
}

#[test]
fn test_rule_join() {
    let rule1 = legendre::<f64>(4).reseat(-4.0, -1.0);
    let rule2 = legendre::<f64>(4).reseat(-1.0, 1.0);
    let rule3 = legendre::<f64>(4).reseat(1.0, 3.0);

    let joined = Rule::join(&[rule1, rule2, rule3]);

    assert!(joined.validate());
    assert_eq!(joined.a, -4.0);
    assert_eq!(joined.b, 3.0);
}

#[test]
fn test_rule_reseat() {
    let original_rule = legendre::<f64>(4);
    let reseated = original_rule.reseat(-2.0, 2.0);

    assert!(reseated.validate());
    assert_eq!(reseated.a, -2.0);
    assert_eq!(reseated.b, 2.0);
}

#[test]
fn test_rule_scale() {
    let x = vec![0.0, 1.0];
    let w = vec![1.0, 1.0];

    let rule = Rule::new(x, w, -1.0, 1.0);
    let scaled = rule.scale(2.0);

    assert_eq!(scaled.w[0], 2.0);
    assert_eq!(scaled.w[1], 2.0);
}

#[test]
fn test_rule_piecewise() {
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre::<f64>(20).piecewise(&edges);

    assert!(rule.validate());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);
}

#[test]
fn test_gauss_validation_like_cpp() {
    // Test similar to C++ gaussValidate function
    let rule = legendre::<f64>(20);

    // Check interval validity: a <= b
    assert!(rule.a <= rule.b);

    // Check that all points are within [a, b]
    for &xi in rule.x.iter() {
        assert!(xi >= rule.a && xi <= rule.b);
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check that x and w have same length
    assert_eq!(rule.x.len(), rule.w.len());

    // Check x_forward and x_backward consistency
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!((rule.x_forward[i] - expected_forward).abs() < 1e-14);
        assert!((rule.x_backward[i] - expected_backward).abs() < 1e-14);
    }
}

#[test]
fn test_rule_constructor_with_defaults() {
    // Test like C++ Rule constructor with default a, b
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule1 = Rule::new(x.clone(), w.clone(), -1.0, 1.0);
    let rule2 = Rule::new(x, w, -1.0, 1.0);

    assert_eq!(rule1.a, rule2.a);
    assert_eq!(rule1.b, rule2.b);
    assert_eq!(rule1.x, rule2.x);
    assert_eq!(rule1.w, rule2.w);
}

#[test]
fn test_reseat_functionality() {
    // Test reseat functionality first
    let original_rule = legendre::<f64>(4);
    let reseated = original_rule.reseat(-4.0, -1.0);

    assert!(reseated.validate());
    assert_eq!(reseated.a, -4.0);
    assert_eq!(reseated.b, -1.0);
}

#[test]
fn test_join_functionality() {
    // Test join functionality
    let rule1 = legendre::<f64>(4).reseat(-4.0, -1.0);
    let rule2 = legendre::<f64>(4).reseat(-1.0, 1.0);
    let rule3 = legendre::<f64>(4).reseat(1.0, 3.0);

    let joined = Rule::join(&[rule1, rule2, rule3]);

    assert!(joined.validate());
    assert_eq!(joined.a, -4.0);
    assert_eq!(joined.b, 3.0);
}

#[test]
fn test_piecewise_like_cpp() {
    // Test piecewise functionality like C++ test
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre::<f64>(20).piecewise(&edges);

    assert!(rule.validate());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);
}

#[test]
fn test_large_legendre_rule() {
    // Test large rule like C++ test with n=200
    let rule = legendre::<f64>(200);

    assert!(rule.validate());
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
    assert_eq!(rule.x.len(), 200);
    assert_eq!(rule.w.len(), 200);
}

#[test]
fn test_legendre_function() {
    // Test legendre function with different orders
    for n in 1..=5 {
        let rule = legendre::<f64>(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate());
    }

    // Test n=0 case
    let rule = legendre::<f64>(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

// CustomNumeric tests
#[test]
fn test_legendre_custom_f64() {
    // Test legendre_custom function with f64
    for n in 1..=5 {
        let rule = legendre_custom::<f64>(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate_custom());
    }

    // Test n=0 case
    let rule = legendre_custom::<f64>(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

#[test]
fn test_legendre_twofloat() {
    // Test legendre_twofloat function with Df64
    for n in 1..=3 {
        // Smaller range for Df64 due to complexity
        let rule = legendre_twofloat(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate_twofloat());
    }

    // Test n=0 case
    let rule = legendre_twofloat(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

#[test]
fn test_rule_custom_methods() {
    // Test Rule custom methods with f64
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule = Rule::new_custom(x.clone(), w.clone(), -1.0, 1.0);
    assert!(rule.validate_custom());

    let reseated = rule.reseat_custom(-2.0, 0.0);
    assert!(reseated.validate_custom());
    assert_eq!(reseated.a, -2.0);
    assert_eq!(reseated.b, 0.0);

    let scaled = rule.scale_custom(2.0);
    assert!(scaled.validate_custom());
    assert_eq!(scaled.w[0], 1.0);
    assert_eq!(scaled.w[1], 1.0);
}

#[test]
fn test_rule_twofloat_methods() {
    // Test with Df64
    let x_tf = vec![Df64::from(0.0), Df64::from(1.0)];
    let w_tf = vec![Df64::from(0.5), Df64::from(0.5)];

    let rule_tf = Rule::new_twofloat(x_tf, w_tf, Df64::from(-1.0), Df64::from(1.0));
    assert!(rule_tf.validate_twofloat());
}

// ===== Df64 Gauss Integration Precision Tests =====

/// Test function: f(x) = {cos((π/2) * x)}²
/// Integral over [-1, 1] should be exactly 1.0
fn test_function(x: Df64) -> Df64 {
    use nalgebra::RealField;
    let pi_half = Df64::frac_pi_2();
    let cos_val = (pi_half * x).cos();
    cos_val * cos_val
}

/// Analytical integral of f(x) = {cos((π/2) * x)}² over [-1, 1]
/// ∫_{-1}^{1} cos²((π/2) * x) dx = 1.0
fn analytical_integral() -> Df64 {
    Df64::from_f64_unchecked(1.0)
}

#[test]
fn test_twofloat_gauss_rule_validation() {
    println!("Df64 Gauss Rule Validation Test");
    println!("===================================");

    let test_points = vec![5, 10, 20, 50];

    for n in test_points {
        let rule = legendre_twofloat(n);

        println!("Testing rule with {} points:", n);
        println!("  Interval: [{}, {}]", rule.a.to_f64(), rule.b.to_f64());
        println!("  Points: {}", rule.x.len());
        println!("  Weights: {}", rule.w.len());

        // Validate the rule
        let is_valid = rule.validate_twofloat();
        println!(
            "  Validation: {}",
            if is_valid { "✅ PASS" } else { "❌ FAIL" }
        );

        // Check weight sum (should be 2.0 for [-1, 1])
        let mut weight_sum = Df64::from_f64_unchecked(0.0);
        for &w in rule.w.iter() {
            weight_sum += w;
        }
        let expected_sum = Df64::from_f64_unchecked(2.0);
        let weight_error = (weight_sum - expected_sum).abs();

        println!(
            "  Weight sum: {} (expected: 2.0, error: {:.2e})",
            weight_sum.to_f64(),
            weight_error.to_f64()
        );

        // Check symmetry (for even n, should be symmetric)
        if n % 2 == 0 {
            let mid = n / 2;
            let sym_check = (rule.x[mid - 1] + rule.x[mid]).abs() < Df64::epsilon();
            println!(
                "  Symmetry check: {}",
                if sym_check { "✅ PASS" } else { "❌ FAIL" }
            );
        }

        println!();
    }
}

#[test]
fn test_twofloat_integration_convergence_analysis() {
    println!("Df64 Integration Convergence Analysis");
    println!("========================================");

    let analytical = analytical_integral();

    // Test convergence with specific number of points
    let test_points = vec![100, 150, 200];

    for n in test_points {
        let rule = legendre_twofloat(n);
        let mut integral = Df64::from_f64_unchecked(0.0);

        for i in 0..rule.x.len() {
            let f_val = test_function(rule.x[i]);
            integral += f_val * rule.w[i];
        }

        let error = (integral - analytical).abs().to_f64();
        let rel_error = error / analytical.to_f64().abs();

        println!(
            "n={:3}: error={:.2e}, rel_error={:.2e}",
            n, error, rel_error
        );
        assert!(rel_error < 1e-30);
    }
}

/// Evaluate Legendre polynomial P_n(x) at point x
fn evaluate_legendre_polynomial<T: CustomNumeric>(x: T, n: usize) -> T {
    if n == 0 {
        T::from_f64_unchecked(1.0)
    } else if n == 1 {
        x
    } else {
        let mut p_prev2 = T::from_f64_unchecked(1.0);
        let mut p_prev1 = x;

        for i in 2..=n {
            let i_f64 = i as f64;
            let p_curr = ((T::from_f64_unchecked(2.0 * i_f64 - 1.0) * x * p_prev1)
                - (T::from_f64_unchecked(i_f64 - 1.0) * p_prev2))
                / T::from_f64_unchecked(i_f64);
            p_prev2 = p_prev1;
            p_prev1 = p_curr;
        }

        p_prev1
    }
}

#[test]
fn test_legendre_vandermonde_basic() {
    // Test with simple 3-point grid
    let x = vec![-1.0, 0.0, 1.0];
    let v = legendre_vandermonde(&x, 2);

    // Check dimensions
    assert_eq!(v.shape().0, 3);
    assert_eq!(v.shape().1, 3);

    // Check first column (P_0 = 1)
    for i in 0..3 {
        assert!((v[[i, 0]] - 1.0).abs() < 1e-12);
    }

    // Check second column (P_1 = x)
    for i in 0..3 {
        assert!((v[[i, 1]] - x[i]).abs() < 1e-12);
    }

    // Check third column (P_2 = (3x^2 - 1)/2)
    for i in 0..3 {
        let expected = (3.0 * x[i] * x[i] - 1.0) / 2.0;
        assert!((v[[i, 2]] - expected).abs() < 1e-12);
    }
}

/// Generic test function for 1D Legendre interpolation of sin(x) - MOVED TO interpolation1d_tests.rs
fn test_interpolate_1d_legendre_sin_generic<T: CustomNumeric + 'static>(
    n_points: usize,
    tolerance: T,
    test_points: Vec<T>,
) where
    T: std::fmt::Display,
{
    // Create Gauss rule using generic function
    let gauss_rule = legendre_generic::<T>(n_points).reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));

    // Sample sin(x) at Gauss points
    let values: Vec<T> = gauss_rule.x.iter().map(|&x| x.sin()).collect();

    // Get interpolation coefficients
    let coeffs = interpolate_1d_legendre(&values, &gauss_rule);

    // Test interpolation at grid points (should be exact)
    for &x_grid in &gauss_rule.x {
        let expected = x_grid.sin();
        let interpolated = evaluate_interpolated_polynomial(x_grid, &coeffs);
        assert!(
            (interpolated - expected).abs_as_same_type() < T::from_f64_unchecked(1e-12),
            "Interpolation failed at grid point {}: expected {}, got {}",
            x_grid,
            expected,
            interpolated
        );
    }

    // Test interpolation at interior points
    for &x_test in &test_points {
        let expected = x_test.sin();
        let interpolated = evaluate_interpolated_polynomial(x_test, &coeffs);
        let error = (interpolated - expected).abs_as_same_type();
        assert!(
            error < tolerance,
            "High-precision interpolation failed at point {}: expected {}, got {}, error={} > tolerance={}",
            x_test,
            expected,
            interpolated,
            error,
            tolerance
        );
    }
}

#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_sin_f64_high_precision() {
    // Test high-precision interpolation of sin(x) with f64
    test_interpolate_1d_legendre_sin_generic::<f64>(
        100,                                        // n_points
        f64::EPSILON * 100.0,                       // tolerance: EPSILON * 100
        vec![-0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 0.9], // test_points
    );
}

#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_sin_twofloat_ultra_high_precision() {
    // Test ultra high-precision interpolation of sin(x) with Df64
    test_interpolate_1d_legendre_sin_generic::<Df64>(
        200,                       // n_points (higher for better precision)
        Df64::from_f64_unchecked(1e-19), // tolerance: 1e-19 (achieved maximum precision)
        vec![
            Df64::from_f64_unchecked(-0.8),
            Df64::from_f64_unchecked(-0.5),
            Df64::from_f64_unchecked(-0.2),
            Df64::from_f64_unchecked(0.1),
            Df64::from_f64_unchecked(0.4),
            Df64::from_f64_unchecked(0.7),
            Df64::from_f64_unchecked(0.9),
        ], // test_points
    );
}

/// Test that the collocation matrix is approximately the inverse of the Vandermonde matrix - MOVED TO interpolation1d_tests.rs
#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_legendre_collocation_matrix_inverse() {
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

/// Test the new fast interpolation method - MOVED TO interpolation1d_tests.rs
#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_fast() {
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
            for (i, &x_grid) in gauss_rule.x.iter().enumerate() {
                let expected = func(x_grid);
                let interpolated = evaluate_interpolated_polynomial(x_grid, &coeffs);
                let error = (interpolated - expected).abs_as_same_type();

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

/// Helper function to check if two vectors are approximately equal within tolerance
fn vecs_approx_equal<T>(a: &[T], b: &[T], tolerance: T) -> bool
where
    T: Copy + std::ops::Sub<Output = T> + PartialOrd,
    T: std::fmt::Display,
{
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        let diff = if a[i] > b[i] {
            a[i] - b[i]
        } else {
            b[i] - a[i]
        };
        if diff > tolerance {
            return false;
        }
    }
    true
}

/// Helper function to compute Legendre polynomial P_n(x)
fn legendre_polynomial(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;

            for k in 2..=n {
                let k_f = k as f64;
                let k1_f = (k - 1) as f64;

                let p2 = ((2.0 * k1_f + 1.0) * x * p1 - k1_f * p0) / k_f;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Helper function to compute Legendre polynomial with Df64
fn legendre_polynomial_twofloat(n: usize, x: Df64) -> Df64 {
    match n {
        0 => Df64::from(1.0),
        1 => x,
        _ => {
            let mut p0 = Df64::from(1.0);
            let mut p1 = x;

            for k in 2..=n {
                let k_f = Df64::from(k as f64);
                let k1_f = Df64::from((k - 1) as f64);

                let p2 =
                    ((Df64::from(2.0) * k1_f + Df64::from(1.0)) * x * p1 - k1_f * p0) / k_f;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Test high-precision Gauss-Legendre rule with f64
/// Similar to C++ test but using f64 with 1e-13 tolerance
#[test]
fn test_high_precision_legendre_f64() {
    let n = 16;
    let rule = legendre_custom::<f64>(n);

    // Expected values computed with high precision (similar to C++ DDouble test)
    let x_expected = [
        -0.9894009349916499,
        -0.9445750230732325,
        -0.8656312023878318,
        -0.755404408355003,
        -0.6178762444026438,
        -0.45801677765722737,
        -0.2816035507792589,
        -0.09501250983763743,
        0.09501250983763743,
        0.2816035507792589,
        0.45801677765722737,
        0.6178762444026438,
        0.755404408355003,
        0.8656312023878318,
        0.9445750230732325,
        0.9894009349916499,
    ];

    let w_expected = [
        0.027152459411754124,
        0.06225352393864806,
        0.0951585116824928,
        0.12462897125553389,
        0.14959598881657682,
        0.16915651939500254,
        0.18260341504492367,
        0.18945061045506834,
        0.18945061045506834,
        0.18260341504492367,
        0.16915651939500254,
        0.14959598881657682,
        0.12462897125553389,
        0.0951585116824928,
        0.06225352393864806,
        0.027152459411754124,
    ];

    // Check with high precision tolerance (1e-13)
    let tolerance = 1e-13;

    // Check x values
    for i in 0..n {
        assert!(
            (rule.x[i] - x_expected[i]).abs() < tolerance,
            "x[{}] mismatch: expected {}, got {}",
            i,
            x_expected[i],
            rule.x[i]
        );
    }

    // Check w values
    for i in 0..n {
        assert!(
            (rule.w[i] - w_expected[i]).abs() < tolerance,
            "w[{}] mismatch: expected {}, got {}",
            i,
            w_expected[i],
            rule.w[i]
        );
    }

    // Check interval
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);

    // Check x_forward and x_backward consistency with high precision
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }
}

/// Test high-precision Gauss-Legendre rule with Df64
/// Similar to C++ DDouble test but using Df64
#[test]
fn test_high_precision_legendre_twofloat() {
    let n = 6; // Smaller n for Df64 due to complexity
    let rule = legendre_twofloat(n);

    // Check that the rule is valid
    assert!(rule.validate_twofloat());

    // Check that all points are within [-1, 1]
    for &xi in rule.x.iter() {
        assert!(xi >= Df64::from(-1.0) && xi <= Df64::from(1.0));
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check x_forward and x_backward consistency with Df64 precision
    let tolerance = Df64::from(1e-15); // Higher precision for Df64
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }

    // Test orthogonality property: sum of weights should be 2.0
    let weight_sum: Df64 = rule.w.iter().fold(Df64::from(0.0), |acc, &w| acc + w);
    assert!(
        (weight_sum - Df64::from(2.0)).abs() < Df64::from(1e-14),
        "Sum of weights should be 2.0, got {}",
        weight_sum
    );
}

/// Test Legendre polynomial evaluation at Gauss-Legendre nodes
/// This tests the orthogonality property
#[test]
fn test_legendre_polynomial_at_nodes() {
    let n = 8;
    let rule = legendre_custom::<f64>(n);

    // Test that P_0(x) = 1 at all nodes
    for i in 0..n {
        let p0 = legendre_polynomial(0, rule.x[i]);
        assert!((p0 - 1.0).abs() < 1e-14, "P_0(x[{}]) should be 1.0", i);
    }

    // Test that P_1(x) = x at all nodes
    for i in 0..n {
        let p1 = legendre_polynomial(1, rule.x[i]);
        assert!(
            (p1 - rule.x[i]).abs() < 1e-14,
            "P_1(x[{}]) should equal x[{}]",
            i,
            i
        );
    }

    // Test that P_n(x) = 0 at all nodes (where n is the order of the rule)
    // This is the defining property of Gauss-Legendre nodes
    for i in 0..n {
        let pn = legendre_polynomial(n, rule.x[i]);
        assert!(
            pn.abs() < 1e-12,
            "P_{}(x[{}]) should be approximately 0, got {}",
            n,
            i,
            pn
        );
    }
}

/// Test Legendre polynomial evaluation with Df64
#[test]
fn test_legendre_polynomial_twofloat_at_nodes() {
    let n = 4; // Smaller n for Df64
    let rule = legendre_twofloat(n);

    // Test that P_0(x) = 1 at all nodes
    for i in 0..n {
        let p0 = legendre_polynomial_twofloat(0, rule.x[i]);
        assert!(
            (p0 - Df64::from(1.0)).abs() < Df64::from(1e-15),
            "P_0(x[{}]) should be 1.0",
            i
        );
    }

    // Test that P_1(x) = x at all nodes
    for i in 0..n {
        let p1 = legendre_polynomial_twofloat(1, rule.x[i]);
        assert!(
            (p1 - rule.x[i]).abs() < Df64::from(1e-15),
            "P_1(x[{}]) should equal x[{}]",
            i,
            i
        );
    }

    // Test that P_n(x) = 0 at all nodes
    for i in 0..n {
        let pn = legendre_polynomial_twofloat(n, rule.x[i]);
        assert!(
            pn.abs() < Df64::from(1e-14),
            "P_{}(x[{}]) should be approximately 0, got {}",
            n,
            i,
            pn
        );
    }
}

/// Test large Gauss-Legendre rule like C++ test with n=200
#[test]
fn test_large_legendre_rule_high_precision() {
    let n = 200;
    let rule = legendre_custom::<f64>(n);

    // Check basic properties
    assert!(rule.validate_custom());
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
    assert_eq!(rule.x.len(), n);
    assert_eq!(rule.w.len(), n);

    // Check that all points are within [-1, 1]
    for &xi in rule.x.iter() {
        assert!((-1.0..=1.0).contains(&xi));
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check sum of weights should be 2.0
    let weight_sum: f64 = rule.w.iter().sum();
    assert!(
        (weight_sum - 2.0).abs() < 1e-14,
        "Sum of weights should be 2.0, got {}",
        weight_sum
    );

    // Check x_forward and x_backward consistency with high precision
    let tolerance = 1e-14;
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }
}

/// Test piecewise functionality with high precision
#[test]
fn test_piecewise_high_precision() {
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre_custom::<f64>(20).piecewise(&edges);

    assert!(rule.validate_custom());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);

    // Check that all points are within the overall interval
    for &xi in rule.x.iter() {
        assert!((-4.0..=3.0).contains(&xi));
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check sum of weights should be 7.0 (length of interval)
    let weight_sum: f64 = rule.w.iter().sum();
    assert!(
        (weight_sum - 7.0).abs() < 1e-13,
        "Sum of weights should be 7.0, got {}",
        weight_sum
    );
}
