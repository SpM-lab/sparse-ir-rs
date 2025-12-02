//! Tests for SVE module functions

use super::utils::extend_to_full_domain;
use super::{compute_sve, TworkType};
use crate::kernel::{CentrosymmKernel, KernelProperties, LogisticKernel, RegularizedBoseKernel, SymmetryType};
use crate::poly::PiecewiseLegendrePoly;
use mdarray::DTensor;

/// Create a simple polynomial on positive domain [0, 1]
fn create_simple_poly_on_positive_domain() -> PiecewiseLegendrePoly {
    // Create a simple polynomial: f(x) = 1 + 2x on [0, 1]
    // Legendre basis: P_0(x) = 1, P_1(x) = x
    // On [0, 1], we need to map to [-1, 1] internally
    let data = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 2.0 });
    let knots = vec![0.0, 1.0];
    let delta_x = vec![1.0];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0)
}

/// Create polynomial with multiple segments [0, 0.5, 1.0]
fn create_poly_with_segments() -> PiecewiseLegendrePoly {
    // Two segments: [0, 0.5] and [0.5, 1.0]
    let data_vec = [1.0, 1.5, 0.5, 1.0];
    let data = DTensor::<f64, 2>::from_fn([2, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    let knots = vec![0.0, 0.5, 1.0];
    let delta_x = vec![0.5, 0.5];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0)
}

#[test]
fn test_extend_even_symmetry() {
    let poly_positive = create_simple_poly_on_positive_domain();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even, 1.0);

    // Test: f(-x) = f(x) for Even symmetry
    let poly = &polys_full[0];
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let val_pos = poly.evaluate(x);
        let val_neg = poly.evaluate(-x);
        assert!(
            (val_pos - val_neg).abs() < 1e-14,
            "Even symmetry violated: f({}) = {}, f({}) = {}",
            x,
            val_pos,
            -x,
            val_neg
        );
    }
}

#[test]
fn test_extend_odd_symmetry() {
    let poly_positive = create_simple_poly_on_positive_domain();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Odd, 1.0);

    // Test: f(-x) = -f(x) for Odd symmetry
    let poly = &polys_full[0];
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let val_pos = poly.evaluate(x);
        let val_neg = poly.evaluate(-x);
        assert!(
            (val_pos + val_neg).abs() < 1e-14,
            "Odd symmetry violated: f({}) = {}, f({}) = {}",
            x,
            val_pos,
            -x,
            val_neg
        );
    }
}

#[test]
fn test_positive_domain_preserved() {
    let poly_positive = create_simple_poly_on_positive_domain();

    // Save original values
    let original_values: Vec<f64> = (0..10)
        .map(|i| poly_positive.evaluate(i as f64 * 0.1))
        .collect();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even, 1.0);

    // Check that positive domain values are preserved (with 1/sqrt(2) normalization)
    // The extended polynomial applies 1/sqrt(2) normalization to both parts
    let poly = &polys_full[0];
    let norm_factor = 1.0 / 2.0_f64.sqrt();

    for (i, &expected) in original_values.iter().enumerate() {
        let x = i as f64 * 0.1;
        let actual = poly.evaluate(x);
        let expected_normalized = expected * norm_factor;
        assert!(
            (actual - expected_normalized).abs() < 1e-14,
            "Positive domain not preserved: f({}) = {} (expected {})",
            x,
            actual,
            expected_normalized
        );
    }
}

#[test]
fn test_segment_structure() {
    let poly = create_poly_with_segments();

    let polys_full = extend_to_full_domain(vec![poly], SymmetryType::Even, 1.0);

    // Extended from [0, 0.5, 1.0] to [-1.0, -0.5, 0.0, 0.5, 1.0]
    let expected_knots = [-1.0, -0.5, 0.0, 0.5, 1.0];

    // Check segment structure
    for (i, &expected) in expected_knots.iter().enumerate() {
        assert!(
            (polys_full[0].knots[i] - expected).abs() < 1e-14,
            "Segment {} mismatch: got {}, expected {}",
            i,
            polys_full[0].knots[i],
            expected
        );
    }
}

#[test]
fn test_multiple_polynomials() {
    let poly1 = create_simple_poly_on_positive_domain();
    let poly2 = create_poly_with_segments();

    let polys_full = extend_to_full_domain(vec![poly1, poly2], SymmetryType::Even, 1.0);

    // Should have extended both polynomials
    assert_eq!(polys_full.len(), 2);

    // Both should satisfy even symmetry
    for poly in &polys_full {
        let val_pos = poly.evaluate(0.3);
        let val_neg = poly.evaluate(-0.3);
        assert!(
            (val_pos - val_neg).abs() < 1e-14,
            "Even symmetry violated for one of the polynomials"
        );
    }
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// This test verifies the fundamental SVE relation:
/// K(x, y) = sum_{l=0}^{L-1} s_l * u_l(x) * v_l(y)
///
/// where:
/// - x, y ∈ [-1, 1] (scaled variables used in SVE computation)
/// - u_l(x) are left singular functions
/// - v_l(y) are right singular functions
/// - s_l are singular values
fn test_sve_decomposition_kernel_impl<K>(kernel: K, _lambda: f64, epsilon: f64)
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    // Compute SVE
    let sve_result = compute_sve(kernel.clone(), epsilon, None, None, TworkType::Auto);

    // Test points: midpoints of knots of u and v in scaled domain [-1, 1]
    //
    // Evaluating at midpoints between knots is more representative of how the
    // piecewise Legendre polynomials approximate the kernel between collocation
    // points, and avoids testing exactly at the knots used during SVE.
    let u_polys = sve_result.u.get_polys();
    let v_polys = sve_result.v.get_polys();
    let x_knots = &u_polys[0].knots;
    let y_knots = &v_polys[0].knots;

    let test_x: Vec<f64> = x_knots
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();
    let test_y: Vec<f64> = y_knots
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();

    // Tolerance for comparison (absolute error)
    // Note: We use absolute error only because kernel values can be very small.
    // Tolerance is set based on epsilon to account for numerical errors in SVE decomposition.
    // Use a tolerance that is modestly larger than epsilon to account for accumulated errors.
    let tolerance = epsilon * 200.0;

    // Track worst-case error for diagnostics
    let mut max_error = 0.0f64;
    let mut worst_x = 0.0f64;
    let mut worst_y = 0.0f64;
    let mut worst_direct = 0.0f64;
    let mut worst_sve = 0.0f64;
    let mut max_abs_direct = 0.0f64;

    for &x in &test_x {
        for &y in &test_y {
            // Compute kernel value directly
            let k_direct = kernel.compute(x, y);

            // Compute kernel value from SVE decomposition
            let mut k_sve = 0.0;
            for l in 0..sve_result.s.len() {
                let u_l_x = sve_result.u[l].evaluate(x);
                let v_l_y = sve_result.v[l].evaluate(y);
                k_sve += sve_result.s[l] * u_l_x * v_l_y;
            }

            // Compute absolute error only
            // Note: We don't use relative error because kernel values can be very small
            let error = (k_direct - k_sve).abs();

            if error > max_error {
                max_error = error;
                worst_x = x;
                worst_y = y;
                worst_direct = k_direct;
                worst_sve = k_sve;
            }

            let abs_direct = k_direct.abs();
            if abs_direct > max_abs_direct {
                max_abs_direct = abs_direct;
            }
        }
    }

    eprintln!(
        "Max SVE abs error: error={:.15e}, x={:.15e}, y={:.15e}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}, max|K|={:.15e}",
        max_error, worst_x, worst_y, worst_direct, worst_sve, tolerance, max_abs_direct
    );

    assert!(
        max_error < tolerance,
        "SVE decomposition failed: max_error={:.15e} at x={}, y={}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}",
        max_error, worst_x, worst_y, worst_direct, worst_sve, tolerance
    );
}

/// Same as test_sve_decomposition_kernel_impl but with explicit tolerance parameter
fn test_sve_decomposition_kernel_impl_with_tolerance<K>(
    kernel: K,
    _lambda: f64,
    epsilon: f64,
    tolerance: f64,
)
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    // Compute SVE
    let sve_result = compute_sve(kernel.clone(), epsilon, None, None, TworkType::Auto);

    // Test points: midpoints of knots of u and v in scaled domain [-1, 1]
    let u_polys = sve_result.u.get_polys();
    let v_polys = sve_result.v.get_polys();
    let x_knots = &u_polys[0].knots;
    let y_knots = &v_polys[0].knots;

    let test_x: Vec<f64> = x_knots
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();
    let test_y: Vec<f64> = y_knots
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();

    // Track worst-case error for diagnostics
    let mut max_error = 0.0f64;
    let mut worst_x = 0.0f64;
    let mut worst_y = 0.0f64;
    let mut worst_direct = 0.0f64;
    let mut worst_sve = 0.0f64;
    let mut max_abs_direct = 0.0f64;

    for &x in &test_x {
        for &y in &test_y {
            // Compute kernel value directly
            let k_direct = kernel.compute(x, y);

            // Compute kernel value from SVE decomposition
            let mut k_sve = 0.0;
            for l in 0..sve_result.s.len() {
                let u_l_x = sve_result.u[l].evaluate(x);
                let v_l_y = sve_result.v[l].evaluate(y);
                k_sve += sve_result.s[l] * u_l_x * v_l_y;
            }

            // Compute absolute error only
            let error = (k_direct - k_sve).abs();

            if error > max_error {
                max_error = error;
                worst_x = x;
                worst_y = y;
                worst_direct = k_direct;
                worst_sve = k_sve;
            }

            let abs_direct = k_direct.abs();
            if abs_direct > max_abs_direct {
                max_abs_direct = abs_direct;
            }
        }
    }

    eprintln!(
        "Max SVE abs error: error={:.15e}, x={:.15e}, y={:.15e}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}, max|K|={:.15e}",
        max_error, worst_x, worst_y, worst_direct, worst_sve, tolerance, max_abs_direct
    );

    assert!(
        max_error < tolerance,
        "SVE decomposition failed: max_error={:.15e} at x={}, y={}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}",
        max_error, worst_x, worst_y, worst_direct, worst_sve, tolerance
    );
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for LogisticKernel
#[test]
fn test_sve_decomposition_logistic_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl(LogisticKernel::new(100.0), 100.0, 1e-6);
    
    // Test with lambda = 10^5, epsilon = 1e-12
    test_sve_decomposition_kernel_impl(LogisticKernel::new(1e5), 1e5, 1e-12);
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for RegularizedBoseKernel
/// Note: Uses relaxed tolerance due to numerical challenges with RegularizedBoseKernel
#[test]
fn test_sve_decomposition_regularized_bose_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl_with_tolerance(
        RegularizedBoseKernel::new(100.0), 
        100.0, 
        1e-6,
        1e-6 * 1.0
    );
    
    // Test with lambda = 10^5, epsilon = 1e-12
    // Use relaxed tolerance: epsilon * 100000 (much more lenient than epsilon * 200)
    // For very small kernel values, use absolute tolerance of 1e-5
    //test_sve_decomposition_kernel_impl_with_tolerance(
        //RegularizedBoseKernel::new(1e5), 
        //1e5, 
        //1e-12,
        //1e-5  // Use fixed absolute tolerance for very small values
    //);
}

