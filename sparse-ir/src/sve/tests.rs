//! Tests for SVE module functions

use super::utils::extend_to_full_domain;
use crate::kernel::SymmetryType;
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
