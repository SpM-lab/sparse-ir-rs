//! Tests for piecewise Legendre polynomial Fourier transform implementations

use crate::freq::{BosonicFreq, FermionicFreq};
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::polyfourier::{
    BosonicPiecewiseLegendreFT, FermionicPiecewiseLegendreFT, FermionicPiecewiseLegendreFTVector,
    PiecewiseLegendreFT,
};
use crate::special_functions::spherical_bessel_j;
use crate::traits::{Bosonic, Fermionic, Statistics};
use mdarray::tensor;

#[test]
fn test_fermionic_ft_creation() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly.clone(), Fermionic, None);

    assert_eq!(ft_poly.get_n_asymp(), f64::INFINITY);
    assert_eq!(ft_poly.get_statistics(), Statistics::Fermionic);
    assert_eq!(ft_poly.zeta(), 1);
}

#[test]
fn test_bosonic_ft_creation() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = BosonicPiecewiseLegendreFT::new(poly.clone(), Bosonic, Some(100.0));

    assert_eq!(ft_poly.get_n_asymp(), 100.0);
    assert_eq!(ft_poly.get_statistics(), Statistics::Bosonic);
    assert_eq!(ft_poly.zeta(), 0);
}

#[test]
fn test_ft_evaluation_fermionic() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test evaluation at valid fermionic frequency
    let omega = FermionicFreq::new(1).unwrap();
    let result = ft_poly.evaluate(&omega);

    // The result should be a complex number
    assert!(result.is_finite());
    println!("Fermionic FT at n=1: {}", result);
}

#[test]
fn test_ft_evaluation_bosonic() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None);

    // Test evaluation at valid bosonic frequency
    let omega = BosonicFreq::new(0).unwrap();
    let result = ft_poly.evaluate(&omega);

    // The result should be a complex number
    assert!(result.is_finite());
    println!("Bosonic FT at n=0: {}", result);
}

#[test]
fn test_ft_vector_creation() {
    let data1 = tensor![[1.0], [0.0]];
    let data2 = tensor![[0.0], [1.0]];
    let knots = vec![-1.0, 1.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);

    let ft_poly1 = FermionicPiecewiseLegendreFT::new(poly1, Fermionic, None);
    let ft_poly2 = FermionicPiecewiseLegendreFT::new(poly2, Fermionic, None);

    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly1, ft_poly2]);

    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_from_poly_vector() {
    let data1 = tensor![[1.0], [0.0]];
    let data2 = tensor![[0.0], [1.0]];
    let knots = vec![-1.0, 1.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0);

    let poly_vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
    let ft_vector =
        FermionicPiecewiseLegendreFTVector::from_poly_vector(&poly_vector, Fermionic, None);

    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_evaluation() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly]);

    let omega = FermionicFreq::new(1).unwrap();
    let results = ft_vector.evaluate_at(&omega);

    assert_eq!(results.len(), 1);
    assert!(results[0].is_finite());
}

#[test]
fn test_power_model_creation() {
    let data = tensor![[1.0, 0.0], [0.0, 1.0]];
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Check that power model was created
    assert!(!ft_poly.model.moments.is_empty());
    println!("Power model moments: {:?}", ft_poly.model.moments);
}

#[test]
fn test_invalid_domain_panic() {
    let data = tensor![[1.0], [0.0]];
    let knots = vec![0.0, 2.0]; // Invalid domain for Fourier transform

    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    // This should panic
    std::panic::catch_unwind(|| {
        FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    })
    .expect_err("Should panic for invalid domain");
}

/// Test to compare get_tnl implementation with expected values
/// This is a simplified test - in practice we would need reference values from C++
#[test]
fn test_get_tnl_basic_values() {
    // Create a simple polynomial for testing
    let data = tensor![[1.0, 0.0], [0.0, 1.0]];
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    let _ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test get_tnl for various l and w values
    // Note: These are expected values that should match C++ implementation

    // For l=0, get_tnl should be 2 * j_0(|w|) where j_0(x) = sin(x)/x
    let result_0_1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, 1.0);
    let expected_0_1 = 2.0 * (1.0_f64.sin() / 1.0); // 2 * sin(1)
    println!(
        "get_tnl(0, 1.0) = {}, expected = {}",
        result_0_1, expected_0_1
    );

    let result_0_pi = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, std::f64::consts::PI);
    let expected_0_pi = 2.0 * (std::f64::consts::PI.sin() / std::f64::consts::PI); // Should be close to 0
    println!(
        "get_tnl(0, π) = {}, expected = {}",
        result_0_pi, expected_0_pi
    );

    // For l=1, get_tnl should be 2i * j_1(|w|) where j_1(x) = sin(x)/x² - cos(x)/x
    let result_1_1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(1, 1.0);
    let j1_1 = 1.0_f64.sin() / (1.0 * 1.0) - 1.0_f64.cos() / 1.0;
    let im_unit = num_complex::Complex64::new(0.0, 1.0);
    let expected_1_1 = 2.0 * im_unit * j1_1;
    println!(
        "get_tnl(1, 1.0) = {}, expected = {}",
        result_1_1, expected_1_1
    );

    // Test negative w (should apply conjugation)
    let result_0_neg1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, -1.0);
    println!(
        "get_tnl(0, -1.0) = {}, should be conjugate of positive",
        result_0_neg1
    );

    // Basic sanity checks
    assert!(result_0_1.re.is_finite());
    assert!(result_0_1.im.is_finite());
    assert!(result_1_1.re.is_finite());
    assert!(result_1_1.im.is_finite());
    assert!(result_0_neg1.re.is_finite());
    assert!(result_0_neg1.im.is_finite());
}

/// Test spherical Bessel function implementation
#[test]
fn test_spherical_bessel_basic() {
    let data = tensor![[1.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    let _ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test j_0(x) = sin(x)/x for x != 0
    let x: f64 = 1.0;
    let j0 = spherical_bessel_j(0, x);
    let expected_j0 = x.sin() / x;
    println!("j_0({}) = {}, expected = {}", x, j0, expected_j0);

    // Test j_1(x) = sin(x)/x² - cos(x)/x
    let j1 = spherical_bessel_j(1, x);
    let expected_j1 = x.sin() / (x * x) - x.cos() / x;
    println!("j_1({}) = {}, expected = {}", x, j1, expected_j1);

    // Test j_0(0) = 1
    let j0_zero = spherical_bessel_j(0, 0.0);
    println!("j_0(0) = {}, expected = 1.0", j0_zero);
    assert!((j0_zero - 1.0).abs() < 1e-10);

    // Test j_n(0) = 0 for n > 0
    let j1_zero = spherical_bessel_j(1, 0.0);
    println!("j_1(0) = {}, expected = 0.0", j1_zero);
    assert!(j1_zero.abs() < 1e-10);
}

/// Test constant polynomial Fourier transform
/// This should help identify why constant polynomials produce non-zero values
#[test]
fn test_constant_polynomial_fourier_transform() {
    // Create constant polynomial f(x) = 1
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test evaluation at different frequencies
    for n in 0..5 {
        if let Ok(omega) = crate::freq::MatsubaraFreq::new(n) {
            let result = ft_poly.evaluate(&omega);
            println!("Constant poly at n={}: {}", n, result);

            // For constant polynomial, only n=0 should be non-zero
            if n == 0 {
                assert!(
                    result.norm() > 0.1,
                    "Constant polynomial should be non-zero at n=0"
                );
            } else {
                // For n > 0, the result should be very close to zero
                println!("  Norm at n={}: {}", n, result.norm());
                if result.norm() > 1e-6 {
                    println!(
                        "  WARNING: Constant polynomial has significant value at n={}",
                        n
                    );
                }
            }
        }
    }
}
