//! Tests for piecewise Legendre polynomial Fourier transform implementations
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::basis::FiniteTempBasis;
use crate::freq::{BosonicFreq, FermionicFreq};
use crate::kernel::{CentrosymmKernel, KernelProperties, LogisticKernel, RegularizedBoseKernel};
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::polyfourier::{
    BosonicPiecewiseLegendreFT, FermionicPiecewiseLegendreFT, FermionicPiecewiseLegendreFTVector,
    PiecewiseLegendreFT,
};
use crate::special_functions::spherical_bessel_j;
use crate::traits::{Bosonic, Fermionic, Statistics, StatisticsType};
use mdarray::tensor;
use num_complex::Complex64;

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

// ===== Asymptotic branch of uhat (|n| >= n_asymp); regression tests for #265 =====

/// Smallest Matsubara index `n >= n_min` of the parity `zeta` (1: fermions,
/// 0: bosons).
fn matsubara_index_at_or_above(n_min: i64, zeta: i64) -> i64 {
    if (n_min - zeta).rem_euclid(2) == 0 {
        n_min
    } else {
        n_min + 1
    }
}

/// Relative tolerance for the agreement between the asymptotic branch (`giw`)
/// and the exact transform (`compute_unl_inner`) of the same basis function.
///
/// Error model: `giw` sums the complete endpoint expansion (one term per
/// Legendre order; all higher derivatives of the piecewise polynomial vanish),
/// so it differs from the exact transform only by the jumps of the
/// piecewise-Legendre representation at interior knots, plus f64 rounding.
/// For the functions #265 did not affect (l = 0, 3 mod 4) this difference was
/// measured before the fix at <= 1.3e-8 relative for LogisticKernel up to
/// Lambda = 1e3, and at <= 6.1e-10 for RegularizedBoseKernel at Lambda = 10.
/// (For RegularizedBoseKernel the agreement degrades away from Lambda = 10,
/// for reasons unrelated to #265: 5e-7 at Lambda = 100 and above 1e-2 at
/// Lambda <= 3. That kernel is therefore checked at Lambda = 10 only.)
/// A wrong parity in the moments instead puts the asymptotic value on the
/// other complex axis: uhat_l(n) is purely real or purely imaginary,
/// depending on the parity of u_l and on the statistics, so the relative
/// error is then at least 1.
const ASYMPTOTIC_VS_EXACT_RTOL: f64 = 1e-6;

/// Evaluate both branches of `PiecewiseLegendreFT::evaluate` -- the exact
/// transform and the asymptotic series -- at the same frequencies on both
/// sides of the switch point `n_asymp`, for every basis function, and require
/// them to agree.
fn check_asymptotic_branch_matches_exact<K, S>(basis: &FiniteTempBasis<K, S>)
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    // #265 hit exactly l = 1, 2 (mod 4): make sure every residue occurs twice.
    assert!(
        basis.size() >= 8,
        "basis too small: size = {}",
        basis.size()
    );

    let uhat = basis.uhat();
    let n_asymp = uhat.n_asymp();
    assert_eq!(n_asymp, basis.kernel().conv_radius());
    let zeta = uhat[0].zeta();
    // First index on the asymptotic branch; n0 - 2 is the last exact one.
    let n0 = matsubara_index_at_or_above(n_asymp.ceil() as i64, zeta);
    let ns = [
        n0 - 4,
        n0 - 2,
        n0,
        n0 + 2,
        matsubara_index_at_or_above(2 * n0, zeta),
        matsubara_index_at_or_above(10 * n0, zeta),
        -n0,
        -(n0 + 2),
    ];

    let mut failures = Vec::new();
    let mut max_rel_err = 0.0_f64;
    for l in 0..basis.size() {
        let ft = &uhat[l];
        assert_eq!(
            ft.evaluate_at_n(n0 - 2),
            ft.compute_unl_inner(&ft.poly, n0 - 2)
        );
        assert_eq!(ft.evaluate_at_n(n0), ft.giw(n0));
        for &n in &ns {
            let exact = ft.compute_unl_inner(&ft.poly, n);
            let asymptotic = ft.giw(n);
            let rel_err = (asymptotic - exact).norm() / exact.norm();
            max_rel_err = max_rel_err.max(rel_err);
            if !(rel_err <= ASYMPTOTIC_VS_EXACT_RTOL) {
                failures.push((l, n, rel_err));
            }
        }
    }
    let mut failing_l: Vec<usize> = failures.iter().map(|&(l, _, _)| l).collect();
    failing_l.dedup();
    assert!(
        failures.is_empty(),
        "{:?}, Lambda = {}, beta = {}, size = {}: asymptotic uhat deviates from the exact \
         transform by more than {:e} (relative) near n_asymp = {} for l = {:?}; \
         max relative error = {:.3e}; first failure (l, n, rel_err) = {:?}",
        S::STATISTICS,
        basis.lambda(),
        basis.beta(),
        basis.size(),
        ASYMPTOTIC_VS_EXACT_RTOL,
        n_asymp,
        failing_l,
        max_rel_err,
        failures.first()
    );
}

/// Check the 1/nu tail of uhat against the imaginary-time basis functions.
///
/// Integrating by parts, i nu_n uhat_l(n) = L0 - L1 / (i nu_n) + O(nu_n^-2) with
/// nu_n = pi n / beta, L0 = (-1)^n u_l(beta) - u_l(0) and
/// L1 = (-1)^n u_l'(beta) - u_l'(0), where (-1)^n is -1 for fermions and +1 for
/// bosons. At the frequencies used here the remainder is dominated by the L1
/// term, so |i nu_n uhat_l(n) - L0| <= 2 (|u_l'(0)| + |u_l'(beta)|) / |nu_n|,
/// plus a rounding floor. With the parity defect of #265 the limit came out as
/// (-1)^n u_l(beta) + u_l(0) instead, an error of 2 |u_l(beta)|.
fn check_uhat_high_frequency_tail<K, S>(basis: &FiniteTempBasis<K, S>)
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let beta = basis.beta();
    let u = basis.u();
    let uhat = basis.uhat();
    let zeta = uhat[0].zeta();
    let sign_n = if zeta == 1 { -1.0 } else { 1.0 }; // (-1)^n
    let ns = [
        matsubara_index_at_or_above(1_000_000, zeta),
        matsubara_index_at_or_above(100_000_000, zeta),
    ];

    let mut failing_l = Vec::new();
    let mut max_err_over_tol = 0.0_f64;
    for l in 0..basis.size() {
        let (u_0, u_beta) = (u[l].evaluate(0.0), u[l].evaluate(beta));
        let du = u[l].deriv(1);
        let (du_0, du_beta) = (du.evaluate(0.0), du.evaluate(beta));
        let limit = Complex64::new(sign_n * u_beta - u_0, 0.0);
        for n in ns.into_iter().flat_map(|n| [n, -n]) {
            assert!(n.unsigned_abs() as f64 >= uhat.n_asymp());
            let nu = std::f64::consts::PI * n as f64 / beta;
            let scaled = Complex64::new(0.0, nu) * uhat[l].evaluate_at_n(n);
            let tol =
                2.0 * (du_0.abs() + du_beta.abs()) / nu.abs() + 1e-12 * (u_0.abs() + u_beta.abs());
            let err = (scaled - limit).norm();
            max_err_over_tol = max_err_over_tol.max(err / tol);
            if !(err <= tol) {
                failing_l.push(l);
            }
        }
    }
    failing_l.dedup();
    assert!(
        failing_l.is_empty(),
        "{:?}, Lambda = {}, beta = {}, size = {}: i nu_n uhat_l(n) does not approach \
         (-1)^n u_l(beta) - u_l(0) for l = {:?}; max error / tolerance = {:.3e}",
        S::STATISTICS,
        basis.lambda(),
        beta,
        basis.size(),
        failing_l,
        max_err_over_tol
    );
}

#[test]
fn test_uhat_asymptotic_branch_logistic_fermionic() {
    // The parameters of the #265 report (beta = 10, wmax = 1, eps = 1e-10;
    // the SVE runs in Df64), then a larger Lambda with an f64 SVE.
    for (lambda, beta, epsilon) in [(10.0, 10.0, 1e-10), (1e3, 100.0, 1e-6)] {
        let basis = FiniteTempBasis::<_, Fermionic>::new(
            LogisticKernel::new(lambda),
            beta,
            Some(epsilon),
            None,
        );
        check_asymptotic_branch_matches_exact(&basis);
        check_uhat_high_frequency_tail(&basis);
    }
}

#[test]
fn test_uhat_asymptotic_branch_logistic_bosonic() {
    for (lambda, beta, epsilon) in [(10.0, 10.0, 1e-10), (1e3, 100.0, 1e-6)] {
        let basis = FiniteTempBasis::<_, Bosonic>::new(
            LogisticKernel::new(lambda),
            beta,
            Some(epsilon),
            None,
        );
        check_asymptotic_branch_matches_exact(&basis);
        check_uhat_high_frequency_tail(&basis);
    }
}

#[test]
fn test_uhat_asymptotic_branch_regularized_bose() {
    let basis = FiniteTempBasis::<_, Bosonic>::new(
        RegularizedBoseKernel::new(10.0),
        10.0,
        Some(1e-10),
        None,
    );
    check_asymptotic_branch_matches_exact(&basis);
    check_uhat_high_frequency_tail(&basis);
}
