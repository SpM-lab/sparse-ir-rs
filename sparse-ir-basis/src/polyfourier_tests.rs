//! Tests for piecewise Legendre polynomial Fourier transform implementations
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::basis::FiniteTempBasis;
use crate::freq::{BosonicFreq, FermionicFreq};
use crate::kernel::{BasisKernel, LogisticKernel, RegularizedBoseKernel};
use crate::matrix::mat;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::polyfourier::{
    BosonicPiecewiseLegendreFT, FermionicPiecewiseLegendreFT, FermionicPiecewiseLegendreFTVector,
    PiecewiseLegendreFT,
};
use crate::special_functions::spherical_bessel_j;
use crate::traits::{Bosonic, Fermionic, Statistics, StatisticsType};
use num_complex::Complex64;

#[test]
fn test_fermionic_ft_creation() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly.clone(), Fermionic, None).unwrap();

    assert_eq!(ft_poly.get_n_asymp(), f64::INFINITY);
    assert_eq!(ft_poly.get_statistics(), Statistics::Fermionic);
    assert_eq!(ft_poly.zeta(), 1);
}

#[test]
fn test_bosonic_ft_creation() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = BosonicPiecewiseLegendreFT::new(poly.clone(), Bosonic, Some(100.0)).unwrap();

    assert_eq!(ft_poly.get_n_asymp(), 100.0);
    assert_eq!(ft_poly.get_statistics(), Statistics::Bosonic);
    assert_eq!(ft_poly.zeta(), 0);
}

#[test]
fn test_ft_evaluation_fermionic() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();

    // Test evaluation at valid fermionic frequency
    let omega = FermionicFreq::new(1).unwrap();
    let result = ft_poly.evaluate(&omega);

    // The result should be a complex number
    assert!(result.is_finite());
    println!("Fermionic FT at n=1: {}", result);
}

#[test]
fn test_ft_evaluation_bosonic() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None).unwrap();

    // Test evaluation at valid bosonic frequency
    let omega = BosonicFreq::new(0).unwrap();
    let result = ft_poly.evaluate(&omega);

    // The result should be a complex number
    assert!(result.is_finite());
    println!("Bosonic FT at n=0: {}", result);
}

#[test]
fn test_ft_vector_creation() {
    let data1 = mat![[1.0], [0.0]];
    let data2 = mat![[0.0], [1.0]];
    let knots = vec![-1.0, 1.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0).unwrap();
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0).unwrap();

    let ft_poly1 = FermionicPiecewiseLegendreFT::new(poly1, Fermionic, None).unwrap();
    let ft_poly2 = FermionicPiecewiseLegendreFT::new(poly2, Fermionic, None).unwrap();

    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly1, ft_poly2]);

    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_from_poly_vector() {
    let data1 = mat![[1.0], [0.0]];
    let data2 = mat![[0.0], [1.0]];
    let knots = vec![-1.0, 1.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0).unwrap();
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0).unwrap();

    let poly_vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]).unwrap();
    let ft_vector =
        FermionicPiecewiseLegendreFTVector::from_poly_vector(&poly_vector, Fermionic, None)
            .unwrap();

    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_evaluation() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();
    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly]);

    let omega = FermionicFreq::new(1).unwrap();
    let results = ft_vector.evaluate_at(&omega);

    assert_eq!(results.len(), 1);
    assert!(results[0].is_finite());
}

#[test]
fn test_power_model_creation() {
    let data = mat![[1.0, 0.0], [0.0, 1.0]];
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0).unwrap();

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();

    // Check that power model was created
    assert!(!ft_poly.model.moments.is_empty());
    println!("Power model moments: {:?}", ft_poly.model.moments);
}

#[test]
fn test_invalid_domain_panic() {
    let data = mat![[1.0], [0.0]];
    let knots = vec![0.0, 2.0]; // Invalid domain for Fourier transform

    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    // This should panic
    std::panic::catch_unwind(|| {
        FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();
    })
    .expect_err("Should panic for invalid domain");
}

/// Test to compare get_tnl implementation with expected values
/// This is a simplified test - in practice we would need reference values from C++
#[test]
fn test_get_tnl_basic_values() {
    // Create a simple polynomial for testing
    let data = mat![[1.0, 0.0], [0.0, 1.0]];
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0).unwrap();

    let _ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();

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
    let data = mat![[1.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();
    let _ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();

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
    let data = mat![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0).unwrap();

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();

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
    K: BasisKernel,
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
            ft.evaluate_at_n(n0 - 2).unwrap(),
            ft.compute_unl_inner(&ft.poly, n0 - 2)
        );
        assert_eq!(ft.evaluate_at_n(n0).unwrap(), ft.giw(n0));
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
    K: BasisKernel,
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
            let scaled = Complex64::new(0.0, nu) * uhat[l].evaluate_at_n(n).unwrap();
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
            LogisticKernel::new(lambda).unwrap(),
            beta,
            Some(epsilon),
            None,
        )
        .unwrap();
        check_asymptotic_branch_matches_exact(&basis);
        check_uhat_high_frequency_tail(&basis);
    }
}

#[test]
fn test_uhat_asymptotic_branch_logistic_bosonic() {
    for (lambda, beta, epsilon) in [(10.0, 10.0, 1e-10), (1e3, 100.0, 1e-6)] {
        let basis = FiniteTempBasis::<_, Bosonic>::new(
            LogisticKernel::new(lambda).unwrap(),
            beta,
            Some(epsilon),
            None,
        )
        .unwrap();
        check_asymptotic_branch_matches_exact(&basis);
        check_uhat_high_frequency_tail(&basis);
    }
}

#[test]
fn test_uhat_asymptotic_branch_regularized_bose() {
    let basis = FiniteTempBasis::<_, Bosonic>::new(
        RegularizedBoseKernel::new(10.0).unwrap(),
        10.0,
        Some(1e-10),
        None,
    )
    .unwrap();
    check_asymptotic_branch_matches_exact(&basis);
    check_uhat_high_frequency_tail(&basis);
}

/// Functions without a parity (symm = 0, as from `compute_sve_general`) have
/// default Matsubara sampling points (they were NotSupported, #183): every
/// entry point returns the same symmetric set, and the samplings built from
/// it, through the Basis trait, recover the coefficients.
#[test]
fn test_default_matsubara_points_of_functions_without_parity() {
    use crate::basis_trait::Basis;
    use crate::matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
    use crate::sve::{TworkType, compute_sve_general};
    use num_complex::Complex64;

    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve_general(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    assert!(sve.u().get_polys().iter().all(|u| u.get_symm() == 0));
    let basis =
        FiniteTempBasis::<_, Fermionic>::from_sve_result(kernel, 1.0, sve, Some(1e-6), None)
            .unwrap();
    for positive_only in [false, true] {
        let points = basis
            .default_matsubara_sampling_points(positive_only)
            .unwrap();
        let ns: Vec<i64> = points.iter().map(|f| f.n()).collect();
        assert!(!ns.is_empty() && ns.iter().all(|n| n % 2 != 0), "{ns:?}");
        assert_eq!(
            Basis::default_matsubara_sampling_points(&basis, positive_only).unwrap(),
            points
        );
        assert_eq!(
            basis
                .default_matsubara_sampling_points_i64(positive_only)
                .unwrap(),
            ns
        );
        assert_eq!(
            basis
                .default_matsubara_sampling_points_i64_with_mitigate(
                    positive_only,
                    false,
                    basis.size()
                )
                .unwrap(),
            ns
        );
    }

    let coeffs: Vec<f64> = (0..basis.size()).map(|l| 1.0 / (l as f64 + 1.0)).collect();
    let complex: Vec<Complex64> = coeffs.iter().map(|&c| Complex64::new(c, 0.0)).collect();
    let sampling = MatsubaraSampling::new(&basis).unwrap();
    let fitted = sampling.fit(&sampling.evaluate(&complex).unwrap()).unwrap();
    for (l, (a, b)) in fitted.iter().zip(&complex).enumerate() {
        assert!((a - b).norm() < 1e-10, "l = {l}: {a} vs {b}");
    }
    let sampling = MatsubaraSamplingPositiveOnly::new(&basis).unwrap();
    let fitted = sampling.fit(&sampling.evaluate(&coeffs).unwrap()).unwrap();
    for (l, (a, b)) in fitted.iter().zip(&coeffs).enumerate() {
        assert!((a - b).abs() < 1e-10, "l = {l}: {a} vs {b}");
    }
}

/// A bosonic basis of a nearly one-sided window (review of #342: β = 10,
/// ε = 1e-6, window (-0.01, 1), L = 8) gets well-conditioned default
/// Matsubara points from the pair selection over the whole grid. Selecting
/// among the sign changes of û_L instead (8277719, reverted) gave condition
/// numbers of 1.37e3 (1.93e3 with positive_only), and dropped n = ±2.
#[test]
fn test_default_matsubara_points_of_a_nearly_one_sided_bosonic_window() {
    use crate::kernel::LogisticKernelWindowed;
    use crate::matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};

    let kernel = LogisticKernelWindowed::from_physical(10.0, -0.01, 1.0).unwrap();
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, 10.0, Some(1e-6), None).unwrap();
    assert_eq!(basis.size(), 8);
    let ns = basis.default_matsubara_sampling_points_i64(false).unwrap();
    assert!(ns.contains(&2) && ns.contains(&-2), "{ns:?}");
    // Measured: 3.84 for both
    let cond = MatsubaraSampling::new(&basis)
        .unwrap()
        .condition_number()
        .unwrap();
    assert!(cond < 10.0, "condition number {cond:e}, points {ns:?}");
    let cond = MatsubaraSamplingPositiveOnly::new(&basis)
        .unwrap()
        .condition_number()
        .unwrap();
    assert!(cond < 10.0, "positive_only: condition number {cond:e}");
}

/// The method of the Matsubara points can be chosen: the pair selection for
/// a basis with parity gives a symmetric set of at least L points, as well
/// conditioned as the sign changes (within a factor 2 here); the sign
/// changes of functions without parity are NotSupported
#[test]
fn test_matsubara_point_method_can_be_chosen() {
    use crate::basis::MatsubaraPointMethod;
    use crate::error::Error;
    use crate::freq::MatsubaraFreq;
    use crate::kernel::LogisticKernelWindowed;
    use crate::traits::StatisticsType;
    use nalgebra::DMatrix;

    fn check<S: StatisticsType + std::fmt::Debug + 'static>() {
        let basis =
            FiniteTempBasis::<_, S>::new(LogisticKernel::new(1e3).unwrap(), 10.0, Some(1e-8), None)
                .unwrap();
        let points = |method| {
            basis
                .matsubara_sampling_points(false, false, method)
                .unwrap()
                .iter()
                .map(|f| f.n())
                .collect::<Vec<i64>>()
        };
        assert_eq!(
            points(MatsubaraPointMethod::SignChanges),
            basis.default_matsubara_sampling_points_i64(false).unwrap()
        );
        let pairs = points(MatsubaraPointMethod::PairSelection);
        assert!(pairs.len() >= basis.size(), "{pairs:?}");
        assert!(pairs.iter().all(|n| pairs.contains(&-n)), "{pairs:?}");
        let positive = basis
            .matsubara_sampling_points(true, false, MatsubaraPointMethod::PairSelection)
            .unwrap();
        assert!(
            positive
                .iter()
                .all(|f| f.n() >= 0 && pairs.contains(&f.n()))
        );

        let cond = |ns: &[i64]| {
            let a = DMatrix::from_fn(2 * ns.len(), basis.size(), |i, l| {
                let v = basis.uhat_full()[l].evaluate(&MatsubaraFreq::<S>::new(ns[i / 2]).unwrap());
                if i % 2 == 0 { v.re } else { v.im }
            });
            let sv = a.singular_values();
            sv.max() / sv.min()
        };
        let signs = points(MatsubaraPointMethod::SignChanges);
        let (c_pairs, c_signs) = (cond(&pairs), cond(&signs));
        assert!(
            c_pairs < 2.0 * c_signs,
            "{:?}: {c_pairs:e} vs {c_signs:e}",
            S::STATISTICS
        );

        let window = LogisticKernelWindowed::from_physical(10.0, -0.3, 1.0).unwrap();
        let basis = FiniteTempBasis::<_, S>::new(window, 10.0, Some(1e-6), None).unwrap();
        let err = basis
            .matsubara_sampling_points(false, false, MatsubaraPointMethod::SignChanges)
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
        assert_eq!(
            basis
                .matsubara_sampling_points(false, false, MatsubaraPointMethod::PairSelection)
                .unwrap(),
            basis.default_matsubara_sampling_points(false).unwrap()
        );
    }
    check::<Fermionic>();
    check::<Bosonic>();
}

/// `set` rejects an index past the end, including `index == len`, and then
/// leaves the vector unchanged.
#[test]
fn test_ft_vector_set_rejects_index_past_the_end() {
    use crate::error::Error;

    let knots = vec![-1.0, 1.0];
    let poly0 = PiecewiseLegendrePoly::new(mat![[1.0], [0.0]], knots.clone(), 0, None, 0).unwrap();
    let poly1 = PiecewiseLegendrePoly::new(mat![[0.0], [1.0]], knots, 1, None, 0).unwrap();
    let ft0 = FermionicPiecewiseLegendreFT::new(poly0, Fermionic, None).unwrap();
    let ft1 = FermionicPiecewiseLegendreFT::new(poly1, Fermionic, None).unwrap();
    let mut ft_vector =
        FermionicPiecewiseLegendreFTVector::from_vector(vec![ft0.clone(), ft1.clone()]);

    for index in [2, 3] {
        let err = ft_vector.set(index, ft0.clone()).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidParameter {
                name: "index",
                value: index.to_string(),
                reason: "must be less than the length 2".to_string(),
            }
        );
    }
    assert_eq!(ft_vector.size(), 2);
    assert_eq!(ft_vector.get(0).unwrap().poly.get_l(), 0);
    assert_eq!(ft_vector.get(1).unwrap().poly.get_l(), 1);

    // A valid index replaces the element.
    ft_vector.set(0, ft1).unwrap();
    assert_eq!(ft_vector.get(0).unwrap().poly.get_l(), 1);
}

/// `new` checks the domain and n_asymp with typed errors (before: a panic for
/// the domain, and any n_asymp was accepted), and evaluates the power model
/// at the end of the domain: an xmax just below 1, which the 1e-12 tolerance
/// accepts, panicked in derivs(1.0).
#[test]
fn test_ft_new_checks_its_arguments() {
    use crate::error::Error;

    let poly_on = |lo: f64, hi: f64| {
        PiecewiseLegendrePoly::new(mat![[1.0], [0.5]], vec![lo, hi], 0, None, 1).unwrap()
    };
    let err = FermionicPiecewiseLegendreFT::new(poly_on(-2.0, 2.0), Fermionic, None).unwrap_err();
    assert_eq!(
        err,
        Error::InvalidParameter {
            name: "poly",
            value: "a polynomial on [-2.0, 2.0]".to_string(),
            reason: "must be defined on [-1, 1] (within 1e-12)".to_string(),
        }
    );
    for n_asymp in [-1.0, f64::NAN] {
        let err = FermionicPiecewiseLegendreFT::new(poly_on(-1.0, 1.0), Fermionic, Some(n_asymp))
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidParameter {
                    name: "n_asymp",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    assert!(
        FermionicPiecewiseLegendreFT::new(poly_on(-1.0, 1.0), Fermionic, Some(f64::INFINITY))
            .is_ok()
    );

    let near = poly_on(-1.0, 1.0 - 1e-13);
    let ft = FermionicPiecewiseLegendreFT::new(near.clone(), Fermionic, None).unwrap();
    assert_eq!(ft.model.moments.len(), near.get_polyorder());
}

/// `evaluate_at_n` rejects an index of the wrong parity; before it returned 0.
#[test]
fn test_evaluate_at_n_rejects_the_wrong_parity() {
    use crate::error::Error;

    let poly = PiecewiseLegendrePoly::new(mat![[1.0], [0.5]], vec![-1.0, 1.0], 0, None, 1).unwrap();
    let ft = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None).unwrap();
    assert_eq!(
        ft.evaluate_at_n(2),
        Err(Error::InvalidMatsubaraIndex {
            n: 2,
            statistics: Statistics::Fermionic,
        })
    );
    assert_eq!(
        ft.evaluate_at_n(1),
        Ok(ft.evaluate(&FermionicFreq::new(1).unwrap()))
    );
    assert!(ft.evaluate_at_ns(&[1, 2]).is_err());
    assert_eq!(ft.evaluate_at_ns(&[1, -1]).unwrap().len(), 2);
}

/// Functions without a definite parity (symm = 0) have no sign changes and
/// extrema to choose Matsubara points by: these are NotSupported, with both
/// statistics (#183); the default points then come from the pair selection.
#[test]
fn test_sign_changes_of_functions_without_parity_are_not_supported() {
    use crate::error::Error;
    use crate::freq::MatsubaraFreq;
    use crate::polyfourier::{find_extrema, sign_changes};

    let poly = PiecewiseLegendrePoly::new(mat![[1.0], [0.5]], vec![-1.0, 1.0], 0, None, 0).unwrap();
    let not_supported = |r: Result<Vec<i64>, Error>| {
        let err = r.unwrap_err();
        assert!(
            matches!(&err, Error::NotSupported { what } if what.contains("got symm = 0")),
            "{err:?}"
        );
    };
    let ns =
        |v: Vec<MatsubaraFreq<Fermionic>>| -> Vec<i64> { v.into_iter().map(|f| f.n()).collect() };
    let ft = FermionicPiecewiseLegendreFT::new(poly.clone(), Fermionic, None).unwrap();
    for positive_only in [false, true] {
        not_supported(sign_changes(&ft, positive_only).map(ns));
        not_supported(find_extrema(&ft, positive_only).map(ns));
    }
    let nb =
        |v: Vec<MatsubaraFreq<Bosonic>>| -> Vec<i64> { v.into_iter().map(|f| f.n()).collect() };
    let ft = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None).unwrap();
    for positive_only in [false, true] {
        not_supported(sign_changes(&ft, positive_only).map(nb));
        not_supported(find_extrema(&ft, positive_only).map(nb));
    }
}

/// A function without definite parity (`symm == 0`, mixing even P0/P2 and odd
/// P1 content) must get correct Matsubara values from the asymptotic model at
/// large |n|: the model agrees with the exact inner transform. (A one-sided
/// moment formula that takes the parity from `(-1)^l` is ~100% wrong here.)
#[test]
fn test_parityless_asymptotic_model_matches_inner_transform() {
    let poly =
        PiecewiseLegendrePoly::new(mat![[0.5], [0.4], [0.3]], vec![-1.0, 1.0], 1, None, 0).unwrap();
    // n_asymp = inf: always the inner transform; n_asymp = 3: the model at |n| >= 3
    let exact = FermionicPiecewiseLegendreFT::new(poly.clone(), Fermionic, None).unwrap();
    let asymp = FermionicPiecewiseLegendreFT::new(poly, Fermionic, Some(3.0)).unwrap();

    for n in [5_i64, 11, 21, 51, 101, 501] {
        let e = exact.evaluate_at_n(n).unwrap();
        let a = asymp.evaluate_at_n(n).unwrap();
        let rel = (a - e).norm() / e.norm();
        assert!(
            rel < 1e-9,
            "n={n}: asymptotic {a} vs exact {e} (rel_err={rel:.2e})"
        );
    }
}

/// The basis functions of an asymmetric window have no parity: the
/// asymptotic model at |n| >= n_asymp, which takes the boundary derivatives
/// at both ends, must still agree with the exact inner transform.
#[test]
fn test_uhat_asymptotic_branch_logistic_windowed() {
    use crate::kernel::LogisticKernelWindowed;

    let (beta, omega_min, omega_max) = (10.0, -0.3, 1.0);
    let kernel = LogisticKernelWindowed::from_physical(beta, omega_min, omega_max).unwrap();
    let fermionic = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(1e-10), None).unwrap();
    assert!(fermionic.uhat()[0].poly.get_symm() == 0);
    check_asymptotic_branch_matches_exact(&fermionic);
    let bosonic = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, Some(1e-10), None).unwrap();
    check_asymptotic_branch_matches_exact(&bosonic);
}

/// The Matsubara candidates are evaluated for n >= 0 only, relying on
/// û_l(-n) = conj(û_l(n)) for real u_l(τ); check it on a basis without
/// parity (asymmetric window), at low and asymptotic frequencies.
#[test]
fn test_uhat_negative_frequency_is_conjugate() {
    use crate::freq::MatsubaraFreq;
    use crate::kernel::LogisticKernelWindowed;

    let kernel = LogisticKernelWindowed::from_physical(100.0, -0.2, 1.0).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, 100.0, Some(1e-8), None).unwrap();
    let uhat = basis.uhat_full();
    for n in [1i64, 3, 51, 999, 100_001, 10_000_001] {
        let pos = uhat.evaluate_at(&MatsubaraFreq::<Fermionic>::new(n).unwrap());
        let neg = uhat.evaluate_at(&MatsubaraFreq::<Fermionic>::new(-n).unwrap());
        for (p, q) in pos.iter().zip(&neg) {
            assert_eq!(*p, q.conj(), "n = {n}");
        }
    }
}

/// Default Matsubara sampling points are symmetric (`±n_k`, the full set is
/// the positive-only set and its negatives), with an even number of points
/// for fermions and an odd number including n = 0 for bosons; the number of
/// points is the basis size, or one more when the parity requires it, and
/// the fitting matrix is well conditioned. mitigate adds points: symmetric
/// ones (oversampling) without parity, fencing points with parity. Checked
/// with parity (symmetric kernel: the sign changes of SparseIR.jl) and
/// without (asymmetric window: the pair selection).
#[test]
fn test_default_matsubara_points_symmetric() {
    use crate::freq::MatsubaraFreq;
    use crate::kernel::LogisticKernelWindowed;

    fn check<K, S>(kernel: K)
    where
        K: BasisKernel,
        S: StatisticsType + 'static,
    {
        let basis = FiniteTempBasis::<K, S>::new(kernel, 100.0, Some(1e-8), None).unwrap();
        let l = basis.size();
        let full = basis.default_matsubara_sampling_points(false).unwrap();
        let pos = basis.default_matsubara_sampling_points(true).unwrap();
        let mut expected: Vec<MatsubaraFreq<S>> = pos.clone();
        expected.extend(pos.iter().filter(|f| f.n() != 0).map(|&f| -f));
        expected.sort();
        assert!(full == expected, "full set is not ±(positive-only set)");
        assert!(pos.iter().all(|f| f.n() >= 0));
        let bosonic = S::STATISTICS == Statistics::Bosonic;
        assert_eq!(full.len() % 2, usize::from(bosonic));
        assert_eq!(pos.first().unwrap().n() == 0, bosonic);
        assert!(
            full.len() == l || full.len() == l + 1,
            "{} points for L = {l}",
            full.len()
        );
        // Conditioning of the fit at these points
        let values = basis.uhat().evaluate_at_many(&full);
        let a = nalgebra::DMatrix::from_fn(full.len(), l, |i, j| values[i][j]);
        let sv = a.singular_values();
        assert!(sv[0] / sv[l - 1] < 100.0, "cond = {}", sv[0] / sv[l - 1]);
        // mitigate adds points
        let full_n: Vec<i64> = full.iter().map(|f| f.n()).collect();
        let over = basis
            .default_matsubara_sampling_points_i64_with_mitigate(false, true, l)
            .unwrap();
        assert!(over.len() > full_n.len());
        assert!(full_n.iter().all(|n| over.contains(n)));
        if basis.uhat()[0].poly.get_symm() == 0 {
            assert!(over.iter().all(|n| over.contains(&-n)));
        }
    }
    check::<_, Fermionic>(LogisticKernel::new(100.0).unwrap());
    check::<_, Bosonic>(LogisticKernel::new(100.0).unwrap());
    let window = LogisticKernelWindowed::from_physical(100.0, -0.2, 1.0).unwrap();
    check::<_, Fermionic>(window);
    check::<_, Bosonic>(window);
}

/// The inner transform, contracted segment by segment from shared factors,
/// agrees with the term-by-term sum of `data · T_nl · phase / norm` to
/// rounding, for every function of a basis, at low and high frequencies
#[test]
fn test_inner_transform_matches_term_by_term_sum() {
    let beta = 10.0;
    let kernel = LogisticKernel::new(100.0).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(1e-10), None).unwrap();
    let uhat = basis.uhat_full();
    for n in [1i64, -3, 51, 999, 100_001] {
        let freq = FermionicFreq::new(n).unwrap();
        let shared = uhat.evaluate_at(&freq);
        let max = shared.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for (l, ft) in uhat.get_polys().iter().enumerate() {
            if (n as f64).abs() >= ft.n_asymp {
                continue;
            }
            let poly = &ft.poly;
            let wred = std::f64::consts::PI / 4.0 * n as f64;
            let phase = FermionicPiecewiseLegendreFT::phase_stable(poly, n);
            let mut naive = Complex64::new(0.0, 0.0);
            for order in 0..poly.get_polyorder() {
                for (j, ph) in phase.iter().enumerate() {
                    let tnl = FermionicPiecewiseLegendreFT::get_tnl(
                        order as i32,
                        wred * poly.get_delta_x()[j],
                    );
                    naive += poly.get_data()[[order, j]] * tnl * ph / poly.get_norms()[j];
                }
            }
            naive /= 2.0_f64.sqrt();
            assert!(
                (shared[l] - naive).norm() <= 1e-13 * max,
                "n = {n}, l = {l}: {} vs {naive}",
                shared[l]
            );
        }
    }
}

/// Evaluating all functions at once (sharing the segment-wise factors) gives
/// bitwise the values of evaluating each function on its own, with and
/// without parity, on both branches of the transform
#[test]
fn test_evaluate_at_matches_each_function_bitwise() {
    use crate::kernel::LogisticKernelWindowed;

    fn check<K: BasisKernel>(kernel: K) {
        let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, 10.0, Some(1e-10), None).unwrap();
        let uhat = basis.uhat_full();
        for n in [1i64, -3, 51, 999, 100_001, 10_000_001] {
            let freq = FermionicFreq::new(n).unwrap();
            let shared = uhat.evaluate_at(&freq);
            for (l, ft) in uhat.get_polys().iter().enumerate() {
                assert_eq!(shared[l], ft.evaluate(&freq), "n = {n}, l = {l}");
            }
        }
    }
    check(LogisticKernel::new(100.0).unwrap());
    check(LogisticKernelWindowed::from_physical(10.0, -3.0, 10.0).unwrap());
}

/// A bosonic pair selection always has n = 0, a fermionic one never, and as
/// many rows as requested functions rounded up to the parity of the
/// statistics, also when `uhat` has no next function to round up with
/// (review of #342: β = 100, window (-0.3, 1), ε = 1e-8, L = 22). Requesting
/// more points than functions is an error, not a smaller set.
#[test]
fn test_pair_selection_keeps_n0_and_the_requested_size() {
    use crate::error::Error;
    use crate::kernel::LogisticKernelWindowed;
    use crate::polyfourier::{PiecewiseLegendreFTVector, pair_sampling_points};

    fn check<S: StatisticsType + 'static>() {
        let kernel = LogisticKernelWindowed::from_physical(100.0, -0.3, 1.0).unwrap();
        let basis = FiniteTempBasis::<_, S>::new(kernel, 100.0, Some(1e-8), None).unwrap();
        let l = basis.size();
        let bosonic = S::STATISTICS == Statistics::Bosonic;
        for n_funcs in [l, l + 1, l + 2] {
            let uhat = PiecewiseLegendreFTVector::from_vector(
                basis.uhat_full().polyvec[..n_funcs].to_vec(),
            );
            for n_points in [l - 1, l, n_funcs] {
                for (positive_only, mitigate) in [(false, false), (true, false), (false, true)] {
                    let ns: Vec<i64> =
                        pair_sampling_points(&uhat, n_points, positive_only, mitigate)
                            .unwrap()
                            .iter()
                            .map(|f| f.n())
                            .collect();
                    let case = format!(
                        "{:?}, {n_funcs} functions, {n_points} points, positive_only = \
                         {positive_only}, mitigate = {mitigate}: {ns:?}",
                        S::STATISTICS
                    );
                    let positive = ns.iter().filter(|&&n| n > 0).count();
                    let rows = 2 * positive + usize::from(bosonic);
                    assert_eq!(ns.contains(&0), bosonic, "{case}");
                    if !positive_only {
                        assert_eq!(ns.len(), rows, "{case}");
                    }
                    if !mitigate {
                        // n_points rounded up to the parity of the statistics
                        assert!(rows >= n_points && rows <= n_points + 1, "{case}");
                        assert_eq!(rows % 2 == 1, bosonic, "{case}");
                    }
                }
            }
            assert!(matches!(
                pair_sampling_points(&uhat, n_funcs + 1, false, false),
                Err(Error::InvalidParameter {
                    name: "n_points",
                    ..
                })
            ));
        }
    }
    check::<Bosonic>();
    check::<Fermionic>();
}

/// `mitigate` adds frequencies that lower the condition number of the
/// sampling matrix (by about half, measured over the bases of the comparison
/// in the review of #342); oversampling with the opposite choice would raise
/// it.
#[test]
fn test_pair_selection_mitigate_improves_conditioning() {
    use crate::basis::MatsubaraPointMethod;
    use crate::kernel::LogisticKernelWindowed;
    use crate::matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};

    fn check<K: BasisKernel, S: StatisticsType + 'static>(kernel: K, name: &str) {
        let basis = FiniteTempBasis::<_, S>::new(kernel, 1e3, Some(1e-8), None).unwrap();
        for positive_only in [false, true] {
            let cond = |mitigate| {
                let points = basis
                    .matsubara_sampling_points(
                        positive_only,
                        mitigate,
                        MatsubaraPointMethod::PairSelection,
                    )
                    .unwrap();
                if positive_only {
                    MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, points)
                        .unwrap()
                        .condition_number()
                        .unwrap()
                } else {
                    MatsubaraSampling::with_sampling_points(&basis, points)
                        .unwrap()
                        .condition_number()
                        .unwrap()
                }
            };
            let (plain, mitigated) = (cond(false), cond(true));
            assert!(
                mitigated < 0.8 * plain,
                "{name} {:?}, positive_only = {positive_only}: condition number {mitigated:e} \
                 with mitigate, {plain:e} without",
                S::STATISTICS
            );
        }
    }
    check::<_, Fermionic>(LogisticKernel::new(1e3).unwrap(), "logistic");
    check::<_, Bosonic>(LogisticKernel::new(1e3).unwrap(), "logistic");
    let windowed = LogisticKernelWindowed::from_physical(1e3, -0.3, 1.0).unwrap();
    check::<_, Fermionic>(windowed, "windowed");
    check::<_, Bosonic>(windowed, "windowed");
}

/// The default Matsubara points of a basis come from the sign changes when
/// its functions have a parity and from the pair selection when they do not,
/// and the pair selection of a small windowed basis is pinned (a change of
/// the method or of its candidate grid shows here; the centrosymmetric
/// bases are pinned to main in sparse-ir/tests/default_points_main.rs).
#[test]
fn test_default_matsubara_points_branch_and_pinned_pair_selection() {
    use crate::basis::MatsubaraPointMethod;
    use crate::kernel::LogisticKernelWindowed;

    fn points<K: BasisKernel, S: StatisticsType + 'static>(
        basis: &FiniteTempBasis<K, S>,
        positive_only: bool,
        mitigate: bool,
        method: Option<MatsubaraPointMethod>,
    ) -> Vec<i64> {
        match method {
            Some(m) => basis
                .matsubara_sampling_points(positive_only, mitigate, m)
                .unwrap()
                .iter()
                .map(|f| f.n())
                .collect(),
            None => basis
                .default_matsubara_sampling_points_i64_with_mitigate(
                    positive_only,
                    mitigate,
                    basis.size(),
                )
                .unwrap(),
        }
    }
    fn check<S: StatisticsType + 'static>(pinned: &[&[i64]; 3]) {
        let logistic = FiniteTempBasis::<_, S>::new(
            LogisticKernel::new(10.0).unwrap(),
            10.0,
            Some(1e-6),
            None,
        )
        .unwrap();
        let kernel = LogisticKernelWindowed::from_physical(10.0, -0.3, 1.0).unwrap();
        let windowed = FiniteTempBasis::<_, S>::new(kernel, 10.0, Some(1e-6), None).unwrap();
        for (k, (positive_only, mitigate)) in [(false, false), (true, false), (false, true)]
            .into_iter()
            .enumerate()
        {
            let sign = Some(MatsubaraPointMethod::SignChanges);
            let pair = Some(MatsubaraPointMethod::PairSelection);
            assert_eq!(
                points(&logistic, positive_only, mitigate, None),
                points(&logistic, positive_only, mitigate, sign)
            );
            let default = points(&windowed, positive_only, mitigate, None);
            assert_eq!(default, points(&windowed, positive_only, mitigate, pair));
            assert_eq!(
                default,
                pinned[k],
                "{:?}, positive_only = {positive_only}, mitigate = {mitigate}",
                S::STATISTICS
            );
        }
    }
    check::<Fermionic>(&PINNED_FERMIONIC);
    check::<Bosonic>(&PINNED_BOSONIC);
}

/// Default Matsubara points of the windowed basis (β = 10, window (-0.3, 1),
/// ε = 1e-6): all, positive_only, mitigate
const PINNED_FERMIONIC: [&[i64]; 3] = [
    &[-13, -7, -5, -3, -1, 1, 3, 5, 7, 13],
    &[1, 3, 5, 7, 13],
    &[-15, -13, -7, -5, -3, -1, 1, 3, 5, 7, 13, 15],
];
const PINNED_BOSONIC: [&[i64]; 3] = [
    &[-10, -6, -4, -2, 0, 2, 4, 6, 10],
    &[0, 2, 4, 6, 10],
    &[-12, -10, -6, -4, -2, 0, 2, 4, 6, 10, 12],
];
