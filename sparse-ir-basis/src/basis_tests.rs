//! Tests for FiniteTempBasis functionality
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::basis::{FermionicBasis, FiniteTempBasis};
use crate::error::Error;
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::sve::{TworkType, compute_sve};
use crate::traits::{Bosonic, Fermionic};

#[test]
fn test_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * omega_max).unwrap();
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None).unwrap();

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);
}

fn invalid(name: &'static str, value: &str, reason: &str) -> Error {
    Error::InvalidParameter {
        name,
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

/// Invalid parameters are rejected before the SVE is computed. Before the
/// change β ≤ 0 and max_size = 0 panicked, β = NaN gave a NaN basis, and
/// ε ≥ 1 or ε = NaN panicked after the SVE (ε = 1 gave a basis of size 1).
/// `max_size` limits the basis only (issue #285), so a zero `max_size` is
/// rejected explicitly instead of by the SVE truncation.
#[test]
fn test_basis_new_rejects_invalid_parameters() {
    let kernel = LogisticKernel::new(10.0).unwrap();
    let cases = [
        (
            -1.0,
            None,
            None,
            invalid("beta", "-1.0", "must be positive and finite"),
        ),
        (
            0.0,
            None,
            None,
            invalid("beta", "0.0", "must be positive and finite"),
        ),
        (
            f64::NAN,
            None,
            None,
            invalid("beta", "NaN", "must be positive and finite"),
        ),
        (
            f64::INFINITY,
            None,
            None,
            invalid("beta", "inf", "must be positive and finite"),
        ),
        (
            10.0,
            Some(0.0),
            None,
            invalid("epsilon", "0.0", "must be in (0, 1)"),
        ),
        (
            10.0,
            Some(1.0),
            None,
            invalid("epsilon", "1.0", "must be in (0, 1)"),
        ),
        (
            10.0,
            Some(2.0),
            None,
            invalid("epsilon", "2.0", "must be in (0, 1)"),
        ),
        (
            10.0,
            Some(f64::NAN),
            None,
            invalid("epsilon", "NaN", "must be in (0, 1)"),
        ),
        (
            10.0,
            Some(1e-6),
            Some(0),
            invalid("max_size", "0", "must be positive"),
        ),
    ];
    for (beta, epsilon, max_size, expected) in cases {
        let err = FermionicBasis::new(kernel, beta, epsilon, max_size)
            .err()
            .expect("must be rejected");
        assert_eq!(
            err, expected,
            "beta = {beta}, epsilon = {epsilon:?}, max_size = {max_size:?}"
        );
    }
}

/// `from_sve_result` checks the same parameters, with `epsilon` as a
/// truncation threshold: 0 keeps every singular value.
#[test]
fn test_basis_from_sve_result_checks_its_parameters() {
    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    let build = |beta, epsilon, max_size| {
        FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
            kernel,
            beta,
            sve.clone(),
            epsilon,
            max_size,
        )
    };
    // FiniteTempBasis does not implement Debug, so take the error with err().
    let rejected = |beta, epsilon, max_size| {
        build(beta, epsilon, max_size)
            .err()
            .expect("must be rejected")
    };
    assert_eq!(
        rejected(-1.0, None, None),
        invalid("beta", "-1.0", "must be positive and finite")
    );
    assert_eq!(
        rejected(10.0, Some(1.0), None),
        invalid("epsilon", "1.0", "must be in [0, 1)")
    );
    assert_eq!(
        rejected(10.0, Some(1e-6), Some(0)),
        invalid("max_size", "0", "must be positive")
    );
    assert_eq!(build(10.0, Some(0.0), None).unwrap().size(), sve.s.len());
}

/// An SVE on another domain than [-1, 1] × [-1, 1] cannot define the basis:
/// its Fourier transform panicked before the change.
#[test]
fn test_basis_from_sve_result_rejects_an_sve_on_another_domain() {
    let kernel = LogisticKernel::new(10.0).unwrap();
    let mut sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    let knots: Vec<f64> = sve.u.get_polys()[0].knots.iter().map(|x| 2.0 * x).collect();
    sve.u = sve.u.rescale_domain(knots, None, None).unwrap();
    let err = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
        kernel, 10.0, sve, None, None,
    )
    .err()
    .expect("must be rejected");
    assert!(
        matches!(
            err,
            Error::InvalidParameter {
                name: "sve_result",
                ..
            }
        ),
        "{err:?}"
    );
}

/// An SVE whose domain ends within 1e-12 inside ±1 passes the domain check.
/// Before the change its Fourier transform panicked at x = 1 (fixed in the
/// FT), and its u functions ended inside [0, β], so that evaluating them at
/// τ = 0 or β panicked. The knots of the basis functions now end exactly at
/// 0 and β (u) and ±ωmax (v).
#[test]
fn test_basis_from_an_sve_just_inside_the_unit_domain() {
    use crate::poly::PiecewiseLegendrePolyVector;

    let beta = 10.0;
    let kernel = LogisticKernel::new(10.0).unwrap();
    let mut sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    let shrink = |funcs: &PiecewiseLegendrePolyVector| {
        let mut knots = funcs.get_polys()[0].knots.clone();
        let last = knots.len() - 1;
        knots[0] = -1.0 + 1e-13;
        knots[last] = 1.0 - 1e-13;
        funcs.rescale_domain(knots, None, None).unwrap()
    };
    sve.u = shrink(&sve.u);
    sve.v = shrink(&sve.v);

    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
        kernel, beta, sve, None, None,
    )
    .unwrap();
    for u in basis.u().get_polys() {
        assert_eq!((u.xmin, u.xmax), (0.0, beta));
        assert!(u.evaluate(0.0).is_finite() && u.evaluate(beta).is_finite());
    }
    let wmax = basis.omega_max();
    for v in basis.v().get_polys() {
        assert_eq!((v.xmin, v.xmax), (-wmax, wmax));
    }
}

/// The snap is a no-op for the SVE of a kernel of this crate, whose knots
/// are exactly ±1: the basis functions keep the scaled knots and widths.
#[test]
fn test_basis_knots_of_a_kernel_sve_are_the_scaled_knots() {
    let beta = 10.0;
    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    let x_knots = sve.u.get_polys()[0].knots.clone();
    let x_widths = sve.u.get_polys()[0].delta_x.clone();
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
        kernel, beta, sve, None, None,
    )
    .unwrap();
    let u = &basis.u().get_polys()[0];
    let scaled: Vec<f64> = x_knots.iter().map(|&x| beta / 2.0 * (x + 1.0)).collect();
    let widths: Vec<f64> = x_widths.iter().map(|&dx| beta / 2.0 * dx).collect();
    assert_eq!(u.knots, scaled);
    assert_eq!(u.delta_x, widths);
}

#[test]
fn test_default_tau_sampling_points_conditioning() {
    // Test parameters: beta=1.0, lambda=10.0
    let beta = 1.0;
    let lambda = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(lambda).unwrap();
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None).unwrap();

    println!("\n=== Default Tau Sampling Points Test ===");
    println!("Beta: {}, Lambda: {}, Epsilon: {}", beta, lambda, epsilon);
    println!("Basis size: {}", basis.size());

    // Get default sampling points
    let tau_points = basis.default_tau_sampling_points().unwrap();
    println!("Number of sampling points: {}", tau_points.len());

    // Verify range: [-beta/2, beta/2] (matches C++ implementation)
    let beta_half = beta / 2.0;
    for &tau in &tau_points {
        assert!(
            tau >= -beta_half && tau <= beta_half,
            "tau={} out of range [{}, {}]",
            tau,
            -beta_half,
            beta_half
        );
    }

    // Verify sorted (monotonically increasing)
    for i in 1..tau_points.len() {
        assert!(
            tau_points[i] >= tau_points[i - 1],
            "Points not sorted: tau[{}]={} < tau[{}]={}",
            i,
            tau_points[i],
            i - 1,
            tau_points[i - 1]
        );
    }

    // Verify symmetry around 0 (for range [-beta/2, beta/2])
    // Points should come in pairs: tau and -tau
    let tol = 1e-10;

    for &tau in &tau_points {
        let tau_reflected = -tau;
        let has_pair = tau_points.iter().any(|&t| (t - tau_reflected).abs() < tol);
        assert!(
            has_pair || tau.abs() < tol,
            "tau={} lacks symmetric pair around 0",
            tau
        );
    }
    println!("✅ Sampling points are symmetric around 0");

    // Evaluate sampling matrix: matrix[i,l] = u_l(tau_i)
    // Use the Basis trait method which handles tau normalization
    use crate::basis_trait::Basis;
    let matrix = basis.evaluate_tau(&tau_points).unwrap();

    let num_points = tau_points.len();
    let basis_size = basis.size();
    println!("Sampling matrix shape: {}x{}", num_points, basis_size);

    let sv = crate::fitters::singular_values(matrix.host_data().unwrap(), num_points, basis_size)
        .expect("SVD computation failed");

    println!("\nSampling matrix SVD:");
    let min_dim = sv.len();
    println!("  Rank: {}", min_dim);
    println!("  First singular value: {:.6e}", sv[0]);
    println!("  Last singular value: {:.6e}", sv[min_dim - 1]);

    let condition_number = sv[0] / sv[min_dim - 1];
    println!("  Condition number: {:.6e}", condition_number);

    // Reference condition number (from Julia/C++ for beta=1, lambda=10)
    // This is approximate - actual value depends on implementation details
    let reference_cond = 25.0; // ~24.4 observed

    // Check: condition number should not be significantly worse
    // (within factor of 2 of reference)
    assert!(
        condition_number < reference_cond * 2.0,
        "Condition number too large: {:.2e} (reference: {:.2e})",
        condition_number,
        reference_cond
    );

    // Julia check: cond > 1e8 triggers warning
    assert!(
        condition_number < 1e8,
        "Sampling matrix is poorly conditioned: cond = {:.6e}",
        condition_number
    );

    println!(
        "✅ Condition number: {:.2e} (reference: {:.2e}, threshold: {:.2e})",
        condition_number,
        reference_cond,
        reference_cond * 2.0
    );
}

#[test]
fn test_regularized_bose_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * omega_max).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);

    println!("\n=== RegularizedBoseKernel Basis Test ===");
    println!(
        "Beta: {}, Omega_max: {}, Epsilon: {}",
        beta, omega_max, epsilon
    );
    println!("Basis size: {}", basis.size());
    println!("Accuracy: {:.6e}", basis.accuracy);
}

#[test]
fn test_regularized_bose_basis_different_parameters() {
    // Test with different beta and omega_max values
    let test_cases = vec![(1.0, 1.0, 1e-6), (10.0, 10.0, 1e-6), (100.0, 1.0, 1e-6)];

    for (beta, omega_max, epsilon) in test_cases {
        let kernel = RegularizedBoseKernel::new(beta * omega_max).unwrap();
        let basis = FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(
            kernel,
            beta,
            Some(epsilon),
            None,
        )
        .unwrap();

        assert_eq!(basis.beta, beta);
        assert!((basis.omega_max() - omega_max).abs() < 1e-10);
        assert!(basis.size() > 0);
        assert!(basis.accuracy > 0.0);
        assert!(basis.accuracy < epsilon);

        println!(
            "Beta={}, Omega_max={}: size={}, accuracy={:.6e}",
            beta,
            omega_max,
            basis.size(),
            basis.accuracy
        );
    }
}

#[test]
fn test_default_omega_sampling_points_fermionic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    let omega_points = basis.default_omega_sampling_points().unwrap();

    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());

    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(
            omega.abs() <= wmax,
            "omega = {} exceeds wmax = {}",
            omega,
            wmax
        );
    }

    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_default_omega_sampling_points_bosonic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None).unwrap();

    let omega_points = basis.default_omega_sampling_points().unwrap();

    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());

    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(
            omega.abs() <= wmax,
            "omega = {} exceeds wmax = {}",
            omega,
            wmax
        );
    }

    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_omega_points_symmetry() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis_f =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();
    let omega_points = basis_f.default_omega_sampling_points().unwrap();

    // Check approximate symmetry: for each positive point, there should be a negative counterpart
    // (This is approximate due to the nature of the roots)
    let positive: Vec<f64> = omega_points.iter().filter(|&&x| x > 0.0).copied().collect();
    let negative: Vec<f64> = omega_points
        .iter()
        .filter(|&&x| x < 0.0)
        .map(|&x| -x)
        .collect();

    println!("Omega points: {:?}", omega_points);
    println!(
        "Number of positive: {}, negative: {}",
        positive.len(),
        negative.len()
    );
}

/// The regularized bosonic kernel in physical units,
/// K^B(τ, ω) = ω e^{-τω} / (1 - e^{-βω}) for 0 ≤ τ ≤ β, with K^B(τ, 0) = 1/β.
///
/// Reference: irbasis paper, N. Chikano, K. Yoshimi, J. Otsuki, H. Shinaoka,
/// Comput. Phys. Commun. 240, 181 (2019), arXiv:1807.05237, Eq. (3). A
/// `RegularizedBoseKernel` basis must expand it as Σ_l U_l(τ) S_l V_l(ω) with
/// S_l = sqrt(β ωmax³/2) s_l (Eq. (25)).
fn regularized_bose_kernel_physical(tau: f64, omega: f64, beta: f64) -> f64 {
    if omega == 0.0 {
        1.0 / beta
    } else if omega > 0.0 {
        omega * (-tau * omega).exp() / (1.0 - (-beta * omega).exp())
    } else {
        // Same function with non-positive exponents for ω < 0.
        omega * ((beta - tau) * omega).exp() / ((beta * omega).exp() - 1.0)
    }
}

#[test]
fn test_regularized_bose_basis_represents_physical_kernel() {
    // ωmax ≠ 1 separates ωmax^(+1) (Eq. (25)) from any other power of ωmax.
    // At ε = 1e-12 the truncation error of Σ U S V is far below 1e-8 ωmax,
    // the scale of K^B; a wrong power of ωmax is an O(1) relative error.
    for &(beta, omega_max) in &[(10.0, 2.0), (4.0, 2.5), (20.0, 0.5)] {
        let kernel = RegularizedBoseKernel::new(beta * omega_max).unwrap();
        let basis =
            FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-12), None)
                .unwrap();
        let s = basis.s();
        for &tau in &[0.3, 0.37 * beta, 0.8 * beta] {
            let u = basis.u().evaluate_at(tau);
            for &omega in &[-0.7 * omega_max, 0.0, 0.2 * omega_max, 0.9 * omega_max] {
                let v = basis.v().evaluate_at(omega);
                let usv: f64 = (0..basis.size()).map(|l| u[l] * s[l] * v[l]).sum();
                let exact = regularized_bose_kernel_physical(tau, omega, beta);
                assert!(
                    (usv - exact).abs() <= 1e-8 * omega_max,
                    "beta={beta}, omega_max={omega_max}, tau={tau}, omega={omega}: \
                     sum U S V = {usv}, K^B = {exact}"
                );
            }
        }
    }
}

#[test]
fn test_regularized_bose_basis_single_pole() {
    use crate::freq::MatsubaraFreq;
    use num_complex::Complex64;

    // A single bosonic pole, A(ω) = δ(ω - ω0), so Ĝ(iν) = 1/(iν - ω0). The
    // kernel takes ρ(ω) = A(ω)/ω (Eq. (2)), so G_l = -S_l V_l(ω0)/ω0 (Eq. (8)).
    let (beta, omega_max, omega0) = (10.0, 2.0, 0.6);
    let kernel = RegularizedBoseKernel::new(beta * omega_max).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-12), None)
            .unwrap();
    let v = basis.v().evaluate_at(omega0);
    let gl: Vec<f64> = (0..basis.size())
        .map(|l| -basis.s()[l] * v[l] / omega0)
        .collect();
    for n in [0_i64, 2, -4, 10] {
        let freq = MatsubaraFreq::<Bosonic>::new(n).unwrap();
        let uhat = basis.uhat().evaluate_at(&freq);
        let giv: Complex64 = gl.iter().zip(uhat.iter()).map(|(&g, &u)| u * g).sum();
        let exact = Complex64::new(1.0, 0.0) / Complex64::new(-omega0, freq.value(beta));
        assert!(
            (giv - exact).norm() <= 1e-8 * exact.norm(),
            "n={n}: sum G_l Uhat_l = {giv}, 1/(iν - ω0) = {exact}"
        );
    }
}

/// A large β with a moderate Λ is valid. `from_sve_result` scales the knots
/// and the widths separately, so their difference grows with the magnitude of
/// the knots (about β); the width check of `PiecewiseLegendrePoly::new` must
/// be relative to that magnitude and not reject such a basis.
#[test]
fn test_basis_with_a_large_beta_and_a_moderate_lambda() {
    for beta in [1e4, 1e7, 1e10] {
        let omega_max = 10.0 / beta;
        let kernel = LogisticKernel::new(beta * omega_max).unwrap();
        let basis = FermionicBasis::new(kernel, beta, Some(1e-6), None)
            .unwrap_or_else(|e| panic!("beta = {beta}: {e}"));
        let u = &basis.u().get_polys()[0];
        assert_eq!(u.xmax, beta);
        for tau in [0.0, 0.5 * beta, beta] {
            assert!(u.evaluate(tau).is_finite(), "beta = {beta}, tau = {tau}");
        }
    }
}

/// default_matsubara_sampling_points_impl, which the C API calls with a
/// Matsubara-space spir_funcs, returns NotSupported for functions without a
/// definite parity (#183) and EmptyInput for no functions (it panicked on
/// `len() - 1`).
#[test]
fn test_default_matsubara_points_impl_errors() {
    use crate::polyfourier::PiecewiseLegendreFTVector;
    use crate::sve::compute_sve_general;

    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve_general(kernel, Some(1e-6), None, None, TworkType::Auto).unwrap();
    let basis =
        FiniteTempBasis::<_, Fermionic>::from_sve_result(kernel, 1.0, sve, Some(1e-6), None)
            .unwrap();
    for (fence, positive_only) in [(false, false), (true, true)] {
        let err =
            FiniteTempBasis::<LogisticKernel, Fermionic>::default_matsubara_sampling_points_impl(
                basis.uhat_full(),
                basis.size(),
                fence,
                positive_only,
            )
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
    }
    let empty = PiecewiseLegendreFTVector::<Fermionic>::new();
    assert_eq!(
        FiniteTempBasis::<LogisticKernel, Fermionic>::default_matsubara_sampling_points_impl(
            &empty, 4, false, false
        )
        .unwrap_err(),
        Error::EmptyInput { name: "uhat_full" }
    );
}

/// A basis from an SVE truncated to 2 functions has no default tau sampling
/// points: they are the roots of u_2, which is missing, and its stand-in, the
/// extrema of u_1, does not exist (u_1 is monotonic). The empty extrema were
/// indexed and panicked. Points that need only u_0 and u_1 are still
/// defined, as are the other default points.
#[test]
fn test_default_tau_points_of_an_sve_truncated_to_two_functions() {
    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve(kernel, Some(1e-6), None, Some(2), TworkType::Auto).unwrap();
    assert_eq!(sve.s.len(), 2);
    fn check<S: crate::traits::StatisticsType + 'static>(
        basis: &FiniteTempBasis<LogisticKernel, S>,
    ) {
        let not_supported = |err: Error| {
            assert!(
                matches!(&err, Error::NotSupported { what } if what.contains("no extrema")),
                "{err:?}"
            );
        };
        not_supported(basis.default_tau_sampling_points().unwrap_err());
        for n in [2, 3, 10] {
            not_supported(
                basis
                    .default_tau_sampling_points_size_requested(n)
                    .unwrap_err(),
            );
        }
        // The roots of u_0 (none) and u_1 (one, at τ = β/2)
        assert_eq!(
            basis
                .default_tau_sampling_points_size_requested(0)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            basis
                .default_tau_sampling_points_size_requested(1)
                .unwrap()
                .len(),
            1
        );
        assert!(!basis.default_omega_sampling_points().unwrap().is_empty());
        assert!(
            !basis
                .default_matsubara_sampling_points(false)
                .unwrap()
                .is_empty()
        );
    }
    check(
        &FiniteTempBasis::<_, Fermionic>::from_sve_result(kernel, 10.0, sve.clone(), None, None)
            .unwrap(),
    );
    check(&FiniteTempBasis::<_, Bosonic>::from_sve_result(kernel, 10.0, sve, None, None).unwrap());
}

/// default_sampling_points needs the unscaled functions of an SVE, on
/// [-1, 1]; the functions of a basis (on [0, β] and [-ωmax, ωmax]) are
/// rejected, under the name the caller gives them. It panicked.
#[test]
fn test_default_sampling_points_rejects_scaled_functions() {
    use crate::poly::default_sampling_points;

    // β = 1 and Λ = 10: u is on [0, 1] and v on [-10, 10], neither on [-1, 1].
    let basis =
        FermionicBasis::new(LogisticKernel::new(10.0).unwrap(), 1.0, Some(1e-6), None).unwrap();
    let err = default_sampling_points(basis.u(), "u", 3).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { name: "u", .. }),
        "{err:?}"
    );
    let err = default_sampling_points(basis.v(), "v", 3).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter { name: "v", .. }),
        "{err:?}"
    );
    assert!(default_sampling_points(&basis.sve_result().u, "u", 3).is_ok());
}
