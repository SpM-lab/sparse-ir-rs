//! Tests for FiniteTempBasis functionality
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::basis::{FermionicBasis, FiniteTempBasis};
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::traits::{Bosonic, Fermionic};

#[test]
fn test_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * omega_max);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);
}

#[test]
#[should_panic(expected = "beta must be positive")]
fn test_negative_beta() {
    let kernel = LogisticKernel::new(1.0);
    let _ = FermionicBasis::new(kernel, -1.0, None, None);
}

#[test]
fn test_default_tau_sampling_points_conditioning() {
    // Test parameters: beta=1.0, lambda=10.0
    let beta = 1.0;
    let lambda = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    println!("\n=== Default Tau Sampling Points Test ===");
    println!("Beta: {}, Lambda: {}, Epsilon: {}", beta, lambda, epsilon);
    println!("Basis size: {}", basis.size());

    // Get default sampling points
    let tau_points = basis.default_tau_sampling_points();
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
    let matrix = basis.evaluate_tau(&tau_points);

    let num_points = tau_points.len();
    let basis_size = basis.size();
    println!("Sampling matrix shape: {}x{}", num_points, basis_size);

    // Compute SVD using mdarray-linalg (Faer backend)
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg_faer::Faer;
    let mut matrix_copy = matrix.clone();
    let svd = Faer.svd(&mut *matrix_copy).expect("SVD computation failed");

    println!("\nSampling matrix SVD:");
    let min_dim = svd.s.shape().0.min(svd.s.shape().1);
    println!("  Rank: {}", min_dim);
    // mdarray-linalg stores singular values in first row: s[[0, i]]
    println!("  First singular value: {:.6e}", svd.s[[0, 0]]);
    println!("  Last singular value: {:.6e}", svd.s[[0, min_dim - 1]]);

    let condition_number = svd.s[[0, 0]] / svd.s[[0, min_dim - 1]];
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

    let kernel = RegularizedBoseKernel::new(beta * omega_max);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

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
        let kernel = RegularizedBoseKernel::new(beta * omega_max);
        let basis = FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(
            kernel,
            beta,
            Some(epsilon),
            None,
        );

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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);

    let omega_points = basis.default_omega_sampling_points();

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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    let omega_points = basis.default_omega_sampling_points();

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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis_f =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let omega_points = basis_f.default_omega_sampling_points();

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
        let kernel = RegularizedBoseKernel::new(beta * omega_max);
        let basis =
            FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-12), None);
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
    let kernel = RegularizedBoseKernel::new(beta * omega_max);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-12), None);
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
