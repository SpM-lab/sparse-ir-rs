// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]
use crate::freq::MatsubaraFreq;
use crate::matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
use crate::test_utils::{ErrorNorm, generate_test_data_tau_and_matsubara};
use crate::traits::{Bosonic, Fermionic, StatisticsType};
use crate::{FiniteTempBasis, LogisticKernel, RegularizedBoseKernel};
use num_complex::Complex;

/// Test MatsubaraSampling (symmetric mode, complex coefficients) roundtrip
#[test]
fn test_matsubara_sampling_roundtrip_fermionic() {
    test_matsubara_sampling_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_roundtrip_bosonic() {
    test_matsubara_sampling_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_roundtrip_generic<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);

    // Create symmetric Matsubara sampling points (positive and negative)
    let sampling_points = basis.default_matsubara_sampling_points(false);

    // Create sampling
    let sampling = MatsubaraSampling::with_sampling_points(&basis, sampling_points.clone());

    // Generate test data (we only need Matsubara values)
    let (_coeffs_random, _gtau_values, giwn_values) =
        generate_test_data_tau_and_matsubara::<Complex<f64>, S, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &sampling_points,
            12345,
        );

    // Fit to get coefficients
    let coeffs_fitted = sampling.fit(&giwn_values);

    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);

    // Check roundtrip accuracy
    let max_error = giwn_values
        .iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);

    println!(
        "MatsubaraSampling {:?} roundtrip max error: {}",
        S::STATISTICS,
        max_error
    );
    assert!(max_error < 1e-7, "Roundtrip error too large: {}", max_error);
}

/// Test MatsubaraSamplingPositiveOnly (positive frequencies only, real coefficients) roundtrip
#[test]
fn test_matsubara_sampling_positive_only_roundtrip_fermionic() {
    test_matsubara_sampling_positive_only_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_positive_only_roundtrip_bosonic() {
    test_matsubara_sampling_positive_only_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_positive_only_roundtrip_generic<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);

    // Use positive-only sampling points
    let sampling_points = basis.default_matsubara_sampling_points(true);
    let _n_matsubara = sampling_points.len();

    // Create sampling
    let sampling =
        MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());

    // Generate test data (we only need Matsubara values)
    let (_coeffs_random, _gtau_values, giwn_values) =
        generate_test_data_tau_and_matsubara::<f64, S, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &sampling_points,
            12345,
        );

    // Fit to get real coefficients
    let coeffs_fitted = sampling.fit(&giwn_values);

    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);

    // Check roundtrip accuracy
    let max_error = giwn_values
        .iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);

    println!(
        "MatsubaraSamplingPositiveOnly {:?} roundtrip max error: {}",
        S::STATISTICS,
        max_error
    );
    assert!(max_error < 1e-7, "Roundtrip error too large: {}", max_error);
}

/// Test that basis sizes are consistent
#[test]
fn test_matsubara_sampling_dimensions() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);

    let sampling_points = basis.default_matsubara_sampling_points(true);

    let sampling =
        MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());

    assert_eq!(sampling.basis_size(), basis.size());
    assert_eq!(sampling.n_sampling_points(), sampling_points.len());
}

/// Test MatsubaraSampling evaluate_nd/fit_nd roundtrip
#[test]
fn test_matsubara_sampling_nd_roundtrip_fermionic() {
    test_matsubara_sampling_nd_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_nd_roundtrip_bosonic() {
    test_matsubara_sampling_nd_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_nd_roundtrip_generic<S: StatisticsType + 'static>() {
    use crate::test_utils::generate_nd_test_data;

    use num_complex::Complex;

    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);

    let sampling_points = basis.default_matsubara_sampling_points(false); // Symmetric (positive and negative)

    let sampling = MatsubaraSampling::with_sampling_points(&basis, sampling_points.clone());

    let n_k = 4;
    let n_omega = 5;

    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = generate_nd_test_data::<Complex<f64>, _, _>(
            &basis,
            &[],
            &sampling_points,
            42 + dim as u64,
            &[n_k, n_omega],
        );

        // Move to target dimension
        let coeffs_dim = crate::test_utils::movedim(&coeffs_0, 0, dim);

        // Evaluate and fit along target dimension
        let values_dim = sampling.evaluate_nd(None, &coeffs_dim, dim);
        let coeffs_fitted_dim = sampling.fit_nd(None, &values_dim, dim);

        // Move back to dim=0 for comparison
        let coeffs_fitted_0 = crate::test_utils::movedim(&coeffs_fitted_dim, dim, 0);

        // Check roundtrip
        let max_error = coeffs_0
            .iter()
            .zip(coeffs_fitted_0.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0, f64::max);

        println!(
            "MatsubaraSampling {:?} dim={} roundtrip error: {}",
            S::STATISTICS,
            dim,
            max_error
        );
        assert!(
            max_error < 1e-10,
            "ND roundtrip (dim={}) error too large: {}",
            dim,
            max_error
        );
    }
}

/// Test MatsubaraSamplingPositiveOnly evaluate_nd/fit_nd roundtrip
#[test]
fn test_matsubara_sampling_positive_only_nd_roundtrip_fermionic() {
    test_matsubara_sampling_positive_only_nd_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_positive_only_nd_roundtrip_bosonic() {
    test_matsubara_sampling_positive_only_nd_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_positive_only_nd_roundtrip_generic<S: StatisticsType + 'static>() {
    use crate::test_utils::generate_nd_test_data;

    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);

    // Use positive-only sampling points
    let sampling_points = basis.default_matsubara_sampling_points(true);

    let sampling =
        MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());

    let n_k = 4;
    let n_omega = 5;

    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = generate_nd_test_data::<f64, _, _>(
            &basis,
            &[],
            &sampling_points,
            42 + dim as u64,
            &[n_k, n_omega],
        );

        // Move to target dimension
        let coeffs_dim = crate::test_utils::movedim(&coeffs_0, 0, dim);

        // Evaluate and fit along target dimension
        let values_dim = sampling.evaluate_nd(None, &coeffs_dim, dim);
        let coeffs_fitted_dim = sampling.fit_nd(None, &values_dim, dim);

        // Move back to dim=0 for comparison
        let coeffs_fitted_0 = crate::test_utils::movedim(&coeffs_fitted_dim, dim, 0);

        // Check roundtrip
        let max_error = coeffs_0
            .iter()
            .zip(coeffs_fitted_0.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);

        println!(
            "MatsubaraSamplingPositiveOnly {:?} dim={} roundtrip error: {}",
            S::STATISTICS,
            dim,
            max_error
        );
        assert!(
            max_error < 1e-7,
            "ND roundtrip (dim={}) error too large: {}",
            dim,
            max_error
        );
    }
}

// ====================
// RegularizedBoseKernel MatsubaraSampling Tests
// ====================

/// Generic test for RegularizedBoseKernel MatsubaraSampling roundtrip
fn test_regularized_bose_matsubara_sampling_roundtrip_generic() {
    let beta = 10.0;
    let wmax = 1.0; // Use smaller wmax for better numerics
    let epsilon = 1e-4; // Looser tolerance for RegularizedBoseKernel

    // Create basis
    let kernel = RegularizedBoseKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, Some(epsilon), None);

    let basis_size = basis.size();

    // Create custom Matsubara sampling points (ensure >= basis_size points)
    // Use simple uniform distribution: n = 0, 1, 2, ..., basis_size (for positive-only)
    let sampling_points: Vec<MatsubaraFreq<Bosonic>> = (0..basis_size + 2)
        .map(|n| MatsubaraFreq::new((2 * n) as i64).unwrap()) // Bosonic: n must be even
        .collect();

    // Also create negative frequencies for symmetric sampling
    let mut symmetric_points = sampling_points.clone();
    for n in 1..basis_size + 2 {
        symmetric_points.push(MatsubaraFreq::new(-((2 * n) as i64)).unwrap());
    }

    println!("\n=== RegularizedBose MatsubaraSampling Test ===");
    println!("Basis size: {}", basis_size);
    println!("Sampling points (symmetric): {}", symmetric_points.len());

    // Create sampling
    let sampling = MatsubaraSampling::with_sampling_points(&basis, symmetric_points.clone());

    // Generate test data (we only need Matsubara values)
    let (_coeffs_random, _gtau_values, giwn_values) =
        generate_test_data_tau_and_matsubara::<Complex<f64>, Bosonic, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &symmetric_points,
            12345,
        );

    // Fit to get coefficients
    let coeffs_fitted = sampling.fit(&giwn_values);

    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);

    // Check roundtrip accuracy
    let max_error = giwn_values
        .iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);

    println!("MatsubaraSampling roundtrip max error: {:.2e}", max_error);
    // RegularizedBoseKernel has lower numerical precision due to y=0 singularity
    // Looser tolerance required
    assert!(
        max_error < 2.0,
        "RegularizedBose Matsubara roundtrip error too large: {}",
        max_error
    );
}

/// Generic test for RegularizedBoseKernel MatsubaraSamplingPositiveOnly roundtrip
fn test_regularized_bose_matsubara_sampling_positive_only_roundtrip_generic() {
    let beta = 10.0;
    let wmax = 1.0; // Use smaller wmax for better numerics
    let epsilon = 1e-4; // Looser tolerance for RegularizedBoseKernel

    // Create basis
    let kernel = RegularizedBoseKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, Some(epsilon), None);

    let basis_size = basis.size();

    // Create custom positive-only Matsubara sampling points (ensure >= basis_size points)
    let sampling_points: Vec<MatsubaraFreq<Bosonic>> = (0..basis_size + 2)
        .map(|n| MatsubaraFreq::new((2 * n) as i64).unwrap()) // Bosonic: n must be even
        .collect();

    println!("\n=== RegularizedBose MatsubaraSamplingPositiveOnly Test ===");
    println!("Basis size: {}", basis_size);
    println!("Sampling points (positive-only): {}", sampling_points.len());

    // Create sampling
    let sampling =
        MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());

    // Generate test data (real coefficients for positive-only)
    let (_coeffs_random, _gtau_values, giwn_values) =
        generate_test_data_tau_and_matsubara::<Complex<f64>, Bosonic, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &sampling_points,
            54321,
        );

    // Fit to get coefficients (should be real)
    let coeffs_fitted = sampling.fit(&giwn_values);

    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);

    // Check roundtrip accuracy
    let max_error = giwn_values
        .iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);

    println!(
        "MatsubaraSamplingPositiveOnly roundtrip max error: {:.2e}",
        max_error
    );
    // RegularizedBoseKernel has lower numerical precision due to y=0 singularity
    // Looser tolerance required
    assert!(
        max_error < 1e-2,
        "RegularizedBose Matsubara positive-only roundtrip error too large: {}",
        max_error
    );
}

#[test]
fn test_regularized_bose_matsubara_sampling_roundtrip() {
    test_regularized_bose_matsubara_sampling_roundtrip_generic();
}

#[test]
fn test_regularized_bose_matsubara_sampling_positive_only_roundtrip() {
    test_regularized_bose_matsubara_sampling_positive_only_roundtrip_generic();
}

// ============================================================================
// In-place method tests
// ============================================================================

use mdarray::{Shape, Tensor};

/// Test MatsubaraSampling::evaluate_nd_to matches evaluate_nd
#[test]
fn test_matsubara_sampling_evaluate_nd_to_matches() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let sampling = MatsubaraSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients (complex)
    let coeffs =
        Tensor::<Complex<f64>, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                (idx[2] as f64) * 0.3,
            )
        });

    // Test for dim = 0
    let expected = sampling.evaluate_nd(None, &coeffs, 0);

    let mut actual = Tensor::<Complex<f64>, crate::DynRank>::from_elem(
        &[n_points, n_k, n_omega][..],
        Complex::new(0.0, 0.0),
    );
    sampling.evaluate_nd_to(None, &coeffs, 0, &mut actual);

    // Compare
    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

/// Test MatsubaraSampling::fit_nd_to matches fit_nd
#[test]
fn test_matsubara_sampling_fit_nd_to_matches() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let sampling = MatsubaraSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test values (complex)
    let values =
        Tensor::<Complex<f64>, crate::DynRank>::from_fn(&[n_points, n_k, n_omega][..], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                (idx[2] as f64) * 0.2,
            )
        });

    // Test for dim = 0
    let expected = sampling.fit_nd(None, &values, 0);

    let mut actual = Tensor::<Complex<f64>, crate::DynRank>::from_elem(
        &[basis_size, n_k, n_omega][..],
        Complex::new(0.0, 0.0),
    );
    sampling.fit_nd_to(None, &values, 0, &mut actual);

    // Compare
    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..basis_size {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

/// Test MatsubaraSamplingPositiveOnly::evaluate_nd_to matches evaluate_nd
#[test]
fn test_matsubara_sampling_positive_only_evaluate_nd_to_matches() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let sampling = MatsubaraSamplingPositiveOnly::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients (real)
    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    // Test for dim = 0
    let expected = sampling.evaluate_nd(None, &coeffs, 0);

    let mut actual = Tensor::<Complex<f64>, crate::DynRank>::from_elem(
        &[n_points, n_k, n_omega][..],
        Complex::new(0.0, 0.0),
    );
    sampling.evaluate_nd_to(None, &coeffs, 0, &mut actual);

    // Compare
    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

/// Test MatsubaraSamplingPositiveOnly::fit_nd_to matches fit_nd
#[test]
fn test_matsubara_sampling_positive_only_fit_nd_to_matches() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let sampling = MatsubaraSamplingPositiveOnly::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test values (complex)
    let values =
        Tensor::<Complex<f64>, crate::DynRank>::from_fn(&[n_points, n_k, n_omega][..], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                (idx[2] as f64) * 0.2,
            )
        });

    // Test for dim = 0
    let expected = sampling.fit_nd(None, &values, 0);

    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[basis_size, n_k, n_omega][..], 0.0);
    sampling.fit_nd_to(None, &values, 0, &mut actual);

    // Compare
    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..basis_size {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={}, actual={}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

/// Test MatsubaraSampling creation with specific parameters matching debug.rs
///
/// This test verifies that MatsubaraSampling can be created correctly with
/// the same parameters used in the debug example, and that the resulting
/// basis size and sampling point count match expected values (consistent with C++).
#[test]
fn test_matsubara_sampling_debug_parameters() {
    let t = 0.1;
    let wmax = 1.0;

    let beta = 1.0 / t;
    // construction of the Kernel K

    // Fermionic Basis
    // Step 1: Create kernel
    let lambda_ = beta * wmax;
    let kernel = LogisticKernel::new(lambda_);

    // Step 2: Compute SVE and create basis
    // FiniteTempBasis::new automatically computes SVE internally
    let eps = f64::EPSILON;
    let basisf: FiniteTempBasis<LogisticKernel, Fermionic> =
        FiniteTempBasis::new(kernel, beta, Some(eps), None);

    // Step 3: Create Matsubara sampling
    let matsf = MatsubaraSampling::new(&basisf);

    // Verify expected values (matching C++ implementation)
    // Basis size should be 19 for these parameters
    assert_eq!(
        basisf.size(),
        19,
        "Basis size should be 19 for T=0.1, wmax=1.0"
    );

    // Number of Matsubara sampling points should be 20
    // (l_requested = 20 for Fermionic with L=19)
    assert_eq!(
        matsf.sampling_points().len(),
        20,
        "Number of Matsubara sampling points should be 20 for Fermionic basis with L=19"
    );

    // Verify sampling points are sorted
    let sampling_points = matsf.sampling_points();
    for i in 1..sampling_points.len() {
        assert!(
            sampling_points[i - 1] <= sampling_points[i],
            "Sampling points should be sorted"
        );
    }

    // Verify basis size consistency
    assert_eq!(
        matsf.basis_size(),
        basisf.size(),
        "MatsubaraSampling basis_size() should match basis.size()"
    );

    // Verify matrix dimensions
    let matrix = matsf.matrix();
    assert_eq!(
        matrix.shape().0,
        matsf.n_sampling_points(),
        "Matrix rows should match number of sampling points"
    );
    assert_eq!(
        matrix.shape().1,
        basisf.size(),
        "Matrix columns should match basis size"
    );
}

// ============================================================================
// condition_number (SpM-lab/sparse-ir-rs#270)
// ============================================================================

/// `A[i, l] = uhat_l(iν_i)`, built from the basis functions independently of
/// any sampling object
fn uhat_matrix<S: StatisticsType + 'static>(
    basis: &FiniteTempBasis<LogisticKernel, S>,
    points: &[MatsubaraFreq<S>],
) -> mdarray::DTensor<Complex<f64>, 2> {
    let uhat = basis.uhat();
    mdarray::DTensor::<Complex<f64>, 2>::from_fn([points.len(), basis.size()], |idx| {
        uhat[idx[1]].evaluate(&points[idx[0]])
    })
}

/// Positive-only sampling reports the condition number of the stacked real
/// matrix `[Re A; Im A]` that `fit` solves, not that of the complex matrix `A`
fn check_positive_only_condition_number<S: StatisticsType + 'static>(
    beta: f64,
    wmax: f64,
    epsilon: f64,
) {
    use crate::test_utils::{assert_condition_number_close, oracle_condition_number, stack_re_im};

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    let points = basis.default_matsubara_sampling_points(true);
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, points.clone());

    let oracle = oracle_condition_number(&stack_re_im(&uhat_matrix(&basis, &points)));
    let label = format!(
        "positive-only {:?}, beta={beta}, wmax={wmax}, eps={epsilon:e}, L={}, n={}",
        S::STATISTICS,
        basis.size(),
        points.len()
    );
    assert_condition_number_close(&label, sampling.condition_number(), oracle);
}

/// Full-set sampling reports the condition number of the complex matrix `A`
fn check_full_condition_number<S: StatisticsType + 'static>(beta: f64, wmax: f64, epsilon: f64) {
    use crate::test_utils::{assert_condition_number_close, oracle_condition_number, realify};

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    let points = basis.default_matsubara_sampling_points(false);
    let sampling = MatsubaraSampling::with_sampling_points(&basis, points.clone());

    let oracle = oracle_condition_number(&realify(&uhat_matrix(&basis, &points)));
    let label = format!(
        "full-set {:?}, beta={beta}, wmax={wmax}, eps={epsilon:e}, L={}, n={}",
        S::STATISTICS,
        basis.size(),
        points.len()
    );
    assert_condition_number_close(&label, sampling.condition_number(), oracle);
}

#[test]
fn test_positive_only_condition_number_fermionic() {
    check_positive_only_condition_number::<Fermionic>(10.0, 1.0, 1e-6);
}

#[test]
fn test_positive_only_condition_number_bosonic() {
    check_positive_only_condition_number::<Bosonic>(10.0, 1.0, 1e-6);
    check_positive_only_condition_number::<Bosonic>(1000.0, 1.0, 1e-10);
}

#[test]
fn test_full_condition_number_fermionic() {
    check_full_condition_number::<Fermionic>(10.0, 1.0, 1e-6);
}

#[test]
fn test_full_condition_number_bosonic() {
    check_full_condition_number::<Bosonic>(10.0, 1.0, 1e-6);
    check_full_condition_number::<Bosonic>(1000.0, 1.0, 1e-10);
}

/// `from_matrix` samplings follow the same rule. The matrix is wide (5 points,
/// 8 coefficients, so `[Re A; Im A]` is 10 × 8), as for positive-only
/// sampling points, and chosen so that `cond(A)` over its 5 singular values
/// and `cond([Re A; Im A])` differ.
#[test]
fn test_positive_only_condition_number_from_matrix() {
    use crate::test_utils::{
        assert_condition_number_close, oracle_condition_number, realify, stack_re_im,
    };

    let (n, l) = (5, 8);
    let a = mdarray::DTensor::<Complex<f64>, 2>::from_fn([n, l], |idx| {
        let x = (idx[0] as f64 + 1.0) / n as f64;
        let j = idx[1] as i32;
        Complex::new(x.powi(j), (x * (j as f64 + 1.0)).sin())
    });
    let points: Vec<MatsubaraFreq<Bosonic>> = (0..n as i64)
        .map(|k| MatsubaraFreq::new(2 * k).unwrap())
        .collect();
    let sampling = MatsubaraSamplingPositiveOnly::from_matrix(points, a.clone());

    let oracle = oracle_condition_number(&stack_re_im(&a));
    let cond_complex = oracle_condition_number(&realify(&a));
    assert!(
        (oracle - cond_complex).abs() > 1e-3 * oracle,
        "test matrix must separate cond([Re A; Im A]) = {oracle:.6e} from cond(A) = {cond_complex:.6e}"
    );
    assert_condition_number_close(
        "from_matrix positive-only",
        sampling.condition_number(),
        oracle,
    );
}

/// Batches with a zero extent, as (batch extents, target axis)
fn empty_batches() -> Vec<(Vec<usize>, usize)> {
    vec![
        (vec![0], 1),
        (vec![0], 0),
        (vec![0, 3], 1),
        (vec![2, 0], 2),
        (vec![0, 3], 0),
    ]
}

fn with_target(batch: &[usize], dim: usize, n: usize) -> Vec<usize> {
    let mut dims = batch.to_vec();
    dims.insert(dim, n);
    dims
}

/// Evaluating or fitting an empty batch gives an empty result of the right
/// shape. Before the fix these segfaulted in `movedim` (mdarray#21) when the
/// zero extent ended up before the last axis of a permuted view.
#[test]
fn test_matsubara_nd_with_empty_batch() {
    use mdarray::{DynRank, Tensor};

    let basis =
        FiniteTempBasis::<_, Fermionic>::new(LogisticKernel::new(10.0), 1.0, Some(1e-6), None);
    let sampling = MatsubaraSampling::new(&basis);
    let positive = MatsubaraSamplingPositiveOnly::new(&basis);
    let l = sampling.basis_size();

    for (batch, dim) in empty_batches() {
        let dims_l = with_target(&batch, dim, l);
        let coeffs = Tensor::<f64, DynRank>::zeros(&dims_l[..]);
        let coeffs_z = Tensor::<Complex<f64>, DynRank>::zeros(&dims_l[..]);

        for (s_np, values) in [
            (
                sampling.n_sampling_points(),
                sampling.evaluate_nd::<f64>(None, &coeffs, dim),
            ),
            (
                sampling.n_sampling_points(),
                sampling.evaluate_nd::<Complex<f64>>(None, &coeffs_z, dim),
            ),
            (
                sampling.n_sampling_points(),
                sampling.evaluate_nd_real(None, &coeffs, dim),
            ),
            (
                positive.n_sampling_points(),
                positive.evaluate_nd(None, &coeffs, dim),
            ),
        ] {
            assert_eq!(values.shape().dims(), &with_target(&batch, dim, s_np)[..]);
        }

        let dims_np = with_target(&batch, dim, sampling.n_sampling_points());
        let values = Tensor::<Complex<f64>, DynRank>::zeros(&dims_np[..]);
        assert_eq!(
            sampling.fit_nd(None, &values, dim).shape().dims(),
            &dims_l[..]
        );
        assert_eq!(
            sampling.fit_nd_real(None, &values, dim).shape().dims(),
            &dims_l[..]
        );
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&dims_np[..]);
        sampling.evaluate_nd_to::<f64>(None, &coeffs, dim, &mut out);
        let mut out_l = Tensor::<Complex<f64>, DynRank>::zeros(&dims_l[..]);
        sampling.fit_nd_to(None, &values, dim, &mut out_l);

        let dims_np = with_target(&batch, dim, positive.n_sampling_points());
        let values = Tensor::<Complex<f64>, DynRank>::zeros(&dims_np[..]);
        assert_eq!(
            positive.fit_nd(None, &values, dim).shape().dims(),
            &dims_l[..]
        );
    }
}

/// The InplaceFitter methods of both Matsubara samplings accept an empty
/// batch. Before the fix, the zd fit of MatsubaraSampling along a middle
/// axis iterated a permuted view of the empty output, which mdarray 0.7.2
/// does out of bounds (mdarray#21), and wrote past its end.
#[test]
fn test_matsubara_inplace_fitter_with_empty_batch() {
    use crate::fitters::InplaceFitter;
    use mdarray::{DynRank, Tensor};

    let basis =
        FiniteTempBasis::<_, Fermionic>::new(LogisticKernel::new(10.0), 1.0, Some(1e-6), None);
    let full = MatsubaraSampling::new(&basis);
    let positive = MatsubaraSamplingPositiveOnly::new(&basis);

    fn check<F: InplaceFitter>(f: &F, batch: &[usize], dim: usize) {
        let (l, np) = (f.basis_size(), f.n_points());
        let (dims_l, dims_np) = (with_target(batch, dim, l), with_target(batch, dim, np));
        let coeffs = Tensor::<f64, DynRank>::zeros(&dims_l[..]);
        let coeffs_z = Tensor::<Complex<f64>, DynRank>::zeros(&dims_l[..]);
        let values_z = Tensor::<Complex<f64>, DynRank>::zeros(&dims_np[..]);
        let mut out_np = Tensor::<Complex<f64>, DynRank>::zeros(&dims_np[..]);
        let mut out_l = Tensor::<f64, DynRank>::zeros(&dims_l[..]);
        let mut out_l_z = Tensor::<Complex<f64>, DynRank>::zeros(&dims_l[..]);
        assert!(f.evaluate_nd_dz_to(None, &coeffs, dim, &mut out_np.expr_mut()));
        assert!(f.evaluate_nd_zz_to(None, &coeffs_z, dim, &mut out_np.expr_mut()));
        assert!(f.fit_nd_zd_to(None, &values_z, dim, &mut out_l.expr_mut()));
        assert!(f.fit_nd_zz_to(None, &values_z, dim, &mut out_l_z.expr_mut()));
    }

    for (batch, dim) in empty_batches() {
        check(&full, &batch, dim);
        check(&positive, &batch, dim);
    }
}

#[test]
#[should_panic(expected = "No sampling points given")]
fn test_matsubara_sampling_rejects_empty_sampling_points() {
    let basis =
        FiniteTempBasis::<_, Fermionic>::new(LogisticKernel::new(10.0), 1.0, Some(1e-6), None);
    MatsubaraSampling::with_sampling_points(&basis, vec![]);
}

#[test]
#[should_panic(expected = "No sampling points given")]
fn test_matsubara_sampling_positive_only_rejects_empty_sampling_points() {
    let basis =
        FiniteTempBasis::<_, Fermionic>::new(LogisticKernel::new(10.0), 1.0, Some(1e-6), None);
    MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, vec![]);
}

#[test]
#[should_panic(expected = "Matrix must have at least one column")]
fn test_matsubara_from_matrix_rejects_zero_columns() {
    let matrix = mdarray::DTensor::<Complex<f64>, 2>::zeros([1, 0]);
    MatsubaraSampling::<Fermionic>::from_matrix(vec![MatsubaraFreq::new(1).unwrap()], matrix);
}

#[test]
#[should_panic(expected = "Matrix must have at least one column")]
fn test_matsubara_positive_only_from_matrix_rejects_zero_columns() {
    let matrix = mdarray::DTensor::<Complex<f64>, 2>::zeros([1, 0]);
    MatsubaraSamplingPositiveOnly::<Fermionic>::from_matrix(
        vec![MatsubaraFreq::new(1).unwrap()],
        matrix,
    );
}
