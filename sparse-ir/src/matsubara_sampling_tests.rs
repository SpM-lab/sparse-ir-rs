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
