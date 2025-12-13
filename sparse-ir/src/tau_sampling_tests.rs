use crate::basis::FiniteTempBasis;
use crate::kernel::{CentrosymmKernel, KernelProperties};
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::sampling::TauSampling;
use crate::traits::{Bosonic, Fermionic, StatisticsType};
use num_complex::Complex;

use crate::test_utils::{ConvertFromReal, ErrorNorm, RandomGenerate, movedim};

/// Generic test for evaluate_nd/fit_nd roundtrip (generic over element type and statistics)
fn test_evaluate_nd_roundtrip<T, S>()
where
    T: RandomGenerate
        + num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + ErrorNorm
        + 'static
        + ConvertFromReal
        + std::ops::Mul<f64, Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
    S: StatisticsType + 'static,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let n_k = 5;
    let n_omega = 7;

    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate random test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = crate::test_utils::generate_nd_test_data::<T, _, _>(
            &basis,
            sampling.sampling_points(),
            &[],
            42 + dim as u64,
            &[n_k, n_omega],
        );

        // Move to target dimension
        // coeffs: [basis_size, n_k, n_omega] → move axis 0 to position dim
        let coeffs_dim = movedim(&coeffs_0, 0, dim);

        // Evaluate along target dimension (using generic API)
        let evaluated_values = sampling.evaluate_nd::<T>(None, &coeffs_dim, dim);

        // Fit back along target dimension (using generic API)
        let fitted_coeffs_dim = sampling.fit_nd::<T>(None, &evaluated_values, dim);

        // Move back to dim=0 for comparison
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();

        // Check roundtrip (compare in dim=0 format)
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    // ErrorNorm returns f64 for both f64 and Complex<f64>
                    let abs_error = (orig - fitted).error_norm();

                    assert!(
                        abs_error < 1e-10,
                        "ND roundtrip (dim={}) error at ({},{},{}): error={}",
                        dim,
                        l,
                        k,
                        omega,
                        abs_error
                    );
                }
            }
        }
    }
}

#[test]
fn test_evaluate_nd_fermionic_real() {
    test_evaluate_nd_roundtrip::<f64, Fermionic>();
}

#[test]
fn test_evaluate_nd_fermionic_complex() {
    test_evaluate_nd_roundtrip::<Complex<f64>, Fermionic>();
}

#[test]
fn test_evaluate_nd_bosonic_real() {
    test_evaluate_nd_roundtrip::<f64, Bosonic>();
}

#[test]
fn test_evaluate_nd_bosonic_complex() {
    test_evaluate_nd_roundtrip::<Complex<f64>, Bosonic>();
}

// ====================
// RegularizedBoseKernel TauSampling Tests
// ====================

/// Generic test for RegularizedBoseKernel evaluate_nd/fit_nd roundtrip
fn test_regularized_bose_evaluate_nd_roundtrip<T>()
where
    T: RandomGenerate
        + num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + ErrorNorm
        + 'static
        + ConvertFromReal
        + std::ops::Mul<f64, Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
{
    let beta = 10.0;
    let wmax = 1.0; // Use smaller wmax for better numerics
    let epsilon = Some(1e-4); // Looser tolerance for RegularizedBoseKernel

    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let n_k = 5;
    let n_omega = 7;

    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate random test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = crate::test_utils::generate_nd_test_data::<T, _, _>(
            &basis,
            sampling.sampling_points(),
            &[],
            42 + dim as u64,
            &[n_k, n_omega],
        );

        // Move to target dimension
        let coeffs_dim = movedim(&coeffs_0, 0, dim);

        // Evaluate along target dimension
        let evaluated_values = sampling.evaluate_nd::<T>(None, &coeffs_dim, dim);

        // Fit back along target dimension
        let fitted_coeffs_dim = sampling.fit_nd::<T>(None, &evaluated_values, dim);

        // Move back to dim=0 for comparison
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();

        // Check roundtrip (compare in dim=0 format)
        let mut max_error = 0.0;
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    let abs_error = (orig - fitted).error_norm();
                    if abs_error > max_error {
                        max_error = abs_error;
                    }
                }
            }
        }

        println!(
            "RegularizedBose TauSampling {} ND roundtrip (dim={}): max_error = {:.2e}",
            std::any::type_name::<T>(),
            dim,
            max_error
        );
        assert!(
            max_error < 1e-7,
            "RegularizedBose ND roundtrip (dim={}) error too large: {}",
            dim,
            max_error
        );
    }
}

#[test]
fn test_regularized_bose_evaluate_nd_real() {
    test_regularized_bose_evaluate_nd_roundtrip::<f64>();
}

#[test]
fn test_regularized_bose_evaluate_nd_complex() {
    test_regularized_bose_evaluate_nd_roundtrip::<Complex<f64>>();
}

/// Test that evaluate_nd_to produces identical results to evaluate_nd
fn test_evaluate_nd_to_matches<T, S>()
where
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + 'static
        + std::fmt::Debug
        + ErrorNorm,
    S: StatisticsType + 'static,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients
    let coeffs = Tensor::<T, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
        <T as From<f64>>::from((idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3))
    });

    // Test for dim = 0
    let expected = sampling.evaluate_nd(None, &coeffs, 0);

    let mut actual = Tensor::<T, crate::DynRank>::from_elem(&[n_points, n_k, n_omega][..], <T as From<f64>>::from(0.0));
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
                let diff = (e - a).error_norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i, j, k, e, a
                );
            }
        }
    }
}

#[test]
fn test_evaluate_nd_to_matches_fermionic_real() {
    test_evaluate_nd_to_matches::<f64, Fermionic>();
}

#[test]
fn test_evaluate_nd_to_matches_fermionic_complex() {
    test_evaluate_nd_to_matches::<Complex<f64>, Fermionic>();
}

/// Test that fit_nd_to produces identical results to fit_nd
fn test_fit_nd_to_matches<T, S>()
where
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + 'static
        + std::fmt::Debug
        + ErrorNorm,
    S: StatisticsType + 'static,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test values (at sampling points)
    let values = Tensor::<T, crate::DynRank>::from_fn(&[n_points, n_k, n_omega][..], |idx| {
        <T as From<f64>>::from((idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3))
    });

    // Test for dim = 0
    let expected = sampling.fit_nd(None, &values, 0);

    let mut actual = Tensor::<T, crate::DynRank>::from_elem(&[basis_size, n_k, n_omega][..], <T as From<f64>>::from(0.0));
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
                let diff = (e - a).error_norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i, j, k, e, a
                );
            }
        }
    }
}

#[test]
fn test_fit_nd_to_matches_fermionic_real() {
    test_fit_nd_to_matches::<f64, Fermionic>();
}

#[test]
fn test_fit_nd_to_matches_fermionic_complex() {
    test_fit_nd_to_matches::<Complex<f64>, Fermionic>();
}

// ============================================================================
// Tests for evaluate_nd_with_context
// ============================================================================

#[test]
fn test_evaluate_nd_with_context_dim0() {
    use crate::working_buffer::SamplingContext;
    use mdarray::Tensor;

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients
    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    // Expected result using existing method
    let expected = sampling.evaluate_nd(None, &coeffs, 0);

    // Actual result using context
    let mut ctx = SamplingContext::new();
    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_points, n_k, n_omega][..], 0.0);
    sampling.evaluate_nd_with_context(&mut ctx, None, &coeffs, 0, &mut actual);

    // Compare
    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i, j, k, e, a
                );
            }
        }
    }
}

#[test]
fn test_evaluate_nd_with_context_dim1() {
    use crate::working_buffer::SamplingContext;
    use mdarray::Tensor;

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients with basis_size in middle dimension
    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[n_k, basis_size, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    // Expected result using existing method
    let expected = sampling.evaluate_nd(None, &coeffs, 1);

    // Actual result using context
    let mut ctx = SamplingContext::new();
    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_k, n_points, n_omega][..], 0.0);
    sampling.evaluate_nd_with_context(&mut ctx, None, &coeffs, 1, &mut actual);

    // Compare
    for i in 0..n_k {
        for j in 0..n_points {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i, j, k, e, a
                );
            }
        }
    }
}
