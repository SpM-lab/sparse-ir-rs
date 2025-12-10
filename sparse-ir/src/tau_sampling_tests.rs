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
        + std::ops::Mul<Output = T>
        + num_traits::MulAdd<Output = T>,
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
        + std::ops::Mul<Output = T>
        + num_traits::MulAdd<Output = T>,
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
