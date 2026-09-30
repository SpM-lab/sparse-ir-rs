//! Tests for DiscreteLehmannRepresentation
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::matrix::Mat;
#[allow(unused_imports)]
use crate::test_utils::At;
use crate::{
    AbstractKernel, Basis, Bosonic, DiscreteLehmannRepresentation, Error, Fermionic,
    FiniteTempBasis, LogisticKernel, MatsubaraFreq, MatsubaraSampling, RegularizedBoseKernel,
    Statistics, StatisticsType, TauSampling, bosonic_single_pole, giwn_single_pole,
    gtau_single_pole,
};
use num_complex::Complex;

fn max_relative_error_real(lhs: &crate::TypedTensor<f64>, rhs: &Mat<f64>) -> f64 {
    assert_eq!(lhs.rank(), 2);
    assert_eq!(*rhs.shape(), (lhs.shape()[0], lhs.shape()[1]));

    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..lhs.shape()[0] {
        for j in 0..lhs.shape()[1] {
            let a = lhs.at(&[i, j]);
            let b = rhs.at(&[i, j]);
            max_diff = max_diff.max((a - b).abs());
            max_ref = max_ref.max(a.abs());
        }
    }

    if max_ref == 0.0 {
        max_diff
    } else {
        max_diff / max_ref
    }
}

fn max_relative_error_complex(
    lhs: &crate::TypedTensor<Complex<f64>>,
    rhs: &Mat<Complex<f64>>,
) -> f64 {
    assert_eq!(lhs.rank(), 2);
    assert_eq!(*rhs.shape(), (lhs.shape()[0], lhs.shape()[1]));

    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..lhs.shape()[0] {
        for j in 0..lhs.shape()[1] {
            let a = lhs.at(&[i, j]);
            let b = rhs.at(&[i, j]);
            max_diff = max_diff.max((a - b).norm());
            max_ref = max_ref.max(a.norm());
        }
    }

    if max_ref == 0.0 {
        max_diff
    } else {
        max_diff / max_ref
    }
}

#[test]
fn test_dlr_construction_fermionic() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis).unwrap();

    assert_eq!(dlr.poles.len(), basis.size());
    assert_eq!(dlr.beta, beta);
    assert_eq!(dlr.wmax, wmax);

    // Poles should be in [-wmax, wmax]
    for &pole in &dlr.poles {
        assert!(
            pole.abs() <= wmax,
            "pole = {} exceeds wmax = {}",
            pole,
            wmax
        );
    }
}

#[test]
fn test_dlr_with_custom_poles() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None).unwrap();

    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&basis, poles.clone())
        .unwrap();

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);

    let tau_values = dlr.evaluate_tau(&[0.0, beta / 3.0, beta]).unwrap();
    for i in 0..3 {
        assert!(
            tau_values.at(&[i, 2]).is_finite(),
            "tau basis value for zero pole must be finite"
        );
        assert!(
            (tau_values.at(&[i, 2]) + 0.5).abs() < 1e-12,
            "zero-pole tau basis should match the logistic limit"
        );
    }

    let freqs = [
        crate::MatsubaraFreq::<Bosonic>::new(0).unwrap(),
        crate::MatsubaraFreq::<Bosonic>::new(2).unwrap(),
    ];
    let matsubara_values = dlr.evaluate_matsubara(&freqs).unwrap();
    assert!(
        matsubara_values.at(&[0, 2]).re.is_finite() && matsubara_values.at(&[0, 2]).im.is_finite(),
        "zero-pole Matsubara basis at n=0 must be finite"
    );
    assert!(
        (matsubara_values.at(&[0, 2]).re + 0.5 * beta).abs() < 1e-12,
        "zero-pole Matsubara basis should match the logistic limit"
    );
    assert!(
        matsubara_values.at(&[1, 2]).norm() < 1e-12,
        "zero-pole Matsubara basis should vanish away from n=0"
    );
}

/// Generic test for from_IR_nd/to_IR_nd roundtrip
fn test_dlr_nd_roundtrip_generic<T, S>()
where
    T: crate::fitters::FitScalar
        + num_traits::One
        + num_traits::Zero
        + std::ops::Sub<Output = T>
        + 'static
        + crate::test_utils::ErrorNorm
        + crate::test_utils::ConvertFromReal,
    S: crate::StatisticsType + 'static,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None).unwrap();

    let dlr = DiscreteLehmannRepresentation::<S>::from_ir(&basis).unwrap();

    let basis_size = basis.size();

    // Create reference 3D tensor with basis_size at dim=0
    let shape_ref = [basis_size, 3, 4];
    let gl_ref = {
        let mut tensor = crate::test_utils::tensor_filled(&shape_ref, T::zero());
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let mag = ((l + 1) as f64).powi(-2) * (i + j + 1) as f64;
                    *tensor.get_mut(&[l, i, j]).unwrap() = T::from_real(mag);
                }
            }
        }
        tensor
    };

    // Test transformation along each dimension
    for dim in 0..3 {
        // Move basis dimension from 0 to dim
        let gl_3d = crate::test_utils::movedim(&gl_ref, 0, dim);

        // Transform: IR → DLR → IR
        let g_dlr = dlr.from_ir_nd::<T>(None, &gl_3d, dim).unwrap();
        let gl_reconst = dlr.to_ir_nd::<T>(None, &g_dlr, dim).unwrap();

        // Move back to dim=0 for comparison
        let gl_reconst_dim0 = crate::test_utils::movedim(&gl_reconst, dim, 0);

        // Check shape
        assert_eq!(gl_reconst_dim0.rank(), gl_ref.rank());

        // Check roundtrip - compare with reference
        let mut max_error = 0.0;
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let val_orig = gl_ref.at(&[l, i, j]);
                    let val_reconst = gl_reconst_dim0.at(&[l, i, j]);
                    let error = (val_orig - val_reconst).error_norm();
                    if error > max_error {
                        max_error = error;
                    }
                }
            }
        }

        println!(
            "DLR {:?} {} ND roundtrip (dim={}): error = {:.2e}",
            S::STATISTICS,
            std::any::type_name::<T>(),
            dim,
            max_error
        );
        assert!(
            max_error < 1e-7,
            "ND roundtrip error too large for dim {}: {:.2e}",
            dim,
            max_error
        );
    }
}

#[test]
fn test_dlr_nd_roundtrip_real_fermionic() {
    test_dlr_nd_roundtrip_generic::<f64, Fermionic>();
}

#[test]
fn test_dlr_nd_roundtrip_complex_fermionic() {
    test_dlr_nd_roundtrip_generic::<Complex<f64>, Fermionic>();
}

#[test]
fn test_dlr_nd_roundtrip_real_bosonic() {
    test_dlr_nd_roundtrip_generic::<f64, Bosonic>();
}

#[test]
fn test_dlr_nd_roundtrip_complex_bosonic() {
    test_dlr_nd_roundtrip_generic::<Complex<f64>, Bosonic>();
}

#[test]
fn test_dlr_basis_trait() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis_ir =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis_ir).unwrap();

    // Test Basis trait methods
    assert_eq!(dlr.beta(), beta);
    assert_eq!(dlr.wmax(), wmax);
    assert_eq!(dlr.lambda(), beta * wmax);
    assert_eq!(dlr.size(), dlr.poles.len());
    assert_eq!(dlr.accuracy(), basis_ir.accuracy());

    let sig = dlr.significance();
    assert_eq!(sig.len(), dlr.size());
    assert!(
        sig.iter().all(|&s| (s - 1.0).abs() < 1e-10),
        "All significance should be 1.0"
    );

    // Test evaluate_tau
    let tau_points = vec![0.0, beta / 4.0, beta / 2.0, 3.0 * beta / 4.0];
    let matrix_tau = dlr.evaluate_tau(&tau_points).unwrap();
    assert_eq!(matrix_tau.shape(), &[tau_points.len(), dlr.size()]);

    // Test evaluate_matsubara
    use crate::MatsubaraFreq;
    let freqs = vec![
        MatsubaraFreq::<Fermionic>::new(1).unwrap(),
        MatsubaraFreq::<Fermionic>::new(3).unwrap(),
        MatsubaraFreq::<Fermionic>::new(-1).unwrap(),
    ];
    let matrix_matsu = dlr.evaluate_matsubara(&freqs).unwrap();
    assert_eq!(matrix_matsu.shape(), &[freqs.len(), dlr.size()]);
}

#[test]
fn test_dlr_with_tau_sampling() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis_ir =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Create DLR
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis_ir).unwrap();

    // Create TauSampling from DLR (using Basis trait)
    let tau_points = basis_ir.default_tau_sampling_points().unwrap();
    let n_tau_points = tau_points.len();
    let sampling_dlr = TauSampling::<Fermionic>::with_sampling_points(&dlr, tau_points).unwrap();

    // Test that it works
    println!(
        "IR tau sampling points: {}, DLR size: {}",
        n_tau_points,
        dlr.size()
    );
    assert_eq!(sampling_dlr.n_sampling_points(), n_tau_points);
    assert_eq!(sampling_dlr.basis_size(), dlr.size());

    println!("TauSampling with DLR created successfully!");
}

// ====================
// RegularizedBoseKernel DLR Tests
// ====================

#[test]
fn test_dlr_regularized_bose_construction() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();

    // Note: With improved SVEHints (proper segments_x/y), DLR now has ~60% of expected poles
    // Previous: basis=11, poles=1 (9% coverage, error=3.66e0)
    // Current:  basis=55, poles=33 (60% coverage, error=2.6e-2) ✅ Major improvement!
    // Future:   Need Df64 SVE for full precision
    println!("\n=== RegularizedBoseKernel DLR Test ===");
    println!("Beta: {}, Wmax: {}", beta, wmax);
    println!("Basis size: {}", basis.size());
    println!(
        "DLR poles: {} (expected: {}, coverage: {:.1}%)",
        dlr.poles.len(),
        basis.size(),
        100.0 * dlr.poles.len() as f64 / basis.size() as f64
    );

    assert!(
        dlr.poles.len() > basis.size() / 2,
        "DLR should have at least 50% of basis size poles"
    );
    assert_eq!(dlr.beta, beta);
    assert_eq!(dlr.wmax, wmax);

    // Poles should be in [-wmax, wmax]
    for &pole in &dlr.poles {
        assert!(
            pole.abs() <= wmax,
            "pole = {} exceeds wmax = {}",
            pole,
            wmax
        );
    }
}

#[test]
fn test_dlr_regularized_bose_with_custom_poles() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&basis, poles.clone())
        .unwrap();

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);

    let tau_values = dlr.evaluate_tau(&[0.0, beta / 3.0, beta]).unwrap();
    for i in 0..3 {
        assert!(
            tau_values.at(&[i, 2]).is_finite(),
            "tau basis value for zero pole must be finite"
        );
        // -K^B(τ, 0) = -lim ω e^{-τω}/(1 - e^{-βω}) = -1/β
        assert!(
            (tau_values.at(&[i, 2]) + 1.0 / beta).abs() < 1e-12,
            "zero-pole tau basis should match the regularized limit -1/beta"
        );
    }

    let freqs = [
        crate::MatsubaraFreq::<Bosonic>::new(0).unwrap(),
        crate::MatsubaraFreq::<Bosonic>::new(2).unwrap(),
    ];
    let matsubara_values = dlr.evaluate_matsubara(&freqs).unwrap();
    assert!(
        matsubara_values.at(&[0, 2]).re.is_finite() && matsubara_values.at(&[0, 2]).im.is_finite(),
        "zero-pole Matsubara basis at n=0 must be finite"
    );
    // ω/(iν - ω) at ν = 0 is -1 for every ω, including the limit ω → 0
    assert!(
        (matsubara_values.at(&[0, 2]).re + 1.0).abs() < 1e-12,
        "zero-pole Matsubara basis should match the regularized limit -1"
    );
    assert!(
        matsubara_values.at(&[1, 2]).norm() < 1e-12,
        "zero-pole Matsubara basis should vanish away from n=0"
    );

    println!("\n=== RegularizedBoseKernel DLR with Custom Poles ===");
    println!("Successfully created DLR with {} custom poles", poles.len());
}

#[test]
fn test_dlr_regularized_bose_basis_functions_match_physical_kernel() {
    // For a RegularizedBoseKernel basis the DLR functions are u_p(τ) =
    // -K^B(τ, ω_p) with K^B(τ, ω) = ω e^{-τω}/(1 - e^{-βω}), and
    // û_p(iν) = ω_p/(iν - ω_p): irbasis paper (Chikano et al., CPC 240, 181
    // (2019), arXiv:1807.05237) Eqs. (3) and (16). Both are closed forms, so
    // they must hold to rounding; wmax ≠ 1 exposes any extra power of wmax.
    let beta = 10.0;
    let wmax = 2.0;
    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-10), None)
            .unwrap();
    let poles = vec![-1.5, -0.4, 0.3, 1.8];
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&basis, poles.clone())
        .unwrap();

    let taus = [0.25, 3.7, 8.9];
    let tau_values = dlr.evaluate_tau(&taus).unwrap();
    for (i, &tau) in taus.iter().enumerate() {
        for (p, &pole) in poles.iter().enumerate() {
            let exact = -pole * (-tau * pole).exp() / (1.0 - (-beta * pole).exp());
            assert!(
                (tau_values.at(&[i, p]) - exact).abs() <= 1e-13 * exact.abs().max(1.0),
                "tau={tau}, pole={pole}: u_p = {}, -K^B = {exact}",
                tau_values.at(&[i, p])
            );
        }
    }

    let freqs = [0_i64, 2, -6].map(|n| crate::MatsubaraFreq::<Bosonic>::new(n).unwrap());
    let matsubara_values = dlr.evaluate_matsubara(&freqs).unwrap();
    for (i, freq) in freqs.iter().enumerate() {
        for (p, &pole) in poles.iter().enumerate() {
            let exact = Complex::new(pole, 0.0) / Complex::new(-pole, freq.value(beta));
            assert!(
                (matsubara_values.at(&[i, p]) - exact).norm() <= 1e-13 * exact.norm().max(1.0),
                "n={}, pole={pole}: uhat_p = {}, pole/(iν - pole) = {exact}",
                freq.get_n(),
                matsubara_values.at(&[i, p])
            );
        }
    }
}

#[test]
fn test_dlr_regularized_bose_nd_roundtrip_f64() {
    test_dlr_regularized_bose_nd_roundtrip_generic::<f64>();
}

#[test]
fn test_dlr_regularized_bose_nd_roundtrip_complex() {
    test_dlr_regularized_bose_nd_roundtrip_generic::<Complex<f64>>();
}

/// Generic test for RegularizedBoseKernel DLR from_IR_nd/to_IR_nd roundtrip
fn test_dlr_regularized_bose_nd_roundtrip_generic<T>()
where
    T: crate::fitters::FitScalar
        + num_traits::One
        + num_traits::Zero
        + std::ops::Sub<Output = T>
        + 'static
        + crate::test_utils::ErrorNorm
        + crate::test_utils::ConvertFromReal,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();

    let basis_size = basis.size();

    // Create reference 3D tensor with basis_size at dim=0
    let shape_ref = [basis_size, 3, 4];
    let gl_ref = {
        let mut tensor = crate::test_utils::tensor_filled(&shape_ref, T::zero());
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let mag = ((l + 1) as f64).powi(-2) * (i + j + 1) as f64;
                    *tensor.get_mut(&[l, i, j]).unwrap() = T::from_real(mag);
                }
            }
        }
        tensor
    };

    // Test transformation along each dimension
    for dim in 0..3 {
        // Move basis dimension from 0 to dim
        let gl_3d = crate::test_utils::movedim(&gl_ref, 0, dim);

        // Transform: IR → DLR → IR
        let g_dlr = dlr.from_ir_nd::<T>(None, &gl_3d, dim).unwrap();
        let gl_reconst = dlr.to_ir_nd::<T>(None, &g_dlr, dim).unwrap();

        // Move back to dim=0 for comparison
        let gl_reconst_dim0 = crate::test_utils::movedim(&gl_reconst, dim, 0);

        // Check shape
        assert_eq!(gl_reconst_dim0.rank(), gl_ref.rank());

        // Check roundtrip - compare with reference
        let mut max_error = 0.0;
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let val_orig = gl_ref.at(&[l, i, j]);
                    let val_reconst = gl_reconst_dim0.at(&[l, i, j]);
                    let error = (val_orig - val_reconst).error_norm();
                    if error > max_error {
                        max_error = error;
                    }
                }
            }
        }

        println!(
            "RegularizedBose DLR {} ND roundtrip (dim={}): error = {:.2e}",
            std::any::type_name::<T>(),
            dim,
            max_error
        );
        // With sinh formula fix (minus sign) and alpha=4 root finding:
        // Actual error: ~1.24e-12 (excellent precision!)
        assert!(
            max_error < 2e-12,
            "RegularizedBose ND roundtrip error too large for dim {}: {:.2e}",
            dim,
            max_error
        );
    }
}

#[test]
fn test_dlr_regularized_bose_matches_ir_evaluations() {
    let beta = 1e4;
    let lambda = 1e2;
    let epsilon = 1e-10;

    let kernel = RegularizedBoseKernel::new(lambda).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();

    let tau_points = basis.default_tau_sampling_points().unwrap();
    let tau_sampling =
        TauSampling::<Bosonic>::with_sampling_points(&basis, tau_points.clone()).unwrap();

    let matsubara_points = basis.default_matsubara_sampling_points(false).unwrap();
    let matsubara_sampling =
        MatsubaraSampling::<Bosonic>::with_sampling_points(&basis, matsubara_points.clone())
            .unwrap();

    let n_poles = dlr.poles.len();
    let dlr_coeffs_2d = Mat::<f64>::from_fn([n_poles, 1], |idx| {
        let pole = dlr.poles[idx[0]];
        (idx[0] as f64 + 1.0) / (1.0 + pole.abs())
    });
    let dlr_coeffs =
        crate::TypedTensor::from_vec_col_major(vec![n_poles, 1], dlr_coeffs_2d.as_slice().to_vec())
            .unwrap();

    let ir_coeffs = dlr.to_ir_nd::<f64>(None, &dlr_coeffs, 0).unwrap();

    let g_tau_ir = tau_sampling.evaluate_nd(None, &ir_coeffs, 0).unwrap();
    let dlr_tau = dlr.evaluate_tau(&tau_points).unwrap();
    let g_tau_dlr = Mat::<f64>::from_fn([tau_points.len(), 1], |idx| {
        let i = idx[0];
        let mut sum = 0.0;
        for p in 0..n_poles {
            sum += dlr_tau.at(&[i, p]) * dlr_coeffs_2d.at(&[p, 0]);
        }
        sum
    });

    let g_iw_ir = matsubara_sampling
        .evaluate_nd_real(None, &ir_coeffs, 0)
        .unwrap();
    let dlr_iw = dlr.evaluate_matsubara(&matsubara_points).unwrap();
    let g_iw_dlr = Mat::<Complex<f64>>::from_fn([matsubara_points.len(), 1], |idx| {
        let i = idx[0];
        let mut sum = Complex::new(0.0, 0.0);
        for p in 0..n_poles {
            sum += dlr_iw.at(&[i, p]) * dlr_coeffs_2d.at(&[p, 0]);
        }
        sum
    });

    let tau_error = max_relative_error_real(&g_tau_ir, &g_tau_dlr);
    let matsubara_error = max_relative_error_complex(&g_iw_ir, &g_iw_dlr);

    assert!(
        tau_error < 1e-8,
        "RegularizedBose DLR tau evaluation mismatch: {:.3e}",
        tau_error
    );
    assert!(
        matsubara_error < 1e-8,
        "RegularizedBose DLR Matsubara evaluation mismatch: {:.3e}",
        matsubara_error
    );
}

#[test]
fn test_fermionic_dlr_tau_sampling_matrix_matches_stable_kernel() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis).unwrap();
    let tau_points = basis.default_tau_sampling_points().unwrap();
    let tau_sampling =
        TauSampling::<Fermionic>::with_sampling_points(&dlr, tau_points.clone()).unwrap();

    let expected = Mat::<f64>::from_fn([tau_points.len(), dlr.poles.len()], |idx| {
        let tau = tau_points[idx[0]];
        let pole = dlr.poles[idx[1]];
        let (tau_norm, sign) = crate::taufuncs::normalize_tau::<Fermionic>(tau, beta).unwrap();
        let x = 2.0 * tau_norm / beta - 1.0;
        let y = pole / wmax;
        sign * (-kernel.compute(x, y))
    });

    let matrix = tau_sampling.matrix();
    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..tau_points.len() {
        for p in 0..dlr.poles.len() {
            let actual = matrix.at(&[i, p]);
            let reference = expected.at(&[i, p]);
            assert!(
                actual.is_finite(),
                "tau sampling matrix contains non-finite value at ({}, {})",
                i,
                p
            );
            max_diff = max_diff.max((actual - reference).abs());
            max_ref = max_ref.max(reference.abs());
        }
    }

    let rel_error = if max_ref == 0.0 {
        max_diff
    } else {
        max_diff / max_ref
    };
    assert!(
        rel_error < 1e-12,
        "fermionic DLR tau sampling matrix mismatch: {:.3e}",
        rel_error
    );
}

#[test]
fn test_bosonic_logistic_dlr_tau_sampling_matrix_matches_stable_kernel() {
    let beta = 10_000.0;
    let wmax = 1.0;
    let epsilon = 1e-12;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None).unwrap();
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();
    let tau_points = basis.default_tau_sampling_points().unwrap();
    let tau_sampling =
        TauSampling::<Bosonic>::with_sampling_points(&dlr, tau_points.clone()).unwrap();

    let expected = Mat::<f64>::from_fn([tau_points.len(), dlr.poles.len()], |idx| {
        let tau = tau_points[idx[0]];
        let pole = dlr.poles[idx[1]];
        let (tau_norm, sign) = crate::taufuncs::normalize_tau::<Bosonic>(tau, beta).unwrap();
        let x = 2.0 * tau_norm / beta - 1.0;
        let y = pole / wmax;
        sign * (-kernel.compute(x, y))
    });

    let matrix = tau_sampling.matrix();
    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..tau_points.len() {
        for p in 0..dlr.poles.len() {
            let actual = matrix.at(&[i, p]);
            let reference = expected.at(&[i, p]);
            assert!(
                actual.is_finite(),
                "tau sampling matrix contains non-finite value at ({}, {}) for pole {} and tau {}",
                i,
                p,
                dlr.poles[p],
                tau_points[i]
            );
            max_diff = max_diff.max((actual - reference).abs());
            max_ref = max_ref.max(reference.abs());
        }
    }

    let rel_error = if max_ref == 0.0 {
        max_diff
    } else {
        max_diff / max_ref
    };
    assert!(
        rel_error < 1e-12,
        "bosonic logistic DLR tau sampling matrix mismatch: {:.3e}",
        rel_error
    );
}

// ============================================================================
// Single-pole helpers: gtau_single_pole / giwn_single_pole (#262)
// ============================================================================

/// `(beta, omega)` cases for the single-pole helper tests: both signs of
/// `omega`, three temperatures, and `|beta * omega|` from 1e-3 (close to the
/// bosonic pole at `omega = 0`) to 50 (strongly suppressed tails).
const SINGLE_POLE_CASES: [(f64, f64); 18] = [
    (1.0, -5.0),
    (1.0, -1.0),
    (1.0, -1e-3),
    (1.0, 1e-3),
    (1.0, 1.0),
    (1.0, 5.0),
    (10.0, -5.0),
    (10.0, -0.1),
    (10.0, -1e-4),
    (10.0, 1e-4),
    (10.0, 0.1),
    (10.0, 5.0),
    (100.0, -0.5),
    (100.0, -0.01),
    (100.0, -1e-5),
    (100.0, 1e-5),
    (100.0, 0.01),
    (100.0, 0.5),
];

/// `zeta` in `G(tau - beta) = zeta * G(tau)`: -1 for fermions, +1 for bosons.
fn statistics_zeta<S: StatisticsType>() -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => -1.0,
        Statistics::Bosonic => 1.0,
    }
}

/// Single-pole `G(tau) = -exp(-omega tau) / (1 + zeta' exp(-beta omega))` for
/// `tau` in `[0, beta]` (`zeta' = +1` fermions, `-1` bosons).
///
/// Written without the overflow-avoiding rearrangement the implementation uses
/// for `omega < 0`, so it independently checks both branches. Valid while
/// `|beta * omega|` stays far below the `exp` overflow threshold (~709).
fn single_pole_tau_reference<S: StatisticsType>(tau: f64, omega: f64, beta: f64) -> f64 {
    let boltzmann = (-beta * omega).exp();
    let denominator = match S::STATISTICS {
        Statistics::Fermionic => 1.0 + boltzmann,
        Statistics::Bosonic => 1.0 - boltzmann,
    };
    -(-omega * tau).exp() / denominator
}

/// `gtau_single_pole` against the closed form, its sign, and the jump
/// `G(0+) - G(0-) = -1` fixed by the canonical (anti)commutator.
fn check_single_pole_tau_closed_form<S: StatisticsType>() {
    let zeta = statistics_zeta::<S>();
    for (beta, omega) in SINGLE_POLE_CASES {
        let x = (beta * omega).abs();
        // Error model: exp amplifies the rounding of its argument by |beta*omega|
        // on both sides; the bosonic reference loses ~eps/|beta*omega| to the
        // cancellation in 1 - exp(-beta*omega). Factor 16 is headroom.
        let rel_tol = 16.0 * f64::EPSILON * (1.0 + x + 1.0 / x);

        // G(tau) = -<T c(tau) c^dag>: negative for fermions at every omega;
        // for bosons negative for omega > 0 and positive for omega < 0.
        let expected_sign = match S::STATISTICS {
            Statistics::Fermionic => -1.0,
            Statistics::Bosonic => -omega.signum(),
        };

        for frac in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let tau = frac * beta;
            let value = gtau_single_pole::<S>(tau, omega, beta).unwrap();
            let reference = single_pole_tau_reference::<S>(tau, omega, beta);
            assert!(
                (value - reference).abs() <= rel_tol * reference.abs(),
                "{:?} G(tau={}) for omega={}, beta={}: got {:.16e}, expected {:.16e} \
                 (rel. error {:.3e}, tol {:.3e})",
                S::STATISTICS,
                tau,
                omega,
                beta,
                value,
                reference,
                ((value - reference) / reference).abs(),
                rel_tol
            );
            assert_eq!(
                value.signum(),
                expected_sign,
                "{:?} G(tau={}) for omega={}, beta={} has the wrong sign: {:.16e}",
                S::STATISTICS,
                tau,
                omega,
                beta,
                value
            );
        }

        // Periodic (bosons) / antiperiodic (fermions) extension to tau < 0.
        let tau = -0.25 * beta;
        let value = gtau_single_pole::<S>(tau, omega, beta).unwrap();
        let reference = zeta * single_pole_tau_reference::<S>(tau + beta, omega, beta);
        assert!(
            (value - reference).abs() <= rel_tol * reference.abs(),
            "{:?} G(tau={}) for omega={}, beta={}: got {:.16e}, expected {:.16e}",
            S::STATISTICS,
            tau,
            omega,
            beta,
            value,
            reference
        );

        // G(0+) - G(0-) = -1 with G(0-) = zeta * G(beta-). Both terms carry a
        // few ulps of relative error, so the bound scales with their magnitude.
        let g_0 = gtau_single_pole::<S>(0.0, omega, beta).unwrap();
        let g_beta = gtau_single_pole::<S>(beta, omega, beta).unwrap();
        let jump = g_0 - zeta * g_beta;
        let jump_tol = 16.0 * f64::EPSILON * g_0.abs().max(g_beta.abs()).max(1.0);
        assert!(
            (jump + 1.0).abs() <= jump_tol,
            "{:?} G(0+) - G(0-) for omega={}, beta={}: got {:.16e}, expected -1 (tol {:.3e})",
            S::STATISTICS,
            omega,
            beta,
            jump,
            jump_tol
        );
    }
}

#[test]
fn test_single_pole_tau_matches_closed_form_fermionic() {
    check_single_pole_tau_closed_form::<Fermionic>();
}

#[test]
fn test_single_pole_tau_matches_closed_form_bosonic() {
    check_single_pole_tau_closed_form::<Bosonic>();
}

/// `gtau_single_pole` and `giwn_single_pole` must be a Fourier pair:
/// `G(iv_n) = int_0^beta dtau exp(i v_n tau) G(tau)`.
fn check_single_pole_fourier_pair<S: StatisticsType>() {
    let n_segments: usize = 32;
    let rule = crate::gauss::legendre::<f64>(16);
    // Lowest Matsubara index of the right parity: 1 for fermions, 0 for bosons.
    let parity = match S::STATISTICS {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    };

    for (beta, omega) in SINGLE_POLE_CASES {
        let edges: Vec<f64> = (0..=n_segments)
            .map(|k| beta * k as f64 / n_segments as f64)
            .collect();
        let quad = rule.piecewise(&edges).unwrap();
        let gtau: Vec<f64> = quad
            .x
            .iter()
            .map(|&tau| gtau_single_pole::<S>(tau, omega, beta).unwrap())
            .collect();

        // Error model: every quadrature term carries O(100) ulps of relative
        // rounding error (exp/cos/sin of arguments up to ~50) and the N-term sum
        // adds at most N/2 ulps of sum_i |w_i G(tau_i)|; 2 N eps bounds both.
        // The quadrature truncation error is negligible: 16-point Gauss-Legendre
        // on 32 panels resolves exp((i v_n - omega) tau) far below machine
        // precision for |v_n| <= 17 pi / beta and |beta omega| <= 50.
        let l1: f64 = quad.w.iter().zip(&gtau).map(|(&w, &g)| (w * g).abs()).sum();
        let tol = 2.0 * quad.x.len() as f64 * f64::EPSILON * l1;

        for m in -8..=8 {
            let freq = MatsubaraFreq::<S>::new(2 * m + parity).unwrap();
            let nu = freq.value(beta);
            let transform: Complex<f64> = quad
                .x
                .iter()
                .zip(&quad.w)
                .zip(&gtau)
                .map(|((&tau, &w), &g)| Complex::new(0.0, nu * tau).exp() * (w * g))
                .sum();
            let reference = giwn_single_pole::<S>(&freq, omega, beta).unwrap();
            assert!(
                (transform - reference).norm() <= tol,
                "{:?} Fourier transform of G(tau) at n={} for omega={}, beta={}: \
                 got {:.16e}, giwn_single_pole gives {:.16e} (|diff| {:.3e}, tol {:.3e})",
                S::STATISTICS,
                freq.n(),
                omega,
                beta,
                transform,
                reference,
                (transform - reference).norm(),
                tol
            );
        }
    }
}

#[test]
fn test_single_pole_fourier_pair_fermionic() {
    check_single_pole_fourier_pair::<Fermionic>();
}

#[test]
fn test_single_pole_fourier_pair_bosonic() {
    check_single_pole_fourier_pair::<Bosonic>();
}

/// `gtau_single_pole` times the pole weight must reproduce the DLR tau
/// functions, which carry the same convention as the C ABI DLR evaluation.
fn check_single_pole_matches_dlr_evaluate_tau<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 5.0;
    let epsilon = 1e-10;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None).unwrap();
    let dlr = DiscreteLehmannRepresentation::<S>::from_ir(&basis).unwrap();
    assert!(
        dlr.poles.iter().any(|&pole| pole > 0.0) && dlr.poles.iter().any(|&pole| pole < 0.0),
        "DLR poles must cover both signs of omega: {:?}",
        dlr.poles
    );

    let taus = [
        -0.75 * beta,
        -0.1 * beta,
        0.0,
        0.1 * beta,
        0.5 * beta,
        0.9 * beta,
        beta,
    ];
    let dlr_tau = dlr.evaluate_tau(&taus).unwrap();

    for (p, (&pole, &weight)) in dlr.poles.iter().zip(dlr.pole_weights()).enumerate() {
        // An exact bosonic zero pole is a genuine pole of the unweighted
        // single-pole function; the DLR evaluates it through its finite
        // regularized limit instead.
        if S::STATISTICS == Statistics::Bosonic && pole == 0.0 {
            continue;
        }
        for (i, &tau) in taus.iter().enumerate() {
            let expected = gtau_single_pole::<S>(tau, pole, beta).unwrap() * weight;
            let actual = dlr_tau.at(&[i, p]);
            // Both sides combine the same rounded exp, weight and denominator in
            // a different order, so they agree to a few ulps.
            assert!(
                (actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs(),
                "{:?} DLR u_p(tau={}) for pole {}: evaluate_tau gives {:.16e}, \
                 gtau_single_pole * weight gives {:.16e}",
                S::STATISTICS,
                tau,
                pole,
                actual,
                expected
            );
        }
    }
}

#[test]
fn test_single_pole_matches_dlr_evaluate_tau_fermionic() {
    check_single_pole_matches_dlr_evaluate_tau::<Fermionic>();
}

#[test]
fn test_single_pole_matches_dlr_evaluate_tau_bosonic() {
    check_single_pole_matches_dlr_evaluate_tau::<Bosonic>();
}

/// `omega = 0` is a genuine pole of the Bose factor (#209). The result stays
/// infinite, with the sign of the one-sided limit selected by the sign of the
/// zero: `G -> -inf` as `omega -> 0+` and `G -> +inf` as `omega -> 0-`.
#[test]
fn test_bosonic_single_pole_diverges_at_zero_omega() {
    let beta = 10.0;
    for tau in [0.0, 0.5 * beta, beta] {
        assert_eq!(
            bosonic_single_pole(tau, 0.0, beta).unwrap(),
            f64::NEG_INFINITY,
            "omega = +0.0 at tau = {}",
            tau
        );
        assert_eq!(
            bosonic_single_pole(tau, -0.0, beta).unwrap(),
            f64::INFINITY,
            "omega = -0.0 at tau = {}",
            tau
        );
    }
}

// ====================
// DLR error paths (#237)
// ====================

/// Basis whose default ω sampling points, i.e. the default DLR poles, are
/// truncated to `n_poles`; everything else is delegated to `inner`.
///
/// This stands in for a basis where root finding yields fewer default poles
/// than the basis size, as reported for `RegularizedBoseKernel` at large Λ in
/// #114. That case no longer reproduces, so the truncation is synthetic.
struct TruncatedDefaultPoles<'a, B> {
    inner: &'a B,
    n_poles: usize,
}

impl<S, B> Basis<S> for TruncatedDefaultPoles<'_, B>
where
    S: StatisticsType,
    B: Basis<S>,
{
    type Kernel = B::Kernel;

    fn kernel(&self) -> &Self::Kernel {
        self.inner.kernel()
    }

    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    fn wmax(&self) -> f64 {
        self.inner.wmax()
    }

    fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn accuracy(&self) -> f64 {
        self.inner.accuracy()
    }

    fn significance(&self) -> Vec<f64> {
        self.inner.significance()
    }

    fn svals(&self) -> Vec<f64> {
        self.inner.svals()
    }

    fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error> {
        self.inner.default_tau_sampling_points()
    }

    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<MatsubaraFreq<S>>, Error>
    where
        S: 'static,
    {
        self.inner.default_matsubara_sampling_points(positive_only)
    }

    fn evaluate_tau(&self, tau: &[f64]) -> Result<crate::Matrix<f64>, Error> {
        self.inner.evaluate_tau(tau)
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[MatsubaraFreq<S>],
    ) -> Result<crate::Matrix<Complex<f64>>, Error>
    where
        S: 'static,
    {
        self.inner.evaluate_matsubara(freqs)
    }

    fn evaluate_omega(&self, omega: &[f64]) -> Result<crate::Matrix<f64>, Error> {
        self.inner.evaluate_omega(omega)
    }

    fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error> {
        let mut poles = self.inner.default_omega_sampling_points()?;
        poles.truncate(self.n_poles);
        Ok(poles)
    }
}

fn check_dlr_new_insufficient_default_poles<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 1.0;
    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(1e-6), None).unwrap();
    let basis_size = basis.size();
    assert_eq!(
        basis.default_omega_sampling_points().unwrap().len(),
        basis_size
    );

    // Exactly as many default poles as basis functions is enough.
    let enough = TruncatedDefaultPoles {
        inner: &basis,
        n_poles: basis_size,
    };
    let dlr = DiscreteLehmannRepresentation::<S>::from_ir(&enough).unwrap();
    assert_eq!(dlr.poles.len(), basis_size);

    // One pole fewer than the basis size is rejected with a typed error
    // carrying both counts, instead of a panic.
    let too_few = TruncatedDefaultPoles {
        inner: &basis,
        n_poles: basis_size - 1,
    };
    let err = DiscreteLehmannRepresentation::<S>::from_ir(&too_few)
        .err()
        .expect("DLR construction must fail with too few default poles");
    assert_eq!(
        err,
        Error::InsufficientDefaultPoles {
            basis_size,
            n_poles: basis_size - 1,
        }
    );
}

#[test]
fn test_dlr_new_insufficient_default_poles_fermionic() {
    check_dlr_new_insufficient_default_poles::<Fermionic>();
}

#[test]
fn test_dlr_new_insufficient_default_poles_bosonic() {
    check_dlr_new_insufficient_default_poles::<Bosonic>();
}

/// `RegularizedBoseKernel` supports only bosonic statistics. A fermionic basis
/// built from it must be rejected by both DLR constructors with a typed error
/// instead of panicking in `RegularizedBoseKernel::regularizer` (#237, #241).
#[test]
fn test_dlr_regularized_bose_fermionic_is_kernel_statistics_mismatch() {
    let beta = 1.0;
    let wmax = 10.0;
    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Fermionic>::new(kernel, beta, Some(1e-6), None)
            .unwrap();
    // The default poles are sufficient, so `new` reaches the statistics check.
    assert!(basis.default_omega_sampling_points().unwrap().len() >= basis.size());

    let err = DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(
        &basis,
        vec![-2.0, 0.5, 3.0],
    )
    .err()
    .expect("with_poles must reject RegularizedBoseKernel with fermionic statistics");
    assert_eq!(err, Error::KernelStatisticsMismatch);

    let err = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis)
        .err()
        .expect("new must reject RegularizedBoseKernel with fermionic statistics");
    assert_eq!(err, Error::KernelStatisticsMismatch);

    // The same kernel with bosonic statistics is supported.
    let bosonic =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-6), None)
            .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(
        &bosonic,
        vec![-2.0, 0.5, 3.0],
    )
    .unwrap();
    assert_eq!(dlr.poles, vec![-2.0, 0.5, 3.0]);
}

#[test]
fn test_dlr_error_display_and_error_trait() {
    let err = Error::InsufficientDefaultPoles {
        basis_size: 12,
        n_poles: 7,
    };
    assert_eq!(
        err.to_string(),
        "number of default poles (7) is less than the basis size (12)"
    );

    let err = Error::KernelStatisticsMismatch;
    assert_eq!(
        err.to_string(),
        "kernel does not support the requested statistics: kernels with ypower = 1 \
         (e.g. RegularizedBoseKernel) require bosonic statistics"
    );

    // Usable as a boxed `std::error::Error`, e.g. with `?` in functions
    // returning `Box<dyn Error>`. Neither variant wraps an underlying cause.
    let boxed: Box<dyn std::error::Error> = Box::new(err);
    assert!(boxed.source().is_none());
    assert_eq!(
        boxed.to_string(),
        Error::KernelStatisticsMismatch.to_string()
    );
}

/// Converting an empty batch gives an empty result of the right shape.
/// Before the fix `from_ir_nd` and `to_ir_nd` segfaulted in `movedim`
/// (mdarray#21) for a zero extent before the last axis of a permuted view.
#[test]
fn test_dlr_nd_with_empty_batch() {
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis).unwrap();
    let (l, n_poles) = (basis.size(), dlr.poles.len());

    for (batch, dim) in [
        (vec![0usize], 1),
        (vec![0], 0),
        (vec![0, 3], 1),
        (vec![2, 0], 2),
    ] {
        let with_target = |n: usize| {
            let mut dims = batch.clone();
            dims.insert(dim, n);
            dims
        };
        let gl = crate::test_utils::tensor_filled::<f64>(&with_target(l), 0.0);
        let g_dlr = dlr.from_ir_nd::<f64>(None, &gl, dim).unwrap();
        assert_eq!(g_dlr.shape(), &with_target(n_poles)[..]);
        let back = dlr.to_ir_nd::<f64>(None, &g_dlr, dim).unwrap();
        assert_eq!(back.shape(), &with_target(l)[..]);

        let gl_z = crate::test_utils::tensor_filled::<Complex<f64>>(
            &with_target(l),
            Complex::new(0.0, 0.0),
        );
        let g_dlr_z = dlr.from_ir_nd::<Complex<f64>>(None, &gl_z, dim).unwrap();
        assert_eq!(g_dlr_z.shape(), &with_target(n_poles)[..]);
        let back_z = dlr.to_ir_nd::<Complex<f64>>(None, &g_dlr_z, dim).unwrap();
        assert_eq!(back_z.shape(), &with_target(l)[..]);
    }
}

/// from_ir_nd and to_ir_nd check the axis and the extent along it before
/// anything else, also for an empty batch. They panicked: an index out of
/// bounds for dim >= rank, an assertion for a wrong extent.
#[test]
fn test_dlr_transforms_report_the_axis_and_the_input_shape() {
    use crate::ArrayRole;

    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis).unwrap();
    let (l, n_poles) = (basis.size(), dlr.poles.len());
    assert_eq!(dlr.ir_basis_size(), Some(l));

    let gl = crate::test_utils::tensor_filled::<f64>(&[l, 3], 0.0);
    assert_eq!(
        dlr.from_ir_nd::<f64>(None, &gl, 2).err(),
        Some(Error::AxisOutOfRange { axis: 2, rank: 2 })
    );
    let g = crate::test_utils::tensor_filled::<Complex<f64>>(&[3, n_poles], Complex::new(0.0, 0.0));
    assert_eq!(
        dlr.to_ir_nd::<Complex<f64>>(None, &g, 2).err(),
        Some(Error::AxisOutOfRange { axis: 2, rank: 2 })
    );

    for batch in [3usize, 0] {
        let bad_gl = crate::test_utils::tensor_filled::<f64>(&[l + 1, batch], 0.0);
        assert_eq!(
            dlr.from_ir_nd::<f64>(None, &bad_gl, 0).err(),
            Some(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected: vec![l, batch],
                actual: vec![l + 1, batch],
            }),
            "batch = {batch}"
        );
        let bad_g = crate::test_utils::tensor_filled::<Complex<f64>>(
            &[batch, n_poles - 1],
            Complex::new(0.0, 0.0),
        );
        assert_eq!(
            dlr.to_ir_nd::<Complex<f64>>(None, &bad_g, 1).err(),
            Some(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected: vec![batch, n_poles],
                actual: vec![batch, n_poles - 1],
            }),
            "batch = {batch}"
        );
    }
}

/// A DLR without poles has no basis functions. Before the fix `with_poles`
/// panicked with an index out of bounds (evaluating V at no poles); a
/// sampling built on it would have segfaulted in the SVD (mdarray#21).
#[test]
fn test_dlr_with_no_poles_is_an_error() {
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let result = DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&basis, vec![]);
    assert!(matches!(result, Err(Error::EmptyInput { name: "poles" })));
    assert_eq!(
        Error::EmptyInput { name: "poles" }.to_string(),
        "poles must not be empty"
    );
}

/// The single-pole functions reject τ outside [-β, β], a β that is not
/// positive and finite, and a non-finite ω. Before the change the first
/// panicked and the others gave NaN or infinite values silently. ω = 0 is a
/// genuine pole of the bosonic function and stays infinite.
#[test]
fn test_single_pole_functions_check_their_arguments() {
    let beta = 2.0;
    assert!(matches!(
        gtau_single_pole::<Fermionic>(2.5, 1.0, beta),
        Err(Error::OutOfDomain { name: "tau", .. })
    ));
    assert!(matches!(
        crate::fermionic_single_pole(0.5, 1.0, 0.0),
        Err(Error::InvalidParameter { name: "beta", .. })
    ));
    assert!(matches!(
        bosonic_single_pole(0.5, f64::NAN, beta),
        Err(Error::InvalidParameter { name: "omega", .. })
    ));
    assert!(matches!(
        gtau_single_pole::<Bosonic>(0.5, f64::INFINITY, beta),
        Err(Error::InvalidParameter { name: "omega", .. })
    ));

    let freq = MatsubaraFreq::<Fermionic>::new(1).unwrap();
    assert!(matches!(
        giwn_single_pole(&freq, f64::INFINITY, beta),
        Err(Error::InvalidParameter { name: "omega", .. })
    ));
    assert!(matches!(
        giwn_single_pole(&freq, 1.0, -1.0),
        Err(Error::InvalidParameter { name: "beta", .. })
    ));

    assert_eq!(
        bosonic_single_pole(0.5, 0.0, beta).unwrap(),
        f64::NEG_INFINITY
    );
}

/// The DLR has no real-frequency functions: evaluate_omega is NotSupported
/// (it hit unimplemented!), and so is a DLR built from a DLR. Its default tau
/// and Matsubara sampling points are the interpolation nodes selected by a
/// row interpolative decomposition, one per pole (they were NotSupported
/// before the independent DLR). Its tau functions reject τ outside [-β, β]
/// and NaN for every pole, including the bosonic pole at 0, which returned
/// its limit without looking at τ.
#[test]
fn test_dlr_basis_methods_report_errors() {
    let beta = 10.0;
    let ir = FiniteTempBasis::<_, Bosonic>::new(
        LogisticKernel::new(beta).unwrap(),
        beta,
        Some(1e-6),
        None,
    )
    .unwrap();
    let dlr =
        DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&ir, vec![-0.5, 0.0, 0.5])
            .unwrap();

    let not_supported = |err: Error| assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
    assert_eq!(Basis::default_tau_sampling_points(&dlr).unwrap().len(), 3);
    assert_eq!(
        Basis::default_matsubara_sampling_points(&dlr, false)
            .unwrap()
            .len(),
        3
    );
    assert_eq!(TauSampling::new(&dlr).unwrap().n_sampling_points(), 3);
    not_supported(Basis::evaluate_omega(&dlr, &[0.1]).unwrap_err());
    not_supported(
        DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&dlr, vec![0.1])
            .err()
            .unwrap(),
    );

    for tau in [2.0 * beta, f64::NAN] {
        let err = Basis::evaluate_tau(&dlr, &[0.0, tau]).unwrap_err();
        assert!(
            matches!(err, Error::OutOfDomain { name: "tau", .. }),
            "{err:?}"
        );
    }
    assert_eq!(Basis::evaluate_tau(&dlr, &[]).unwrap().shape(), &[0, 3]);
    // Unchanged values: the pole at 0 gives its limit -1/2 at any τ.
    let values = Basis::evaluate_tau(&dlr, &[0.0, 0.5 * beta, beta]).unwrap();
    for i in 0..3 {
        assert_eq!(values.at(&[i, 1]), -0.5);
    }
}

/// with_poles checks the poles: a pole outside [-ωmax, ωmax] is OutOfDomain
/// and NaN or an infinity NonFiniteInput, both named "poles". They panicked
/// (SPIR_INTERNAL_ERROR in spir_dlr_new_with_poles) and were then reported
/// as evaluate_omega's OutOfDomain of "omega". ±ωmax are valid poles.
#[test]
fn test_with_poles_rejects_poles_outside_the_frequency_domain() {
    let ir = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        10.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    // Λ = 10 and β = 10: ωmax = 1.
    for pole in [2.0, -2.0] {
        let err =
            DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&ir, vec![0.5, pole])
                .err()
                .unwrap();
        assert_eq!(
            err,
            Error::OutOfDomain {
                name: "poles",
                value: pole,
                domain: (-1.0, 1.0),
            }
        );
    }
    for pole in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err =
            DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&ir, vec![0.5, pole])
                .err()
                .unwrap();
        assert!(
            matches!(
                &err,
                Error::NonFiniteInput { name: "poles", index, value }
                    if index == &vec![1] && value.to_bits() == pole.to_bits()
            ),
            "{err:?}"
        );
    }
    DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&ir, vec![-1.0, 1.0]).unwrap();
}

/// Duplicate poles are accepted, like duplicate sampling points (#291):
/// they only make the fit of from_ir_nd ill-conditioned (the coefficients of
/// equal poles are not unique), and to_ir_nd(from_ir_nd(gl)) still gives gl.
#[test]
fn test_with_poles_accepts_duplicate_poles() {
    let ir = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        10.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let mut poles = ir.default_omega_sampling_points().unwrap();
    poles.push(poles[0]);
    let dlr =
        DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&ir, poles.clone()).unwrap();
    assert_eq!(dlr.poles, poles);

    let l = ir.size();
    let gl = crate::test_utils::tensor_from_fn::<f64>(&[l], |i| 1.0 / ((i[0] + 1) as f64).powi(2));
    let g_dlr = dlr.from_ir_nd::<f64>(None, &gl, 0).unwrap();
    assert!(g_dlr.host_data().unwrap().iter().all(|x| x.is_finite()));
    let back = dlr.to_ir_nd::<f64>(None, &g_dlr, 0).unwrap();
    for (l, (x, y)) in back
        .host_data()
        .unwrap()
        .iter()
        .zip(gl.host_data().unwrap())
        .enumerate()
    {
        assert!((x - y).abs() < 1e-10, "l = {l}: {x} vs {y}");
    }
}

use crate::kernel::{KernelProperties, LogisticSVEHints};

/// LogisticKernel that reports ypower = 2, as a kernel of another crate
/// could; with_poles only reads its ypower and its regularizer
#[derive(Clone, Copy)]
struct OtherYpowerKernel(LogisticKernel);

impl KernelProperties for OtherYpowerKernel {
    type SVEHintsType<T>
        = LogisticSVEHints<T>
    where
        T: Copy + std::fmt::Debug + Send + Sync + crate::CustomNumeric + 'static;

    fn ypower(&self) -> i32 {
        2
    }

    fn conv_radius(&self) -> f64 {
        self.0.conv_radius()
    }

    fn xmax(&self) -> f64 {
        self.0.xmax()
    }

    fn ymax(&self) -> f64 {
        self.0.ymax()
    }

    fn regularizer<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        self.0.regularizer::<S>(beta, omega)
    }

    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + std::fmt::Debug + Send + Sync + crate::CustomNumeric + 'static,
    {
        self.0.sve_hints(epsilon)
    }
}

/// `inner` with its kernel replaced by `kernel`
struct WithKernel<'a, B, K> {
    inner: &'a B,
    kernel: K,
}

impl<S, B, K> Basis<S> for WithKernel<'_, B, K>
where
    S: StatisticsType,
    B: Basis<S>,
    K: KernelProperties,
{
    type Kernel = K;

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    fn wmax(&self) -> f64 {
        self.inner.wmax()
    }

    fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn accuracy(&self) -> f64 {
        self.inner.accuracy()
    }

    fn significance(&self) -> Vec<f64> {
        self.inner.significance()
    }

    fn svals(&self) -> Vec<f64> {
        self.inner.svals()
    }

    fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error> {
        self.inner.default_tau_sampling_points()
    }

    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<MatsubaraFreq<S>>, Error>
    where
        S: 'static,
    {
        self.inner.default_matsubara_sampling_points(positive_only)
    }

    fn evaluate_tau(&self, tau: &[f64]) -> Result<crate::Matrix<f64>, Error> {
        self.inner.evaluate_tau(tau)
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[MatsubaraFreq<S>],
    ) -> Result<crate::Matrix<Complex<f64>>, Error>
    where
        S: 'static,
    {
        self.inner.evaluate_matsubara(freqs)
    }

    fn evaluate_omega(&self, omega: &[f64]) -> Result<crate::Matrix<f64>, Error> {
        self.inner.evaluate_omega(omega)
    }

    fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error> {
        self.inner.default_omega_sampling_points()
    }
}

/// A bosonic pole at 0 is evaluated through its finite limit, which is known
/// for ypower 0 and 1 only; for another ypower, evaluate_tau and
/// evaluate_matsubara panicked. with_poles reports NotSupported now. Poles
/// away from 0, and fermionic poles (which need no limit), are unaffected.
#[test]
fn test_with_poles_rejects_a_bosonic_zero_pole_of_other_ypower() {
    let kernel = LogisticKernel::new(10.0).unwrap();
    let other = OtherYpowerKernel(kernel);

    let ir_b = FiniteTempBasis::<_, Bosonic>::new(kernel, 10.0, Some(1e-6), None).unwrap();
    let b = WithKernel {
        inner: &ir_b,
        kernel: other,
    };
    for zero in [0.0, -0.0] {
        let err =
            DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&b, vec![-0.5, zero, 0.5])
                .err()
                .unwrap();
        assert!(
            matches!(&err, Error::NotSupported { what } if what.contains("ypower = 2")),
            "{err:?}"
        );
    }
    let dlr =
        DiscreteLehmannRepresentation::<Bosonic>::from_ir_with_poles(&b, vec![-0.5, 0.5]).unwrap();
    assert!(
        dlr.evaluate_tau(&[0.0, 5.0])
            .unwrap()
            .host_data()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );

    let ir_f = FiniteTempBasis::<_, Fermionic>::new(kernel, 10.0, Some(1e-6), None).unwrap();
    let f = WithKernel {
        inner: &ir_f,
        kernel: other,
    };
    let dlr =
        DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&f, vec![-0.5, 0.0, 0.5])
            .unwrap();
    assert!(
        dlr.evaluate_tau(&[0.0, 5.0])
            .unwrap()
            .host_data()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}
