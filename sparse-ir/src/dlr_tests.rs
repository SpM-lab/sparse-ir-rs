//! Tests for DiscreteLehmannRepresentation
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::{
    AbstractKernel, Basis, Bosonic, DiscreteLehmannRepresentation, Error, Fermionic,
    FiniteTempBasis, LogisticKernel, MatsubaraFreq, MatsubaraSampling, RegularizedBoseKernel,
    Statistics, StatisticsType, TauSampling, bosonic_single_pole, giwn_single_pole,
    gtau_single_pole,
};
use mdarray::{DTensor, Shape, Tensor};
use num_complex::Complex;

fn max_relative_error_real(lhs: &Tensor<f64, mdarray::DynRank>, rhs: &DTensor<f64, 2>) -> f64 {
    assert_eq!(lhs.rank(), 2);
    assert_eq!(*rhs.shape(), (lhs.shape().dim(0), lhs.shape().dim(1)));

    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..lhs.shape().dim(0) {
        for j in 0..lhs.shape().dim(1) {
            let a = lhs[&[i, j][..]];
            let b = rhs[[i, j]];
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
    lhs: &Tensor<Complex<f64>, mdarray::DynRank>,
    rhs: &DTensor<Complex<f64>, 2>,
) -> f64 {
    assert_eq!(lhs.rank(), 2);
    assert_eq!(*rhs.shape(), (lhs.shape().dim(0), lhs.shape().dim(1)));

    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..lhs.shape().dim(0) {
        for j in 0..lhs.shape().dim(1) {
            let a = lhs[&[i, j][..]];
            let b = rhs[[i, j]];
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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);

    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis).unwrap();

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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone()).unwrap();

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);

    let tau_values = dlr.evaluate_tau(&[0.0, beta / 3.0, beta]);
    for i in 0..3 {
        assert!(
            tau_values[[i, 2]].is_finite(),
            "tau basis value for zero pole must be finite"
        );
        assert!(
            (tau_values[[i, 2]] + 0.5).abs() < 1e-12,
            "zero-pole tau basis should match the logistic limit"
        );
    }

    let freqs = [
        crate::MatsubaraFreq::<Bosonic>::new(0).unwrap(),
        crate::MatsubaraFreq::<Bosonic>::new(2).unwrap(),
    ];
    let matsubara_values = dlr.evaluate_matsubara(&freqs);
    assert!(
        matsubara_values[[0, 2]].re.is_finite() && matsubara_values[[0, 2]].im.is_finite(),
        "zero-pole Matsubara basis at n=0 must be finite"
    );
    assert!(
        (matsubara_values[[0, 2]].re + 0.5 * beta).abs() < 1e-12,
        "zero-pole Matsubara basis should match the logistic limit"
    );
    assert!(
        matsubara_values[[1, 2]].norm() < 1e-12,
        "zero-pole Matsubara basis should vanish away from n=0"
    );
}

/// Generic test for from_IR_nd/to_IR_nd roundtrip
fn test_dlr_nd_roundtrip_generic<T, S>()
where
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + 'static
        + crate::test_utils::ErrorNorm
        + crate::test_utils::ConvertFromReal,
    S: crate::StatisticsType + 'static,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None);

    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis).unwrap();

    let basis_size = basis.size();

    // Create reference 3D tensor with basis_size at dim=0
    let shape_ref = [basis_size, 3, 4];
    let gl_ref = {
        let mut tensor = Tensor::<T, mdarray::DynRank>::zeros(&shape_ref[..]);
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let mag = ((l + 1) as f64).powi(-2) * (i + j + 1) as f64;
                    tensor[&[l, i, j][..]] = T::from_real(mag);
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
        let g_dlr = dlr.from_ir_nd::<T>(None, &gl_3d, dim);
        let gl_reconst = dlr.to_ir_nd::<T>(None, &g_dlr, dim);

        // Move back to dim=0 for comparison
        let gl_reconst_dim0 = crate::test_utils::movedim(&gl_reconst, dim, 0);

        // Check shape
        assert_eq!(gl_reconst_dim0.rank(), gl_ref.rank());

        // Check roundtrip - compare with reference
        let mut max_error = 0.0;
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let val_orig = gl_ref[&[l, i, j][..]];
                    let val_reconst = gl_reconst_dim0[&[l, i, j][..]];
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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis_ir =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis_ir).unwrap();

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
    let matrix_tau = dlr.evaluate_tau(&tau_points);
    assert_eq!(*matrix_tau.shape(), (tau_points.len(), dlr.size()));

    // Test evaluate_matsubara
    use crate::MatsubaraFreq;
    let freqs = vec![
        MatsubaraFreq::<Fermionic>::new(1).unwrap(),
        MatsubaraFreq::<Fermionic>::new(3).unwrap(),
        MatsubaraFreq::<Fermionic>::new(-1).unwrap(),
    ];
    let matrix_matsu = dlr.evaluate_matsubara(&freqs);
    assert_eq!(*matrix_matsu.shape(), (freqs.len(), dlr.size()));
}

#[test]
fn test_dlr_with_tau_sampling() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis_ir =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);

    // Create DLR
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis_ir).unwrap();

    // Create TauSampling from DLR (using Basis trait)
    let tau_points = basis_ir.default_tau_sampling_points();
    let n_tau_points = tau_points.len();
    let sampling_dlr = TauSampling::<Fermionic>::with_sampling_points(&dlr, tau_points);

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

    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis).unwrap();

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

    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone()).unwrap();

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);

    let tau_values = dlr.evaluate_tau(&[0.0, beta / 3.0, beta]);
    for i in 0..3 {
        assert!(
            tau_values[[i, 2]].is_finite(),
            "tau basis value for zero pole must be finite"
        );
        // -K^B(τ, 0) = -lim ω e^{-τω}/(1 - e^{-βω}) = -1/β
        assert!(
            (tau_values[[i, 2]] + 1.0 / beta).abs() < 1e-12,
            "zero-pole tau basis should match the regularized limit -1/beta"
        );
    }

    let freqs = [
        crate::MatsubaraFreq::<Bosonic>::new(0).unwrap(),
        crate::MatsubaraFreq::<Bosonic>::new(2).unwrap(),
    ];
    let matsubara_values = dlr.evaluate_matsubara(&freqs);
    assert!(
        matsubara_values[[0, 2]].re.is_finite() && matsubara_values[[0, 2]].im.is_finite(),
        "zero-pole Matsubara basis at n=0 must be finite"
    );
    // ω/(iν - ω) at ν = 0 is -1 for every ω, including the limit ω → 0
    assert!(
        (matsubara_values[[0, 2]].re + 1.0).abs() < 1e-12,
        "zero-pole Matsubara basis should match the regularized limit -1"
    );
    assert!(
        matsubara_values[[1, 2]].norm() < 1e-12,
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
    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-10), None);
    let poles = vec![-1.5, -0.4, 0.3, 1.8];
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone()).unwrap();

    let taus = [0.25, 3.7, 8.9];
    let tau_values = dlr.evaluate_tau(&taus);
    for (i, &tau) in taus.iter().enumerate() {
        for (p, &pole) in poles.iter().enumerate() {
            let exact = -pole * (-tau * pole).exp() / (1.0 - (-beta * pole).exp());
            assert!(
                (tau_values[[i, p]] - exact).abs() <= 1e-13 * exact.abs().max(1.0),
                "tau={tau}, pole={pole}: u_p = {}, -K^B = {exact}",
                tau_values[[i, p]]
            );
        }
    }

    let freqs = [0_i64, 2, -6].map(|n| crate::MatsubaraFreq::<Bosonic>::new(n).unwrap());
    let matsubara_values = dlr.evaluate_matsubara(&freqs);
    for (i, freq) in freqs.iter().enumerate() {
        for (p, &pole) in poles.iter().enumerate() {
            let exact = Complex::new(pole, 0.0) / Complex::new(-pole, freq.value(beta));
            assert!(
                (matsubara_values[[i, p]] - exact).norm() <= 1e-13 * exact.norm().max(1.0),
                "n={}, pole={pole}: uhat_p = {}, pole/(iν - pole) = {exact}",
                freq.get_n(),
                matsubara_values[[i, p]]
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
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + From<f64>
        + Copy
        + Default
        + 'static
        + crate::test_utils::ErrorNorm
        + crate::test_utils::ConvertFromReal,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis).unwrap();

    let basis_size = basis.size();

    // Create reference 3D tensor with basis_size at dim=0
    let shape_ref = [basis_size, 3, 4];
    let gl_ref = {
        let mut tensor = Tensor::<T, mdarray::DynRank>::zeros(&shape_ref[..]);
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let mag = ((l + 1) as f64).powi(-2) * (i + j + 1) as f64;
                    tensor[&[l, i, j][..]] = T::from_real(mag);
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
        let g_dlr = dlr.from_ir_nd::<T>(None, &gl_3d, dim);
        let gl_reconst = dlr.to_ir_nd::<T>(None, &g_dlr, dim);

        // Move back to dim=0 for comparison
        let gl_reconst_dim0 = crate::test_utils::movedim(&gl_reconst, dim, 0);

        // Check shape
        assert_eq!(gl_reconst_dim0.rank(), gl_ref.rank());

        // Check roundtrip - compare with reference
        let mut max_error = 0.0;
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let val_orig = gl_ref[&[l, i, j][..]];
                    let val_reconst = gl_reconst_dim0[&[l, i, j][..]];
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

    let kernel = RegularizedBoseKernel::new(lambda);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis).unwrap();

    let tau_points = basis.default_tau_sampling_points();
    let tau_sampling = TauSampling::<Bosonic>::with_sampling_points(&basis, tau_points.clone());

    let matsubara_points = basis.default_matsubara_sampling_points(false);
    let matsubara_sampling =
        MatsubaraSampling::<Bosonic>::with_sampling_points(&basis, matsubara_points.clone());

    let n_poles = dlr.poles.len();
    let dlr_coeffs_2d = DTensor::<f64, 2>::from_fn([n_poles, 1], |idx| {
        let pole = dlr.poles[idx[0]];
        (idx[0] as f64 + 1.0) / (1.0 + pole.abs())
    });
    let dlr_coeffs = dlr_coeffs_2d.clone().into_dyn().to_tensor();

    let ir_coeffs = dlr.to_ir_nd::<f64>(None, &dlr_coeffs, 0);

    let g_tau_ir = tau_sampling.evaluate_nd(None, &ir_coeffs, 0);
    let dlr_tau = dlr.evaluate_tau(&tau_points);
    let g_tau_dlr = DTensor::<f64, 2>::from_fn([tau_points.len(), 1], |idx| {
        let i = idx[0];
        let mut sum = 0.0;
        for p in 0..n_poles {
            sum += dlr_tau[[i, p]] * dlr_coeffs_2d[[p, 0]];
        }
        sum
    });

    let g_iw_ir = matsubara_sampling.evaluate_nd_real(None, &ir_coeffs, 0);
    let dlr_iw = dlr.evaluate_matsubara(&matsubara_points);
    let g_iw_dlr = DTensor::<Complex<f64>, 2>::from_fn([matsubara_points.len(), 1], |idx| {
        let i = idx[0];
        let mut sum = Complex::new(0.0, 0.0);
        for p in 0..n_poles {
            sum += dlr_iw[[i, p]] * dlr_coeffs_2d[[p, 0]];
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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis).unwrap();
    let tau_points = basis.default_tau_sampling_points();
    let tau_sampling = TauSampling::<Fermionic>::with_sampling_points(&dlr, tau_points.clone());

    let expected = DTensor::<f64, 2>::from_fn([tau_points.len(), dlr.poles.len()], |idx| {
        let tau = tau_points[idx[0]];
        let pole = dlr.poles[idx[1]];
        let (tau_norm, sign) = crate::taufuncs::normalize_tau::<Fermionic>(tau, beta);
        let x = 2.0 * tau_norm / beta - 1.0;
        let y = pole / wmax;
        sign * (-kernel.compute(x, y))
    });

    let matrix = tau_sampling.matrix();
    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..tau_points.len() {
        for p in 0..dlr.poles.len() {
            let actual = matrix[[i, p]];
            let reference = expected[[i, p]];
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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis).unwrap();
    let tau_points = basis.default_tau_sampling_points();
    let tau_sampling = TauSampling::<Bosonic>::with_sampling_points(&dlr, tau_points.clone());

    let expected = DTensor::<f64, 2>::from_fn([tau_points.len(), dlr.poles.len()], |idx| {
        let tau = tau_points[idx[0]];
        let pole = dlr.poles[idx[1]];
        let (tau_norm, sign) = crate::taufuncs::normalize_tau::<Bosonic>(tau, beta);
        let x = 2.0 * tau_norm / beta - 1.0;
        let y = pole / wmax;
        sign * (-kernel.compute(x, y))
    });

    let matrix = tau_sampling.matrix();
    let mut max_diff = 0.0_f64;
    let mut max_ref = 0.0_f64;
    for i in 0..tau_points.len() {
        for p in 0..dlr.poles.len() {
            let actual = matrix[[i, p]];
            let reference = expected[[i, p]];
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
            let value = gtau_single_pole::<S>(tau, omega, beta);
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
        let value = gtau_single_pole::<S>(tau, omega, beta);
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
        let g_0 = gtau_single_pole::<S>(0.0, omega, beta);
        let g_beta = gtau_single_pole::<S>(beta, omega, beta);
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
        let quad = rule.piecewise(&edges);
        let gtau: Vec<f64> = quad
            .x
            .iter()
            .map(|&tau| gtau_single_pole::<S>(tau, omega, beta))
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
            let reference = giwn_single_pole::<S>(&freq, omega, beta);
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

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None);
    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis).unwrap();
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
    let dlr_tau = dlr.evaluate_tau(&taus);

    for (p, (&pole, &weight)) in dlr.poles.iter().zip(dlr.pole_weights()).enumerate() {
        // An exact bosonic zero pole is a genuine pole of the unweighted
        // single-pole function; the DLR evaluates it through its finite
        // regularized limit instead.
        if S::STATISTICS == Statistics::Bosonic && pole == 0.0 {
            continue;
        }
        for (i, &tau) in taus.iter().enumerate() {
            let expected = gtau_single_pole::<S>(tau, pole, beta) * weight;
            let actual = dlr_tau[[i, p]];
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
            bosonic_single_pole(tau, 0.0, beta),
            f64::NEG_INFINITY,
            "omega = +0.0 at tau = {}",
            tau
        );
        assert_eq!(
            bosonic_single_pole(tau, -0.0, beta),
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

    fn default_tau_sampling_points(&self) -> Vec<f64> {
        self.inner.default_tau_sampling_points()
    }

    fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>>
    where
        S: 'static,
    {
        self.inner.default_matsubara_sampling_points(positive_only)
    }

    fn evaluate_tau(&self, tau: &[f64]) -> DTensor<f64, 2> {
        self.inner.evaluate_tau(tau)
    }

    fn evaluate_matsubara(&self, freqs: &[MatsubaraFreq<S>]) -> DTensor<Complex<f64>, 2>
    where
        S: 'static,
    {
        self.inner.evaluate_matsubara(freqs)
    }

    fn evaluate_omega(&self, omega: &[f64]) -> DTensor<f64, 2> {
        self.inner.evaluate_omega(omega)
    }

    fn default_omega_sampling_points(&self) -> Vec<f64> {
        let mut poles = self.inner.default_omega_sampling_points();
        poles.truncate(self.n_poles);
        poles
    }
}

fn check_dlr_new_insufficient_default_poles<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 1.0;
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(1e-6), None);
    let basis_size = basis.size();
    assert_eq!(basis.default_omega_sampling_points().len(), basis_size);

    // Exactly as many default poles as basis functions is enough.
    let enough = TruncatedDefaultPoles {
        inner: &basis,
        n_poles: basis_size,
    };
    let dlr = DiscreteLehmannRepresentation::<S>::new(&enough).unwrap();
    assert_eq!(dlr.poles.len(), basis_size);

    // One pole fewer than the basis size is rejected with a typed error
    // carrying both counts, instead of a panic.
    let too_few = TruncatedDefaultPoles {
        inner: &basis,
        n_poles: basis_size - 1,
    };
    let err = DiscreteLehmannRepresentation::<S>::new(&too_few)
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
    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Fermionic>::new(kernel, beta, Some(1e-6), None);
    // The default poles are sufficient, so `new` reaches the statistics check.
    assert!(basis.default_omega_sampling_points().len() >= basis.size());

    let err = DiscreteLehmannRepresentation::<Fermionic>::with_poles(&basis, vec![-2.0, 0.5, 3.0])
        .err()
        .expect("with_poles must reject RegularizedBoseKernel with fermionic statistics");
    assert_eq!(err, Error::KernelStatisticsMismatch);

    let err = DiscreteLehmannRepresentation::<Fermionic>::new(&basis)
        .err()
        .expect("new must reject RegularizedBoseKernel with fermionic statistics");
    assert_eq!(err, Error::KernelStatisticsMismatch);

    // The same kernel with bosonic statistics is supported.
    let bosonic =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(1e-6), None);
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&bosonic, vec![-2.0, 0.5, 3.0])
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
        LogisticKernel::new(10.0),
        1.0,
        Some(1e-6),
        None,
    );
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis).unwrap();
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
        let gl = Tensor::<f64, mdarray::DynRank>::zeros(&with_target(l)[..]);
        let g_dlr = dlr.from_ir_nd::<f64>(None, &gl, dim);
        assert_eq!(g_dlr.shape().dims(), &with_target(n_poles)[..]);
        let back = dlr.to_ir_nd::<f64>(None, &g_dlr, dim);
        assert_eq!(back.shape().dims(), &with_target(l)[..]);

        let gl_z = Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&with_target(l)[..]);
        let g_dlr_z = dlr.from_ir_nd::<Complex<f64>>(None, &gl_z, dim);
        assert_eq!(g_dlr_z.shape().dims(), &with_target(n_poles)[..]);
        let back_z = dlr.to_ir_nd::<Complex<f64>>(None, &g_dlr_z, dim);
        assert_eq!(back_z.shape().dims(), &with_target(l)[..]);
    }
}

/// A DLR without poles has no basis functions. Before the fix `with_poles`
/// panicked with an index out of bounds (evaluating V at no poles); a
/// sampling built on it would have segfaulted in the SVD (mdarray#21).
#[test]
fn test_dlr_with_no_poles_is_an_error() {
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(10.0),
        1.0,
        Some(1e-6),
        None,
    );
    let result = DiscreteLehmannRepresentation::<Fermionic>::with_poles(&basis, vec![]);
    assert!(matches!(result, Err(Error::EmptyInput { name: "poles" })));
    assert_eq!(
        Error::EmptyInput { name: "poles" }.to_string(),
        "poles must not be empty"
    );
}
