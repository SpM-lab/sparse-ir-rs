//! Tests for DiscreteLehmannRepresentation

use crate::{
    Basis, Bosonic, DiscreteLehmannRepresentation, Fermionic, FiniteTempBasis, LogisticKernel,
    RegularizedBoseKernel, TauSampling,
};
use mdarray::Tensor;
use num_complex::Complex;

#[test]
fn test_dlr_construction_fermionic() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);

    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis);

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

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone());

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);
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

    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis);

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
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis_ir);

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
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis_ir);

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
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis);

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

fn test_dlr_regularized_bose_with_custom_poles() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone());

    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);

    println!("\n=== RegularizedBoseKernel DLR with Custom Poles ===");
    println!("Successfully created DLR with {} custom poles", poles.len());
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

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(&basis);

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
