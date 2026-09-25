//! Integration tests for sparseir-capi
//!
//! Port of libsparseir's cinterface_integration.cxx
//! Tests the complete workflow: IR basis → sampling → DLR

use num_complex::Complex64;
use rstest::rstest;
use sparse_ir_capi::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INPUT_DIMENSION_MISMATCH, SPIR_INTERNAL_ERROR,
    SPIR_INVALID_DIMENSION, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR,
    SPIR_STATISTICS_FERMIONIC, spir_basis, spir_basis_get_default_matsus,
    spir_basis_get_default_taus, spir_basis_get_n_default_matsus, spir_basis_get_n_default_taus,
    spir_basis_get_size, spir_basis_get_svals, spir_basis_get_u, spir_basis_get_uhat,
    spir_basis_new, spir_basis_release, spir_dlr_get_npoles, spir_dlr_get_poles, spir_dlr_new,
    spir_dlr_new_with_poles, spir_dlr2ir_dd, spir_dlr2ir_zz, spir_funcs_eval,
    spir_funcs_eval_matsu, spir_funcs_release, spir_ir2dlr_dd, spir_ir2dlr_zz, spir_kernel,
    spir_kernel_release, spir_logistic_kernel_new, spir_matsu_sampling_new,
    spir_reg_bose_kernel_new, spir_sampling, spir_sampling_eval_dd, spir_sampling_eval_dz,
    spir_sampling_eval_zz, spir_sampling_fit_dd, spir_sampling_fit_zd, spir_sampling_fit_zz,
    spir_sampling_release, spir_sve_result, spir_sve_result_get_size, spir_sve_result_get_svals,
    spir_sve_result_new, spir_sve_result_release, spir_tau_sampling_new,
};
use std::sync::{Arc, Barrier};
use std::thread;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a logistic kernel
fn create_logistic_kernel(lambda: f64) -> *mut spir_kernel {
    unsafe {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());
        kernel
    }
}

/// Create an IR basis
fn create_ir_basis(
    statistics: i32,
    beta: f64,
    wmax: f64,
    epsilon: f64,
) -> (*mut spir_kernel, *mut spir_sve_result, *mut spir_basis) {
    unsafe {
        let kernel = create_logistic_kernel(beta * wmax);

        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, epsilon, -1, -1, 0, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            statistics,
            beta,
            wmax,
            epsilon,
            kernel,
            sve,
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        (kernel, sve, basis)
    }
}

/// Create a regularized-bose IR basis
fn create_regularized_bose_ir_basis(
    statistics: i32,
    beta: f64,
    wmax: f64,
    epsilon: f64,
) -> (*mut spir_kernel, *mut spir_sve_result, *mut spir_basis) {
    unsafe {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(beta * wmax, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, epsilon, -1, -1, 0, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            statistics,
            beta,
            wmax,
            epsilon,
            kernel,
            sve,
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        (kernel, sve, basis)
    }
}

/// Get basis size
fn get_basis_size(basis: *const spir_basis) -> i32 {
    unsafe {
        let mut size = 0;
        let status = spir_basis_get_size(basis, &mut size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        size
    }
}

/// Get default tau sampling points
fn get_default_tau_points(basis: *const spir_basis) -> Vec<f64> {
    unsafe {
        let mut num_points = 0;
        let status = spir_basis_get_n_default_taus(basis, &mut num_points);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut points = vec![0.0; num_points as usize];
        let status = spir_basis_get_default_taus(basis, points.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        points
    }
}

/// Get default Matsubara sampling points
fn get_default_matsubara_points(basis: *const spir_basis, positive_only: bool) -> Vec<i64> {
    unsafe {
        let mut num_points = 0;
        let status = spir_basis_get_n_default_matsus(basis, positive_only, &mut num_points);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut points = vec![0; num_points as usize];
        let status = spir_basis_get_default_matsus(basis, positive_only, points.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        points
    }
}

// ============================================================================
// Tests
// ============================================================================

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_integration_1d_fermionic(#[case] epsilon: f64) {
    let beta = 100.0;
    let wmax = 2.0;
    let tol = 10.0 * epsilon;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis);
        println!("IR basis size: {}", basis_size);

        // Get tau sampling points
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;
        println!("Tau sampling: {} points", num_tau);

        // Create tau sampling
        let mut status = SPIR_INTERNAL_ERROR;
        let tau_sampling = spir_tau_sampling_new(basis, num_tau, tau_points.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!tau_sampling.is_null());

        // Get Matsubara sampling points
        let matsu_points = get_default_matsubara_points(basis, false);
        let num_matsu = matsu_points.len() as i32;
        println!("Matsubara sampling: {} points", num_matsu);

        // Create Matsubara sampling
        let mut status = SPIR_INTERNAL_ERROR;
        let matsu_sampling =
            spir_matsu_sampling_new(basis, false, num_matsu, matsu_points.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!matsu_sampling.is_null());

        // Create test coefficients (simple case: all ones)
        let coeffs = vec![1.0; basis_size as usize];

        // Test tau sampling: evaluate
        let mut gtau = vec![0.0; num_tau as usize];
        let dims = [basis_size];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ Tau evaluate succeeded");

        // Test tau sampling: fit (roundtrip)
        let mut coeffs_fit = vec![0.0; basis_size as usize];
        let dims_tau = [num_tau];
        let status = spir_sampling_fit_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims_tau.as_ptr(),
            0,
            gtau.as_ptr(),
            coeffs_fit.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check roundtrip error
        let max_error: f64 = coeffs
            .iter()
            .zip(&coeffs_fit)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);
        println!("✓ Tau fit roundtrip error: {:.2e}", max_error);
        assert!(max_error < tol);

        // Test Matsubara sampling: evaluate
        let mut giw = vec![Complex64::new(0.0, 0.0); num_matsu as usize];
        let status = spir_sampling_eval_dz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            giw.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ Matsubara evaluate succeeded");

        // Test Matsubara sampling: fit (roundtrip)
        let mut coeffs_fit_matsu = vec![0.0; basis_size as usize];
        let dims_matsu = [num_matsu];
        let status = spir_sampling_fit_zd(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims_matsu.as_ptr(),
            0,
            giw.as_ptr(),
            coeffs_fit_matsu.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check roundtrip error (real part only for fit_zd)
        let max_error_matsu: f64 = coeffs
            .iter()
            .zip(&coeffs_fit_matsu)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);
        println!("✓ Matsubara fit roundtrip error: {:.2e}", max_error_matsu);
        assert!(max_error_matsu < tol);

        // Cleanup
        spir_sampling_release(matsu_sampling);
        spir_sampling_release(tau_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_dlr_conversion_1d(#[case] epsilon: f64) {
    let beta = 100.0;
    let wmax = 2.0;
    let tol = 10.0 * epsilon;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis);

        // Create DLR
        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr.is_null());

        // Get number of poles
        let mut npoles = 0;
        let status = spir_dlr_get_npoles(dlr, &mut npoles);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("DLR has {} poles (IR basis size: {})", npoles, basis_size);
        assert!(npoles >= basis_size);

        // Get poles
        let mut poles = vec![0.0; npoles as usize];
        let status = spir_dlr_get_poles(dlr, poles.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Generate random DLR coefficients
        let dlr_coeffs: Vec<f64> = (0..npoles)
            .map(|i| {
                let pole: f64 = poles[i as usize];
                (i as f64 * 0.1) * pole.abs().sqrt()
            })
            .collect();

        // Test DLR → IR conversion
        let mut ir_coeffs = vec![0.0; basis_size as usize];
        let dlr_dims = [npoles];
        let status = spir_dlr2ir_dd(
            dlr,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dlr_dims.as_ptr(),
            0,
            dlr_coeffs.as_ptr(),
            ir_coeffs.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ DLR → IR conversion succeeded");

        // Test IR → DLR conversion (roundtrip)
        let mut dlr_coeffs_reconst = vec![0.0; npoles as usize];
        let ir_dims = [basis_size];
        let status = spir_ir2dlr_dd(
            dlr,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            ir_dims.as_ptr(),
            0,
            ir_coeffs.as_ptr(),
            dlr_coeffs_reconst.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check roundtrip error
        let max_error = dlr_coeffs
            .iter()
            .zip(&dlr_coeffs_reconst)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("✓ DLR roundtrip error: {:.2e}", max_error);
        assert!(max_error < tol);

        // Test DLR funcs evaluation
        let mut u_status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut u_status);
        assert_eq!(u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_u.is_null());

        let mut uhat_status = SPIR_INTERNAL_ERROR;
        let dlr_uhat = spir_basis_get_uhat(dlr, &mut uhat_status);
        assert_eq!(uhat_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_uhat.is_null());

        // Evaluate DLR u at tau=0.5
        let tau = 0.5;
        let mut u_values = vec![0.0; npoles as usize];
        let status = spir_funcs_eval(dlr_u, tau, u_values.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!(
            "✓ DLR u evaluation at τ={}: first value = {:.6e}",
            tau, u_values[0]
        );

        // Evaluate DLR uhat at n=1
        let n = 1i64;
        let mut uhat_values = vec![Complex64::new(0.0, 0.0); npoles as usize];
        let status = spir_funcs_eval_matsu(dlr_uhat, n, uhat_values.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!(
            "✓ DLR uhat evaluation at n={}: first value = {:.6e}",
            n,
            uhat_values[0].norm()
        );

        // Cleanup
        spir_funcs_release(dlr_uhat);
        spir_funcs_release(dlr_u);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_dlr_sampling_integration(#[case] epsilon: f64) {
    let beta = 1000.0; // Match C++ test
    let wmax = 2.0;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis);

        // Get tau points
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;

        // Create DLR
        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);

        let mut npoles = 0;
        spir_dlr_get_npoles(dlr, &mut npoles);

        // Get poles
        let mut poles = vec![0.0; npoles as usize];
        spir_dlr_get_poles(dlr, poles.as_mut_ptr());

        // Get DLR u funcs
        let mut dlr_u_status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut dlr_u_status);
        assert_eq!(dlr_u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_u.is_null());

        // Get IR u funcs
        let mut ir_u_status = SPIR_INTERNAL_ERROR;
        let ir_u = spir_basis_get_u(basis, &mut ir_u_status);
        assert_eq!(ir_u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!ir_u.is_null());

        // Create DLR coefficients
        let dlr_coeffs: Vec<f64> = (0..npoles).map(|i| (i as f64 + 1.0) * 0.1).collect();

        // DLR → IR
        let mut ir_coeffs = vec![0.0; basis_size as usize];
        let dlr_dims = [npoles];
        let status = spir_dlr2ir_dd(
            dlr,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dlr_dims.as_ptr(),
            0,
            dlr_coeffs.as_ptr(),
            ir_coeffs.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Evaluate DLR coeffs using DLR u funcs
        let mut gtau_from_dlr = vec![0.0; num_tau as usize];
        for (i, &tau) in tau_points.iter().enumerate() {
            let mut u_values = vec![0.0; npoles as usize];
            let status = spir_funcs_eval(dlr_u, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // g(tau) = sum_l coeffs[l] * u[l](tau)
            gtau_from_dlr[i] = dlr_coeffs.iter().zip(&u_values).map(|(c, u)| c * u).sum();
        }

        // Evaluate IR coeffs using IR u funcs
        let mut gtau_from_ir = vec![0.0; num_tau as usize];
        for (i, &tau) in tau_points.iter().enumerate() {
            let mut u_values = vec![0.0; basis_size as usize];
            let status = spir_funcs_eval(ir_u, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            gtau_from_ir[i] = ir_coeffs.iter().zip(&u_values).map(|(c, u)| c * u).sum();
        }

        // Compare results (DLR and IR should give same Green's function)
        let max_diff: f64 = gtau_from_dlr
            .iter()
            .zip(&gtau_from_ir)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);
        println!(
            "✓ Max difference between DLR and IR evaluation: {:.2e}",
            max_diff
        );
        assert!(max_diff < 1e-4, "DLR and IR should give similar results");

        // Cleanup
        spir_funcs_release(ir_u);
        spir_funcs_release(dlr_u);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_fermionic_dlr_tau_sampling_matches_dlr_funcs() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    unsafe {
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;

        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);

        let mut npoles = 0;
        spir_dlr_get_npoles(dlr, &mut npoles);
        assert!(npoles > 0);

        let mut sampling_status = SPIR_INTERNAL_ERROR;
        let tau_sampling =
            spir_tau_sampling_new(dlr, num_tau, tau_points.as_ptr(), &mut sampling_status);
        assert_eq!(sampling_status, SPIR_COMPUTATION_SUCCESS);

        let mut u_status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut u_status);
        assert_eq!(u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_u.is_null());

        let coeffs: Vec<f64> = (0..npoles).map(|i| (i as f64 + 1.0) * 0.1).collect();

        let mut gtau_from_funcs = vec![0.0; num_tau as usize];
        for (i, &tau) in tau_points.iter().enumerate() {
            let mut u_values = vec![0.0; npoles as usize];
            let status = spir_funcs_eval(dlr_u, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            gtau_from_funcs[i] = coeffs.iter().zip(&u_values).map(|(c, u)| c * u).sum();
        }

        let mut gtau_from_sampling = vec![0.0; num_tau as usize];
        let dims = [npoles];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau_from_sampling.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let max_diff = gtau_from_funcs
            .iter()
            .zip(&gtau_from_sampling)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "Fermionic DLR tau funcs and tau sampling should match, max diff = {:.3e}",
            max_diff
        );

        spir_funcs_release(dlr_u);
        spir_sampling_release(tau_sampling);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_bosonic_dlr_tau_sampling_matches_dlr_funcs() {
    let beta = 100.0;
    let wmax = 2.0;
    let epsilon = 1e-7;

    unsafe {
        let (kernel, sve, basis) = create_regularized_bose_ir_basis(0, beta, wmax, epsilon);
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;

        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);

        let mut npoles = 0;
        spir_dlr_get_npoles(dlr, &mut npoles);
        assert!(npoles > 0);

        let mut sampling_status = SPIR_INTERNAL_ERROR;
        let tau_sampling =
            spir_tau_sampling_new(dlr, num_tau, tau_points.as_ptr(), &mut sampling_status);
        assert_eq!(sampling_status, SPIR_COMPUTATION_SUCCESS);

        let mut u_status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut u_status);
        assert_eq!(u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_u.is_null());

        let coeffs: Vec<f64> = (0..npoles).map(|i| (i as f64 + 1.0) * 0.1).collect();

        let mut gtau_from_funcs = vec![0.0; num_tau as usize];
        for (i, &tau) in tau_points.iter().enumerate() {
            let mut u_values = vec![0.0; npoles as usize];
            let status = spir_funcs_eval(dlr_u, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            gtau_from_funcs[i] = coeffs.iter().zip(&u_values).map(|(c, u)| c * u).sum();
        }

        let mut gtau_from_sampling = vec![0.0; num_tau as usize];
        let dims = [npoles];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau_from_sampling.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let max_diff = gtau_from_funcs
            .iter()
            .zip(&gtau_from_sampling)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "DLR tau funcs and tau sampling should match, max diff = {:.3e}",
            max_diff
        );

        spir_funcs_release(dlr_u);
        spir_sampling_release(tau_sampling);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_bosonic_logistic_dlr_tau_sampling_matches_dlr_funcs() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    unsafe {
        let (kernel, sve, basis) = create_ir_basis(0, beta, wmax, epsilon);
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;

        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);

        let mut npoles = 0;
        spir_dlr_get_npoles(dlr, &mut npoles);
        assert!(npoles > 0);

        let mut sampling_status = SPIR_INTERNAL_ERROR;
        let tau_sampling =
            spir_tau_sampling_new(dlr, num_tau, tau_points.as_ptr(), &mut sampling_status);
        assert_eq!(sampling_status, SPIR_COMPUTATION_SUCCESS);

        let mut u_status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut u_status);
        assert_eq!(u_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_u.is_null());

        let coeffs: Vec<f64> = (0..npoles).map(|i| (i as f64 + 1.0) * 0.1).collect();

        let mut gtau_from_funcs = vec![0.0; num_tau as usize];
        for (i, &tau) in tau_points.iter().enumerate() {
            let mut u_values = vec![0.0; npoles as usize];
            let status = spir_funcs_eval(dlr_u, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert!(
                u_values.iter().all(|value| value.is_finite()),
                "DLR tau funcs must stay finite at tau = {}",
                tau
            );
            gtau_from_funcs[i] = coeffs.iter().zip(&u_values).map(|(c, u)| c * u).sum();
            assert!(
                gtau_from_funcs[i].is_finite(),
                "DLR tau funcs produced non-finite sampled value at tau = {}",
                tau
            );
        }

        let mut gtau_from_sampling = vec![0.0; num_tau as usize];
        let dims = [npoles];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau_from_sampling.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(
            gtau_from_sampling.iter().all(|value| value.is_finite()),
            "DLR tau sampling must stay finite for logistic bosons"
        );

        let max_diff = gtau_from_funcs
            .iter()
            .zip(&gtau_from_sampling)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "Logistic bosonic DLR tau funcs and tau sampling should match, max diff = {:.3e}",
            max_diff
        );

        spir_funcs_release(dlr_u);
        spir_sampling_release(tau_sampling);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_bosonic_dlr_matsubara_sampling_matches_dlr_funcs() {
    let beta = 100.0;
    let wmax = 2.0;
    let epsilon = 1e-7;

    unsafe {
        let (kernel, sve, basis) = create_regularized_bose_ir_basis(0, beta, wmax, epsilon);
        let matsu_points = get_default_matsubara_points(basis, false);
        let num_matsu = matsu_points.len() as i32;

        let mut dlr_status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut dlr_status);
        assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);

        let mut npoles = 0;
        spir_dlr_get_npoles(dlr, &mut npoles);
        assert!(npoles > 0);

        let mut sampling_status = SPIR_INTERNAL_ERROR;
        let matsu_sampling = spir_matsu_sampling_new(
            dlr,
            false,
            num_matsu,
            matsu_points.as_ptr(),
            &mut sampling_status,
        );
        assert_eq!(sampling_status, SPIR_COMPUTATION_SUCCESS);

        let mut uhat_status = SPIR_INTERNAL_ERROR;
        let dlr_uhat = spir_basis_get_uhat(dlr, &mut uhat_status);
        assert_eq!(uhat_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!dlr_uhat.is_null());

        let coeffs: Vec<f64> = (0..npoles).map(|i| (i as f64 + 1.0) * 0.1).collect();

        let mut giw_from_funcs = vec![Complex64::new(0.0, 0.0); num_matsu as usize];
        for (i, &n) in matsu_points.iter().enumerate() {
            let mut uhat_values = vec![Complex64::new(0.0, 0.0); npoles as usize];
            let status = spir_funcs_eval_matsu(dlr_uhat, n, uhat_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            giw_from_funcs[i] = coeffs.iter().zip(&uhat_values).map(|(c, u)| *c * *u).sum();
        }

        let mut giw_from_sampling = vec![Complex64::new(0.0, 0.0); num_matsu as usize];
        let dims = [npoles];
        let status = spir_sampling_eval_dz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            giw_from_sampling.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let max_diff = giw_from_funcs
            .iter()
            .zip(&giw_from_sampling)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-8,
            "DLR Matsubara funcs and Matsubara sampling should match, max diff = {:.3e}",
            max_diff
        );

        spir_funcs_release(dlr_uhat);
        spir_sampling_release(matsu_sampling);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_regularized_bose_svals_and_dlr_funcs_match_physical_kernel() {
    // RegularizedBoseKernel is K^B(τ, ω) = ω e^{-τω}/(1 - e^{-βω}) in physical
    // units (irbasis paper, Chikano et al., CPC 240, 181 (2019),
    // arXiv:1807.05237, Eq. (3)), so S_l = sqrt(β wmax³/2) s_l (Eq. (25)), the
    // DLR τ functions are -K^B(τ, ω_p) and the Matsubara functions are
    // ω_p/(iν - ω_p) (Eq. (16)); the ω_p = 0 limits are -1/β and -δ_{n,0}.
    // wmax ≠ 1 exposes any extra power of wmax.
    let beta = 10.0;
    let wmax = 2.0;
    let epsilon = 1e-10;
    let poles = [-1.5, 0.0, 1.8];
    let k_b = |tau: f64, w: f64| -> f64 {
        if w == 0.0 {
            1.0 / beta
        } else {
            w * (-tau * w).exp() / (1.0 - (-beta * w).exp())
        }
    };

    unsafe {
        let (kernel, sve, basis) = create_regularized_bose_ir_basis(0, beta, wmax, epsilon);

        let size = get_basis_size(basis) as usize;
        let mut basis_svals = vec![0.0; size];
        let status = spir_basis_get_svals(basis, basis_svals.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut n_sve = 0;
        let status = spir_sve_result_get_size(sve, &mut n_sve);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut sve_svals = vec![0.0; n_sve as usize];
        let status = spir_sve_result_get_svals(sve, sve_svals.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let scale = (beta * wmax.powi(3) / 2.0).sqrt();
        for l in 0..size {
            let expected = scale * sve_svals[l];
            assert!(
                (basis_svals[l] - expected).abs() <= 1e-14 * expected,
                "l={l}: S_l = {}, sqrt(beta wmax^3/2) s_l = {expected}",
                basis_svals[l]
            );
        }

        let mut status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new_with_poles(basis, poles.len() as i32, poles.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut status = SPIR_INTERNAL_ERROR;
        let dlr_u = spir_basis_get_u(dlr, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut status = SPIR_INTERNAL_ERROR;
        let dlr_uhat = spir_basis_get_uhat(dlr, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        for &tau in &[0.25, 3.7, 8.9] {
            let mut values = vec![0.0; poles.len()];
            let status = spir_funcs_eval(dlr_u, tau, values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            for (p, &pole) in poles.iter().enumerate() {
                let exact = -k_b(tau, pole);
                assert!(
                    (values[p] - exact).abs() <= 1e-13 * exact.abs().max(1.0),
                    "tau={tau}, pole={pole}: u_p = {}, -K^B = {exact}",
                    values[p]
                );
            }
        }

        for n in [0_i64, 2, -6] {
            let mut values = vec![Complex64::new(0.0, 0.0); poles.len()];
            let status = spir_funcs_eval_matsu(dlr_uhat, n, values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            let iv = Complex64::new(0.0, n as f64 * std::f64::consts::PI / beta);
            for (p, &pole) in poles.iter().enumerate() {
                let exact = if pole == 0.0 {
                    Complex64::new(if n == 0 { -1.0 } else { 0.0 }, 0.0)
                } else {
                    Complex64::new(pole, 0.0) / (iv - pole)
                };
                assert!(
                    (values[p] - exact).norm() <= 1e-13 * exact.norm().max(1.0),
                    "n={n}, pole={pole}: uhat_p = {}, pole/(iν - pole) = {exact}",
                    values[p]
                );
            }
        }

        spir_funcs_release(dlr_uhat);
        spir_funcs_release(dlr_u);
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_column_major_order(#[case] epsilon: f64) {
    let beta = 50.0;
    let wmax = 2.0;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis);

        // Get tau sampling points
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len() as i32;

        // Create tau sampling
        let mut status = SPIR_INTERNAL_ERROR;
        let tau_sampling = spir_tau_sampling_new(basis, num_tau, tau_points.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Test coefficients
        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.1).collect();

        // Evaluate with row-major
        let mut gtau_row = vec![0.0; num_tau as usize];
        let dims = [basis_size];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau_row.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Evaluate with column-major
        let mut gtau_col = vec![0.0; num_tau as usize];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_COLUMN_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            gtau_col.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Results should be identical for 1D
        let max_diff: f64 = gtau_row
            .iter()
            .zip(&gtau_col)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);
        println!("✓ Row vs Column-major difference (1D): {:.2e}", max_diff);
        assert!(max_diff < 1e-14);

        // Cleanup
        spir_sampling_release(tau_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_2d_tensor_operations(#[case] epsilon: f64) {
    let beta = 50.0;
    let wmax = 2.0;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis) as usize;

        // Get tau sampling points
        let tau_points = get_default_tau_points(basis);
        let num_tau = tau_points.len();

        // Create tau sampling
        let mut status = SPIR_INTERNAL_ERROR;
        let tau_sampling =
            spir_tau_sampling_new(basis, num_tau as i32, tau_points.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Test 2D array (batch_size=3, target_dim=0)
        let batch_size = 3;
        let total_size = basis_size * batch_size;

        // Create 2D coefficients: [basis_size, batch_size]
        let coeffs_2d: Vec<f64> = (0..total_size).map(|i| (i as f64 + 1.0) * 0.1).collect();

        // Evaluate with row-major (dims = [basis_size, batch_size], target_dim=0)
        let mut gtau_2d = vec![0.0; num_tau * batch_size];
        let dims = [basis_size as i32, batch_size as i32];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            2,
            dims.as_ptr(),
            0,
            coeffs_2d.as_ptr(),
            gtau_2d.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ 2D evaluation succeeded (row-major)");

        // Fit back
        let mut coeffs_fit_2d = vec![0.0; total_size];
        let dims_tau = [num_tau as i32, batch_size as i32];
        let status = spir_sampling_fit_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            2,
            dims_tau.as_ptr(),
            0,
            gtau_2d.as_ptr(),
            coeffs_fit_2d.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check roundtrip
        let max_error = coeffs_2d
            .iter()
            .zip(&coeffs_fit_2d)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("✓ 2D roundtrip error: {:.2e}", max_error);
        assert!(max_error < 1e-10);

        // Test with target_dim=1
        let dims_swapped = [batch_size as i32, basis_size as i32];

        // Need to transpose input data for target_dim=1
        let mut coeffs_2d_transposed = vec![0.0; total_size];
        for i in 0..basis_size {
            for j in 0..batch_size {
                coeffs_2d_transposed[j * basis_size + i] = coeffs_2d[i * batch_size + j];
            }
        }

        let mut gtau_2d_t1 = vec![0.0; num_tau * batch_size];
        let status = spir_sampling_eval_dd(
            tau_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            2,
            dims_swapped.as_ptr(),
            1,
            coeffs_2d_transposed.as_ptr(),
            gtau_2d_t1.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ 2D evaluation with target_dim=1 succeeded");

        // Cleanup
        spir_sampling_release(tau_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[rstest]
#[case(1e-6)]
#[case(1e-9)]
fn test_complex_coefficients(#[case] epsilon: f64) {
    let beta = 50.0;
    let wmax = 2.0;

    unsafe {
        // Create IR basis (Fermionic)
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis) as usize;

        // Get Matsubara points
        let matsu_points = get_default_matsubara_points(basis, false);
        let num_matsu = matsu_points.len();

        // Create Matsubara sampling
        let mut status = SPIR_INTERNAL_ERROR;
        let matsu_sampling = spir_matsu_sampling_new(
            basis,
            false,
            num_matsu as i32,
            matsu_points.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Complex coefficients
        let coeffs: Vec<Complex64> = (0..basis_size)
            .map(|i| Complex64::new((i as f64 + 1.0) * 0.1, (i as f64 + 1.0) * 0.05))
            .collect();

        // Evaluate
        let mut giw = vec![Complex64::new(0.0, 0.0); num_matsu];
        let dims = [basis_size as i32];
        let status = spir_sampling_eval_zz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            giw.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        println!("✓ Complex evaluation succeeded");

        // Fit
        let mut coeffs_fit = vec![Complex64::new(0.0, 0.0); basis_size];
        let dims_matsu = [num_matsu as i32];
        let status = spir_sampling_fit_zz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims_matsu.as_ptr(),
            0,
            giw.as_ptr(),
            coeffs_fit.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check roundtrip
        let max_error = coeffs
            .iter()
            .zip(&coeffs_fit)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        println!("✓ Complex roundtrip error: {:.2e}", max_error);
        assert!(max_error < 1e-10);

        // Cleanup
        spir_sampling_release(matsu_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_concurrent_matsubara_fit_zz_is_thread_safe() {
    let beta = 0.7;
    let wmax = 8.0;
    let epsilon = 1e-2;

    unsafe {
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis) as usize;
        let matsu_points = get_default_matsubara_points(basis, false);
        let num_matsu = matsu_points.len();

        let mut status = SPIR_INTERNAL_ERROR;
        let matsu_sampling = spir_matsu_sampling_new(
            basis,
            false,
            num_matsu as i32,
            matsu_points.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!matsu_sampling.is_null());

        let coeffs: Vec<Complex64> = (0..basis_size)
            .map(|i| Complex64::new((i as f64 + 1.0) * 0.1, (i as f64 + 1.0) * 0.05))
            .collect();

        let mut giw = vec![Complex64::new(0.0, 0.0); num_matsu];
        let dims = [basis_size as i32];
        let status = spir_sampling_eval_zz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            giw.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let barrier = Arc::new(Barrier::new(8));
        let sampling_addr = matsu_sampling as usize;
        let giw = Arc::new(giw);

        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let barrier = Arc::clone(&barrier);
                let giw = Arc::clone(&giw);
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..20 {
                        let mut coeffs_fit = vec![Complex64::new(0.0, 0.0); basis_size];
                        let dims_matsu = [num_matsu as i32];
                        let status = unsafe {
                            spir_sampling_fit_zz(
                                sampling_addr as *const _,
                                std::ptr::null(),
                                SPIR_ORDER_ROW_MAJOR,
                                1,
                                dims_matsu.as_ptr(),
                                0,
                                giw.as_ptr(),
                                coeffs_fit.as_mut_ptr(),
                            )
                        };
                        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                    }
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });

        spir_sampling_release(matsu_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_concurrent_matsubara_fit_zd_positive_only_is_thread_safe() {
    let beta = 0.7;
    let wmax = 8.0;
    let epsilon = 1e-2;

    unsafe {
        let (kernel, sve, basis) = create_ir_basis(1, beta, wmax, epsilon);
        let basis_size = get_basis_size(basis) as usize;
        let matsu_points = get_default_matsubara_points(basis, true);
        let num_matsu = matsu_points.len();

        let mut status = SPIR_INTERNAL_ERROR;
        let matsu_sampling = spir_matsu_sampling_new(
            basis,
            true,
            num_matsu as i32,
            matsu_points.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!matsu_sampling.is_null());

        let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64 + 1.0) * 0.25).collect();

        let mut giw = vec![Complex64::new(0.0, 0.0); num_matsu];
        let dims = [basis_size as i32];
        let status = spir_sampling_eval_dz(
            matsu_sampling,
            std::ptr::null(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            dims.as_ptr(),
            0,
            coeffs.as_ptr(),
            giw.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let barrier = Arc::new(Barrier::new(8));
        let sampling_addr = matsu_sampling as usize;
        let giw = Arc::new(giw);

        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let barrier = Arc::clone(&barrier);
                let giw = Arc::clone(&giw);
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..20 {
                        let mut coeffs_fit = vec![0.0; basis_size];
                        let dims_matsu = [num_matsu as i32];
                        let status = unsafe {
                            spir_sampling_fit_zd(
                                sampling_addr as *const _,
                                std::ptr::null(),
                                SPIR_ORDER_ROW_MAJOR,
                                1,
                                dims_matsu.as_ptr(),
                                0,
                                giw.as_ptr(),
                                coeffs_fit.as_mut_ptr(),
                            )
                        };
                        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                    }
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });

        spir_sampling_release(matsu_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

// ============================================================================
// Validation of `input_dims` (SpM-lab/sparse-ir-rs#245)
// ============================================================================

/// Written to output buffers before each call and checked afterwards.
const OUT_SENTINEL: f64 = -12345.0;

/// Output elements past a valid result; they must never be written.
const OUT_SLACK: usize = 64;

/// Malformed `(input_dims, target_dim, order)` triples for an entry point whose
/// target axis must hold `n_in` elements. Each must be rejected with
/// `SPIR_INVALID_DIMENSION` before `input` is read or `out` is written. Before
/// the fix these crashed the process (SIGSEGV, or an allocation-failure abort)
/// or returned `SPIR_INTERNAL_ERROR` / `SPIR_INPUT_DIMENSION_MISMATCH`.
fn malformed_input_dims(n_in: i32) -> Vec<(Vec<i32>, i32, i32)> {
    let rm = SPIR_ORDER_ROW_MAJOR;
    let cm = SPIR_ORDER_COLUMN_MAJOR;
    vec![
        // Negative extent on a non-target axis, in both memory orders.
        (vec![n_in, -1], 0, rm),
        (vec![n_in, -1], 0, cm),
        (vec![3, n_in, -2], 1, rm),
        (vec![n_in, i32::MIN], 0, rm),
        // Negative extent on the target axis.
        (vec![-1, 2], 0, rm),
        (vec![2, -1], 1, cm),
        // Zero-length axes, on non-target and target axes.
        (vec![n_in, 0], 0, rm),
        (vec![n_in, 0], 0, cm),
        (vec![0, n_in, 3], 1, rm),
        (vec![0, 2], 0, rm),
        // The element count overflows `usize`.
        (vec![n_in, i32::MAX, i32::MAX, i32::MAX], 0, rm),
        // The element count fits in `usize`, but the array would span more than
        // `isize::MAX` bytes.
        (vec![n_in, 1 << 30, (3 << 29) / n_in], 0, rm),
    ]
}

/// Check that an `input_dims` entry point rejects every malformed shape with
/// `SPIR_INVALID_DIMENSION` and leaves `out` untouched, returns
/// `SPIR_INPUT_DIMENSION_MISMATCH` for a well-formed shape with the wrong target
/// extent (when `check_mismatch`), and still accepts well-formed batched shapes
/// in both memory orders, writing exactly the result elements.
///
/// `call(input_dims, target_dim, order, out)` must pass an input buffer holding
/// at least `2 * n_in` elements; `out` holds `2 * n_out + OUT_SLACK` elements.
fn check_input_dims_validation<T: Copy + PartialEq + std::fmt::Debug>(
    name: &str,
    n_in: i32,
    n_out: i32,
    sentinel: T,
    check_mismatch: bool,
    call: impl Fn(&[i32], i32, i32, &mut [T]) -> i32,
) {
    let n_result = 2 * n_out as usize;
    let out_len = n_result + OUT_SLACK;

    for (dims, target_dim, order) in malformed_input_dims(n_in) {
        let mut out = vec![sentinel; out_len];
        let status = call(&dims, target_dim, order, &mut out);
        assert_eq!(
            status, SPIR_INVALID_DIMENSION,
            "{name}: input_dims={dims:?}, target_dim={target_dim}, order={order}"
        );
        assert!(
            out.iter().all(|&x| x == sentinel),
            "{name}: output written for rejected input_dims={dims:?}"
        );
    }

    if check_mismatch {
        let dims = vec![n_in + 1, 2];
        let mut out = vec![sentinel; out_len];
        let status = call(&dims, 0, SPIR_ORDER_ROW_MAJOR, &mut out);
        assert_eq!(
            status, SPIR_INPUT_DIMENSION_MISMATCH,
            "{name}: input_dims={dims:?}"
        );
        assert!(
            out.iter().all(|&x| x == sentinel),
            "{name}: output written for mismatched input_dims={dims:?}"
        );
    }

    for (dims, target_dim, order) in [
        (vec![n_in, 2], 0, SPIR_ORDER_ROW_MAJOR),
        (vec![2, n_in], 1, SPIR_ORDER_ROW_MAJOR),
        (vec![n_in, 2], 0, SPIR_ORDER_COLUMN_MAJOR),
        (vec![2, n_in], 1, SPIR_ORDER_COLUMN_MAJOR),
    ] {
        let mut out = vec![sentinel; out_len];
        let status = call(&dims, target_dim, order, &mut out);
        assert_eq!(
            status, SPIR_COMPUTATION_SUCCESS,
            "{name}: input_dims={dims:?}, target_dim={target_dim}, order={order}"
        );
        assert!(
            out[..n_result].iter().all(|&x| x != sentinel),
            "{name}: result not fully written for input_dims={dims:?}"
        );
        assert!(
            out[n_result..].iter().all(|&x| x == sentinel),
            "{name}: wrote past the result for input_dims={dims:?}"
        );
    }
}

/// A fermionic IR basis with default tau and Matsubara samplings and a DLR.
struct InputDimsFixture {
    kernel: *mut spir_kernel,
    sve: *mut spir_sve_result,
    basis: *mut spir_basis,
    tau: *mut spir_sampling,
    matsu: *mut spir_sampling,
    dlr: *mut spir_basis,
    basis_size: i32,
    n_tau: i32,
    n_matsu: i32,
    n_poles: i32,
}

impl InputDimsFixture {
    fn new() -> Self {
        let (kernel, sve, basis) = create_ir_basis(SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6);
        let basis_size = get_basis_size(basis);

        let taus = get_default_tau_points(basis);
        let mut status = SPIR_INTERNAL_ERROR;
        let tau = spir_tau_sampling_new(basis, taus.len() as i32, taus.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let matsus = get_default_matsubara_points(basis, false);
        let mut status = SPIR_INTERNAL_ERROR;
        let matsu = spir_matsu_sampling_new(
            basis,
            false,
            matsus.len() as i32,
            matsus.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut n_poles = 0;
        assert_eq!(
            spir_dlr_get_npoles(dlr, &mut n_poles),
            SPIR_COMPUTATION_SUCCESS
        );

        Self {
            kernel,
            sve,
            basis,
            tau,
            matsu,
            dlr,
            basis_size,
            n_tau: taus.len() as i32,
            n_matsu: matsus.len() as i32,
            n_poles,
        }
    }
}

impl Drop for InputDimsFixture {
    fn drop(&mut self) {
        spir_basis_release(self.dlr);
        spir_sampling_release(self.matsu);
        spir_sampling_release(self.tau);
        spir_basis_release(self.basis);
        spir_sve_result_release(self.sve);
        spir_kernel_release(self.kernel);
    }
}

fn complex_sentinel() -> Complex64 {
    Complex64::new(OUT_SENTINEL, OUT_SENTINEL)
}

#[test]
fn test_sampling_eval_dd_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![1.0; 2 * fx.basis_size as usize];
    check_input_dims_validation(
        "spir_sampling_eval_dd",
        fx.basis_size,
        fx.n_tau,
        OUT_SENTINEL,
        true,
        |dims, target_dim, order, out| {
            spir_sampling_eval_dd(
                fx.tau,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_sampling_eval_dz_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![1.0; 2 * fx.basis_size as usize];
    check_input_dims_validation(
        "spir_sampling_eval_dz",
        fx.basis_size,
        fx.n_matsu,
        complex_sentinel(),
        true,
        |dims, target_dim, order, out| {
            spir_sampling_eval_dz(
                fx.matsu,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_sampling_eval_zz_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![Complex64::new(1.0, 0.5); 2 * fx.basis_size as usize];
    check_input_dims_validation(
        "spir_sampling_eval_zz",
        fx.basis_size,
        fx.n_matsu,
        complex_sentinel(),
        true,
        |dims, target_dim, order, out| {
            spir_sampling_eval_zz(
                fx.matsu,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_sampling_fit_dd_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![1.0; 2 * fx.n_tau as usize];
    check_input_dims_validation(
        "spir_sampling_fit_dd",
        fx.n_tau,
        fx.basis_size,
        OUT_SENTINEL,
        true,
        |dims, target_dim, order, out| {
            spir_sampling_fit_dd(
                fx.tau,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_sampling_fit_zz_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![Complex64::new(1.0, 0.5); 2 * fx.n_matsu as usize];
    check_input_dims_validation(
        "spir_sampling_fit_zz",
        fx.n_matsu,
        fx.basis_size,
        complex_sentinel(),
        true,
        |dims, target_dim, order, out| {
            spir_sampling_fit_zz(
                fx.matsu,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_sampling_fit_zd_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![Complex64::new(1.0, 0.5); 2 * fx.n_matsu as usize];
    check_input_dims_validation(
        "spir_sampling_fit_zd",
        fx.n_matsu,
        fx.basis_size,
        OUT_SENTINEL,
        true,
        |dims, target_dim, order, out| {
            spir_sampling_fit_zd(
                fx.matsu,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

// The DLR conversions cannot check the target extent at the boundary (the IR
// basis size of a DLR is not exposed by the core), so `check_mismatch` is off.

#[test]
fn test_ir2dlr_dd_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![1.0; 2 * fx.basis_size as usize];
    check_input_dims_validation(
        "spir_ir2dlr_dd",
        fx.basis_size,
        fx.n_poles,
        OUT_SENTINEL,
        false,
        |dims, target_dim, order, out| {
            spir_ir2dlr_dd(
                fx.dlr,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_ir2dlr_zz_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![Complex64::new(1.0, 0.5); 2 * fx.basis_size as usize];
    check_input_dims_validation(
        "spir_ir2dlr_zz",
        fx.basis_size,
        fx.n_poles,
        complex_sentinel(),
        false,
        |dims, target_dim, order, out| {
            spir_ir2dlr_zz(
                fx.dlr,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_dlr2ir_dd_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![1.0; 2 * fx.n_poles as usize];
    check_input_dims_validation(
        "spir_dlr2ir_dd",
        fx.n_poles,
        fx.basis_size,
        OUT_SENTINEL,
        false,
        |dims, target_dim, order, out| {
            spir_dlr2ir_dd(
                fx.dlr,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}

#[test]
fn test_dlr2ir_zz_validates_input_dims() {
    let fx = InputDimsFixture::new();
    let input = vec![Complex64::new(1.0, 0.5); 2 * fx.n_poles as usize];
    check_input_dims_validation(
        "spir_dlr2ir_zz",
        fx.n_poles,
        fx.basis_size,
        complex_sentinel(),
        false,
        |dims, target_dim, order, out| {
            spir_dlr2ir_zz(
                fx.dlr,
                std::ptr::null(),
                order,
                dims.len() as i32,
                dims.as_ptr(),
                target_dim,
                input.as_ptr(),
                out.as_mut_ptr(),
            )
        },
    );
}
