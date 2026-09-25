//! Integration tests for sparseir-capi
//!
//! Port of libsparseir's cinterface_integration.cxx
//! Tests the complete workflow: IR basis → sampling → DLR

use num_complex::Complex64;
use rstest::rstest;
use sparse_ir::tsvd::compute_svd_dtensor;
use sparse_ir::{CustomNumeric, DTensor, Df64};
use sparse_ir_capi::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, SPIR_ORDER_COLUMN_MAJOR,
    SPIR_ORDER_ROW_MAJOR, SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_FERMIONIC, spir_basis,
    spir_basis_get_default_matsus, spir_basis_get_default_matsus_ext, spir_basis_get_default_taus,
    spir_basis_get_n_default_matsus, spir_basis_get_n_default_matsus_ext,
    spir_basis_get_n_default_taus, spir_basis_get_size, spir_basis_get_u, spir_basis_get_uhat,
    spir_basis_new, spir_basis_release, spir_dlr_get_npoles, spir_dlr_get_poles, spir_dlr_new,
    spir_dlr2ir_dd, spir_funcs_batch_eval, spir_funcs_batch_eval_matsu, spir_funcs_eval,
    spir_funcs_eval_matsu, spir_funcs_get_size, spir_funcs_release, spir_ir2dlr_dd, spir_kernel,
    spir_kernel_release, spir_logistic_kernel_new, spir_matsu_sampling_new,
    spir_matsu_sampling_new_with_matrix, spir_reg_bose_kernel_new, spir_sampling,
    spir_sampling_eval_dd, spir_sampling_eval_dz, spir_sampling_eval_zz, spir_sampling_fit_dd,
    spir_sampling_fit_zd, spir_sampling_fit_zz, spir_sampling_get_cond_num, spir_sampling_release,
    spir_sve_result, spir_sve_result_new, spir_sve_result_release, spir_tau_sampling_new,
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
// spir_sampling_get_cond_num (SpM-lab/sparse-ir-rs#270)
// ============================================================================
//
// The reported condition number must be that of the matrix the fit functions
// solve the least-squares problem with:
// - tau sampling: the real n x L matrix A[i, l] = u_l(tau_i);
// - full-set Matsubara sampling: the complex n x L matrix A[i, l] = uhat_l(iv_i);
// - positive-only Matsubara sampling: the real 2n x L matrix [Re A; Im A] of the
//   real least-squares problem [Re A; Im A] x = [Re g; Im g].
//
// Oracle: A is rebuilt independently of the sampling object, from the basis
// functions (spir_basis_get_u / spir_basis_get_uhat) or from the matrix passed
// at construction, and its singular values are computed in double-double
// precision by the nalgebra-based `sparse_ir::tsvd::compute_svd_dtensor`, which
// shares no code with the faer SVD that the sampling fitters use.

/// sigma_max / sigma_min of a real row-major `rows x cols` matrix (oracle).
fn oracle_cond(rows: usize, cols: usize, a: &[f64]) -> f64 {
    assert_eq!(a.len(), rows * cols);
    let m = DTensor::<Df64, 2>::from_fn([rows, cols], |idx| Df64::from(a[idx[0] * cols + idx[1]]));
    let (_, s, _) = compute_svd_dtensor(&m);
    // compute_svd_dtensor truncates below 2 eps_Df64 * sigma_max; it must not
    // have dropped a singular value, or s_min below would not be sigma_min.
    assert_eq!(
        s.len(),
        rows.min(cols),
        "oracle SVD dropped a singular value"
    );
    let (mut s_max, mut s_min) = (s[0], s[0]);
    for &x in &s[1..] {
        if x > s_max {
            s_max = x;
        }
        if x < s_min {
            s_min = x;
        }
    }
    (s_max / s_min).to_f64()
}

/// `[Re A; Im A]` (row-major, 2n x L) of a row-major complex n x L matrix `A`.
fn stack_re_im(a: &[Complex64]) -> Vec<f64> {
    a.iter()
        .map(|z| z.re)
        .chain(a.iter().map(|z| z.im))
        .collect()
}

/// Real embedding `[[Re A, -Im A], [Im A, Re A]]` (row-major, 2n x 2L) of a
/// row-major complex n x L matrix `A`. Its singular values are those of `A`,
/// each twice, so its condition number is that of `A`.
fn realify(n: usize, l: usize, a: &[Complex64]) -> Vec<f64> {
    assert_eq!(a.len(), n * l);
    let mut out = vec![0.0; 4 * n * l];
    for i in 0..n {
        for j in 0..l {
            let z = a[i * l + j];
            out[i * 2 * l + j] = z.re;
            out[i * 2 * l + l + j] = -z.im;
            out[(n + i) * 2 * l + j] = z.im;
            out[(n + i) * 2 * l + l + j] = z.re;
        }
    }
    out
}

/// `uhat_l(iv_n)` for every basis function at the reduced Matsubara indices
/// `ns`, row-major `ns.len() x L`, evaluated through the C API.
///
/// # Safety
/// `basis` must be a live basis handle.
unsafe fn eval_uhat_rows(basis: *const spir_basis, ns: &[i64]) -> Vec<Complex64> {
    unsafe {
        let mut status = SPIR_INTERNAL_ERROR;
        let uhat = spir_basis_get_uhat(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!uhat.is_null());
        let mut n_funcs = 0;
        assert_eq!(
            spir_funcs_get_size(uhat, &mut n_funcs),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(n_funcs, get_basis_size(basis));
        let mut out = vec![Complex64::new(0.0, 0.0); ns.len() * n_funcs as usize];
        let status = spir_funcs_batch_eval_matsu(
            uhat,
            SPIR_ORDER_ROW_MAJOR,
            ns.len() as i32,
            ns.as_ptr(),
            out.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        spir_funcs_release(uhat);
        out
    }
}

/// `u_l(tau)` for every basis function at `taus`, row-major `taus.len() x L`,
/// evaluated through the C API.
///
/// # Safety
/// `basis` must be a live basis handle.
unsafe fn eval_u_rows(basis: *const spir_basis, taus: &[f64]) -> Vec<f64> {
    unsafe {
        let mut status = SPIR_INTERNAL_ERROR;
        let u = spir_basis_get_u(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!u.is_null());
        let mut n_funcs = 0;
        assert_eq!(
            spir_funcs_get_size(u, &mut n_funcs),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(n_funcs, get_basis_size(basis));
        let mut out = vec![0.0; taus.len() * n_funcs as usize];
        let status = spir_funcs_batch_eval(
            u,
            SPIR_ORDER_ROW_MAJOR,
            taus.len() as i32,
            taus.as_ptr(),
            out.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        spir_funcs_release(u);
        out
    }
}

fn new_matsu_sampling(
    basis: *const spir_basis,
    positive_only: bool,
    points: &[i64],
) -> *mut spir_sampling {
    let mut status = SPIR_INTERNAL_ERROR;
    let sampling = spir_matsu_sampling_new(
        basis,
        positive_only,
        points.len() as i32,
        points.as_ptr(),
        &mut status,
    );
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    assert!(!sampling.is_null());
    sampling
}

fn get_cond_num(sampling: *const spir_sampling) -> f64 {
    let mut cond = f64::NAN;
    let status = spir_sampling_get_cond_num(sampling, &mut cond);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    cond
}

/// Assert that a reported condition number equals the oracle.
///
/// Tolerance: a backward-stable SVD returns every singular value with an
/// absolute error O(eps_mach * sigma_max), so sigma_min, and with it
/// kappa = sigma_max / sigma_min, carries a relative error O(eps_mach * kappa).
/// The factor 100 covers the dimension-dependent constant of the f64 SVD
/// behind the reported value (the double-double oracle is exact at this
/// level); the 1e-12 floor covers kappa = O(1). The values #270 reports for
/// positive-only samplings differ from the correct ones by at least 1%
/// (4.79 vs 4.84 for the fermionic beta = 10 basis), far outside this bound.
fn assert_cond_close(label: &str, reported: f64, oracle: f64) {
    let rtol = (100.0 * f64::EPSILON * oracle).max(1e-12);
    let rel_err = (reported - oracle).abs() / oracle;
    println!("{label}: relative error {rel_err:.3e} (bound {rtol:.3e})");
    assert!(
        rel_err <= rtol,
        "{label}: spir_sampling_get_cond_num = {reported:.6e}, oracle = {oracle:.6e}, \
         relative error {rel_err:.3e} > {rtol:.3e}"
    );
}

/// Positive-only samplings must report the condition number of the stacked
/// real matrix `[Re A; Im A]` that the fit solves, not that of the complex
/// n x L matrix `A` (#270); full-set samplings keep reporting that of `A`.
#[rstest]
#[case::fermionic_beta10(SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6)]
#[case::bosonic_beta10(SPIR_STATISTICS_BOSONIC, 10.0, 1.0, 1e-6)]
#[case::bosonic_beta1000(SPIR_STATISTICS_BOSONIC, 1000.0, 1.0, 1e-10)]
fn test_sampling_cond_num_matsubara_is_fit_matrix_cond(
    #[case] statistics: i32,
    #[case] beta: f64,
    #[case] wmax: f64,
    #[case] epsilon: f64,
) {
    unsafe {
        let (kernel, sve, basis) = create_ir_basis(statistics, beta, wmax, epsilon);
        let l = get_basis_size(basis) as usize;
        let label =
            format!("statistics={statistics}, beta={beta}, wmax={wmax}, eps={epsilon:e}, L={l}");

        let pos_points = get_default_matsubara_points(basis, true);
        let pos_sampling = new_matsu_sampling(basis, true, &pos_points);
        let pos_reported = get_cond_num(pos_sampling);
        let a_pos = eval_uhat_rows(basis, &pos_points);
        let n_pos = pos_points.len();
        let pos_oracle = oracle_cond(2 * n_pos, l, &stack_re_im(&a_pos));
        let pos_complex = oracle_cond(2 * n_pos, 2 * l, &realify(n_pos, l, &a_pos));

        let full_points = get_default_matsubara_points(basis, false);
        let full_sampling = new_matsu_sampling(basis, false, &full_points);
        let full_reported = get_cond_num(full_sampling);
        let n_full = full_points.len();
        let full_oracle = oracle_cond(
            2 * n_full,
            2 * l,
            &realify(n_full, l, &eval_uhat_rows(basis, &full_points)),
        );

        println!(
            "{label}: positive-only (n={n_pos}) reported {pos_reported:.6e}, \
             [Re A; Im A] {pos_oracle:.6e}, complex A {pos_complex:.6e}; \
             full-set (n={n_full}) reported {full_reported:.6e}, oracle {full_oracle:.6e}"
        );
        assert_cond_close(&format!("{label}, positive-only"), pos_reported, pos_oracle);
        assert_cond_close(&format!("{label}, full-set"), full_reported, full_oracle);

        spir_sampling_release(pos_sampling);
        spir_sampling_release(full_sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

/// Fourier transform of the bosonic `TauConst` augmentation,
/// `1/sqrt(beta)` on [0, beta], at the (even) reduced Matsubara index `n`:
/// `sqrt(beta)` at n = 0 and zero elsewhere.
fn tau_const_hat(beta: f64, n: i64) -> Complex64 {
    if n == 0 {
        Complex64::new(beta.sqrt(), 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Fourier transform of the bosonic `TauLinear` augmentation,
/// `sqrt(3/beta) (2 tau/beta - 1)` on [0, beta], at the (even) reduced
/// Matsubara index `n`: `2 sqrt(3/beta) / (i nu_n)` with `nu_n = pi n / beta`,
/// and zero at n = 0.
fn tau_linear_hat(beta: f64, n: i64) -> Complex64 {
    if n == 0 {
        Complex64::new(0.0, 0.0)
    } else {
        let nu = std::f64::consts::PI * n as f64 / beta;
        Complex64::new(0.0, -2.0 * (3.0 / beta).sqrt() / nu)
    }
}

/// Samplings created from a user-supplied matrix follow the same rule, for
/// both memory orders. The matrix is that of #270's ill-conditioned example:
/// a bosonic basis augmented with `TauConst` and `TauLinear` (columns
/// `[TauConst, TauLinear, uhat_0, ..., uhat_{L-1}]`), sampled at the default
/// Matsubara points of the underlying basis for L + 2 functions, of which the
/// positive-only sampling keeps n >= 0. The augmentations, column order and
/// point selection are convention-matched with Python sparse-ir
/// (`src/sparse_ir/augment.py`: `TauConst.hat`, `TauLinear.hat`,
/// `AugmentedBasis.default_matsubara_sampling_points`; SpM-lab/sparse-ir at
/// 2a42227), which the #270 report used to build this matrix.
#[test]
fn test_sampling_cond_num_custom_matrix_is_fit_matrix_cond() {
    let (beta, wmax, epsilon) = (10.0, 1.0, 1e-6);
    unsafe {
        let (kernel, sve, basis) = create_ir_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon);
        let l = get_basis_size(basis) as usize;
        let l_aug = l + 2;

        let mut n_total = 0;
        let status =
            spir_basis_get_n_default_matsus_ext(basis, false, false, l_aug as i32, &mut n_total);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut full_points = vec![0i64; n_total as usize];
        let mut n_written = 0;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            false,
            false,
            l_aug as i32,
            n_total,
            full_points.as_mut_ptr(),
            &mut n_written,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_written, n_total);
        full_points.sort_unstable();
        let pos_points: Vec<i64> = full_points.iter().copied().filter(|&n| n >= 0).collect();

        for (positive_only, points) in [(true, &pos_points), (false, &full_points)] {
            let n = points.len();
            let uhat = eval_uhat_rows(basis, points);
            let mut a_row = Vec::with_capacity(n * l_aug);
            for (i, &freq) in points.iter().enumerate() {
                a_row.push(tau_const_hat(beta, freq));
                a_row.push(tau_linear_hat(beta, freq));
                a_row.extend_from_slice(&uhat[i * l..(i + 1) * l]);
            }
            let mut a_col = vec![Complex64::new(0.0, 0.0); n * l_aug];
            for i in 0..n {
                for j in 0..l_aug {
                    a_col[j * n + i] = a_row[i * l_aug + j];
                }
            }
            let oracle = if positive_only {
                oracle_cond(2 * n, l_aug, &stack_re_im(&a_row))
            } else {
                oracle_cond(2 * n, 2 * l_aug, &realify(n, l_aug, &a_row))
            };

            for (order, matrix) in [
                (SPIR_ORDER_ROW_MAJOR, &a_row),
                (SPIR_ORDER_COLUMN_MAJOR, &a_col),
            ] {
                let mut status = SPIR_INTERNAL_ERROR;
                let sampling = spir_matsu_sampling_new_with_matrix(
                    order,
                    SPIR_STATISTICS_BOSONIC,
                    l_aug as i32,
                    positive_only,
                    n as i32,
                    points.as_ptr(),
                    matrix.as_ptr(),
                    &mut status,
                );
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                assert!(!sampling.is_null());
                let reported = get_cond_num(sampling);
                let label = format!(
                    "TauConst+TauLinear augmented bosonic basis (L+2={l_aug}), \
                     positive_only={positive_only} (n={n}), order={order}"
                );
                println!("{label}: reported {reported:.6e}, oracle {oracle:.6e}");
                assert_cond_close(&label, reported, oracle);
                spir_sampling_release(sampling);
            }
        }

        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

/// Tau samplings report the condition number of their real n x L matrix.
#[rstest]
#[case::fermionic(SPIR_STATISTICS_FERMIONIC)]
#[case::bosonic(SPIR_STATISTICS_BOSONIC)]
fn test_sampling_cond_num_tau_is_sampling_matrix_cond(#[case] statistics: i32) {
    let (beta, wmax, epsilon) = (10.0, 1.0, 1e-6);
    unsafe {
        let (kernel, sve, basis) = create_ir_basis(statistics, beta, wmax, epsilon);
        let l = get_basis_size(basis) as usize;
        let taus = get_default_tau_points(basis);
        let mut status = SPIR_INTERNAL_ERROR;
        let sampling = spir_tau_sampling_new(basis, taus.len() as i32, taus.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sampling.is_null());

        let reported = get_cond_num(sampling);
        let oracle = oracle_cond(taus.len(), l, &eval_u_rows(basis, &taus));
        let label = format!(
            "tau sampling, statistics={statistics}, L={l}, n={}",
            taus.len()
        );
        println!("{label}: reported {reported:.6e}, oracle {oracle:.6e}");
        assert_cond_close(&label, reported, oracle);

        spir_sampling_release(sampling);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_sampling_cond_num_rejects_null_arguments() {
    let (kernel, sve, basis) = create_ir_basis(SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6);
    let points = get_default_matsubara_points(basis, true);
    let sampling = new_matsu_sampling(basis, true, &points);

    let mut cond = -1.0;
    let status = spir_sampling_get_cond_num(std::ptr::null(), &mut cond);
    assert_eq!(status, SPIR_INVALID_ARGUMENT);
    assert_eq!(cond, -1.0, "cond_num must be left untouched on failure");
    let status = spir_sampling_get_cond_num(sampling, std::ptr::null_mut());
    assert_eq!(status, SPIR_INVALID_ARGUMENT);

    spir_sampling_release(sampling);
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}
