//! Integration tests for sparseir-capi
//!
//! Port of libsparseir's cinterface_integration.cxx
//! Tests the complete workflow: IR basis → sampling → DLR

use num_complex::Complex64;
use rstest::rstest;
use sparseir_capi::{
    spir_basis, spir_basis_get_default_matsus, spir_basis_get_default_taus,
    spir_basis_get_n_default_matsus, spir_basis_get_n_default_taus, spir_basis_get_size,
    spir_basis_get_u, spir_basis_get_uhat, spir_basis_new, spir_basis_release, spir_dlr2ir_dd,
    spir_dlr_get_npoles, spir_dlr_get_poles, spir_dlr_new, spir_funcs_eval, spir_funcs_eval_matsu,
    spir_funcs_release, spir_ir2dlr_dd, spir_kernel, spir_kernel_release, spir_logistic_kernel_new,
    spir_matsu_sampling_new, spir_sampling_eval_dd, spir_sampling_eval_dz, spir_sampling_eval_zz,
    spir_sampling_fit_dd, spir_sampling_fit_zd, spir_sampling_fit_zz, spir_sampling_release,
    spir_sve_result, spir_sve_result_new, spir_sve_result_release, spir_tau_sampling_new,
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR,
};

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

