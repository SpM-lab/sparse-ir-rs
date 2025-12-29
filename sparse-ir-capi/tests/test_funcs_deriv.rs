//! Test for spir_funcs_deriv C-API function
//!
//! This test verifies that the derivative computation works correctly
//! at the C-API level.

use sparse_ir_capi::*;
use std::ptr;

#[test]
fn test_funcs_deriv_basic() {
    unsafe {
        // Create a simple basis for testing
        let lambda = 10.0;
        let eps = 1e-6;
        let mut status = 0;

        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let sve = spir_sve_result_new(kernel, eps, -1, -1, SPIR_TWORK_FLOAT64, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        let beta = 10.0;
        let wmax = 1.0;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            wmax,
            eps,
            kernel,
            sve,
            -1,
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        // Get u functions (imaginary time basis functions)
        let u_funcs = spir_basis_get_u(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!u_funcs.is_null());

        // Test n=0 (should return clone)
        let deriv0 = spir_funcs_deriv(u_funcs, 0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!deriv0.is_null());

        // Test n=1 (first derivative)
        let deriv1 = spir_funcs_deriv(u_funcs, 1, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!deriv1.is_null());

        // Test n=2 (second derivative)
        let deriv2 = spir_funcs_deriv(u_funcs, 2, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!deriv2.is_null());

        // Verify that derivatives are different objects
        assert_ne!(deriv0, deriv1);
        assert_ne!(deriv1, deriv2);

        // Test evaluation at a point to verify derivatives make sense
        let x = 0.5;
        let mut basis_size = 0;
        let size_status = spir_basis_get_size(basis, &mut basis_size);
        assert_eq!(size_status, SPIR_COMPUTATION_SUCCESS);

        let mut values0 = vec![0.0; basis_size as usize];
        let mut values1 = vec![0.0; basis_size as usize];

        let eval_status0 = spir_funcs_eval(deriv0, x, values0.as_mut_ptr());
        assert_eq!(eval_status0, SPIR_COMPUTATION_SUCCESS);

        let eval_status1 = spir_funcs_eval(deriv1, x, values1.as_mut_ptr());
        assert_eq!(eval_status1, SPIR_COMPUTATION_SUCCESS);

        // Verify that derivative values are different from original
        let mut any_different = false;
        for i in 0..basis_size as usize {
            if (values0[i] - values1[i]).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Derivative should differ from original");

        // Cleanup
        spir_funcs_release(deriv2);
        spir_funcs_release(deriv1);
        spir_funcs_release(deriv0);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_funcs_deriv_numerical_consistency() {
    unsafe {
        // Create a basis
        let lambda = 10.0;
        let eps = 1e-6;
        let mut status = 0;

        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert!(!kernel.is_null());

        let sve = spir_sve_result_new(kernel, eps, -1, -1, SPIR_TWORK_FLOAT64, &mut status);
        assert!(!sve.is_null());

        let beta = 10.0;
        let wmax = 1.0;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            wmax,
            eps,
            kernel,
            sve,
            -1,
            &mut status,
        );
        assert!(!basis.is_null());

        let u_funcs = spir_basis_get_u(basis, &mut status);
        assert!(!u_funcs.is_null());

        // Get derivative
        let deriv1 = spir_funcs_deriv(u_funcs, 1, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!deriv1.is_null());

        // Test numerical consistency at several points
        let mut basis_size = 0;
        spir_basis_get_size(basis, &mut basis_size);

        let test_points = [0.2, 0.5, 0.8];
        let h = 1e-8;

        for &x in &test_points {
            let mut f_plus = vec![0.0; basis_size as usize];
            let mut f_minus = vec![0.0; basis_size as usize];
            let mut deriv_analytical = vec![0.0; basis_size as usize];

            spir_funcs_eval(u_funcs, x + h, f_plus.as_mut_ptr());
            spir_funcs_eval(u_funcs, x - h, f_minus.as_mut_ptr());
            spir_funcs_eval(deriv1, x, deriv_analytical.as_mut_ptr());

            // Check first few basis functions (numerical differentiation
            // may be less accurate for higher order functions)
            for i in 0..3.min(basis_size as usize) {
                let numerical = (f_plus[i] - f_minus[i]) / (2.0 * h);
                let analytical = deriv_analytical[i];
                let rel_error = if analytical.abs() > 1e-10 {
                    ((numerical - analytical) / analytical).abs()
                } else {
                    (numerical - analytical).abs()
                };

                assert!(
                    rel_error < 1e-4,
                    "Derivative mismatch at x={}, i={}: numerical={}, analytical={}, rel_error={}",
                    x,
                    i,
                    numerical,
                    analytical,
                    rel_error
                );
            }
        }

        // Cleanup
        spir_funcs_release(deriv1);
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}

#[test]
fn test_funcs_deriv_error_handling() {
    unsafe {
        let mut status = 0;

        // Test NULL pointer
        let result = spir_funcs_deriv(ptr::null(), 1, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(result.is_null());

        // Test negative derivative order
        let lambda = 10.0;
        let eps = 1e-6;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        let sve = spir_sve_result_new(kernel, eps, -1, -1, SPIR_TWORK_FLOAT64, &mut status);
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            10.0,
            1.0,
            eps,
            kernel,
            sve,
            -1,
            &mut status,
        );
        let u_funcs = spir_basis_get_u(basis, &mut status);

        let result = spir_funcs_deriv(u_funcs, -1, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(result.is_null());

        // Cleanup
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}
