//! Basis API
//!
//! Functions for creating and manipulating finite temperature basis objects.

use std::panic::{catch_unwind, AssertUnwindSafe};

use sparseir_rust::basis::FiniteTempBasis;

use crate::types::{spir_basis, spir_funcs, spir_kernel, spir_sve_result};
use crate::{StatusCode, SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT};

/// Manual release function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_release(basis: *mut spir_basis) {
    if !basis.is_null() {
        unsafe {
            let _ = Box::from_raw(basis);
        }
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_basis_clone(src: *const spir_basis) -> *mut spir_basis {
    if src.is_null() {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let src_ref = &*src;
        let cloned = (*src_ref).clone();
        Box::into_raw(Box::new(cloned))
    }));

    result.unwrap_or(std::ptr::null_mut())
}

/// Manual is_assigned function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_is_assigned(obj: *const spir_basis) -> i32 {
    if obj.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    }));

    result.unwrap_or(0)
}

/// Create a finite temperature basis (libsparseir compatible)
///
/// # Arguments
/// * `statistics` - 0 for Bosonic, 1 for Fermionic
/// * `beta` - Inverse temperature (must be > 0)
/// * `omega_max` - Frequency cutoff (must be > 0)
/// * `epsilon` - Accuracy target (must be > 0)
/// * `k` - Kernel object (can be NULL if sve is provided)
/// * `sve` - Pre-computed SVE result (can be NULL, will compute if needed)
/// * `max_size` - Maximum basis size (-1 for no limit)
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to basis object, or NULL on failure
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_new(
    statistics: libc::c_int,
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    k: *const spir_kernel,
    sve: *const spir_sve_result,
    max_size: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_basis {
    if status.is_null() {
        return std::ptr::null_mut();
    }

    // Validate inputs
    if beta <= 0.0 || omega_max <= 0.0 || epsilon <= 0.0 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Validate statistics
    if statistics != 0 && statistics != 1 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Must have kernel (SVE can be provided for optimization but kernel is required for type info)
    if k.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Convert max_size
    let max_size_opt = if max_size < 0 {
        None
    } else {
        Some(max_size as usize)
    };

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let kernel_ref = &*k;

        // Check that kernel's lambda matches beta * omega_max
        let expected_lambda = beta * omega_max;
        let kernel_lambda = kernel_ref.lambda();
        if (kernel_lambda - expected_lambda).abs() > 1e-10 {
            return Err(format!(
                "Kernel lambda ({}) does not match beta * omega_max ({})",
                kernel_lambda, expected_lambda
            ));
        }

        // Dispatch based on kernel type and statistics
        if let Some(logistic) = kernel_ref.as_logistic() {
            if statistics == 1 {
                // Fermionic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        **logistic,
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(**logistic, beta, Some(epsilon), max_size_opt)
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_logistic_fermionic(
                    basis,
                ))))
            } else {
                // Bosonic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        **logistic,
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(**logistic, beta, Some(epsilon), max_size_opt)
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_logistic_bosonic(
                    basis,
                ))))
            }
        } else if let Some(reg_bose) = kernel_ref.as_regularized_bose() {
            if statistics == 1 {
                // Fermionic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        **reg_bose,
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(**reg_bose, beta, Some(epsilon), max_size_opt)
                };
                Ok(Box::into_raw(Box::new(
                    spir_basis::new_regularized_bose_fermionic(basis),
                )))
            } else {
                // Bosonic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        **reg_bose,
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(**reg_bose, beta, Some(epsilon), max_size_opt)
                };
                Ok(Box::into_raw(Box::new(
                    spir_basis::new_regularized_bose_bosonic(basis),
                )))
            }
        } else {
            Err("Unknown kernel type".to_string())
        }
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(_)) | Err(_) => {
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Get the number of basis functions
///
/// # Arguments
/// * `b` - Basis object
/// * `size` - Pointer to store the size
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or size is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_size(b: *const spir_basis, size: *mut libc::c_int) -> StatusCode {
    if b.is_null() || size.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        *size = basis.size() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get singular values from a basis
///
/// # Arguments
/// * `b` - Basis object
/// * `svals` - Pre-allocated array to store singular values (size must be >= basis size)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or svals is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_svals(b: *const spir_basis, svals: *mut f64) -> StatusCode {
    if b.is_null() || svals.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let sval_vec = basis.svals();
        std::ptr::copy_nonoverlapping(sval_vec.as_ptr(), svals, sval_vec.len());
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get statistics type (Fermionic or Bosonic) of a basis
///
/// # Arguments
/// * `b` - Basis object
/// * `statistics` - Pointer to store statistics (0 = Bosonic, 1 = Fermionic)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or statistics is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_stats(
    b: *const spir_basis,
    statistics: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || statistics.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        *statistics = basis.statistics();
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get singular values (alias for spir_basis_get_svals for libsparseir compatibility)
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_singular_values(
    b: *const spir_basis,
    svals: *mut f64,
) -> StatusCode {
    spir_basis_get_svals(b, svals)
}

/// Get the number of default tau sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `num_points` - Pointer to store the number of points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_n_default_taus(
    b: *const spir_basis,
    num_points: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let points = basis.default_tau_sampling_points();
        *num_points = points.len() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default tau sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `points` - Pre-allocated array to store tau points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_taus(
    b: *const spir_basis,
    points: *mut f64,
) -> StatusCode {
    if b.is_null() || points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let tau_points = basis.default_tau_sampling_points();
        std::ptr::copy_nonoverlapping(tau_points.as_ptr(), points, tau_points.len());
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get the number of default Matsubara sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `num_points` - Pointer to store the number of points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_n_default_matsus(
    b: *const spir_basis,
    positive_only: bool,
    num_points: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let points = basis.default_matsubara_sampling_points(positive_only);
        *num_points = points.len() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default Matsubara sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `points` - Pre-allocated array to store Matsubara indices
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_matsus(
    b: *const spir_basis,
    positive_only: bool,
    points: *mut i64,
) -> StatusCode {
    if b.is_null() || points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let matsu_points = basis.default_matsubara_sampling_points(positive_only);
        std::ptr::copy_nonoverlapping(matsu_points.as_ptr(), points, matsu_points.len());
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the basis functions in imaginary time (τ) domain
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
///
/// # Safety
/// The caller must ensure that `b` is a valid pointer, and must call
/// `spir_funcs_release()` on the returned pointer when done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_basis_get_u(
    b: *const spir_basis,
    status: *mut StatusCode,
) -> *mut spir_funcs {
    use crate::types::{spir_funcs, BasisType};
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if b.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis_ref = &*b;
        let beta = basis_ref.beta();

        let funcs = match basis_ref.inner() {
            BasisType::LogisticFermionic(basis) => {
                spir_funcs::from_u_fermionic(basis.u.clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => spir_funcs::from_u_bosonic(basis.u.clone(), beta),
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_u_fermionic(basis.u.clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_u_bosonic(basis.u.clone(), beta)
            }
            // DLR: tau-domain functions using discrete poles
            // Note: DLR always uses LogisticKernel
            BasisType::DLRFermionic(dlr) => spir_funcs::from_dlr_tau_fermionic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.inv_weights.clone(),
            ),
            BasisType::DLRBosonic(dlr) => spir_funcs::from_dlr_tau_bosonic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.inv_weights.clone(),
            ),
        };

        Result::<*mut spir_funcs, String>::Ok(Box::into_raw(Box::new(funcs)))
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(msg)) => {
            eprintln!("Error in spir_basis_get_u: {}", msg);
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Gets the basis functions in real frequency (ω) domain
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
///
/// # Safety
/// The caller must ensure that `b` is a valid pointer, and must call
/// `spir_funcs_release()` on the returned pointer when done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_basis_get_v(
    b: *const spir_basis,
    status: *mut StatusCode,
) -> *mut spir_funcs {
    use crate::types::{spir_funcs, BasisType};
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if b.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis_ref = &*b;
        let beta = basis_ref.beta();

        let funcs = match basis_ref.inner() {
            BasisType::LogisticFermionic(basis) => spir_funcs::from_v(basis.v.clone(), beta),
            BasisType::LogisticBosonic(basis) => spir_funcs::from_v(basis.v.clone(), beta),
            BasisType::RegularizedBoseFermionic(basis) => spir_funcs::from_v(basis.v.clone(), beta),
            BasisType::RegularizedBoseBosonic(basis) => spir_funcs::from_v(basis.v.clone(), beta),
            // DLR: no continuous functions (v)
            BasisType::DLRFermionic(_)
            | BasisType::DLRBosonic(_) => {
                return Result::<*mut spir_funcs, String>::Err(
                    "DLR does not support continuous functions".to_string(),
                );
            }
        };

        Result::<*mut spir_funcs, String>::Ok(Box::into_raw(Box::new(funcs)))
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(msg)) => {
            eprintln!("Error in spir_basis_get_v: {}", msg);
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Gets the number of default omega (real frequency) sampling points
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object
/// * `num_points` - Pointer to store the number of sampling points
///
/// # Returns
/// Status code (SPIR_COMPUTATION_SUCCESS on success)
///
/// # Safety
/// The caller must ensure that `b` and `num_points` are valid pointers
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_n_default_ws(
    b: *const spir_basis,
    num_points: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let omega_points = basis.default_omega_sampling_points();
        *num_points = omega_points.len() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the default omega (real frequency) sampling points
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object
/// * `points` - Pre-allocated array to store the omega sampling points
///
/// # Returns
/// Status code (SPIR_COMPUTATION_SUCCESS on success)
///
/// # Safety
/// The caller must ensure that `points` has size >= `spir_basis_get_n_default_ws(b)`
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_ws(b: *const spir_basis, points: *mut f64) -> StatusCode {
    if b.is_null() || points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let omega_points = basis.default_omega_sampling_points();
        std::ptr::copy_nonoverlapping(omega_points.as_ptr(), points, omega_points.len());
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the basis functions in Matsubara frequency domain
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the basis functions object (`spir_funcs`), or NULL if creation fails
///
/// # Safety
/// The caller must ensure that `b` is a valid pointer, and must call
/// `spir_funcs_release()` on the returned pointer when done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_basis_get_uhat(
    b: *const spir_basis,
    status: *mut StatusCode,
) -> *mut spir_funcs {
    use crate::types::{spir_funcs, BasisType};
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if b.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis_ref = &*b;
        let beta = basis_ref.beta();

        let funcs = match basis_ref.inner() {
            BasisType::LogisticFermionic(basis) => {
                spir_funcs::from_uhat_fermionic(basis.uhat.clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => {
                spir_funcs::from_uhat_bosonic(basis.uhat.clone(), beta)
            }
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_uhat_fermionic(basis.uhat.clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_uhat_bosonic(basis.uhat.clone(), beta)
            }
            // DLR: Matsubara-domain functions using discrete poles
            BasisType::DLRFermionic(dlr) => spir_funcs::from_dlr_matsubara_fermionic(
                dlr.poles.clone(),
                beta,
                dlr.inv_weights.clone(),
            ),
            BasisType::DLRBosonic(dlr) => spir_funcs::from_dlr_matsubara_bosonic(
                dlr.poles.clone(),
                beta,
                dlr.inv_weights.clone(),
            ),
        };

        Result::<*mut spir_funcs, String>::Ok(Box::into_raw(Box::new(funcs)))
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(msg)) => {
            eprintln!("Error in spir_basis_get_uhat: {}", msg);
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Get default tau sampling points with custom limit (extended version)
///
/// # Arguments
/// * `b` - Basis object
/// * `n_points` - Maximum number of points requested
/// * `points` - Pre-allocated array to store tau points (size >= n_points)
/// * `n_points_returned` - Pointer to store actual number of points returned
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or n_points < 0
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Note
/// Returns min(n_points, actual_default_points) sampling points
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_taus_ext(
    b: *const spir_basis,
    n_points: libc::c_int,
    points: *mut f64,
    n_points_returned: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || points.is_null() || n_points_returned.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if n_points < 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let tau_points = basis.default_tau_sampling_points();

        // Return min(requested, available) points
        let n_to_return = std::cmp::min(n_points as usize, tau_points.len());
        std::ptr::copy_nonoverlapping(tau_points.as_ptr(), points, n_to_return);
        *n_points_returned = n_to_return as libc::c_int;

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get number of default Matsubara sampling points with custom limit (extended version)
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `L` - Requested number of sampling points
/// * `num_points_returned` - Pointer to store actual number of points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or L < 0
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Note
/// Returns min(L, actual_default_points) sampling points
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_n_default_matsus_ext(
    b: *const spir_basis,
    positive_only: bool,
    L: libc::c_int,
    num_points_returned: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points_returned.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if L < 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let points = basis.default_matsubara_sampling_points(positive_only);

        // Return min(requested, available) points
        let n_to_return = std::cmp::min(L as usize, points.len());
        *num_points_returned = n_to_return as libc::c_int;

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default Matsubara sampling points with custom limit (extended version)
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `n_points` - Maximum number of points requested
/// * `points` - Pre-allocated array to store Matsubara indices (size >= n_points)
/// * `n_points_returned` - Pointer to store actual number of points returned
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if any pointer is null or n_points < 0
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Note
/// Returns min(n_points, actual_default_points) sampling points
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_matsus_ext(
    b: *const spir_basis,
    positive_only: bool,
    n_points: libc::c_int,
    points: *mut i64,
    n_points_returned: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || points.is_null() || n_points_returned.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if n_points < 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let matsu_points = basis.default_matsubara_sampling_points(positive_only);

        // Return min(requested, available) points
        let n_to_return = std::cmp::min(n_points as usize, matsu_points.len());
        std::ptr::copy_nonoverlapping(matsu_points.as_ptr(), points, n_to_return);
        *n_points_returned = n_to_return as libc::c_int;

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::*;
    use crate::sve::*;
    use std::ptr;

    #[test]
    fn test_basis_from_sve() {
        // Create kernel
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        // Compute SVE
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1.0, -1, -1, -1, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);

        // Create basis from SVE (kernel is still required for type info)
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,      // Fermionic
            10.0,   // beta
            1.0,    // omega_max
            1e-6,   // epsilon
            kernel, // kernel required (for type info)
            sve,    // SVE provided (optimization)
            -1,     // no max_size
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        // Get size
        let mut size = 0;
        let status = spir_basis_get_size(basis, &mut size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(size > 0);
        println!("Basis size: {}", size);

        // Get statistics
        let mut stats = -1;
        let status = spir_basis_get_stats(basis, &mut stats);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(stats, 1); // Fermionic

        // Cleanup
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_from_kernel() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            0,    // Bosonic
            10.0, // beta
            1.0,  // omega_max
            1e-6, // epsilon
            kernel,
            ptr::null(), // no SVE (compute from kernel)
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        let mut stats = -1;
        spir_basis_get_stats(basis, &mut stats);
        assert_eq!(stats, 0); // Bosonic

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_tau_sampling() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            0,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );

        // Get number of tau points
        let mut n_taus = 0;
        let status = spir_basis_get_n_default_taus(basis, &mut n_taus);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_taus > 0);
        println!("Number of tau points: {}", n_taus);

        // Get tau points
        let mut taus = vec![0.0; n_taus as usize];
        let status = spir_basis_get_default_taus(basis, taus.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        println!("First 5 tau points:");
        for i in 0..std::cmp::min(5, taus.len()) {
            println!("  tau[{}] = {}", i, taus[i]);
        }

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_matsubara_sampling() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );

        // Get number of Matsubara points (positive only)
        let mut n_matsus = 0;
        let status = spir_basis_get_n_default_matsus(basis, true, &mut n_matsus);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_matsus > 0);
        println!("Number of Matsubara points (positive): {}", n_matsus);

        // Get Matsubara points
        let mut matsus = vec![0i64; n_matsus as usize];
        let status = spir_basis_get_default_matsus(basis, true, matsus.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        println!("First 5 Matsubara indices:");
        for i in 0..std::cmp::min(5, matsus.len()) {
            println!("  n[{}] = {}", i, matsus[i]);
        }

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_omega_sampling() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );

        // Get number of omega points
        let mut n_ws = 0;
        let status = spir_basis_get_n_default_ws(basis, &mut n_ws);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_ws > 0);
        println!("Number of omega points: {}", n_ws);

        // Get omega points
        let mut ws = vec![0.0; n_ws as usize];
        let status = spir_basis_get_default_ws(basis, ws.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        println!("First 5 omega points:");
        for i in 0..std::cmp::min(5, ws.len()) {
            println!("  w[{}] = {}", i, ws[i]);
        }

        // Test singular_values alias
        let mut size = 0;
        let status = spir_basis_get_size(basis, &mut size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut svals = vec![0.0; size as usize];
        let status = spir_basis_get_singular_values(basis, svals.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Verify it matches get_svals
        let mut svals2 = vec![0.0; size as usize];
        let status2 = spir_basis_get_svals(basis, svals2.as_mut_ptr());
        assert_eq!(status2, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(svals, svals2);
        println!("✓ get_singular_values matches get_svals");

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_ext_functions() {
        use crate::kernel::*;

        // Create kernel and basis
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);

        // Test get_default_taus_ext
        let requested_tau = 5; // Request only 5 points
        let mut tau_points = vec![0.0; requested_tau];
        let mut tau_returned = 0;
        let status = spir_basis_get_default_taus_ext(
            basis,
            requested_tau as libc::c_int,
            tau_points.as_mut_ptr(),
            &mut tau_returned,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(tau_returned, requested_tau as libc::c_int);
        println!(
            "✓ get_default_taus_ext returned {} tau points (requested {})",
            tau_returned, requested_tau
        );
        println!("  First 3: {:?}", &tau_points[..3]);

        // Test get_n_default_matsus_ext
        let requested_matsu = 3; // Request only 3 points
        let mut matsu_count = 0;
        let status = spir_basis_get_n_default_matsus_ext(
            basis,
            true, // positive_only
            requested_matsu,
            &mut matsu_count,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(matsu_count, requested_matsu);
        println!("✓ get_n_default_matsus_ext returned count: {}", matsu_count);

        // Test get_default_matsus_ext
        let mut matsu_points = vec![0i64; matsu_count as usize];
        let mut matsu_returned = 0;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            true, // positive_only
            matsu_count,
            matsu_points.as_mut_ptr(),
            &mut matsu_returned,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(matsu_returned, matsu_count);
        println!(
            "✓ get_default_matsus_ext returned {} matsubara points",
            matsu_returned
        );
        println!("  Points: {:?}", matsu_points);

        // Test error case: negative n_points
        let mut bad_returned = 0;
        let status = spir_basis_get_default_taus_ext(basis, -1, ptr::null_mut(), &mut bad_returned);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        println!("✓ Negative n_points correctly rejected");

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }
}
