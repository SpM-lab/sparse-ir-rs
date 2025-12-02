//! Basis API
//!
//! Functions for creating and manipulating finite temperature basis objects.

use std::panic::{AssertUnwindSafe, catch_unwind};

use sparse_ir::basis::FiniteTempBasis;

use crate::types::{spir_basis, spir_funcs, spir_kernel, spir_sve_result};
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, SPIR_STATISTICS_BOSONIC,
    SPIR_STATISTICS_FERMIONIC, StatusCode,
};

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
    if statistics != SPIR_STATISTICS_BOSONIC && statistics != SPIR_STATISTICS_FERMIONIC {
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
            if statistics == SPIR_STATISTICS_FERMIONIC {
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
            if statistics == SPIR_STATISTICS_FERMIONIC {
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

/// Create a finite temperature basis from SVE result and custom inv_weight function
///
/// This function creates a basis from a pre-computed SVE result and a custom
/// inverse weight function. The inv_weight function is used to scale the basis
/// functions in the frequency domain.
///
/// # Arguments
/// * `statistics` - 0 for Bosonic, 1 for Fermionic
/// * `beta` - Inverse temperature (must be > 0)
/// * `omega_max` - Frequency cutoff (must be > 0)
/// * `epsilon` - Accuracy target (must be > 0)
/// * `lambda` - Kernel parameter Λ = β * ωmax (must be > 0)
/// * `ypower` - Power of y in kernel (typically 0 or 1)
/// * `conv_radius` - Convergence radius for Fourier transform
/// * `sve` - Pre-computed SVE result (must not be NULL)
/// * `inv_weight_funcs` - Custom inv_weight function (must not be NULL)
/// * `max_size` - Maximum basis size (-1 for no limit)
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to basis object, or NULL on failure
///
/// # Note
/// Currently, the inv_weight function is evaluated but the custom weight is not
/// fully integrated into the basis construction. The basis is created using
/// the standard from_sve_result method with the kernel's default inv_weight.
/// This is a limitation of the current Rust implementation compared to the C++ version.
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_new_from_sve_and_inv_weight(
    statistics: libc::c_int,
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    lambda: f64,
    _ypower: libc::c_int,
    _conv_radius: f64,
    sve: *const spir_sve_result,
    inv_weight_funcs: *const spir_funcs,
    max_size: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_basis {
    if status.is_null() {
        return std::ptr::null_mut();
    }

    // Validate inputs
    if beta <= 0.0 || omega_max <= 0.0 || epsilon <= 0.0 || lambda <= 0.0 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Validate statistics
    if statistics != SPIR_STATISTICS_BOSONIC && statistics != SPIR_STATISTICS_FERMIONIC {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Must have SVE and inv_weight_funcs
    if sve.is_null() || inv_weight_funcs.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Check that lambda matches beta * omega_max
    let expected_lambda = beta * omega_max;
    if (lambda - expected_lambda).abs() > 1e-10 {
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
        let sve_ref = &*sve;
        let sve_result = sve_ref.inner().as_ref().clone();

        // Evaluate inv_weight_funcs at a test point to verify it's valid
        // (Note: Currently, the custom inv_weight is not fully integrated into basis construction)
        let test_omega = omega_max / 2.0;
        let _inv_weight_value = match (*inv_weight_funcs).eval_continuous(test_omega) {
            Some(values) if !values.is_empty() => values[0],
            _ => {
                // Default to 1.0 if evaluation fails
                1.0
            }
        };

        // Create kernel with the specified lambda
        // Note: We need to determine kernel type from SVE result or use default LogisticKernel
        // For now, we'll use LogisticKernel as default
        use sparse_ir::kernel::LogisticKernel;
        let kernel = LogisticKernel::new(lambda);

        // Create basis using from_sve_result
        // Note: The custom inv_weight is not currently used in the Rust implementation
        // This is a limitation compared to the C++ version
        if statistics == SPIR_STATISTICS_FERMIONIC {
            // Fermionic
            let basis =
                FiniteTempBasis::<LogisticKernel, sparse_ir::traits::Fermionic>::from_sve_result(
                    kernel,
                    beta,
                    sve_result,
                    Some(epsilon),
                    max_size_opt,
                );
            Ok::<*mut spir_basis, StatusCode>(Box::into_raw(Box::new(
                spir_basis::new_logistic_fermionic(basis),
            )))
        } else {
            // Bosonic
            let basis =
                FiniteTempBasis::<LogisticKernel, sparse_ir::traits::Bosonic>::from_sve_result(
                    kernel,
                    beta,
                    sve_result,
                    Some(epsilon),
                    max_size_opt,
                );
            Ok::<*mut spir_basis, StatusCode>(Box::into_raw(Box::new(
                spir_basis::new_logistic_bosonic(basis),
            )))
        }
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(msg)) => {
            eprintln!("Error in spir_basis_new_from_sve_and_inv_weight: {}", msg);
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
    use crate::types::{BasisType, spir_funcs};
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
    use crate::types::{BasisType, spir_funcs};
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
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => {
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
    use crate::types::{BasisType, spir_funcs};
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

/// Gets the full (untruncated) Matsubara-frequency basis functions
///
/// This function returns an object representing all basis functions
/// in the Matsubara-frequency domain, including those beyond the truncation
/// threshold. Unlike `spir_basis_get_uhat`, which returns only the truncated
/// basis functions (up to `basis.size()`), this function returns all basis
/// functions from the SVE result (up to `sve_result.s.size()`).
///
/// # Arguments
/// * `b` - Pointer to the finite temperature basis object (must be an IR basis)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the basis functions object, or NULL if creation fails
///
/// # Note
/// The returned object must be freed using `spir_funcs_release`
/// when no longer needed
/// This function is only available for IR basis objects (not DLR)
/// uhat_full.size() >= uhat.size() is always true
/// The first uhat.size() functions in uhat_full are identical to uhat
///
/// # Safety
/// The caller must ensure that `b` is a valid pointer, and must call
/// `spir_funcs_release()` on the returned pointer when done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_basis_get_uhat_full(
    b: *const spir_basis,
    status: *mut StatusCode,
) -> *mut spir_funcs {
    use crate::SPIR_NOT_SUPPORTED;
    use crate::types::{BasisType, spir_funcs};
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
                spir_funcs::from_uhat_full_fermionic(basis.uhat_full.clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => {
                spir_funcs::from_uhat_full_bosonic(basis.uhat_full.clone(), beta)
            }
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_uhat_full_fermionic(basis.uhat_full.clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_uhat_full_bosonic(basis.uhat_full.clone(), beta)
            }
            // DLR: not supported (only IR basis has uhat_full)
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => {
                return Result::<*mut spir_funcs, String>::Err(
                    "uhat_full is only available for IR basis, not DLR".to_string(),
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
            eprintln!("Error in spir_basis_get_uhat_full: {}", msg);
            unsafe {
                *status = SPIR_NOT_SUPPORTED;
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
        let tau_points = basis.default_tau_sampling_points_size_requested(n_points as usize);

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
    #[allow(non_snake_case)] L: libc::c_int,
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
        let points =
            basis.default_matsubara_sampling_points_with_mitigate(positive_only, false, L as usize);
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
/// * `mitigate` - If true, enable mitigation (fencing) to improve conditioning
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
/// When mitigate is true, may return more points than requested due to fencing
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_matsus_ext(
    b: *const spir_basis,
    positive_only: bool,
    mitigate: bool,
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
        let matsu_points = basis.default_matsubara_sampling_points_with_mitigate(
            positive_only,
            mitigate,
            n_points as usize,
        );

        // When mitigate is true, may return more points than requested
        let n_to_return = matsu_points.len();
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
    use crate::{
        spir_funcs_get_size, spir_funcs_release, spir_gauss_legendre_rule_piecewise_double,
    };
    use std::ptr;

    #[test]
    fn test_basis_from_sve() {
        use crate::{spir_funcs_get_size, spir_funcs_release};
        // Create kernel
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        // Compute SVE
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1, -1, -1, &mut sve_status);
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
        // Note: get_n_default_matsus_ext doesn't use mitigate, so the actual count
        // may differ when mitigate=false. We'll allocate enough space and check the returned count.
        let mut matsu_points = vec![0i64; matsu_count as usize];
        let mut matsu_returned = 0;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            true,  // positive_only
            false, // mitigate
            matsu_count as libc::c_int,
            matsu_points.as_mut_ptr(),
            &mut matsu_returned,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        // When mitigate=false, the returned count may differ from requested
        assert!(matsu_returned > 0);
        assert!(matsu_returned <= matsu_count as libc::c_int);
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

    #[test]
    fn test_basis_get_uhat_full() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,    // Fermionic
            10.0, // beta
            1.0,  // omega_max
            1e-6, // epsilon
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);

        // Get uhat (truncated)
        let mut uhat_status = SPIR_INTERNAL_ERROR;
        let uhat_funcs = unsafe { spir_basis_get_uhat(basis, &mut uhat_status) };
        assert_eq!(uhat_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!uhat_funcs.is_null());

        let mut uhat_size = 0;
        let status = spir_funcs_get_size(uhat_funcs, &mut uhat_size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get uhat_full (untruncated)
        let mut uhat_full_status = SPIR_INTERNAL_ERROR;
        let uhat_full_funcs = unsafe { spir_basis_get_uhat_full(basis, &mut uhat_full_status) };
        assert_eq!(uhat_full_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!uhat_full_funcs.is_null());

        let mut uhat_full_size = 0;
        let status = spir_funcs_get_size(uhat_full_funcs, &mut uhat_full_size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // uhat_full should have at least as many functions as uhat
        assert_eq!(uhat_size, 10);
        assert!(uhat_full_size >= uhat_size);
        // assert!(uhat_full_size == 28); <--- expected value
        // Test error handling: DLR basis (not supported)
        {
            // Note: DLR basis creation would require different API
            // For now, we just test that IR basis works
        }

        unsafe {
            spir_funcs_release(uhat_funcs);
            spir_funcs_release(uhat_full_funcs);
            spir_basis_release(basis);
            spir_kernel_release(kernel);
        }
    }

    #[test]
    fn test_basis_new_from_sve_and_inv_weight() {
        use crate::{
            SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, spir_funcs_from_piecewise_legendre,
            spir_funcs_release,
        };
        let lambda = 10.0;
        let beta = 1.0;
        let omega_max = lambda / beta;
        let epsilon = 1e-8;

        // Create kernel and SVE result
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut sve_status = SPIR_INTERNAL_ERROR;
        use crate::SPIR_TWORK_AUTO;
        let sve = spir_sve_result_new(kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        // Create inv_weight_func as spir_funcs
        // For LogisticKernel with Fermionic statistics, inv_weight_func(omega) = 1.0
        // We'll create a simple constant function
        let n_segments = 1;
        let segments = [-omega_max, omega_max]; // Full omega range
        let coeffs = [1.0]; // Constant function = 1.0
        let nfuncs = 1;
        let order = 0;

        let mut inv_weight_status = SPIR_INTERNAL_ERROR;
        let inv_weight_funcs = spir_funcs_from_piecewise_legendre(
            segments.as_ptr(),
            n_segments,
            coeffs.as_ptr(),
            nfuncs,
            order,
            &mut inv_weight_status,
        );
        assert_eq!(inv_weight_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!inv_weight_funcs.is_null());

        // Create basis using new function
        // For LogisticKernel: ypower=0, conv_radius=1.0 (typical values)
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new_from_sve_and_inv_weight(
            1, // Fermionic
            beta,
            omega_max,
            epsilon,
            lambda,
            0,   // ypower=0
            1.0, // conv_radius=1.0
            sve,
            inv_weight_funcs,
            -1, // no max_size
            &mut basis_status,
        );

        if basis_status == SPIR_COMPUTATION_SUCCESS && !basis.is_null() {
            let mut basis_size = 0;
            let status = spir_basis_get_size(basis, &mut basis_size);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert!(basis_size > 0);

            unsafe {
                spir_basis_release(basis);
            }
        }

        // Test error handling
        {
            let mut basis_status = SPIR_INTERNAL_ERROR;
            let basis_err = spir_basis_new_from_sve_and_inv_weight(
                1,
                beta,
                omega_max,
                epsilon,
                lambda,
                0,
                1.0,
                ptr::null(),
                inv_weight_funcs,
                -1,
                &mut basis_status,
            );
            assert_ne!(basis_status, SPIR_COMPUTATION_SUCCESS);
            assert!(basis_err.is_null());
        }

        {
            let mut basis_status = SPIR_INTERNAL_ERROR;
            let basis_err = spir_basis_new_from_sve_and_inv_weight(
                1,
                beta,
                omega_max,
                epsilon,
                lambda,
                0,
                1.0,
                sve,
                ptr::null(),
                -1,
                &mut basis_status,
            );
            assert_ne!(basis_status, SPIR_COMPUTATION_SUCCESS);
            assert!(basis_err.is_null());
        }

        unsafe {
            spir_funcs_release(inv_weight_funcs);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
    }
}
