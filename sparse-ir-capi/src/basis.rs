//! Basis API
//!
//! Functions for creating and manipulating finite temperature basis objects.

use std::panic::{AssertUnwindSafe, catch_unwind};

use sparse_ir::basis::FiniteTempBasis;

use crate::types::{spir_basis, spir_funcs, spir_kernel, spir_sve_result};
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED,
    SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_FERMIONIC, StatusCode,
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

/// Check if the basis pointer is non-null.
///
/// Note: This only performs a null check. It cannot detect dangling
/// pointers; dereferencing an arbitrary non-null pointer would be
/// undefined behaviour that `catch_unwind` cannot reliably catch.
///
/// # Returns
/// 1 if the pointer is non-null, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_is_assigned(obj: *const spir_basis) -> i32 {
    if obj.is_null() { 0 } else { 1 }
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
    if beta <= 0.0
        || omega_max <= 0.0
        || epsilon <= 0.0
        || !beta.is_finite()
        || !omega_max.is_finite()
        || !epsilon.is_finite()
    {
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
            return Err(SPIR_INVALID_ARGUMENT);
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
                // RegularizedBoseKernel is meaningful only for bosonic statistics;
                // reject the combination at the boundary (the fermionic
                // regularizer would panic downstream when building a DLR).
                return Err(SPIR_NOT_SUPPORTED);
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
            Err(SPIR_INTERNAL_ERROR)
        }
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(code)) => {
            unsafe {
                *status = code;
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

/// Create a finite temperature basis from SVE result and custom regularizer function
///
/// This function creates a basis from a pre-computed SVE result and a custom
/// regularizer function. The regularizer function is used to scale the basis
/// functions in the frequency domain.
///
/// # Arguments
/// * `statistics` - 0 for Bosonic, 1 for Fermionic
/// * `beta` - Inverse temperature (must be > 0)
/// * `omega_max` - Frequency cutoff (must be > 0)
/// * `epsilon` - Accuracy target (must be > 0)
/// * `lambda` - Kernel parameter Λ = β * ωmax (must be > 0)
/// * `ypower` - Power of y in kernel: 0 for `LogisticKernel`, 1 for `RegularizedBoseKernel`.
///              Other values return `SPIR_INVALID_ARGUMENT`.
/// * `conv_radius` - Convergence radius for Fourier transform (currently unused)
/// * `sve` - Pre-computed SVE result (must not be NULL)
/// * `regularizer_funcs` - Custom regularizer function (must not be NULL)
/// * `max_size` - Maximum basis size (-1 for no limit)
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to basis object, or NULL on failure
///
/// # Note
/// The kernel type is determined by `ypower`: 0 selects `LogisticKernel`, 1 selects
/// `RegularizedBoseKernel`. The regularizer function is evaluated for validity but
/// the custom weight is not yet fully integrated into basis construction.
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_new_from_sve_and_regularizer(
    statistics: libc::c_int,
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    lambda: f64,
    ypower: libc::c_int,
    _conv_radius: f64,
    sve: *const spir_sve_result,
    regularizer_funcs: *const spir_funcs,
    max_size: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_basis {
    if status.is_null() {
        return std::ptr::null_mut();
    }

    // Validate inputs
    if beta <= 0.0
        || omega_max <= 0.0
        || epsilon <= 0.0
        || lambda <= 0.0
        || !beta.is_finite()
        || !omega_max.is_finite()
        || !epsilon.is_finite()
        || !lambda.is_finite()
    {
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

    // Must have SVE and regularizer_funcs
    if sve.is_null() || regularizer_funcs.is_null() {
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

    // Validate ypower: 0 => LogisticKernel, 1 => RegularizedBoseKernel
    if ypower != 0 && ypower != 1 {
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

        // Evaluate regularizer_funcs at a test point to verify it's valid
        // (Note: Currently, the custom regularizer is not fully integrated into basis construction)
        let test_omega = omega_max / 2.0;
        let _regularizer_value = match (*regularizer_funcs).eval_continuous(test_omega) {
            Some(values) if !values.is_empty() => values[0],
            _ => {
                // Default to 1.0 if evaluation fails
                1.0
            }
        };

        // Select kernel type based on ypower:
        //   ypower == 0 => LogisticKernel
        //   ypower == 1 => RegularizedBoseKernel
        if ypower == 0 {
            use sparse_ir::kernel::LogisticKernel;
            let kernel = LogisticKernel::new(lambda);

            if statistics == SPIR_STATISTICS_FERMIONIC {
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
        } else {
            // ypower == 1: RegularizedBoseKernel is meaningful only for bosonic
            // statistics; reject the combination at the boundary (the fermionic
            // regularizer would panic downstream when building a DLR).
            if statistics == SPIR_STATISTICS_FERMIONIC {
                return Err(SPIR_NOT_SUPPORTED);
            } else {
                use sparse_ir::kernel::RegularizedBoseKernel;
                let kernel = RegularizedBoseKernel::new(lambda);

                let basis =
                    FiniteTempBasis::<RegularizedBoseKernel, sparse_ir::traits::Bosonic>::from_sve_result(
                        kernel,
                        beta,
                        sve_result,
                        Some(epsilon),
                        max_size_opt,
                    );
                Ok::<*mut spir_basis, StatusCode>(Box::into_raw(Box::new(
                    spir_basis::new_regularized_bose_bosonic(basis),
                )))
            }
        }
    }));

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(code)) => {
            unsafe {
                *status = code;
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
    use std::io::Write;

    if b.is_null() || svals.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let sval_vec = basis.svals();
        debug_println!("spir_basis_get_svals: sval_vec.len() = {}", sval_vec.len());
        std::io::stderr().flush().unwrap();

        // Safety: We assume the caller has allocated enough memory
        // The size should match basis.size() which is checked by Julia side
        if sval_vec.len() > 0 {
            std::ptr::copy_nonoverlapping(sval_vec.as_ptr(), svals, sval_vec.len());
        }
        SPIR_COMPUTATION_SUCCESS
    }));

    match result {
        Ok(code) => code,
        Err(panic_payload) => {
            use std::io::Write;
            debug_eprintln!("Panic in spir_basis_get_svals");
            std::io::stderr().flush().unwrap();
            if let Some(s) = panic_payload.downcast_ref::<String>() {
                debug_eprintln!("Panic message: {}", s);
            } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                debug_eprintln!("Panic message: {}", s);
            }
            std::io::stderr().flush().unwrap();
            SPIR_INTERNAL_ERROR
        }
    }
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
                spir_funcs::from_u_fermionic(basis.u().clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => {
                spir_funcs::from_u_bosonic(basis.u().clone(), beta)
            }
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_u_fermionic(basis.u().clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_u_bosonic(basis.u().clone(), beta)
            }
            // DLR: tau-domain functions using kernel-aware pole weights
            BasisType::DLRFermionic(dlr) => spir_funcs::from_dlr_tau_fermionic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.pole_weights().to_vec(),
                dlr.kernel_ypower(),
            ),
            BasisType::DLRBosonic(dlr) => spir_funcs::from_dlr_tau_bosonic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.pole_weights().to_vec(),
                dlr.kernel_ypower(),
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
            debug_eprintln!("Error in spir_basis_get_u: {}", msg);
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
            BasisType::LogisticFermionic(basis) => spir_funcs::from_v(basis.v().clone(), beta),
            BasisType::LogisticBosonic(basis) => spir_funcs::from_v(basis.v().clone(), beta),
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_v(basis.v().clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => spir_funcs::from_v(basis.v().clone(), beta),
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
            debug_eprintln!("Error in spir_basis_get_v: {}", msg);
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
                spir_funcs::from_uhat_fermionic(basis.uhat().clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => {
                spir_funcs::from_uhat_bosonic(basis.uhat().clone(), beta)
            }
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_uhat_fermionic(basis.uhat().clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_uhat_bosonic(basis.uhat().clone(), beta)
            }
            // DLR: Matsubara-domain functions using discrete poles
            BasisType::DLRFermionic(dlr) => spir_funcs::from_dlr_matsubara_fermionic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.pole_weights().to_vec(),
                dlr.kernel_ypower(),
            ),
            BasisType::DLRBosonic(dlr) => spir_funcs::from_dlr_matsubara_bosonic(
                dlr.poles.clone(),
                beta,
                dlr.wmax,
                dlr.pole_weights().to_vec(),
                dlr.kernel_ypower(),
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
            debug_eprintln!("Error in spir_basis_get_uhat: {}", msg);
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
                spir_funcs::from_uhat_full_fermionic(basis.uhat_full().clone(), beta)
            }
            BasisType::LogisticBosonic(basis) => {
                spir_funcs::from_uhat_full_bosonic(basis.uhat_full().clone(), beta)
            }
            BasisType::RegularizedBoseFermionic(basis) => {
                spir_funcs::from_uhat_full_fermionic(basis.uhat_full().clone(), beta)
            }
            BasisType::RegularizedBoseBosonic(basis) => {
                spir_funcs::from_uhat_full_bosonic(basis.uhat_full().clone(), beta)
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
            debug_eprintln!("Error in spir_basis_get_uhat_full: {}", msg);
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

/// Get the number of default Matsubara sampling points for a given basis size
/// (extended version)
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only non-negative frequencies
/// * `fence` - If true, add fencing points to improve conditioning
/// * `basis_size` - Size of the basis the points are chosen for. When sampling
///   an augmented basis, pass the augmented size; it may differ from the size
///   of `b`.
/// * `n_points_total` - Pointer to store the number of sampling points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if `b` or `n_points_total` is null, or
///   `basis_size < 0`
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Note
/// The count generally differs from `basis_size`: the parity adjustment,
/// `fence` and `positive_only` all change it. Pass it as `points_capacity` to
/// `spir_basis_get_default_matsus_ext` with the same `positive_only`, `fence`
/// and `basis_size`.
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_n_default_matsus_ext(
    b: *const spir_basis,
    positive_only: bool,
    fence: bool,
    basis_size: libc::c_int,
    n_points_total: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || n_points_total.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if basis_size < 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let matsu_points = basis.default_matsubara_sampling_points_with_mitigate(
            positive_only,
            fence,
            basis_size as usize,
        );

        *n_points_total = matsu_points.len() as libc::c_int;

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default Matsubara sampling points for a given basis size (extended
/// version)
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only non-negative frequencies
/// * `fence` - If true, add fencing points to improve conditioning
/// * `basis_size` - Size of the basis the points are chosen for. When sampling
///   an augmented basis, pass the augmented size; it may differ from the size
///   of `b`.
/// * `points_capacity` - Number of elements `points` can hold
/// * `points` - Buffer for the Matsubara indices, or NULL to query the number
///   of points only
/// * `n_points_total` - Pointer to store the number of sampling points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success, including a count query
///   (`points` is NULL, `points_capacity` is ignored)
/// * `SPIR_INVALID_ARGUMENT` (-6) if `b` or `n_points_total` is null, or
///   `basis_size` or `points_capacity` is negative; nothing is written
/// * `SPIR_INVALID_ARGUMENT` (-6) if `points_capacity` is smaller than the
///   number of points; `points` is left untouched and `*n_points_total` is
///   set to the required number
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Note
/// The point set depends only on `positive_only`, `fence` and `basis_size`.
/// `points_capacity` never changes it, and the output is never truncated.
#[unsafe(no_mangle)]
pub extern "C" fn spir_basis_get_default_matsus_ext(
    b: *const spir_basis,
    positive_only: bool,
    fence: bool,
    basis_size: libc::c_int,
    points_capacity: libc::c_int,
    points: *mut i64,
    n_points_total: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || n_points_total.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if basis_size < 0 || points_capacity < 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let basis = &*b;
        let matsu_points = basis.default_matsubara_sampling_points_with_mitigate(
            positive_only,
            fence,
            basis_size as usize,
        );

        *n_points_total = matsu_points.len() as libc::c_int;
        if points.is_null() {
            return SPIR_COMPUTATION_SUCCESS;
        }
        if matsu_points.len() > points_capacity as usize {
            return SPIR_INVALID_ARGUMENT;
        }

        std::ptr::copy_nonoverlapping(matsu_points.as_ptr(), points, matsu_points.len());

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
        debug_println!("Basis size: {}", size);

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
        debug_println!("Number of tau points: {}", n_taus);

        // Get tau points
        let mut taus = vec![0.0; n_taus as usize];
        let status = spir_basis_get_default_taus(basis, taus.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        debug_println!("First 5 tau points:");
        for i in 0..std::cmp::min(5, taus.len()) {
            debug_println!("  tau[{}] = {}", i, taus[i]);
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
        debug_println!("Number of Matsubara points (positive): {}", n_matsus);

        // Get Matsubara points
        let mut matsus = vec![0i64; n_matsus as usize];
        let status = spir_basis_get_default_matsus(basis, true, matsus.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        debug_println!("First 5 Matsubara indices:");
        for i in 0..std::cmp::min(5, matsus.len()) {
            debug_println!("  n[{}] = {}", i, matsus[i]);
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
        debug_println!("Number of omega points: {}", n_ws);

        // Get omega points
        let mut ws = vec![0.0; n_ws as usize];
        let status = spir_basis_get_default_ws(basis, ws.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        debug_println!("First 5 omega points:");
        for i in 0..std::cmp::min(5, ws.len()) {
            debug_println!("  w[{}] = {}", i, ws[i]);
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
        debug_println!("✓ get_singular_values matches get_svals");

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
        debug_println!("  First 3: {:?}", &tau_points[..3]);

        // Test get_n_default_matsus_ext
        let matsu_basis_size = 3;
        let mut matsu_count = 0;
        let status = spir_basis_get_n_default_matsus_ext(
            basis,
            true,  // positive_only
            false, // fence
            matsu_basis_size,
            &mut matsu_count,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(matsu_count > 0);
        debug_println!("✓ get_n_default_matsus_ext returned count: {}", matsu_count);

        // Test get_default_matsus_ext
        // Allocate space based on the count returned by get_n_default_matsus_ext
        let mut matsu_points = vec![0i64; matsu_count as usize];
        let mut matsu_returned = 0;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            true,  // positive_only
            false, // fence
            matsu_basis_size,
            matsu_count,
            matsu_points.as_mut_ptr(),
            &mut matsu_returned,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        // The returned count should match what get_n_default_matsus_ext reported
        assert_eq!(matsu_returned, matsu_count);
        println!(
            "✓ get_default_matsus_ext returned {} matsubara points",
            matsu_returned
        );
        debug_println!("  Points: {:?}", matsu_points);

        // Test error case: negative n_points
        let mut bad_returned = 0;
        let status = spir_basis_get_default_taus_ext(basis, -1, ptr::null_mut(), &mut bad_returned);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        debug_println!("✓ Negative n_points correctly rejected");

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
    fn test_basis_new_from_sve_and_regularizer() {
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

        // Create regularizer_func as spir_funcs
        // For LogisticKernel with Fermionic statistics, regularizer_func(omega) = 1.0
        // We'll create a simple constant function
        let n_segments = 1;
        let segments = [-omega_max, omega_max]; // Full omega range
        let coeffs = [1.0]; // Constant function = 1.0
        let nfuncs = 1;
        let order = 0;

        let mut regularizer_status = SPIR_INTERNAL_ERROR;
        let regularizer_funcs = spir_funcs_from_piecewise_legendre(
            segments.as_ptr(),
            n_segments,
            coeffs.as_ptr(),
            nfuncs,
            order,
            &mut regularizer_status,
        );
        assert_eq!(regularizer_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!regularizer_funcs.is_null());

        // Create basis using new function
        // For LogisticKernel: ypower=0, conv_radius=1.0 (typical values)
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new_from_sve_and_regularizer(
            1, // Fermionic
            beta,
            omega_max,
            epsilon,
            lambda,
            0,   // ypower=0
            1.0, // conv_radius=1.0
            sve,
            regularizer_funcs,
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
            let basis_err = spir_basis_new_from_sve_and_regularizer(
                1,
                beta,
                omega_max,
                epsilon,
                lambda,
                0,
                1.0,
                ptr::null(),
                regularizer_funcs,
                -1,
                &mut basis_status,
            );
            assert_ne!(basis_status, SPIR_COMPUTATION_SUCCESS);
            assert!(basis_err.is_null());
        }

        {
            let mut basis_status = SPIR_INTERNAL_ERROR;
            let basis_err = spir_basis_new_from_sve_and_regularizer(
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
            spir_funcs_release(regularizer_funcs);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
    }

    #[test]
    fn test_basis_rejects_fermionic_regularized_bose() {
        use crate::SPIR_NOT_SUPPORTED;
        use crate::{
            spir_funcs_from_piecewise_legendre, spir_funcs_release, spir_sve_result_release,
        };

        let lambda = 10.0;
        let beta = 1.0;
        let omega_max = lambda / beta;
        let epsilon = 1e-8;

        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(lambda, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // spir_basis_new with RegularizedBoseKernel + fermionic statistics must
        // be rejected at the boundary instead of deferring a downstream panic.
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            omega_max,
            epsilon,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_NOT_SUPPORTED);
        assert!(basis.is_null());

        // Same combination via spir_basis_new_from_sve_and_regularizer (ypower = 1).
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, epsilon, -1, -1, -1, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        let n_segments = 1;
        let segments = [-omega_max, omega_max];
        let coeffs = [1.0];
        let nfuncs = 1;
        let order = 0;

        let mut regularizer_status = SPIR_INTERNAL_ERROR;
        let regularizer_funcs = spir_funcs_from_piecewise_legendre(
            segments.as_ptr(),
            n_segments,
            coeffs.as_ptr(),
            nfuncs,
            order,
            &mut regularizer_status,
        );
        assert_eq!(regularizer_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!regularizer_funcs.is_null());

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new_from_sve_and_regularizer(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            omega_max,
            epsilon,
            lambda,
            1,   // ypower = 1 => RegularizedBoseKernel
            1.0, // conv_radius (unused)
            sve,
            regularizer_funcs,
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_NOT_SUPPORTED);
        assert!(basis.is_null());

        // Bosonic statistics with the same kernel must still succeed.
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_BOSONIC,
            beta,
            omega_max,
            epsilon,
            kernel,
            sve,
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!basis.is_null());

        unsafe {
            spir_basis_release(basis);
            spir_funcs_release(regularizer_funcs);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
    }

    #[test]
    fn test_basis_rejects_nan_params() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        // NaN beta / omega_max / epsilon must not pass the <= 0.0 guards.
        for (beta, omega_max, epsilon) in [
            (f64::NAN, 1.0, 1e-6),
            (10.0, f64::NAN, 1e-6),
            (10.0, 1.0, f64::NAN),
        ] {
            let mut basis_status = SPIR_INTERNAL_ERROR;
            let basis = spir_basis_new(
                SPIR_STATISTICS_FERMIONIC,
                beta,
                omega_max,
                epsilon,
                kernel,
                ptr::null(),
                -1,
                &mut basis_status,
            );
            assert_eq!(basis_status, SPIR_INVALID_ARGUMENT);
            assert!(basis.is_null());
        }

        unsafe {
            spir_kernel_release(kernel);
        }
    }

    #[test]
    fn test_get_default_matsus_ext_capacity_protocol() {
        let beta = 10.0;
        let wmax = 10.0;
        let epsilon = 1e-8;

        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(beta * wmax, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, epsilon, -1, -1, -1, &mut sve_status);
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            wmax,
            epsilon,
            kernel,
            sve,
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);

        let mut basis_size = 0;
        let status = spir_basis_get_size(basis, &mut basis_size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let positive_only = false;
        let fence = true;

        // The count query and a NULL-buffer call report the same count.
        let mut n_total = 0;
        let status = spir_basis_get_n_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            &mut n_total,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_total >= basis_size);

        let mut n_query = -1;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            0,
            ptr::null_mut(),
            &mut n_query,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_query, n_total);

        // A buffer of exactly n_total elements receives the full point set.
        let expected = unsafe {
            (*basis).default_matsubara_sampling_points_with_mitigate(
                positive_only,
                fence,
                basis_size as usize,
            )
        };
        assert_eq!(expected.len(), n_total as usize);
        let mut points = vec![0i64; n_total as usize];
        let mut n_written = -1;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            n_total,
            points.as_mut_ptr(),
            &mut n_written,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_written, n_total);
        assert_eq!(points, expected);

        // A larger buffer receives the same points; the rest is untouched.
        let sentinel = i64::MIN;
        let mut points = vec![sentinel; n_total as usize + 5];
        let mut n_written = -1;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            n_total + 5,
            points.as_mut_ptr(),
            &mut n_written,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_written, n_total);
        assert_eq!(&points[..n_total as usize], &expected[..]);
        assert!(points[n_total as usize..].iter().all(|&p| p == sentinel));

        // A buffer one element short is rejected without being written, and
        // the required count is reported.
        let mut points = vec![sentinel; n_total as usize];
        let mut n_required = -1;
        let status = spir_basis_get_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            n_total - 1,
            points.as_mut_ptr(),
            &mut n_required,
        );
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert_eq!(n_required, n_total);
        assert!(points.iter().all(|&p| p == sentinel));

        // Invalid arguments leave n_points_total untouched.
        for (size, capacity) in [(-1, n_total), (basis_size, -1)] {
            let mut n_untouched = -7;
            let status = spir_basis_get_default_matsus_ext(
                basis,
                positive_only,
                fence,
                size,
                capacity,
                points.as_mut_ptr(),
                &mut n_untouched,
            );
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
            assert_eq!(n_untouched, -7);
        }
        let mut n_untouched = -7;
        let status =
            spir_basis_get_n_default_matsus_ext(basis, positive_only, fence, -1, &mut n_untouched);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert_eq!(n_untouched, -7);
        let status = spir_basis_get_default_matsus_ext(
            basis,
            positive_only,
            fence,
            basis_size,
            n_total,
            points.as_mut_ptr(),
            ptr::null_mut(),
        );
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        let status = spir_basis_get_default_matsus_ext(
            ptr::null(),
            positive_only,
            fence,
            basis_size,
            n_total,
            points.as_mut_ptr(),
            &mut n_untouched,
        );
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert_eq!(n_untouched, -7);

        unsafe {
            spir_basis_release(basis);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
    }

    /// Regression test for sparse-ir-rs#268: points for a basis size larger
    /// than the basis (the augmented-basis case) must be returned in full,
    /// independent of the buffer length.
    #[test]
    fn test_get_default_matsus_ext_basis_size_exceeds_basis() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);

        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_BOSONIC,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);

        let mut size = 0;
        let status = spir_basis_get_size(basis, &mut size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(size, 10);

        // Point sets from the reproduction in sparse-ir-rs#268 (basis size 12).
        let cases: [(bool, &[i64]); 2] = [
            (false, &[-38, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 38]),
            (true, &[0, 2, 4, 6, 8, 12, 38]),
        ];
        for (positive_only, expected) in cases {
            let mut n_total = 0;
            let status =
                spir_basis_get_n_default_matsus_ext(basis, positive_only, false, 12, &mut n_total);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(n_total as usize, expected.len());

            let mut points = vec![0i64; n_total as usize];
            let mut n_written = 0;
            let status = spir_basis_get_default_matsus_ext(
                basis,
                positive_only,
                false,
                12,
                n_total,
                points.as_mut_ptr(),
                &mut n_written,
            );
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(n_written, n_total);
            assert_eq!(points, expected);
        }

        unsafe {
            spir_basis_release(basis);
            spir_kernel_release(kernel);
        }
    }
}
