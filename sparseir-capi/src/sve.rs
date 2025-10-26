//! SVE result API
//!
//! Functions for computing and manipulating Singular Value Expansion (SVE) results.

use std::panic::catch_unwind;

use sparseir_rust::sve::{compute_sve, TworkType};

use crate::types::{spir_kernel, spir_sve_result};
use crate::{StatusCode, SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT};

/// Manual release function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_release(sve: *mut spir_sve_result) {
    if !sve.is_null() {
        unsafe {
            let _ = Box::from_raw(sve);
        }
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_sve_result_clone(src: *const spir_sve_result) -> *mut spir_sve_result {
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
pub extern "C" fn spir_sve_result_is_assigned(obj: *const spir_sve_result) -> i32 {
    if obj.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    }));

    result.unwrap_or(0)
}

/// Compute Singular Value Expansion (SVE) of a kernel (libsparseir compatible)
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `cutoff` - Cutoff value for singular values (-1 for default: 2*sqrt(machine_epsilon))
/// * `lmax` - Maximum number of Legendre polynomials (currently ignored, auto-determined)
/// * `n_gauss` - Number of Gauss points for integration (currently ignored, auto-determined)
/// * `Twork` - Working precision: 0=Float64, 1=Float64x2, -1=Auto
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to SVE result, or NULL on failure
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
///
/// # Note
/// Parameters `lmax` and `n_gauss` are accepted for libsparseir compatibility but
/// currently ignored. The Rust implementation automatically determines optimal values.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_new(
    k: *const spir_kernel,
    epsilon: f64,
    cutoff: f64,
    _lmax: libc::c_int,
    _n_gauss: libc::c_int,
    twork: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_sve_result {
    // Input validation
    if status.is_null() {
        return std::ptr::null_mut();
    }

    if k.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if epsilon <= 0.0 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Convert twork
    let twork_type = match twork {
        0 => TworkType::Float64,
        1 => TworkType::Float64X2,
        -1 => TworkType::Auto,
        _ => {
            unsafe {
                *status = SPIR_INVALID_ARGUMENT;
            }
            return std::ptr::null_mut();
        }
    };

    // Convert cutoff (-1 means default)
    let cutoff_opt = if cutoff < 0.0 { None } else { Some(cutoff) };

    // Catch panics to prevent unwinding across FFI boundary
    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Dispatch based on kernel type
        let sve_result = if let Some(logistic) = kernel.as_logistic() {
            compute_sve(
                **logistic, epsilon, cutoff_opt, None, // max_num_svals auto-determined
                twork_type,
            )
        } else if let Some(reg_bose) = kernel.as_regularized_bose() {
            compute_sve(
                **reg_bose, epsilon, cutoff_opt, None, // max_num_svals auto-determined
                twork_type,
            )
        } else {
            return Err("Unknown kernel type");
        };

        let sve_wrapper = spir_sve_result::new(sve_result);
        Ok(Box::into_raw(Box::new(sve_wrapper)))
    });

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

/// Get the number of singular values in an SVE result
///
/// # Arguments
/// * `sve` - SVE result object
/// * `size` - Pointer to store the size
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if sve or size is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_get_size(
    sve: *const spir_sve_result,
    size: *mut libc::c_int,
) -> StatusCode {
    if sve.is_null() || size.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let s = &*sve;
        *size = s.size() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Truncate an SVE result based on epsilon and max_size
///
/// This function creates a new SVE result containing only the singular values
/// that are larger than `epsilon * s[0]`, where `s[0]` is the largest singular value.
/// The result can also be limited to a maximum size.
///
/// # Arguments
/// * `sve` - Source SVE result object
/// * `epsilon` - Relative threshold for truncation (singular values < epsilon * s[0] are removed)
/// * `max_size` - Maximum number of singular values to keep (-1 for no limit)
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to new truncated SVE result, or NULL on failure
/// * Status code:
///   - `SPIR_COMPUTATION_SUCCESS` (0) on success
///   - `SPIR_INVALID_ARGUMENT` (-6) if sve or status is null, or epsilon is invalid
///   - `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
/// The returned pointer must be freed with `spir_sve_result_release()`.
///
/// # Example (C)
/// ```c
/// spir_sve_result* sve = spir_sve_result_new(kernel, 1e-10, -1.0, 0, 0, -1, &status);
///
/// // Truncate to keep only singular values > 1e-8 * s[0], max 50 values
/// spir_sve_result* sve_truncated = spir_sve_result_truncate(sve, 1e-8, 50, &status);
///
/// // Use truncated result...
///
/// spir_sve_result_release(sve_truncated);
/// spir_sve_result_release(sve);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_truncate(
    sve: *const spir_sve_result,
    epsilon: f64,
    max_size: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_sve_result {
    // Input validation
    if status.is_null() {
        return std::ptr::null_mut();
    }

    if sve.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if epsilon < 0.0 || !epsilon.is_finite() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| unsafe {
        let sve_ref = &*sve;

        // Convert max_size (-1 means no limit)
        let max_size_opt = if max_size < 0 {
            None
        } else {
            Some(max_size as usize)
        };

        // Extract truncated parts using SVEResult::part
        let (u_part, s_part, v_part) = sve_ref.inner.part(Some(epsilon), max_size_opt);

        // Create new SVE result with truncated data
        let sve_truncated = sparseir_rust::sve::SVEResult::new(
            u_part, s_part, v_part, epsilon, // Use provided epsilon for new result
        );

        // Wrap in C-API type
        let sve_wrapper = spir_sve_result {
            inner: std::sync::Arc::new(sve_truncated),
        };

        Box::into_raw(Box::new(sve_wrapper))
    });

    match result {
        Ok(ptr) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Get singular values from an SVE result
///
/// # Arguments
/// * `sve` - SVE result object
/// * `svals` - Pre-allocated array to store singular values (size must be >= result size)
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if sve or svals is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_get_svals(
    sve: *const spir_sve_result,
    svals: *mut f64,
) -> StatusCode {
    if sve.is_null() || svals.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let s = &*sve;
        let sval_slice = s.svals();
        std::ptr::copy_nonoverlapping(sval_slice.as_ptr(), svals, sval_slice.len());
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::*;
    use std::ptr;

    #[test]
    fn test_sve_result_logistic() {
        // Create kernel
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Compute SVE
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(
            kernel,
            1e-6, // epsilon
            -1.0, // cutoff (default)
            -1,   // lmax (auto)
            -1,   // n_gauss (auto)
            -1,   // Twork (auto)
            &mut sve_status,
        );
        assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve.is_null());

        // Get size
        let mut size = 0;
        let status = spir_sve_result_get_size(sve, &mut size);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(size > 0);
        println!("SVE size: {}", size);

        // Get singular values
        let mut svals = vec![0.0; size as usize];
        let status = spir_sve_result_get_svals(sve, svals.as_mut_ptr());
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Check singular values are positive and decreasing
        assert!(svals[0] > 0.0);
        for i in 1..svals.len() {
            assert!(svals[i] <= svals[i - 1]);
        }

        // Cleanup
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_sve_result_truncate() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);

        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1.0, -1, -1, -1, &mut sve_status);

        let mut size = 0;
        spir_sve_result_get_size(sve, &mut size);

        // Truncate to half size
        let mut truncate_status = SPIR_INTERNAL_ERROR;
        let sve_truncated = spir_sve_result_truncate(sve, 1e-4, size / 2, &mut truncate_status);
        assert_eq!(truncate_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve_truncated.is_null());

        let mut new_size = 0;
        spir_sve_result_get_size(sve_truncated, &mut new_size);
        assert!(new_size <= size / 2);

        spir_sve_result_release(sve_truncated);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_sve_null_pointers() {
        // Null kernel
        let mut status = SPIR_COMPUTATION_SUCCESS;
        let sve = spir_sve_result_new(ptr::null(), 1e-6, -1.0, -1, -1, -1, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(sve.is_null());

        // Null size pointer
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1.0, -1, -1, -1, &mut sve_status);

        let status = spir_sve_result_get_size(sve, ptr::null_mut());
        assert_eq!(status, SPIR_INVALID_ARGUMENT);

        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
}
