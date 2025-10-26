//! Kernel API for C
//!
//! Functions for creating and manipulating kernel objects.

use std::panic::catch_unwind;

use crate::types::spir_kernel;
use crate::{StatusCode, SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT};

// Generate common opaque type functions: release, clone, is_assigned, get_raw_ptr

/// Create a new Logistic kernel
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// * Pointer to the newly created kernel object, or NULL if creation fails
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
///
/// # Example (C)
/// ```c
/// int status;
/// spir_kernel* kernel = spir_logistic_kernel_new(10.0, &status);
/// if (kernel != NULL) {
///     // Use kernel...
///     spir_kernel_release(kernel);
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn spir_logistic_kernel_new(
    lambda: f64,
    status: *mut StatusCode,
) -> *mut spir_kernel {
    // Input validation
    if status.is_null() {
        return std::ptr::null_mut();
    }

    if lambda <= 0.0 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Catch panics to prevent unwinding across FFI boundary
    let result = catch_unwind(|| {
        let kernel = spir_kernel::new_logistic(lambda);
        Box::into_raw(Box::new(kernel))
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

/// Create a new RegularizedBose kernel
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// * Pointer to the newly created kernel object, or NULL if creation fails
#[unsafe(no_mangle)]
pub extern "C" fn spir_reg_bose_kernel_new(
    lambda: f64,
    status: *mut StatusCode,
) -> *mut spir_kernel {
    if status.is_null() {
        return std::ptr::null_mut();
    }

    if lambda <= 0.0 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let kernel = spir_kernel::new_regularized_bose(lambda);
        Box::into_raw(Box::new(kernel))
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

/// Get the lambda parameter of a kernel
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `lambda_out` - Pointer to store the lambda value
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if kernel or lambda_out is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_lambda(
    kernel: *const spir_kernel,
    lambda_out: *mut f64,
) -> StatusCode {
    if kernel.is_null() || lambda_out.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let k = &*kernel;
        *lambda_out = k.lambda();
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Compute kernel value K(x, y)
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `x` - First argument (typically in [-1, 1])
/// * `y` - Second argument (typically in [-1, 1])
/// * `out` - Pointer to store the result
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if kernel or out is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_compute(
    kernel: *const spir_kernel,
    x: f64,
    y: f64,
    out: *mut f64,
) -> StatusCode {
    if kernel.is_null() || out.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let k = &*kernel;
        *out = k.compute(x, y);
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Manual release function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_release(kernel: *mut spir_kernel) {
    if !kernel.is_null() {
        unsafe {
            let _ = Box::from_raw(kernel);
        }
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_kernel_clone(src: *const spir_kernel) -> *mut spir_kernel {
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
pub extern "C" fn spir_kernel_is_assigned(obj: *const spir_kernel) -> i32 {
    if obj.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    }));

    result.unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_logistic_kernel_creation() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_regularized_bose_kernel_creation() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(10.0, &mut status);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_lambda() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut lambda = 0.0;
        let status = spir_kernel_lambda(kernel, &mut lambda);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(lambda, 10.0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_compute() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut result = 0.0;
        let status = spir_kernel_compute(kernel, 0.5, 0.5, &mut result);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(result > 0.0); // Kernel should be positive

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_null_pointer_errors() {
        // Null status pointer
        let kernel = spir_logistic_kernel_new(10.0, ptr::null_mut());
        assert!(kernel.is_null());

        // Null kernel pointer
        let mut lambda = 0.0;
        let status = spir_kernel_lambda(ptr::null(), &mut lambda);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
    }

    #[test]
    fn test_invalid_lambda() {
        let mut status = SPIR_COMPUTATION_SUCCESS;

        // Zero lambda
        let kernel = spir_logistic_kernel_new(0.0, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());

        // Negative lambda
        let kernel = spir_logistic_kernel_new(-1.0, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());
    }
}
