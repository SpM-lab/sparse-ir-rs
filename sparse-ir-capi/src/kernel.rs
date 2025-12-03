//! Kernel API for C
//!
//! Functions for creating and manipulating kernel objects.

use std::panic::catch_unwind;

use sparse_ir::kernel::SVEHints;

use crate::types::spir_kernel;
use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, StatusCode};

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
pub extern "C" fn spir_kernel_get_lambda(
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
///
/// # Safety
/// This function drops the kernel. The inner KernelType data is automatically freed
/// by the Drop implementation when the spir_kernel structure is dropped.
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_release(kernel: *mut spir_kernel) {
    if !kernel.is_null() {
        unsafe {
            // Drop the spir_kernel structure itself.
            // The Drop implementation will automatically free the inner KernelType data.
            let _ = Box::from_raw(kernel);
        }
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_clone(src: *const spir_kernel) -> *mut spir_kernel {
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

/// Get kernel domain boundaries
///
/// # Arguments
/// * `k` - Kernel object
/// * `xmin` - Pointer to store minimum x value
/// * `xmax` - Pointer to store maximum x value
/// * `ymin` - Pointer to store minimum y value
/// * `ymax` - Pointer to store maximum y value
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if any pointer is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_domain(
    k: *const spir_kernel,
    xmin: *mut f64,
    xmax: *mut f64,
    ymin: *mut f64,
    ymax: *mut f64,
) -> StatusCode {
    if k.is_null() || xmin.is_null() || xmax.is_null() || ymin.is_null() || ymax.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let kernel = &*k;
        let (xmin_val, xmax_val, ymin_val, ymax_val) = kernel.domain();
        *xmin = xmin_val;
        *xmax = xmax_val;
        *ymin = ymin_val;
        *ymax = ymax_val;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get x-segments for SVE discretization hints from a kernel
///
/// This function should be called twice:
/// 1. First call with segments=NULL: set n_segments to the required array size
/// 2. Second call with segments allocated: fill segments[0..n_segments-1] with values
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `segments` - Pointer to store segments array (NULL for first call)
/// * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_segments_x(
    k: *const spir_kernel,
    epsilon: f64,
    segments: *mut f64,
    n_segments: *mut libc::c_int,
) -> StatusCode {
    if k.is_null() || n_segments.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Get SVE hints based on kernel type
        let segs = match kernel.inner() {
            crate::types::KernelType::Logistic(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.segments_x()
            }
            crate::types::KernelType::RegularizedBose(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.segments_x()
            }
        };

        if segments.is_null() {
            // First call: return the number of segments
            *n_segments = (segs.len() - 1) as libc::c_int;
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Second call: copy segments to output array
        if *n_segments < (segs.len() - 1) as libc::c_int {
            return SPIR_INVALID_ARGUMENT;
        }

        for (i, &seg) in segs.iter().enumerate() {
            *segments.add(i) = seg;
        }
        *n_segments = (segs.len() - 1) as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get y-segments for SVE discretization hints from a kernel
///
/// This function should be called twice:
/// 1. First call with segments=NULL: set n_segments to the required array size
/// 2. Second call with segments allocated: fill segments[0..n_segments-1] with values
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `segments` - Pointer to store segments array (NULL for first call)
/// * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_segments_y(
    k: *const spir_kernel,
    epsilon: f64,
    segments: *mut f64,
    n_segments: *mut libc::c_int,
) -> StatusCode {
    if k.is_null() || n_segments.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Get SVE hints based on kernel type
        let segs = match kernel.inner() {
            crate::types::KernelType::Logistic(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.segments_y()
            }
            crate::types::KernelType::RegularizedBose(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.segments_y()
            }
        };

        if segments.is_null() {
            // First call: return the number of segments
            *n_segments = (segs.len() - 1) as libc::c_int;
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Second call: copy segments to output array
        if *n_segments < (segs.len() - 1) as libc::c_int {
            return SPIR_INVALID_ARGUMENT;
        }

        for (i, &seg) in segs.iter().enumerate() {
            *segments.add(i) = seg;
        }
        *n_segments = (segs.len() - 1) as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get the number of singular values hint from a kernel
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `nsvals` - Pointer to store the number of singular values
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or nsvals is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_nsvals(
    k: *const spir_kernel,
    epsilon: f64,
    nsvals: *mut libc::c_int,
) -> StatusCode {
    if k.is_null() || nsvals.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Get SVE hints based on kernel type
        let n = match kernel.inner() {
            crate::types::KernelType::Logistic(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.nsvals()
            }
            crate::types::KernelType::RegularizedBose(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.nsvals()
            }
        };

        *nsvals = n as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get the number of Gauss points hint from a kernel
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `ngauss` - Pointer to store the number of Gauss points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or ngauss is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_ngauss(
    k: *const spir_kernel,
    epsilon: f64,
    ngauss: *mut libc::c_int,
) -> StatusCode {
    if k.is_null() || ngauss.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Get SVE hints based on kernel type
        let n = match kernel.inner() {
            crate::types::KernelType::Logistic(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.ngauss()
            }
            crate::types::KernelType::RegularizedBose(k) => {
                use sparse_ir::kernel::KernelProperties;
                let hints = k.sve_hints::<f64>(epsilon);
                hints.ngauss()
            }
        };

        *ngauss = n as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
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
        let status = spir_kernel_get_lambda(kernel, &mut lambda);

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
        let status = spir_kernel_get_lambda(ptr::null(), &mut lambda);
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

    #[test]
    fn test_kernel_domain() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut xmin = 0.0;
        let mut xmax = 0.0;
        let mut ymin = 0.0;
        let mut ymax = 0.0;
        let status = spir_kernel_get_domain(kernel, &mut xmin, &mut xmax, &mut ymin, &mut ymax);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(xmin, -1.0);
        assert_eq!(xmax, 1.0);
        assert_eq!(ymin, -1.0);
        assert_eq!(ymax, 1.0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_nsvals() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &mut nsvals);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(nsvals > 0);
        assert!(nsvals >= 10);
        assert!(nsvals <= 1000);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_ngauss() {
        let lambda = 10.0;
        let epsilon_coarse = 1e-6;
        let epsilon_fine = 1e-10;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut ngauss_coarse = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_coarse, &mut ngauss_coarse);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss_coarse > 0);
        assert_eq!(ngauss_coarse, 10); // For epsilon >= 1e-8, ngauss should be 10

        let mut ngauss_fine = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_fine, &mut ngauss_fine);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss_fine > 0);
        assert_eq!(ngauss_fine, 16); // For epsilon < 1e-8, ngauss should be 16

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_segments_x() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // First call: get the number of segments
        let mut n_segments = 0;
        let status =
            spir_kernel_get_sve_hints_segments_x(kernel, epsilon, ptr::null_mut(), &mut n_segments);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments > 0);

        // Second call: get the actual segments
        let mut segments = vec![0.0; (n_segments + 1) as usize];
        let mut n_segments_out = n_segments + 1;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            segments.as_mut_ptr(),
            &mut n_segments_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_segments_out, n_segments);

        // Verify segments are valid
        assert_eq!(segments.len(), (n_segments + 1) as usize);
        assert!((segments[0] - (0.0)).abs() < 1e-10);
        assert!((segments[n_segments as usize] - 1.0).abs() < 1e-10);

        // Verify segments are in ascending order
        for i in 1..segments.len() {
            assert!(segments[i] > segments[i - 1]);
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_segments_y() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // First call: get the number of segments
        let mut n_segments = 0;
        let status =
            spir_kernel_get_sve_hints_segments_y(kernel, epsilon, ptr::null_mut(), &mut n_segments);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments > 0);

        // Second call: get the actual segments
        let mut segments = vec![0.0; (n_segments + 1) as usize];
        let mut n_segments_out = n_segments + 1;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            segments.as_mut_ptr(),
            &mut n_segments_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_segments_out, n_segments);

        // Verify segments are valid
        assert_eq!(segments.len(), (n_segments + 1) as usize);
        assert!((segments[0] - (0.0)).abs() < 1e-10);
        assert!((segments[n_segments as usize] - 1.0).abs() < 1e-10);

        // Verify segments are in ascending order
        for i in 1..segments.len() {
            assert!(segments[i] > segments[i - 1]);
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_with_regularized_bose() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Test nsvals
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &mut nsvals);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(nsvals > 0);

        // Test ngauss
        let mut ngauss = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &mut ngauss);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss > 0);

        // Test segments_x
        let mut n_segments_x = 0;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_x,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments_x > 0);

        // Test segments_y
        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_y,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments_y > 0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_error_handling() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Test with nullptr kernel
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(ptr::null(), epsilon, &mut nsvals);
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        // Test with nullptr output parameter
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, ptr::null_mut());
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        // Test with invalid epsilon
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, -1.0, &mut nsvals);
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        spir_kernel_release(kernel);
    }
}
