//! SVE result API
//!
//! Functions for computing and manipulating Singular Value Expansion (SVE) results.

use std::panic::catch_unwind;

use sparse_ir::sve::{TworkType, compute_sve};

use crate::types::{spir_kernel, spir_sve_result};
use crate::utils::checked_len;
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, SPIR_INVALID_DIMENSION,
    StatusCode,
};

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
pub extern "C" fn spir_sve_result_clone(src: *const spir_sve_result) -> *mut spir_sve_result {
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

/// Check if the SVE result pointer is non-null.
///
/// Note: This only performs a null check. It cannot detect dangling
/// pointers; dereferencing an arbitrary non-null pointer would be
/// undefined behaviour that `catch_unwind` cannot reliably catch.
///
/// # Returns
/// 1 if the pointer is non-null, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_is_assigned(obj: *const spir_sve_result) -> i32 {
    if obj.is_null() { 0 } else { 1 }
}

/// Compute Singular Value Expansion (SVE) of a kernel (libsparseir compatible)
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
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
/// The singular value truncation cutoff is automatically set to 2 * machine epsilon
/// of the working precision (about 4.44e-16 for Float64 and 4.93e-32 for Float64x2),
/// as in libsparseir: singular values smaller than this cutoff times the largest
/// singular value are discarded.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_new(
    k: *const spir_kernel,
    epsilon: f64,
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

    if epsilon <= 0.0 || !epsilon.is_finite() {
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

    // Catch panics to prevent unwinding across FFI boundary
    // Force flush debug output before entering catch_unwind
    if crate::is_debug_enabled() {
        eprintln!(
            "[SPARSEIR DEBUG] spir_sve_result_new: called with epsilon={}, twork={}, kernel_ptr={:p}",
            epsilon, twork, k
        );
        std::io::Write::flush(&mut std::io::stderr()).ok();
    }
    let result = catch_unwind(|| unsafe {
        let kernel = &*k;
        if crate::is_debug_enabled() {
            eprintln!("[SPARSEIR DEBUG] spir_sve_result_new: kernel type determined");
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }

        // Dispatch based on kernel type
        // cutoff = None selects the default 2 * machine epsilon of the working precision
        let sve_result = if let Some(logistic) = kernel.as_logistic() {
            debug_println!("spir_sve_result_new: computing SVE for LogisticKernel");
            compute_sve(
                **logistic, epsilon, None,
                None, // cutoff=None (auto), max_num_svals auto-determined
                twork_type,
            )
        } else if let Some(reg_bose) = kernel.as_regularized_bose() {
            debug_println!("spir_sve_result_new: computing SVE for RegularizedBoseKernel");
            compute_sve(
                **reg_bose, epsilon, None,
                None, // cutoff=None (auto), max_num_svals auto-determined
                twork_type,
            )
        } else {
            debug_eprintln!("spir_sve_result_new: Unknown kernel type");
            return Err("Unknown kernel type");
        };

        debug_println!("spir_sve_result_new: SVE computation completed, creating wrapper");
        let sve_wrapper = spir_sve_result::new(sve_result);
        debug_println!("spir_sve_result_new: wrapper created successfully");
        Ok(Box::into_raw(Box::new(sve_wrapper)))
    });

    match result {
        Ok(Ok(ptr)) => {
            unsafe {
                *status = SPIR_COMPUTATION_SUCCESS;
            }
            ptr
        }
        Ok(Err(msg)) => {
            debug_eprintln!("Error in spir_sve_result_new: {}", msg);
            unsafe {
                *status = SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
        Err(panic_payload) => {
            debug_eprintln!("Panic in spir_sve_result_new");
            // Try to extract panic message if possible
            if let Some(s) = panic_payload.downcast_ref::<String>() {
                debug_eprintln!("Panic message: {}", s);
            } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                debug_eprintln!("Panic message: {}", s);
            } else {
                debug_eprintln!("Panic occurred but message could not be extracted");
            }
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
/// that are larger than `epsilon * s\[0\]`, where `s\[0\]` is the largest singular value.
/// The result can also be limited to a maximum size.
///
/// # Arguments
/// * `sve` - Source SVE result object
/// * `epsilon` - Relative threshold for truncation (singular values < epsilon * s\[0\] are removed)
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
/// spir_sve_result* sve = spir_sve_result_new(kernel, 1e-10, 0, 0, -1, &status);
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
        let (u_part, s_part, v_part) = sve_ref.inner().part(Some(epsilon), max_size_opt);

        // Create new SVE result with truncated data
        let sve_truncated = sparse_ir::sve::SVEResult::new(
            u_part, s_part, v_part, epsilon, // Use provided epsilon for new result
        );

        // Wrap in C-API type
        let sve_wrapper = spir_sve_result::new(sve_truncated);

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

/// Read and validate the `nx * ny` entries of a matrix passed to the
/// `spir_sve_result_from_matrix*` functions
///
/// # Errors
/// * `SPIR_INVALID_DIMENSION` if the matrix is too large to be addressed
/// * `SPIR_INVALID_ARGUMENT` if an entry is NaN or infinite: the SVD never
///   converges on such a matrix
///
/// # Safety
/// `matrix` must be non-null and point to `nx * ny` initialized `f64`s that
/// stay valid for `'a`.
unsafe fn validated_kernel_matrix<'a>(
    matrix: *const f64,
    nx: usize,
    ny: usize,
) -> Result<&'a [f64], StatusCode> {
    let len = checked_len::<f64>(&[nx, ny]).ok_or(SPIR_INVALID_DIMENSION)?;
    // SAFETY: `matrix` is non-null and holds `nx * ny` elements (caller
    // contract), and `checked_len` bounds the byte size by `isize::MAX`.
    let values = unsafe { std::slice::from_raw_parts(matrix, len) };
    if values.iter().all(|x| x.is_finite()) {
        Ok(values)
    } else {
        Err(SPIR_INVALID_ARGUMENT)
    }
}

/// Read and validate the `n_segments + 1` segment boundaries passed to the
/// `spir_sve_result_from_matrix*` functions
///
/// # Errors
/// * `SPIR_INVALID_DIMENSION` if the array is too large to be addressed
/// * `SPIR_INVALID_ARGUMENT` if a boundary is NaN or infinite, or the
///   boundaries are not strictly increasing
///
/// # Safety
/// `segments` must be non-null and point to `n_segments + 1` initialized
/// `f64`s that stay valid for `'a`, and `n_segments` must be positive.
unsafe fn validated_segments<'a>(
    segments: *const f64,
    n_segments: libc::c_int,
) -> Result<&'a [f64], StatusCode> {
    // `n_segments` is positive, so neither the cast nor the `+ 1` overflows.
    let len = n_segments as usize + 1;
    checked_len::<f64>(&[len]).ok_or(SPIR_INVALID_DIMENSION)?;
    // SAFETY: `segments` is non-null and holds `len` elements (caller
    // contract), and `checked_len` bounds the byte size by `isize::MAX`.
    let segments = unsafe { std::slice::from_raw_parts(segments, len) };
    let finite = segments.iter().all(|s| s.is_finite());
    if finite && segments.windows(2).all(|w| w[0] < w[1]) {
        Ok(segments)
    } else {
        Err(SPIR_INVALID_ARGUMENT)
    }
}

/// Create a SVE result from a discretized kernel matrix
///
/// This function performs singular value expansion (SVE) on a discretized kernel
/// matrix K. The matrix K should already be in the appropriate form (no weight
/// application needed). The function supports both double and DDouble precision
/// based on whether K_low is provided.
///
/// # Arguments
/// * `K_high` - High part of the kernel matrix (required, size: nx * ny,
///   finite entries)
/// * `K_low` - Low part of the kernel matrix (optional, nullptr for double
///   precision; finite entries)
/// * `nx` - Number of rows in the matrix
/// * `ny` - Number of columns in the matrix
/// * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `segments_x` - X-direction segments (array of boundary points, size:
///   n_segments_x + 1, finite and strictly increasing)
/// * `n_segments_x` - Number of segments in x direction (boundary points - 1)
/// * `segments_y` - Y-direction segments (array of boundary points, size:
///   n_segments_y + 1, finite and strictly increasing)
/// * `n_segments_y` - Number of segments in y direction (boundary points - 1)
/// * `n_gauss` - Number of Gauss points per segment
/// * `epsilon` - Target accuracy
/// * `status` - Pointer to store status code
///
/// # Returns
/// Pointer to SVE result on success, nullptr on failure. If `status` is
/// non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `K_high`, `segments_x` or `segments_y` is NULL,
///   a size is less than 1, `epsilon` is not positive and finite, an entry of
///   `K_high` or `K_low` is NaN or infinite, or the segments are not finite
///   and strictly increasing
/// - SPIR_INVALID_DIMENSION if the matrix is too large to be addressed
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The arrays are validated before the SVE is computed.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_from_matrix(
    #[allow(non_snake_case)] K_high: *const f64,
    #[allow(non_snake_case)] K_low: *const f64,
    nx: libc::c_int,
    ny: libc::c_int,
    order: libc::c_int,
    segments_x: *const f64,
    n_segments_x: libc::c_int,
    segments_y: *const f64,
    n_segments_y: libc::c_int,
    n_gauss: libc::c_int,
    epsilon: f64,
    status: *mut StatusCode,
) -> *mut spir_sve_result {
    use crate::utils::MemoryOrder;
    use sparse_ir::gauss::legendre;
    use sparse_ir::poly::PiecewiseLegendrePolyVector;
    use sparse_ir::sve::SVEResult;
    use sparse_ir::tsvd::compute_svd_dtensor;
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if K_high.is_null() || segments_x.is_null() || segments_y.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if nx < 1 || ny < 1 || n_segments_x < 1 || n_segments_y < 1 || n_gauss < 1 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Read and validate every array before computing anything: the SVD never
    // converges on a NaN or an infinity (it iterated forever before the TSVD
    // bounded its iteration). The sizes are positive (checked above).
    // Note: n_segments_x is the number of segments (boundary points - 1),
    // matching C++ API behavior: C++ uses segments_x[0..n_segments_x] (n_segments_x + 1 elements)
    let (nx_u, ny_u) = (nx as usize, ny as usize);
    let validated = (|| -> Result<_, StatusCode> {
        // SAFETY: the pointers are non-null (checked above, `K_low` here) and
        // hold the documented number of elements (caller contract); the sizes
        // are positive.
        unsafe {
            let k_high = validated_kernel_matrix(K_high, nx_u, ny_u)?;
            let k_low = if K_low.is_null() {
                None
            } else {
                Some(validated_kernel_matrix(K_low, nx_u, ny_u)?)
            };
            let segs_x = validated_segments(segments_x, n_segments_x)?;
            let segs_y = validated_segments(segments_y, n_segments_y)?;
            Ok((k_high, k_low, segs_x, segs_y))
        }
    })();
    let (k_high_slice, k_low_slice, segs_x_slice, segs_y_slice) = match validated {
        Ok(inputs) => inputs,
        Err(code) => {
            unsafe {
                *status = code;
            }
            return std::ptr::null_mut();
        }
    };

    let result = catch_unwind(|| {
        // Reconstruct Gauss rules
        let rule_base_dd = legendre::<sparse_ir::Df64>(n_gauss as usize);

        // DDouble precision if K_low is provided
        if let Some(k_low_slice) = k_low_slice {
            // DDouble precision path
            use sparse_ir::Df64;
            use sparse_ir::numeric::CustomNumeric;

            // Convert segments to DDouble
            let segs_x_dd: Vec<Df64> = segs_x_slice.iter().map(|&x| Df64::from(x)).collect();
            let segs_y_dd: Vec<Df64> = segs_y_slice.iter().map(|&y| Df64::from(y)).collect();

            // Create piecewise Gauss rules
            let gauss_x_dd = rule_base_dd.piecewise(&segs_x_dd);
            let gauss_y_dd = rule_base_dd.piecewise(&segs_y_dd);

            // Convert matrix from C array to DTensor
            let memory_order = MemoryOrder::from_c_int(order).unwrap_or(MemoryOrder::RowMajor);
            let mut matrix =
                mdarray::DTensor::<Df64, 2>::from_elem([nx as usize, ny as usize], Df64::new(0.0));

            match memory_order {
                MemoryOrder::RowMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = i * (ny as usize) + j;
                            // Use unsafe new_full since K_high and K_low are already properly compensated
                            matrix[[i, j]] =
                                unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                        }
                    }
                }
                MemoryOrder::ColumnMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = j * (nx as usize) + i;
                            // Use unsafe new_full since K_high and K_low are already properly compensated
                            matrix[[i, j]] =
                                unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                        }
                    }
                }
            }

            // Prepare f64 segments for polynomial conversion
            let gauss_rule_f64 = legendre::<f64>(n_gauss as usize);
            let segs_x_f64: Vec<f64> = segs_x_slice.to_vec();
            let segs_y_f64: Vec<f64> = segs_y_slice.to_vec();

            // Compute SVD
            let (u, s, v) = compute_svd_dtensor(&matrix);

            // Remove weights from U and V (C++: u_x_(i, j) = u(i, j) / sqrt(gauss_x_w[i]))
            // The input matrix K already has weights applied: sqrt(wx[i]) * K(x[i], y[j]) * sqrt(wy[j])
            // So we need to remove weights from SVD results
            use sparse_ir::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x_dd.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y_dd.w.as_slice(), true);

            // Convert U and V to f64 for polynomial conversion
            let u_f64 = mdarray::DTensor::<f64, 2>::from_fn(*u_unweighted.shape(), |idx| {
                u_unweighted[idx].to_f64()
            });
            let v_f64 = mdarray::DTensor::<f64, 2>::from_fn(*v_unweighted.shape(), |idx| {
                v_unweighted[idx].to_f64()
            });

            let u_polys = sparse_ir::sve::utils::svd_to_polynomials(
                &u_f64,
                &segs_x_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );
            let v_polys = sparse_ir::sve::utils::svd_to_polynomials(
                &v_f64,
                &segs_y_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );

            // Convert singular values to f64 (Df64 -> f64)
            let s_f64: Vec<f64> = s.iter().map(|&sv| sv.to_f64()).collect();

            // Create SVEResult
            let sve_result = SVEResult::new(
                PiecewiseLegendrePolyVector::new(u_polys),
                s_f64,
                PiecewiseLegendrePolyVector::new(v_polys),
                epsilon,
            );

            let sve_wrapper = spir_sve_result::new(sve_result);
            Box::into_raw(Box::new(sve_wrapper))
        } else {
            // Double precision path
            // Convert matrix from C array to DTensor
            let memory_order = MemoryOrder::from_c_int(order).unwrap_or(MemoryOrder::RowMajor);
            let mut matrix = mdarray::DTensor::<f64, 2>::zeros([nx as usize, ny as usize]);

            match memory_order {
                MemoryOrder::RowMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = i * (ny as usize) + j;
                            matrix[[i, j]] = k_high_slice[idx];
                        }
                    }
                }
                MemoryOrder::ColumnMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = j * (nx as usize) + i;
                            matrix[[i, j]] = k_high_slice[idx];
                        }
                    }
                }
            }

            // Reconstruct Gauss rules for weight removal
            let gauss_rule_f64 = legendre::<f64>(n_gauss as usize);
            let segs_x_f64: Vec<f64> = segs_x_slice.to_vec();
            let segs_y_f64: Vec<f64> = segs_y_slice.to_vec();
            let gauss_x = gauss_rule_f64.piecewise(&segs_x_f64);
            let gauss_y = gauss_rule_f64.piecewise(&segs_y_f64);

            // Compute SVD
            let (u, s, v) = compute_svd_dtensor(&matrix);

            // Remove weights from U and V (C++: u_x_(i, j) = u(i, j) / std::sqrt(gauss_x_w[i]))
            // The input matrix K already has weights applied: sqrt(wx[i]) * K(x[i], y[j]) * sqrt(wy[j])
            // So we need to remove weights from SVD results
            use sparse_ir::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y.w.as_slice(), true);

            // Convert to polynomials using svd_to_polynomials
            let u_polys = sparse_ir::sve::utils::svd_to_polynomials(
                &u_unweighted,
                &segs_x_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );
            let v_polys = sparse_ir::sve::utils::svd_to_polynomials(
                &v_unweighted,
                &segs_y_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );

            // Convert singular values to f64 (s is already Vec<f64>)
            let s_f64: Vec<f64> = s;

            // Create SVEResult
            let sve_result = SVEResult::new(
                PiecewiseLegendrePolyVector::new(u_polys),
                s_f64,
                PiecewiseLegendrePolyVector::new(v_polys),
                epsilon,
            );

            let sve_wrapper = spir_sve_result::new(sve_result);
            Box::into_raw(Box::new(sve_wrapper))
        }
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

/// Create a SVE result from centrosymmetric discretized kernel matrices
///
/// This function performs singular value expansion (SVE) on centrosymmetric
/// discretized kernel matrices using even/odd symmetry decomposition. The matrices
/// K_even and K_odd should already be in the appropriate form (no weight
/// application needed). The function supports both double and DDouble precision
/// based on whether K_low is provided.
///
/// # Arguments
/// * `K_even_high` - High part of the even-symmetry kernel matrix (required,
///   size: nx * ny, finite entries)
/// * `K_even_low` - Low part of the even-symmetry kernel matrix (optional,
///   nullptr for double precision; finite entries)
/// * `K_odd_high` - High part of the odd-symmetry kernel matrix (required,
///   size: nx * ny, finite entries)
/// * `K_odd_low` - Low part of the odd-symmetry kernel matrix (optional,
///   nullptr for double precision; finite entries)
/// * `nx` - Number of rows in the matrix
/// * `ny` - Number of columns in the matrix
/// * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `segments_x` - X-direction segments (array of boundary points, size:
///   n_segments_x + 1, finite and strictly increasing)
/// * `n_segments_x` - Number of segments in x direction (boundary points - 1)
/// * `segments_y` - Y-direction segments (array of boundary points, size:
///   n_segments_y + 1, finite and strictly increasing)
/// * `n_segments_y` - Number of segments in y direction (boundary points - 1)
/// * `n_gauss` - Number of Gauss points per segment
/// * `epsilon` - Target accuracy
/// * `status` - Pointer to store status code
///
/// # Returns
/// Pointer to SVE result on success, nullptr on failure. If `status` is
/// non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `K_even_high`, `K_odd_high`, `segments_x` or
///   `segments_y` is NULL, a size is less than 1, `epsilon` is not positive
///   and finite, an entry of a matrix that is read is NaN or infinite, or the
///   segments are not finite and strictly increasing
/// - SPIR_INVALID_DIMENSION if the matrices are too large to be addressed
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The low parts are read only if both are non-NULL. The arrays are validated
/// before the SVE is computed.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_from_matrix_centrosymmetric(
    #[allow(non_snake_case)] K_even_high: *const f64,
    #[allow(non_snake_case)] K_even_low: *const f64,
    #[allow(non_snake_case)] K_odd_high: *const f64,
    #[allow(non_snake_case)] K_odd_low: *const f64,
    nx: libc::c_int,
    ny: libc::c_int,
    order: libc::c_int,
    segments_x: *const f64,
    n_segments_x: libc::c_int,
    segments_y: *const f64,
    n_segments_y: libc::c_int,
    n_gauss: libc::c_int,
    epsilon: f64,
    status: *mut StatusCode,
) -> *mut spir_sve_result {
    use crate::utils::MemoryOrder;
    use sparse_ir::gauss::legendre;
    use sparse_ir::kernel::SymmetryType;
    use sparse_ir::poly::PiecewiseLegendrePolyVector;
    use sparse_ir::sve::utils::{extend_to_full_domain, merge_results};
    use sparse_ir::tsvd::compute_svd_dtensor;
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if K_even_high.is_null() || K_odd_high.is_null() || segments_x.is_null() || segments_y.is_null()
    {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if nx < 1 || ny < 1 || n_segments_x < 1 || n_segments_y < 1 || n_gauss < 1 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    if epsilon <= 0.0 || !epsilon.is_finite() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return std::ptr::null_mut();
    }

    // Determine if using DDouble precision
    let use_ddouble = !K_even_low.is_null() && !K_odd_low.is_null();

    // Read and validate every array that is used before computing anything:
    // the SVD never converges on a NaN or an infinity (it iterated forever
    // before the TSVD bounded its iteration). The sizes are positive (checked
    // above).
    // Note: n_segments_x is the number of segments (boundary points - 1),
    // matching C++ API behavior: C++ uses segments_x[0..n_segments_x] (n_segments_x + 1 elements)
    let (nx_u, ny_u) = (nx as usize, ny as usize);
    let validated = (|| -> Result<_, StatusCode> {
        // SAFETY: the pointers are non-null (checked above; the low parts are
        // read only if both are non-null) and hold the documented number of
        // elements (caller contract); the sizes are positive.
        unsafe {
            let even_high = validated_kernel_matrix(K_even_high, nx_u, ny_u)?;
            let odd_high = validated_kernel_matrix(K_odd_high, nx_u, ny_u)?;
            let lows = if use_ddouble {
                Some((
                    validated_kernel_matrix(K_even_low, nx_u, ny_u)?,
                    validated_kernel_matrix(K_odd_low, nx_u, ny_u)?,
                ))
            } else {
                None
            };
            let segs_x = validated_segments(segments_x, n_segments_x)?;
            let segs_y = validated_segments(segments_y, n_segments_y)?;
            Ok((even_high, odd_high, lows, segs_x, segs_y))
        }
    })();
    let (k_even_high, k_odd_high, k_lows, segs_x_slice, segs_y_slice) = match validated {
        Ok(inputs) => inputs,
        Err(code) => {
            unsafe {
                *status = code;
            }
            return std::ptr::null_mut();
        }
    };

    let result = catch_unwind(|| {
        // Get xmax and ymax from segments
        let xmax = segs_x_slice[segs_x_slice.len() - 1];
        let ymax = segs_y_slice[segs_y_slice.len() - 1];

        // Convert segments to f64 for polynomial conversion
        let segs_x_f64: Vec<f64> = segs_x_slice.to_vec();
        let segs_y_f64: Vec<f64> = segs_y_slice.to_vec();
        let gauss_rule_f64 = legendre::<f64>(n_gauss as usize);

        // Reconstruct Gauss rules for weight removal (reduced domain [0, xmax] x [0, ymax])
        let gauss_x = gauss_rule_f64.piecewise(&segs_x_f64);
        let gauss_y = gauss_rule_f64.piecewise(&segs_y_f64);

        // Helper function to convert matrix and compute SVD
        let compute_svd_for_symmetry = |k_high_slice: &[f64],
                                        k_low_slice: Option<&[f64]>|
         -> (
            mdarray::DTensor<f64, 2>,
            Vec<f64>,
            mdarray::DTensor<f64, 2>,
        ) {
            let memory_order = MemoryOrder::from_c_int(order).unwrap_or(MemoryOrder::RowMajor);
            let matrix = if let Some(k_low_slice) = k_low_slice {
                use sparse_ir::Df64;
                use sparse_ir::numeric::CustomNumeric;
                let mut matrix_dd = mdarray::DTensor::<Df64, 2>::from_elem(
                    [nx as usize, ny as usize],
                    Df64::new(0.0),
                );
                match memory_order {
                    MemoryOrder::RowMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = i * (ny as usize) + j;
                                matrix_dd[[i, j]] =
                                    unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                            }
                        }
                    }
                    MemoryOrder::ColumnMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = j * (nx as usize) + i;
                                matrix_dd[[i, j]] =
                                    unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                            }
                        }
                    }
                }
                // Convert to f64 for SVD
                mdarray::DTensor::<f64, 2>::from_fn(*matrix_dd.shape(), |idx| {
                    matrix_dd[idx].to_f64()
                })
            } else {
                let mut matrix_f64 = mdarray::DTensor::<f64, 2>::zeros([nx as usize, ny as usize]);
                match memory_order {
                    MemoryOrder::RowMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = i * (ny as usize) + j;
                                matrix_f64[[i, j]] = k_high_slice[idx];
                            }
                        }
                    }
                    MemoryOrder::ColumnMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = j * (nx as usize) + i;
                                matrix_f64[[i, j]] = k_high_slice[idx];
                            }
                        }
                    }
                }
                matrix_f64
            };

            // Compute SVD
            let (u, s, v) = compute_svd_dtensor(&matrix);

            // Remove weights from U and V (C++: u_x_(i, j) = u(i, j) / sqrt(gauss_x_w[i]))
            // The input matrix K already has weights applied: sqrt(wx[i]) * K(x[i], y[j]) * sqrt(wy[j])
            // So we need to remove weights from SVD results
            use sparse_ir::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y.w.as_slice(), true);

            // Convert singular values to f64 (s is already Vec<f64>)
            let s_f64: Vec<f64> = s;

            (u_unweighted, s_f64, v_unweighted)
        };

        // Compute SVD for even and odd symmetry
        let (u_even, s_even, v_even) =
            compute_svd_for_symmetry(k_even_high, k_lows.map(|(even, _)| even));
        let (u_odd, s_odd, v_odd) =
            compute_svd_for_symmetry(k_odd_high, k_lows.map(|(_, odd)| odd));

        // Convert to polynomials
        let u_even_polys = sparse_ir::sve::utils::svd_to_polynomials(
            &u_even,
            &segs_x_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );
        let v_even_polys = sparse_ir::sve::utils::svd_to_polynomials(
            &v_even,
            &segs_y_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );

        let u_odd_polys = sparse_ir::sve::utils::svd_to_polynomials(
            &u_odd,
            &segs_x_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );
        let v_odd_polys = sparse_ir::sve::utils::svd_to_polynomials(
            &v_odd,
            &segs_y_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );

        // Extend to full domain
        let u_even_full = extend_to_full_domain(u_even_polys, SymmetryType::Even, xmax);
        let v_even_full = extend_to_full_domain(v_even_polys, SymmetryType::Even, ymax);

        let u_odd_full = extend_to_full_domain(u_odd_polys, SymmetryType::Odd, xmax);
        let v_odd_full = extend_to_full_domain(v_odd_polys, SymmetryType::Odd, ymax);

        // Merge even and odd results. A block of rank 0 (e.g. the odd part of a
        // kernel that is even in y) has no functions: `merge_results` accepts
        // empty blocks, but `PiecewiseLegendrePolyVector::new` would panic.
        let block = |polyvec| PiecewiseLegendrePolyVector { polyvec };
        let result_even = (block(u_even_full), s_even, block(v_even_full));
        let result_odd = (block(u_odd_full), s_odd, block(v_odd_full));

        let sve_result = merge_results(result_even, result_odd, epsilon);

        let sve_wrapper = spir_sve_result::new(sve_result);
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

#[cfg(test)]
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::kernel::*;
    use crate::{
        SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_ORDER_COLUMN_MAJOR,
        SPIR_ORDER_ROW_MAJOR, spir_gauss_legendre_rule_piecewise_double,
    };
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
        debug_println!("SVE size: {}", size);

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
        let sve = spir_sve_result_new(kernel, 1e-6, -1, -1, -1, &mut sve_status);

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

    /// `spir_sve_result_new` truncates at the documented default cutoff,
    /// 2 * machine epsilon of the working precision (issue #249)
    ///
    /// At these lambdas one singular value lies between one and two machine
    /// epsilons times the largest singular value (see the
    /// `test_default_cutoff_is_two_machine_epsilon_*` tests in sparse-ir), so
    /// the result distinguishes 2 * machine epsilon from machine epsilon.
    #[test]
    fn test_sve_result_new_default_cutoff_is_two_machine_epsilon() {
        use sparse_ir::kernel::{LogisticKernel, RegularizedBoseKernel};
        use sparse_ir::numeric::CustomNumeric;
        use sparse_ir::sve::SVEResult;

        let df64_eps = CustomNumeric::to_f64(<sparse_ir::Df64 as CustomNumeric>::epsilon());
        // (lambda, epsilon, C twork value, TworkType, machine epsilon of the working precision)
        let cases = [
            (1.48, 1e-6, 0, TworkType::Float64, f64::EPSILON),
            (1.37, 1e-10, 1, TworkType::Float64X2, df64_eps),
        ];
        for (lambda, epsilon, twork, twork_type, machine_eps) in cases {
            for bosonic in [false, true] {
                let mut status = SPIR_INTERNAL_ERROR;
                let kernel = if bosonic {
                    spir_reg_bose_kernel_new(lambda, &mut status)
                } else {
                    spir_logistic_kernel_new(lambda, &mut status)
                };
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                let reference = |cutoff: f64| -> SVEResult {
                    if bosonic {
                        let k = RegularizedBoseKernel::new(lambda);
                        compute_sve(k, epsilon, Some(cutoff), None, twork_type)
                    } else {
                        let k = LogisticKernel::new(lambda);
                        compute_sve(k, epsilon, Some(cutoff), None, twork_type)
                    }
                };

                let mut status = SPIR_INTERNAL_ERROR;
                let sve = spir_sve_result_new(kernel, epsilon, -1, -1, twork, &mut status);
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                assert!(!sve.is_null());
                let mut size = 0;
                assert_eq!(
                    spir_sve_result_get_size(sve, &mut size),
                    SPIR_COMPUTATION_SUCCESS
                );
                let mut svals = vec![0.0; size as usize];
                assert_eq!(
                    spir_sve_result_get_svals(sve, svals.as_mut_ptr()),
                    SPIR_COMPUTATION_SUCCESS
                );

                assert_eq!(
                    svals,
                    reference(2.0 * machine_eps).s,
                    "default cutoff must be 2 * machine epsilon \
                     (lambda={lambda}, twork={twork}, bosonic={bosonic})"
                );
                assert_eq!(
                    reference(machine_eps).s.len(),
                    svals.len() + 1,
                    "precondition: one singular value must lie in [1, 2) machine epsilons \
                     times s[0] (lambda={lambda}, twork={twork}, bosonic={bosonic})"
                );

                spir_sve_result_release(sve);
                spir_kernel_release(kernel);
            }
        }
    }

    #[test]
    fn test_sve_null_pointers() {
        // Null kernel
        let mut status = SPIR_COMPUTATION_SUCCESS;
        let sve = spir_sve_result_new(ptr::null(), 1e-6, -1, -1, -1, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(sve.is_null());

        // NaN epsilon must not pass the <= 0.0 guard
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, f64::NAN, -1, -1, -1, &mut sve_status);
        assert_eq!(sve_status, SPIR_INVALID_ARGUMENT);
        assert!(sve.is_null());
        spir_kernel_release(kernel);

        // Null size pointer
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1, -1, -1, &mut sve_status);

        let status = spir_sve_result_get_size(sve, ptr::null_mut());
        assert_eq!(status, SPIR_INVALID_ARGUMENT);

        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_sve_result_from_matrix() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        // Create kernel and get SVE hints
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Get SVE hints
        let mut n_gauss = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &mut n_gauss);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_x = 0;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_x,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_y,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get segments
        let mut segments_x = vec![0.0; (n_segments_x + 1) as usize];
        let mut n_segments_x_out = n_segments_x + 1;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            segments_x.as_mut_ptr(),
            &mut n_segments_x_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut segments_y = vec![0.0; (n_segments_y + 1) as usize];
        let mut n_segments_y_out = n_segments_y + 1;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            segments_y.as_mut_ptr(),
            &mut n_segments_y_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        assert_eq!(segments_x.len(), (n_segments_x + 1) as usize);
        assert_eq!(segments_y.len(), (n_segments_y + 1) as usize);
        // Get Gauss points and weights
        // Note: n_segments_x and n_segments_y are the number of boundary points (n_segments + 1),
        // but spir_gauss_legendre_rule_piecewise_double expects the number of segments (n_segments)
        let nx = n_gauss * (n_segments_x); // n_segments_x - 1 is the number of segments
        let ny = n_gauss * (n_segments_y); // n_segments_y - 1 is the number of segments
        let mut x = vec![0.0; nx as usize];
        let mut w_x = vec![0.0; nx as usize];
        let mut y = vec![0.0; ny as usize];
        let mut w_y = vec![0.0; ny as usize];

        let mut status_gauss = SPIR_INTERNAL_ERROR;
        let result = spir_gauss_legendre_rule_piecewise_double(
            n_gauss,
            segments_x.as_ptr(),
            n_segments_x,
            x.as_mut_ptr(),
            w_x.as_mut_ptr(),
            &mut status_gauss,
        );
        assert_eq!(result, SPIR_COMPUTATION_SUCCESS);

        let mut status_gauss = SPIR_INTERNAL_ERROR;
        let result = spir_gauss_legendre_rule_piecewise_double(
            n_gauss,
            segments_y.as_ptr(),
            n_segments_y,
            y.as_mut_ptr(),
            w_y.as_mut_ptr(),
            &mut status_gauss,
        );
        assert_eq!(result, SPIR_COMPUTATION_SUCCESS);

        // Create a simple test kernel matrix
        // Note: In practice, this would be computed from the actual kernel
        let mut k_high = vec![0.0; (nx * ny) as usize];
        for i in 0..(nx as usize) {
            for j in 0..(ny as usize) {
                // Simple test: scaled identity-like matrix
                let k_val = if i == j {
                    (w_x[i] * w_y[j] as f64).sqrt()
                } else {
                    0.0
                };
                k_high[i * ny as usize + j] = k_val;
            }
        }

        // Create SVE result from matrix (row major)
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve_from_matrix = spir_sve_result_from_matrix(
            k_high.as_ptr(),
            ptr::null(),
            nx,
            ny,
            SPIR_ORDER_ROW_MAJOR,
            segments_x.as_ptr(),
            n_segments_x,
            segments_y.as_ptr(),
            n_segments_y,
            n_gauss,
            epsilon,
            &mut sve_status,
        );

        // Note: The test matrix is very simple, so the SVE result may not be meaningful
        // But we can at least verify the function doesn't crash and returns a valid result
        if sve_status == SPIR_COMPUTATION_SUCCESS && !sve_from_matrix.is_null() {
            let mut sve_size = 0;
            let status = spir_sve_result_get_size(sve_from_matrix, &mut sve_size);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            spir_sve_result_release(sve_from_matrix);
        }

        // Test error handling
        {
            let mut sve_status = SPIR_INTERNAL_ERROR;
            let sve_err = spir_sve_result_from_matrix(
                ptr::null(),
                ptr::null(),
                nx,
                ny,
                SPIR_ORDER_ROW_MAJOR,
                segments_x.as_ptr(),
                n_segments_x,
                segments_y.as_ptr(),
                n_segments_y,
                n_gauss,
                epsilon,
                &mut sve_status,
            );
            assert_ne!(sve_status, SPIR_COMPUTATION_SUCCESS);
            assert!(sve_err.is_null());
        }

        // Test column major order
        {
            // Create column-major matrix
            let mut k_col = vec![0.0; (nx * ny) as usize];
            for j in 0..(ny as usize) {
                for i in 0..(nx as usize) {
                    let k_val = if i == j {
                        (w_x[i] * w_y[j] as f64).sqrt()
                    } else {
                        0.0
                    };
                    k_col[j * nx as usize + i] = k_val;
                }
            }

            let mut sve_status = SPIR_INTERNAL_ERROR;
            let sve_col = spir_sve_result_from_matrix(
                k_col.as_ptr(),
                ptr::null(),
                nx,
                ny,
                SPIR_ORDER_COLUMN_MAJOR,
                segments_x.as_ptr(),
                n_segments_x,
                segments_y.as_ptr(),
                n_segments_y,
                n_gauss,
                epsilon,
                &mut sve_status,
            );

            if sve_status == SPIR_COMPUTATION_SUCCESS && !sve_col.is_null() {
                spir_sve_result_release(sve_col);
            }
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_sve_result_from_matrix_centrosymmetric() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        // Create kernel and get SVE hints
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Get SVE hints
        let mut n_gauss = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &mut n_gauss);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_x = 0;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_x,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_y,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get segments
        let mut segments_x = vec![0.0; (n_segments_x + 1) as usize];
        let mut n_segments_x_out = n_segments_x + 1;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            segments_x.as_mut_ptr(),
            &mut n_segments_x_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut segments_y = vec![0.0; (n_segments_y + 1) as usize];
        let mut n_segments_y_out = n_segments_y + 1;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            segments_y.as_mut_ptr(),
            &mut n_segments_y_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get Gauss points and weights
        let nx = n_gauss * n_segments_x;
        let ny = n_gauss * n_segments_y;
        let mut x = vec![0.0; nx as usize];
        let mut w_x = vec![0.0; nx as usize];
        let mut y = vec![0.0; ny as usize];
        let mut w_y = vec![0.0; ny as usize];

        let mut status_gauss = SPIR_INTERNAL_ERROR;
        let result = spir_gauss_legendre_rule_piecewise_double(
            n_gauss,
            segments_x.as_ptr(),
            n_segments_x,
            x.as_mut_ptr(),
            w_x.as_mut_ptr(),
            &mut status_gauss,
        );
        assert_eq!(result, SPIR_COMPUTATION_SUCCESS);

        let mut status_gauss = SPIR_INTERNAL_ERROR;
        let result = spir_gauss_legendre_rule_piecewise_double(
            n_gauss,
            segments_y.as_ptr(),
            n_segments_y - 1,
            y.as_mut_ptr(),
            w_y.as_mut_ptr(),
            &mut status_gauss,
        );
        assert_eq!(result, SPIR_COMPUTATION_SUCCESS);

        // Create even and odd symmetry matrices
        // For centrosymmetric kernel, we decompose into even and odd parts
        let mut k_even_high = vec![0.0; (nx * ny) as usize];
        let mut k_odd_high = vec![0.0; (nx * ny) as usize];

        for i in 0..(nx as usize) {
            for j in 0..(ny as usize) {
                // Simple test: even part is symmetric, odd part is antisymmetric
                let k_val = if i == j {
                    (w_x[i] * w_y[j] as f64).sqrt()
                } else {
                    0.0
                };
                k_even_high[i * ny as usize + j] = k_val;
                k_odd_high[i * ny as usize + j] = k_val * 0.5; // Smaller odd part
            }
        }

        // Create SVE result from centrosymmetric matrices
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve_centrosymm = spir_sve_result_from_matrix_centrosymmetric(
            k_even_high.as_ptr(),
            ptr::null(),
            k_odd_high.as_ptr(),
            ptr::null(),
            nx,
            ny,
            SPIR_ORDER_ROW_MAJOR,
            segments_x.as_ptr(),
            n_segments_x,
            segments_y.as_ptr(),
            n_segments_y,
            n_gauss,
            epsilon,
            &mut sve_status,
        );

        // Note: The test matrices are very simple, so the SVE result may not be meaningful
        // But we can at least verify the function doesn't crash and returns a valid result
        if sve_status == SPIR_COMPUTATION_SUCCESS && !sve_centrosymm.is_null() {
            let mut sve_size = 0;
            let status = spir_sve_result_get_size(sve_centrosymm, &mut sve_size);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            spir_sve_result_release(sve_centrosymm);
        }

        // Test error handling
        {
            let mut sve_status = SPIR_INTERNAL_ERROR;
            let sve_err = spir_sve_result_from_matrix_centrosymmetric(
                ptr::null(),
                ptr::null(),
                k_odd_high.as_ptr(),
                ptr::null(),
                nx,
                ny,
                SPIR_ORDER_ROW_MAJOR,
                segments_x.as_ptr(),
                n_segments_x,
                segments_y.as_ptr(),
                n_segments_y,
                n_gauss,
                epsilon,
                &mut sve_status,
            );
            assert_ne!(sve_status, SPIR_COMPUTATION_SUCCESS);
            assert!(sve_err.is_null());
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_sve_result_from_matrix_centrosymmetric_vs_from_matrix() {
        use sparse_ir::gauss::legendre;
        use sparse_ir::kernel::{KernelProperties, LogisticKernel, SVEHints, SymmetryType};
        use sparse_ir::kernelmatrix::{
            matrix_from_gauss_noncentrosymmetric, matrix_from_gauss_with_segments,
        };

        let lambda = 10.0;
        let epsilon = 1e-6;
        let kernel = LogisticKernel::new(lambda);

        // Get SVE hints
        let hints = kernel.sve_hints::<f64>(epsilon);
        let segments_x = hints.segments_x();
        let segments_y = hints.segments_y();
        let n_gauss = hints.ngauss();

        // Create Gauss rules for reduced domain [0, xmax] x [0, ymax]
        let gauss_rule = legendre::<f64>(n_gauss);
        let gauss_x_reduced = gauss_rule.piecewise(&segments_x);
        let gauss_y_reduced = gauss_rule.piecewise(&segments_y);

        // Compute even and odd matrices (reduced domain)
        let discretized_even = matrix_from_gauss_with_segments(
            &kernel,
            &gauss_x_reduced,
            &gauss_y_reduced,
            SymmetryType::Even,
            &hints,
        );
        let discretized_odd = matrix_from_gauss_with_segments(
            &kernel,
            &gauss_x_reduced,
            &gauss_y_reduced,
            SymmetryType::Odd,
            &hints,
        );

        // Apply weights for SVE
        let k_even_weighted = discretized_even.apply_weights_for_sve();
        let k_odd_weighted = discretized_odd.apply_weights_for_sve();

        // Create full domain segments [-xmax, xmax] x [-ymax, ymax]
        let mut segments_x_full = Vec::new();
        for i in (0..segments_x.len()).rev() {
            segments_x_full.push(-segments_x[i]);
        }
        for i in 1..segments_x.len() {
            segments_x_full.push(segments_x[i]);
        }
        let mut segments_y_full = Vec::new();
        for i in (0..segments_y.len()).rev() {
            segments_y_full.push(-segments_y[i]);
        }
        for i in 1..segments_y.len() {
            segments_y_full.push(segments_y[i]);
        }

        // Create Gauss rules for full domain
        let gauss_x_full = gauss_rule.piecewise(&segments_x_full);
        let gauss_y_full = gauss_rule.piecewise(&segments_y_full);

        // Compute full domain matrix
        let discretized_full =
            matrix_from_gauss_noncentrosymmetric(&kernel, &gauss_x_full, &gauss_y_full, &hints);
        let k_full_weighted = discretized_full.apply_weights_for_sve();

        // Convert matrices to C arrays (row-major)
        let nx = k_even_weighted.shape().0;
        let ny = k_even_weighted.shape().1;
        let nx_full = k_full_weighted.shape().0;
        let ny_full = k_full_weighted.shape().1;

        let mut k_even_vec = vec![0.0; nx * ny];
        let mut k_odd_vec = vec![0.0; nx * ny];
        let mut k_full_vec = vec![0.0; nx_full * ny_full];

        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                k_even_vec[idx] = k_even_weighted[[i, j]];
                k_odd_vec[idx] = k_odd_weighted[[i, j]];
            }
        }

        for i in 0..nx_full {
            for j in 0..ny_full {
                let idx = i * ny_full + j;
                k_full_vec[idx] = k_full_weighted[[i, j]];
            }
        }

        // Convert segments to arrays
        let segments_x_vec = segments_x.clone();
        let segments_y_vec = segments_y.clone();
        let segments_x_full_vec = segments_x_full.clone();
        let segments_y_full_vec = segments_y_full.clone();

        // Compute SVE using centrosymmetric function
        let mut status_centrosymm = SPIR_INTERNAL_ERROR;
        let sve_centrosymm = spir_sve_result_from_matrix_centrosymmetric(
            k_even_vec.as_ptr(),
            ptr::null(), // K_even_low (double precision)
            k_odd_vec.as_ptr(),
            ptr::null(), // K_odd_low (double precision)
            nx as libc::c_int,
            ny as libc::c_int,
            SPIR_ORDER_ROW_MAJOR,
            segments_x_vec.as_ptr(),
            (segments_x_vec.len() - 1) as libc::c_int,
            segments_y_vec.as_ptr(),
            (segments_y_vec.len() - 1) as libc::c_int,
            n_gauss as libc::c_int,
            epsilon,
            &mut status_centrosymm,
        );

        assert_eq!(status_centrosymm, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve_centrosymm.is_null());

        // Compute SVE using non-centrosymmetric function
        let mut status_noncentrosymm = SPIR_INTERNAL_ERROR;
        let sve_noncentrosymm = spir_sve_result_from_matrix(
            k_full_vec.as_ptr(),
            ptr::null(), // K_low (double precision)
            nx_full as libc::c_int,
            ny_full as libc::c_int,
            SPIR_ORDER_ROW_MAJOR,
            segments_x_full_vec.as_ptr(),
            (segments_x_full_vec.len() - 1) as libc::c_int,
            segments_y_full_vec.as_ptr(),
            (segments_y_full_vec.len() - 1) as libc::c_int,
            n_gauss as libc::c_int,
            epsilon,
            &mut status_noncentrosymm,
        );

        assert_eq!(status_noncentrosymm, SPIR_COMPUTATION_SUCCESS);
        assert!(!sve_noncentrosymm.is_null());

        // Compare results
        let mut size_centrosymm = 0;
        let mut size_noncentrosymm = 0;

        spir_sve_result_get_size(sve_centrosymm, &mut size_centrosymm);
        spir_sve_result_get_size(sve_noncentrosymm, &mut size_noncentrosymm);

        // Sizes should be similar (may differ slightly due to numerical precision)
        assert!(
            (size_centrosymm as i32 - size_noncentrosymm as i32).abs() <= 1,
            "Size mismatch: centrosymmetric={}, noncentrosymmetric={}",
            size_centrosymm,
            size_noncentrosymm
        );

        // Get singular values
        let mut svals_centrosymm = vec![0.0; size_centrosymm as usize];
        let mut svals_noncentrosymm = vec![0.0; size_noncentrosymm as usize];

        spir_sve_result_get_svals(sve_centrosymm, svals_centrosymm.as_mut_ptr());
        spir_sve_result_get_svals(sve_noncentrosymm, svals_noncentrosymm.as_mut_ptr());

        // Compare singular values (should match within tolerance)
        let min_size = size_centrosymm.min(size_noncentrosymm) as usize;
        let tolerance = 1e-10;

        for i in 0..min_size {
            let diff = (svals_centrosymm[i] - svals_noncentrosymm[i]).abs();
            let rel_diff = diff / svals_centrosymm[i].max(svals_noncentrosymm[i]);
            assert!(
                diff < tolerance || rel_diff < tolerance,
                "Singular value mismatch at index {}: centrosymmetric={}, noncentrosymmetric={}, diff={}, rel_diff={}",
                i,
                svals_centrosymm[i],
                svals_noncentrosymm[i],
                diff,
                rel_diff
            );
        }

        println!(
            "Comparison successful: {} singular values match within tolerance",
            min_size
        );

        // Cleanup
        spir_sve_result_release(sve_centrosymm);
        spir_sve_result_release(sve_noncentrosymm);
    }

    /// Weighted discretizations of the logistic kernel (Λ = 10, ε = 1e-6) in
    /// the form the `spir_sve_result_from_matrix*` functions take: the matrix
    /// on the full domain and the even and odd matrices on the half domain,
    /// all row-major, with their segments.
    struct KernelMatrices {
        n_gauss: libc::c_int,
        full: Vec<f64>,
        nx_full: usize,
        ny_full: usize,
        segs_x_full: Vec<f64>,
        segs_y_full: Vec<f64>,
        even: Vec<f64>,
        odd: Vec<f64>,
        nx: usize,
        ny: usize,
        segs_x: Vec<f64>,
        segs_y: Vec<f64>,
    }

    const MATRICES_LAMBDA: f64 = 10.0;
    const MATRICES_EPSILON: f64 = 1e-6;

    fn row_major(m: &mdarray::DTensor<f64, 2>) -> Vec<f64> {
        let (rows, cols) = *m.shape();
        (0..rows * cols).map(|k| m[[k / cols, k % cols]]).collect()
    }

    fn column_major(row_major: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols)
            .map(|k| row_major[(k % rows) * cols + k / rows])
            .collect()
    }

    fn logistic_kernel_matrices() -> KernelMatrices {
        use sparse_ir::gauss::legendre;
        use sparse_ir::kernel::{KernelProperties, LogisticKernel, SVEHints, SymmetryType};
        use sparse_ir::kernelmatrix::{
            matrix_from_gauss_noncentrosymmetric, matrix_from_gauss_with_segments,
        };

        let kernel = LogisticKernel::new(MATRICES_LAMBDA);
        let hints = kernel.sve_hints::<f64>(MATRICES_EPSILON);
        let (segs_x, segs_y) = (hints.segments_x(), hints.segments_y());
        let rule = legendre::<f64>(hints.ngauss());
        let (gauss_x, gauss_y) = (rule.piecewise(&segs_x), rule.piecewise(&segs_y));
        let reduced = |symmetry| {
            matrix_from_gauss_with_segments(&kernel, &gauss_x, &gauss_y, symmetry, &hints)
                .apply_weights_for_sve()
        };
        let (even, odd) = (reduced(SymmetryType::Even), reduced(SymmetryType::Odd));

        let mirror = |half: &[f64]| -> Vec<f64> {
            let mut full: Vec<f64> = half.iter().rev().map(|&s| -s).collect();
            full.extend_from_slice(&half[1..]);
            full
        };
        let (segs_x_full, segs_y_full) = (mirror(&segs_x), mirror(&segs_y));
        let full = matrix_from_gauss_noncentrosymmetric(
            &kernel,
            &rule.piecewise(&segs_x_full),
            &rule.piecewise(&segs_y_full),
            &hints,
        )
        .apply_weights_for_sve();

        KernelMatrices {
            n_gauss: hints.ngauss() as libc::c_int,
            nx_full: full.shape().0,
            ny_full: full.shape().1,
            full: row_major(&full),
            segs_x_full,
            segs_y_full,
            nx: even.shape().0,
            ny: even.shape().1,
            even: row_major(&even),
            odd: row_major(&odd),
            segs_x,
            segs_y,
        }
    }

    /// Call `spir_sve_result_from_matrix` on the full-domain matrix `k_high`
    /// (and `k_low`) with the segments of `m` replaced by `segs_x`
    fn sve_from_full_matrix(
        m: &KernelMatrices,
        k_high: &[f64],
        k_low: Option<&[f64]>,
        order: libc::c_int,
        segs_x: &[f64],
    ) -> (StatusCode, *mut spir_sve_result) {
        let mut status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_from_matrix(
            k_high.as_ptr(),
            k_low.map_or(ptr::null(), |low| low.as_ptr()),
            m.nx_full as libc::c_int,
            m.ny_full as libc::c_int,
            order,
            segs_x.as_ptr(),
            (segs_x.len() - 1) as libc::c_int,
            m.segs_y_full.as_ptr(),
            (m.segs_y_full.len() - 1) as libc::c_int,
            m.n_gauss,
            MATRICES_EPSILON,
            &mut status,
        );
        (status, sve)
    }

    /// Call `spir_sve_result_from_matrix_centrosymmetric` on the half-domain
    /// matrices of `m` (row-major), with `k_low` as the low part of both, and
    /// the x segments replaced by `segs_x`
    fn sve_from_reduced_matrices(
        m: &KernelMatrices,
        k_even: &[f64],
        k_odd: &[f64],
        k_low: Option<&[f64]>,
        segs_x: &[f64],
    ) -> (StatusCode, *mut spir_sve_result) {
        let low = k_low.map_or(ptr::null(), |low| low.as_ptr());
        let mut status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_from_matrix_centrosymmetric(
            k_even.as_ptr(),
            low,
            k_odd.as_ptr(),
            low,
            m.nx as libc::c_int,
            m.ny as libc::c_int,
            SPIR_ORDER_ROW_MAJOR,
            segs_x.as_ptr(),
            (segs_x.len() - 1) as libc::c_int,
            m.segs_y.as_ptr(),
            (m.segs_y.len() - 1) as libc::c_int,
            m.n_gauss,
            MATRICES_EPSILON,
            &mut status,
        );
        (status, sve)
    }

    fn largest_singular_value(sve: *const spir_sve_result) -> f64 {
        let mut size = 0;
        assert_eq!(
            spir_sve_result_get_size(sve, &mut size),
            SPIR_COMPUTATION_SUCCESS
        );
        let mut svals = vec![0.0; size as usize];
        assert_eq!(
            spir_sve_result_get_svals(sve, svals.as_mut_ptr()),
            SPIR_COMPUTATION_SUCCESS
        );
        svals[0]
    }

    /// Relative difference of the largest singular value from that of
    /// `compute_sve` for the same kernel and accuracy
    fn s0_relative_error(sve: *const spir_sve_result) -> f64 {
        let reference = compute_sve(
            sparse_ir::kernel::LogisticKernel::new(MATRICES_LAMBDA),
            MATRICES_EPSILON,
            None,
            None,
            TworkType::Auto,
        );
        (largest_singular_value(sve) - reference.s[0]).abs() / reference.s[0]
    }

    /// A NaN or an infinity in the matrix made the SVD iterate forever:
    /// before the fix these calls did not return. They must be rejected as
    /// invalid arguments, in both memory orders and in both parts of a
    /// double-double matrix.
    #[test]
    fn test_sve_result_from_matrix_rejects_non_finite_entries() {
        let m = logistic_kernel_matrices();
        let (rows, cols) = (m.nx_full, m.ny_full);
        let zeros = vec![0.0; m.full.len()];

        // The unmodified matrix is accepted; its singular values are those of
        // the kernel. It discretizes the kernel with the mirrored Gauss points
        // of `compute_sve`, so the largest singular value agrees to rounding.
        for (order, k_high) in [
            (SPIR_ORDER_ROW_MAJOR, m.full.clone()),
            (SPIR_ORDER_COLUMN_MAJOR, column_major(&m.full, rows, cols)),
        ] {
            for k_low in [None, Some(&zeros[..])] {
                let (status, sve) = sve_from_full_matrix(&m, &k_high, k_low, order, &m.segs_x_full);
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                assert!(!sve.is_null());
                let err = s0_relative_error(sve);
                assert!(err < 1e-12, "s_0 relative error {err:e}");
                spir_sve_result_release(sve);
            }
        }

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut k_bad = m.full.clone();
            k_bad[cols + 2] = bad; // entry (1, 2)
            for (order, k_high) in [
                (SPIR_ORDER_ROW_MAJOR, k_bad.clone()),
                (SPIR_ORDER_COLUMN_MAJOR, column_major(&k_bad, rows, cols)),
            ] {
                for k_low in [None, Some(&zeros[..])] {
                    let (status, sve) =
                        sve_from_full_matrix(&m, &k_high, k_low, order, &m.segs_x_full);
                    assert_eq!(status, SPIR_INVALID_ARGUMENT, "bad = {bad}");
                    assert!(sve.is_null());
                }
            }

            // Non-finite low part of a double-double matrix
            let mut low_bad = zeros.clone();
            low_bad[cols + 2] = bad;
            let (status, sve) = sve_from_full_matrix(
                &m,
                &m.full,
                Some(&low_bad),
                SPIR_ORDER_ROW_MAJOR,
                &m.segs_x_full,
            );
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "low part = {bad}");
            assert!(sve.is_null());
        }
    }

    /// Same as above for the even/odd matrices of the centrosymmetric variant
    #[test]
    fn test_sve_result_from_matrix_centrosymmetric_rejects_non_finite_entries() {
        let m = logistic_kernel_matrices();
        let zeros = vec![0.0; m.even.len()];

        for k_low in [None, Some(&zeros[..])] {
            let (status, sve) = sve_from_reduced_matrices(&m, &m.even, &m.odd, k_low, &m.segs_x);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            let err = s0_relative_error(sve);
            assert!(err < 1e-12, "s_0 relative error {err:e}");
            spir_sve_result_release(sve);
        }

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut k_bad = m.even.clone();
            k_bad[m.ny + 2] = bad;
            for k_low in [None, Some(&zeros[..])] {
                for (k_even, k_odd) in [(&k_bad, &m.odd), (&m.even, &k_bad)] {
                    let (status, sve) =
                        sve_from_reduced_matrices(&m, k_even, k_odd, k_low, &m.segs_x);
                    assert_eq!(status, SPIR_INVALID_ARGUMENT, "bad = {bad}");
                    assert!(sve.is_null());
                }
            }

            let mut low_bad = zeros.clone();
            low_bad[m.ny + 2] = bad;
            let (status, sve) =
                sve_from_reduced_matrices(&m, &m.even, &m.odd, Some(&low_bad), &m.segs_x);
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "low part = {bad}");
            assert!(sve.is_null());
        }
    }

    /// Segments that are not strictly increasing were detected, but the status
    /// was then overwritten with SPIR_COMPUTATION_SUCCESS while NULL was
    /// returned. A NaN boundary passed the monotonicity check and failed later
    /// as SPIR_INTERNAL_ERROR.
    #[test]
    fn test_sve_result_from_matrix_rejects_invalid_segments() {
        let m = logistic_kernel_matrices();

        let invalid_segments = |segs: &[f64]| -> Vec<Vec<f64>> {
            let mut swapped = segs.to_vec();
            swapped.swap(1, 2);
            let mut repeated = segs.to_vec();
            repeated[2] = repeated[1];
            let mut nan = segs.to_vec();
            nan[1] = f64::NAN;
            let mut inf = segs.to_vec();
            *inf.last_mut().unwrap() = f64::INFINITY;
            vec![swapped, repeated, nan, inf]
        };

        for segs in invalid_segments(&m.segs_x_full) {
            let (status, sve) =
                sve_from_full_matrix(&m, &m.full, None, SPIR_ORDER_ROW_MAJOR, &segs);
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "segments {segs:?}");
            assert!(sve.is_null());
        }
        for segs in invalid_segments(&m.segs_x) {
            let (status, sve) = sve_from_reduced_matrices(&m, &m.even, &m.odd, None, &segs);
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "segments {segs:?}");
            assert!(sve.is_null());
        }
    }

    fn singular_values(sve: *const spir_sve_result) -> Vec<f64> {
        let mut size = 0;
        assert_eq!(
            spir_sve_result_get_size(sve, &mut size),
            SPIR_COMPUTATION_SUCCESS
        );
        let mut svals = vec![0.0; size as usize];
        assert_eq!(
            spir_sve_result_get_svals(sve, svals.as_mut_ptr()),
            SPIR_COMPUTATION_SUCCESS
        );
        svals
    }

    /// A block of rank 0, such as the odd part of a kernel that is even in y,
    /// leaves no singular functions of that parity. Before the fix the empty
    /// block was wrapped in a `PiecewiseLegendrePolyVector`, which cannot be
    /// empty, and the call failed with SPIR_INTERNAL_ERROR.
    #[test]
    fn test_sve_result_from_matrix_centrosymmetric_with_a_zero_block() {
        let m = logistic_kernel_matrices();
        let zeros = vec![0.0; m.odd.len()];

        let (status, both) = sve_from_reduced_matrices(&m, &m.even, &m.odd, None, &m.segs_x);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let (status, even_only) = sve_from_reduced_matrices(&m, &m.even, &zeros, None, &m.segs_x);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!even_only.is_null());

        // The even block is decomposed as before; for this kernel the even
        // and odd singular values interlace, so the even ones are s_0, s_2, ...
        let (s_both, s_even) = (singular_values(both), singular_values(even_only));
        let expected: Vec<f64> = s_both.iter().step_by(2).copied().collect();
        assert_eq!(s_even, expected);

        spir_sve_result_release(both);
        spir_sve_result_release(even_only);
    }
}
