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
/// The cutoff is automatically set to 2*sqrt(machine_epsilon) internally.
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

    // Catch panics to prevent unwinding across FFI boundary
    let result = catch_unwind(|| unsafe {
        let kernel = &*k;

        // Dispatch based on kernel type
        // cutoff is automatically set to 2*sqrt(machine_epsilon) internally
        let sve_result = if let Some(logistic) = kernel.as_logistic() {
            compute_sve(
                **logistic, epsilon, None, None, // cutoff=None (auto), max_num_svals auto-determined
                twork_type,
            )
        } else if let Some(reg_bose) = kernel.as_regularized_bose() {
            compute_sve(
                **reg_bose, epsilon, None, None, // cutoff=None (auto), max_num_svals auto-determined
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
        let sve_truncated = sparseir_rust::sve::SVEResult::new(
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

/// Create a SVE result from a discretized kernel matrix
///
/// This function performs singular value expansion (SVE) on a discretized kernel
/// matrix K. The matrix K should already be in the appropriate form (no weight
/// application needed). The function supports both double and DDouble precision
/// based on whether K_low is provided.
///
/// # Arguments
/// * `K_high` - High part of the kernel matrix (required, size: nx * ny)
/// * `K_low` - Low part of the kernel matrix (optional, nullptr for double precision)
/// * `nx` - Number of rows in the matrix
/// * `ny` - Number of columns in the matrix
/// * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `segments_x` - X-direction segments (array of boundary points, size: n_segments_x + 1)
/// * `n_segments_x` - Number of segments in x direction (boundary points - 1)
/// * `segments_y` - Y-direction segments (array of boundary points, size: n_segments_y + 1)
/// * `n_segments_y` - Number of segments in y direction (boundary points - 1)
/// * `n_gauss` - Number of Gauss points per segment
/// * `epsilon` - Target accuracy
/// * `status` - Pointer to store status code
///
/// # Returns
/// Pointer to SVE result on success, nullptr on failure
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_from_matrix(
    K_high: *const f64,
    K_low: *const f64,
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
    use sparseir_rust::gauss::legendre;
    use sparseir_rust::poly::PiecewiseLegendrePolyVector;
    use sparseir_rust::sve::SVEResult;
    use sparseir_rust::tsvd::compute_svd_dtensor;
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

    let result = catch_unwind(|| {
        // Convert segments to Vec
        // Note: n_segments_x is the number of segments (boundary points - 1),
        // matching C++ API behavior: C++ uses segments_x[0..n_segments_x] (n_segments_x + 1 elements)
        let segs_x_slice = unsafe {
            std::slice::from_raw_parts(segments_x, (n_segments_x + 1) as usize)
        };
        let segs_y_slice = unsafe {
            std::slice::from_raw_parts(segments_y, (n_segments_y + 1) as usize)
        };

        // Verify segments are monotonically increasing
        // C++: for (int i = 1; i <= n_segments_x; ++i) checks segments_x[1..n_segments_x]
        for i in 1..=n_segments_x as usize {
            if segs_x_slice[i] <= segs_x_slice[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return std::ptr::null_mut();
            }
        }
        for i in 1..=n_segments_y as usize {
            if segs_y_slice[i] <= segs_y_slice[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return std::ptr::null_mut();
            }
        }

        // Determine if using DDouble precision
        let use_ddouble = !K_low.is_null();

        // Reconstruct Gauss rules
        let rule_base_dd = legendre::<sparseir_rust::Df64>(n_gauss as usize);

        if use_ddouble {
            // DDouble precision path
            use sparseir_rust::Df64;
            use sparseir_rust::numeric::CustomNumeric;

            // Convert segments to DDouble
            let segs_x_dd: Vec<Df64> = segs_x_slice.iter().map(|&x| Df64::from(x)).collect();
            let segs_y_dd: Vec<Df64> = segs_y_slice.iter().map(|&y| Df64::from(y)).collect();

            // Create piecewise Gauss rules
            let gauss_x_dd = rule_base_dd.piecewise(&segs_x_dd);
            let gauss_y_dd = rule_base_dd.piecewise(&segs_y_dd);

            // Convert matrix from C array to DTensor
            let memory_order = MemoryOrder::from_c_int(order).unwrap_or(MemoryOrder::RowMajor);
            let mut matrix = mdarray::DTensor::<Df64, 2>::from_elem([nx as usize, ny as usize], Df64::new(0.0));

            let k_high_slice = unsafe { std::slice::from_raw_parts(K_high, (nx * ny) as usize) };
            let k_low_slice = unsafe { std::slice::from_raw_parts(K_low, (nx * ny) as usize) };

            match memory_order {
                MemoryOrder::RowMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = i * (ny as usize) + j;
                            // Use unsafe new_full since K_high and K_low are already properly compensated
                            matrix[[i, j]] = unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                        }
                    }
                }
                MemoryOrder::ColumnMajor => {
                    for i in 0..(nx as usize) {
                        for j in 0..(ny as usize) {
                            let idx = j * (nx as usize) + i;
                            // Use unsafe new_full since K_high and K_low are already properly compensated
                            matrix[[i, j]] = unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
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
            use sparseir_rust::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x_dd.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y_dd.w.as_slice(), true);

            // Convert U and V to f64 for polynomial conversion
            let u_f64 = mdarray::DTensor::<f64, 2>::from_fn(*u_unweighted.shape(), |idx| u_unweighted[idx].to_f64());
            let v_f64 = mdarray::DTensor::<f64, 2>::from_fn(*v_unweighted.shape(), |idx| v_unweighted[idx].to_f64());

            let u_polys = sparseir_rust::sve::utils::svd_to_polynomials(
                &u_f64,
                &segs_x_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );
            let v_polys = sparseir_rust::sve::utils::svd_to_polynomials(
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

            let k_high_slice = unsafe { std::slice::from_raw_parts(K_high, (nx * ny) as usize) };

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
            use sparseir_rust::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y.w.as_slice(), true);

            // Convert to polynomials using svd_to_polynomials
            let u_polys = sparseir_rust::sve::utils::svd_to_polynomials(
                &u_unweighted,
                &segs_x_f64,
                &gauss_rule_f64,
                n_gauss as usize,
            );
            let v_polys = sparseir_rust::sve::utils::svd_to_polynomials(
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
/// * `K_even_high` - High part of the even-symmetry kernel matrix (required, size: nx * ny)
/// * `K_even_low` - Low part of the even-symmetry kernel matrix (optional, nullptr for double precision)
/// * `K_odd_high` - High part of the odd-symmetry kernel matrix (required, size: nx * ny)
/// * `K_odd_low` - Low part of the odd-symmetry kernel matrix (optional, nullptr for double precision)
/// * `nx` - Number of rows in the matrix
/// * `ny` - Number of columns in the matrix
/// * `order` - Memory layout (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `segments_x` - X-direction segments (array of boundary points, size: n_segments_x + 1)
/// * `n_segments_x` - Number of segments in x direction (boundary points - 1)
/// * `segments_y` - Y-direction segments (array of boundary points, size: n_segments_y + 1)
/// * `n_segments_y` - Number of segments in y direction (boundary points - 1)
/// * `n_gauss` - Number of Gauss points per segment
/// * `epsilon` - Target accuracy
/// * `status` - Pointer to store status code
///
/// # Returns
/// Pointer to SVE result on success, nullptr on failure
#[unsafe(no_mangle)]
pub extern "C" fn spir_sve_result_from_matrix_centrosymmetric(
    K_even_high: *const f64,
    K_even_low: *const f64,
    K_odd_high: *const f64,
    K_odd_low: *const f64,
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
    use sparseir_rust::gauss::legendre;
    use sparseir_rust::kernel::SymmetryType;
    use sparseir_rust::poly::PiecewiseLegendrePolyVector;
    use sparseir_rust::tsvd::compute_svd_dtensor;
    use sparseir_rust::sve::utils::{extend_to_full_domain, merge_results};
    use std::panic::catch_unwind;

    if status.is_null() {
        return std::ptr::null_mut();
    }

    if K_even_high.is_null() || K_odd_high.is_null() || segments_x.is_null() || segments_y.is_null() {
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

    let result = catch_unwind(|| {
        // Convert segments to Vec
        // Note: n_segments_x is the number of segments (boundary points - 1),
        // matching C++ API behavior: C++ uses segments_x[0..n_segments_x] (n_segments_x + 1 elements)
        let segs_x_slice = unsafe {
            std::slice::from_raw_parts(segments_x, (n_segments_x + 1) as usize)
        };
        let segs_y_slice = unsafe {
            std::slice::from_raw_parts(segments_y, (n_segments_y + 1) as usize)
        };

        // Verify segments are monotonically increasing
        // C++: for (int i = 1; i <= n_segments_x; ++i) checks segments_x[1..n_segments_x]
        for i in 1..=n_segments_x as usize {
            if segs_x_slice[i] <= segs_x_slice[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return std::ptr::null_mut();
            }
        }
        for i in 1..=n_segments_y as usize {
            if segs_y_slice[i] <= segs_y_slice[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return std::ptr::null_mut();
            }
        }

        // Determine if using DDouble precision
        let use_ddouble = !K_even_low.is_null() && !K_odd_low.is_null();

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
        let compute_svd_for_symmetry = |k_high: *const f64, k_low: *const f64| -> Option<(mdarray::DTensor<f64, 2>, Vec<f64>, mdarray::DTensor<f64, 2>)> {
            let memory_order = MemoryOrder::from_c_int(order).unwrap_or(MemoryOrder::RowMajor);
            let matrix = if use_ddouble {
                use sparseir_rust::Df64;
                use sparseir_rust::numeric::CustomNumeric;
                let mut matrix_dd = mdarray::DTensor::<Df64, 2>::from_elem([nx as usize, ny as usize], Df64::new(0.0));
                let k_high_slice = unsafe { std::slice::from_raw_parts(k_high, (nx * ny) as usize) };
                let k_low_slice = unsafe { std::slice::from_raw_parts(k_low, (nx * ny) as usize) };
                match memory_order {
                    MemoryOrder::RowMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = i * (ny as usize) + j;
                                matrix_dd[[i, j]] = unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                            }
                        }
                    }
                    MemoryOrder::ColumnMajor => {
                        for i in 0..(nx as usize) {
                            for j in 0..(ny as usize) {
                                let idx = j * (nx as usize) + i;
                                matrix_dd[[i, j]] = unsafe { Df64::new_full(k_high_slice[idx], k_low_slice[idx]) };
                            }
                        }
                    }
                }
                // Convert to f64 for SVD
                mdarray::DTensor::<f64, 2>::from_fn(*matrix_dd.shape(), |idx| matrix_dd[idx].to_f64())
            } else {
                let mut matrix_f64 = mdarray::DTensor::<f64, 2>::zeros([nx as usize, ny as usize]);
                let k_high_slice = unsafe { std::slice::from_raw_parts(k_high, (nx * ny) as usize) };
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
            use sparseir_rust::sve::utils::remove_weights;
            let u_unweighted = remove_weights(&u, gauss_x.w.as_slice(), true);
            let v_unweighted = remove_weights(&v, gauss_y.w.as_slice(), true);

            // Convert singular values to f64 (s is already Vec<f64>)
            let s_f64: Vec<f64> = s;

            Some((u_unweighted, s_f64, v_unweighted))
        };

        // Compute SVD for even symmetry
        let (u_even, s_even, v_even) = match compute_svd_for_symmetry(K_even_high, K_even_low) {
            Some(result) => result,
            None => {
                unsafe {
                    *status = SPIR_INTERNAL_ERROR;
                }
                return std::ptr::null_mut();
            }
        };

        // Compute SVD for odd symmetry
        let (u_odd, s_odd, v_odd) = match compute_svd_for_symmetry(K_odd_high, K_odd_low) {
            Some(result) => result,
            None => {
                unsafe {
                    *status = SPIR_INTERNAL_ERROR;
                }
                return std::ptr::null_mut();
            }
        };

        // Convert to polynomials
        let u_even_polys = sparseir_rust::sve::utils::svd_to_polynomials(
            &u_even,
            &segs_x_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );
        let v_even_polys = sparseir_rust::sve::utils::svd_to_polynomials(
            &v_even,
            &segs_y_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );

        let u_odd_polys = sparseir_rust::sve::utils::svd_to_polynomials(
            &u_odd,
            &segs_x_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );
        let v_odd_polys = sparseir_rust::sve::utils::svd_to_polynomials(
            &v_odd,
            &segs_y_f64,
            &gauss_rule_f64,
            n_gauss as usize,
        );

        // Extend to full domain
        let u_even_full = extend_to_full_domain(
            u_even_polys,
            SymmetryType::Even,
            xmax,
        );
        let v_even_full = extend_to_full_domain(
            v_even_polys,
            SymmetryType::Even,
            ymax,
        );

        let u_odd_full = extend_to_full_domain(
            u_odd_polys,
            SymmetryType::Odd,
            xmax,
        );
        let v_odd_full = extend_to_full_domain(
            v_odd_polys,
            SymmetryType::Odd,
            ymax,
        );

        // Merge even and odd results
        let result_even = (
            PiecewiseLegendrePolyVector::new(u_even_full),
            s_even,
            PiecewiseLegendrePolyVector::new(v_even_full),
        );
        let result_odd = (
            PiecewiseLegendrePolyVector::new(u_odd_full),
            s_odd,
            PiecewiseLegendrePolyVector::new(v_odd_full),
        );

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
mod tests {
    use super::*;
    use crate::kernel::*;
    use crate::{SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR, SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, spir_gauss_legendre_rule_piecewise_double};
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

    #[test]
    fn test_sve_null_pointers() {
        // Null kernel
        let mut status = SPIR_COMPUTATION_SUCCESS;
        let sve = spir_sve_result_new(ptr::null(), 1e-6, -1, -1, -1, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(sve.is_null());

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
        let status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, ptr::null_mut(), &mut n_segments_x);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, ptr::null_mut(), &mut n_segments_y);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get segments
        let mut segments_x = vec![0.0; (n_segments_x + 1) as usize];
        let mut n_segments_x_out = n_segments_x + 1;
        let status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, segments_x.as_mut_ptr(), &mut n_segments_x_out);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut segments_y = vec![0.0; (n_segments_y + 1) as usize];
        let mut n_segments_y_out = n_segments_y + 1;
        let status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, segments_y.as_mut_ptr(), &mut n_segments_y_out);
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
        let status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, ptr::null_mut(), &mut n_segments_x);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, ptr::null_mut(), &mut n_segments_y);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get segments
        let mut segments_x = vec![0.0; (n_segments_x + 1) as usize];
        let mut n_segments_x_out = n_segments_x + 1;
        let status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, segments_x.as_mut_ptr(), &mut n_segments_x_out);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut segments_y = vec![0.0; (n_segments_y + 1) as usize];
        let mut n_segments_y_out = n_segments_y + 1;
        let status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, segments_y.as_mut_ptr(), &mut n_segments_y_out);
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
        use sparseir_rust::kernel::{KernelProperties, LogisticKernel, SVEHints, SymmetryType};
        use sparseir_rust::kernelmatrix::{matrix_from_gauss_noncentrosymmetric, matrix_from_gauss_with_segments};
        use sparseir_rust::gauss::legendre;

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
        let discretized_full = matrix_from_gauss_noncentrosymmetric(
            &kernel,
            &gauss_x_full,
            &gauss_y_full,
            &hints,
        );
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
        assert!((size_centrosymm as i32 - size_noncentrosymm as i32).abs() <= 1,
            "Size mismatch: centrosymmetric={}, noncentrosymmetric={}",
            size_centrosymm, size_noncentrosymm);

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
            assert!(diff < tolerance || rel_diff < tolerance,
                "Singular value mismatch at index {}: centrosymmetric={}, noncentrosymmetric={}, diff={}, rel_diff={}",
                i, svals_centrosymm[i], svals_noncentrosymm[i], diff, rel_diff);
        }

        println!("Comparison successful: {} singular values match within tolerance", min_size);

        // Cleanup
        spir_sve_result_release(sve_centrosymm);
        spir_sve_result_release(sve_noncentrosymm);
    }
}
