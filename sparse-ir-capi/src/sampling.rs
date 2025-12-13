//! Sampling API for C
//!
//! This module provides the C API for sparse sampling in imaginary time (τ),
//! Matsubara frequency (iωn), and real frequency (ω) domains.
//!
//! Functions:
//! - Creation: spir_tau_sampling_new, spir_matsu_sampling_new, ...
//! - Introspection: get_npoints, get_taus, get_matsus, get_cond_num
//! - Evaluation: eval_dd, eval_dz, eval_zz (coefficients → sampling points)
//! - Fitting: fit_dd, fit_zz, fit_zd (sampling points → coefficients)
//! - Memory: release, clone, is_assigned (via macro)

use mdarray::Shape;
use num_complex::Complex64;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::gemm::{get_backend_handle, spir_gemm_backend};
use crate::types::{BasisType, SamplingType, spir_basis, spir_sampling};
use crate::utils::{
    MemoryOrder, build_output_dims, convert_dims_for_row_major, create_dview_from_ptr,
    create_dviewmut_from_ptr, read_tensor_nd,
};
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, SPIR_STATISTICS_BOSONIC,
    SPIR_STATISTICS_FERMIONIC, StatusCode,
};
use sparse_ir::fitters::InplaceFitter;
use sparse_ir::{Bosonic, Fermionic};

/// Manual release function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_release(sampling: *mut spir_sampling) {
    if !sampling.is_null() {
        unsafe {
            let _ = Box::from_raw(sampling);
        }
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_clone(src: *const spir_sampling) -> *mut spir_sampling {
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
pub extern "C" fn spir_sampling_is_assigned(obj: *const spir_sampling) -> i32 {
    if obj.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    }));

    result.unwrap_or(0)
}

// ============================================================================
// Creation Functions
// ============================================================================

/// Creates a new tau sampling object for sparse sampling in imaginary time
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `num_points` - Number of sampling points
/// * `points` - Array of sampling points in imaginary time (τ)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `b` is valid and `points` has `num_points` elements
#[unsafe(no_mangle)]
pub extern "C" fn spir_tau_sampling_new(
    b: *const spir_basis,
    num_points: libc::c_int,
    points: *const f64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if b.is_null() || points.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = unsafe { &*b };
        let points_slice = unsafe { std::slice::from_raw_parts(points, num_points as usize) };

        // Convert points to Vec
        let tau_points: Vec<f64> = points_slice.to_vec();

        // Create sampling based on basis statistics
        let sampling_type = match basis_ref.inner() {
            BasisType::LogisticFermionic(ir_basis) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            BasisType::LogisticBosonic(ir_basis) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
            // DLR: tau sampling supported via Basis trait
            BasisType::DLRFermionic(dlr) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    dlr.as_ref(),
                    tau_points,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            BasisType::DLRBosonic(dlr) => {
                let tau_sampling = sparse_ir::sampling::TauSampling::with_sampling_points(
                    dlr.as_ref(),
                    tau_points,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
        };

        let inner = sampling_type;
        let sampling = spir_sampling {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                unsafe {
                    *status = crate::SPIR_INTERNAL_ERROR;
                }
            }
            std::ptr::null_mut()
        }
    }
}

/// Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `positive_only` - If true, only positive frequencies are used
/// * `num_points` - Number of sampling points
/// * `points` - Array of Matsubara frequency indices (n)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
#[unsafe(no_mangle)]
pub extern "C" fn spir_matsu_sampling_new(
    b: *const spir_basis,
    positive_only: bool,
    num_points: libc::c_int,
    points: *const i64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if b.is_null() || points.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = unsafe { &*b };
        let points_slice = unsafe { std::slice::from_raw_parts(points, num_points as usize) };

        // Convert points to Vec
        let matsu_points: Vec<i64> = points_slice.to_vec();

        // Convert i64 indices to MatsubaraFreq
        use sparse_ir::freq::MatsubaraFreq;

        // Helper macro to reduce duplication
        macro_rules! create_matsu_sampling {
            ($basis:expr, Fermionic) => {
                if positive_only {
                    let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyFermionic(Arc::new(matsu_sampling))
                } else {
                    let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraFermionic(Arc::new(matsu_sampling))
                }
            };
            ($basis:expr, Bosonic) => {
                if positive_only {
                    let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
                } else {
                    let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
                }
            };
        }

        // Create sampling based on basis statistics and positive_only flag
        let sampling_type = match basis_ref.inner() {
            BasisType::LogisticFermionic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Fermionic)
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Fermionic)
            }
            BasisType::LogisticBosonic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Bosonic)
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Bosonic)
            }
            // DLR: Matsubara sampling supported via Basis trait
            BasisType::DLRFermionic(dlr) => {
                create_matsu_sampling!(dlr.as_ref(), Fermionic)
            }
            BasisType::DLRBosonic(dlr) => {
                create_matsu_sampling!(dlr.as_ref(), Bosonic)
            }
        };

        let inner = sampling_type;
        let sampling = spir_sampling {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                unsafe {
                    *status = crate::SPIR_INTERNAL_ERROR;
                }
            }
            std::ptr::null_mut()
        }
    }
}

/// Creates a new tau sampling object with custom sampling points and pre-computed matrix
///
/// # Arguments
/// * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size
/// * `num_points` - Number of sampling points
/// * `points` - Array of sampling points in imaginary time (τ)
/// * `matrix` - Pre-computed matrix for the sampling points (num_points x basis_size)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `points` and `matrix` have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_tau_sampling_new_with_matrix(
    order: libc::c_int,
    statistics: libc::c_int,
    basis_size: libc::c_int,
    num_points: libc::c_int,
    points: *const f64,
    matrix: *const f64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if points.is_null() || matrix.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 || basis_size <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        // Convert points to Vec
        let points_slice = unsafe { std::slice::from_raw_parts(points, num_points as usize) };
        let tau_points: Vec<f64> = points_slice.to_vec();

        // Convert matrix to Tensor using the new helper function
        let orig_dims = [num_points as usize, basis_size as usize];
        let dyn_tensor = unsafe { read_tensor_nd(matrix, &orig_dims, mem_order) };

        // Convert DynRank to fixed 2D shape using from_fn (safe conversion)
        let shape_dims = dyn_tensor.shape().with_dims(|dims| dims.to_vec());
        assert_eq!(
            shape_dims.len(),
            2,
            "Expected 2D tensor, got {}D",
            shape_dims.len()
        );
        let num_points_actual = shape_dims[0];
        let basis_size_actual = shape_dims[1];
        let matrix_tensor =
            sparse_ir::DTensor::<f64, 2>::from_fn([num_points_actual, basis_size_actual], |idx| {
                dyn_tensor[&[idx[0], idx[1]][..]]
            });
        // Create sampling based on statistics
        let sampling_type = match statistics {
            SPIR_STATISTICS_FERMIONIC => {
                // SPIR_STATISTICS_FERMIONIC
                let tau_sampling = sparse_ir::sampling::TauSampling::<Fermionic>::from_matrix(
                    tau_points,
                    matrix_tensor,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            SPIR_STATISTICS_BOSONIC => {
                // SPIR_STATISTICS_BOSONIC
                let tau_sampling = sparse_ir::sampling::TauSampling::<Bosonic>::from_matrix(
                    tau_points,
                    matrix_tensor,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
            _ => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        let inner = sampling_type;
        let sampling = spir_sampling {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                unsafe {
                    *status = crate::SPIR_INTERNAL_ERROR;
                }
            }
            std::ptr::null_mut()
        }
    }
}

/// Creates a new Matsubara sampling object with custom sampling points and pre-computed matrix
///
/// # Arguments
/// * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size
/// * `positive_only` - If true, only positive frequencies are used
/// * `num_points` - Number of sampling points
/// * `points` - Array of Matsubara frequency indices (n)
/// * `matrix` - Pre-computed complex matrix (num_points x basis_size)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `points` and `matrix` have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_matsu_sampling_new_with_matrix(
    order: libc::c_int,
    statistics: libc::c_int,
    basis_size: libc::c_int,
    positive_only: bool,
    num_points: libc::c_int,
    points: *const i64,
    matrix: *const Complex64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    use std::io::Write;
    debug_println!(
        "spir_matsu_sampling_new_with_matrix: start, order={}, statistics={}, basis_size={}, positive_only={}, num_points={}",
        order,
        statistics,
        basis_size,
        positive_only,
        num_points
    );
    std::io::stderr().flush().ok();
    let result = catch_unwind(AssertUnwindSafe(|| {
        use std::io::Write;
        debug_println!("spir_matsu_sampling_new_with_matrix: inside catch_unwind");
        std::io::stderr().flush().ok();
        // Validate inputs
        if points.is_null() || matrix.is_null() {
            debug_eprintln!("spir_matsu_sampling_new_with_matrix: null pointer");
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 || basis_size <= 0 {
            debug_eprintln!(
                "spir_matsu_sampling_new_with_matrix: invalid size, num_points={}, basis_size={}",
                num_points,
                basis_size
            );
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        debug_println!("spir_matsu_sampling_new_with_matrix: input validation passed");
        std::io::stderr().flush().ok();

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        // Convert points to Vec<MatsubaraFreq>
        debug_println!("spir_matsu_sampling_new_with_matrix: creating points slice...");
        std::io::stderr().flush().ok();
        let points_slice = unsafe { std::slice::from_raw_parts(points, num_points as usize) };
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: points slice created, len = {}",
            points_slice.len()
        );
        std::io::stderr().flush().ok();
        let matsu_points: Vec<i64> = points_slice.to_vec();
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: matsu_points created, len = {}",
            matsu_points.len()
        );
        std::io::stderr().flush().ok();

        use sparse_ir::freq::MatsubaraFreq;

        // Convert matrix to Tensor using the new helper function
        let orig_dims = [num_points as usize, basis_size as usize];
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: orig_dims = {:?}, mem_order = {:?}",
            orig_dims,
            mem_order
        );
        std::io::stderr().flush().ok();

        debug_println!("spir_matsu_sampling_new_with_matrix: reading tensor from buffer...");
        std::io::stderr().flush().ok();
        let dyn_tensor = unsafe { read_tensor_nd(matrix, &orig_dims, mem_order) };
        let shape_dims = dyn_tensor.shape().with_dims(|dims| dims.to_vec());
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: dyn_tensor created, shape = {:?}",
            shape_dims
        );
        std::io::stderr().flush().ok();

        // Convert DynRank to fixed 2D shape using from_fn (safe conversion)
        debug_println!("spir_matsu_sampling_new_with_matrix: converting to fixed 2D tensor...");
        std::io::stderr().flush().ok();
        assert_eq!(
            shape_dims.len(),
            2,
            "Expected 2D tensor, got {}D",
            shape_dims.len()
        );
        let num_points_actual = shape_dims[0];
        let basis_size_actual = shape_dims[1];
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: converting from shape {:?} to DTensor<Complex64, 2>",
            shape_dims
        );
        std::io::stderr().flush().ok();
        let matrix_tensor = sparse_ir::DTensor::<Complex64, 2>::from_fn(
            [num_points_actual, basis_size_actual],
            |idx| dyn_tensor[&[idx[0], idx[1]][..]],
        );
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: matrix_tensor created, shape = {:?}",
            matrix_tensor.shape()
        );
        std::io::stderr().flush().ok();

        // Create sampling based on statistics and positive_only
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: creating sampling, statistics={}, positive_only={}",
            statistics,
            positive_only
        );
        std::io::stderr().flush().ok();
        let sampling_type = match (statistics, positive_only) {
            (SPIR_STATISTICS_FERMIONIC, true) => {
                debug_println!("spir_matsu_sampling_new_with_matrix: Fermionic, positive-only");
                std::io::stderr().flush().ok();
                // Fermionic, positive-only
                let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                debug_println!(
                    "spir_matsu_sampling_new_with_matrix: matsu_freqs created, len = {}",
                    matsu_freqs.len()
                );
                std::io::stderr().flush().ok();
                debug_println!("spir_matsu_sampling_new_with_matrix: calling from_matrix...");
                std::io::stderr().flush().ok();
                let matsu_sampling =
                    sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::from_matrix(
                        matsu_freqs,
                        matrix_tensor.clone(),
                    );
                debug_println!("spir_matsu_sampling_new_with_matrix: from_matrix returned");
                std::io::stderr().flush().ok();
                SamplingType::MatsubaraPositiveOnlyFermionic(Arc::new(matsu_sampling))
            }
            (SPIR_STATISTICS_FERMIONIC, false) => {
                debug_println!("spir_matsu_sampling_new_with_matrix: Fermionic, full range");
                std::io::stderr().flush().ok();
                // Fermionic, full range
                let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                debug_println!(
                    "spir_matsu_sampling_new_with_matrix: matsu_freqs created, len = {}",
                    matsu_freqs.len()
                );
                std::io::stderr().flush().ok();
                debug_println!("spir_matsu_sampling_new_with_matrix: calling from_matrix...");
                std::io::stderr().flush().ok();
                let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                debug_println!("spir_matsu_sampling_new_with_matrix: from_matrix returned");
                std::io::stderr().flush().ok();
                SamplingType::MatsubaraFermionic(Arc::new(matsu_sampling))
            }
            (SPIR_STATISTICS_BOSONIC, true) => {
                // Bosonic, positive-only
                let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling =
                    sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::from_matrix(
                        matsu_freqs,
                        matrix_tensor.clone(),
                    );
                SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
            }
            (SPIR_STATISTICS_BOSONIC, false) => {
                // Bosonic, full range
                let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
            }
            _ => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        let inner = sampling_type;
        let sampling = spir_sampling {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                unsafe {
                    *status = crate::SPIR_INTERNAL_ERROR;
                }
            }
            std::ptr::null_mut()
        }
    }
}

// ============================================================================
// Introspection Functions
// ============================================================================

/// Gets the number of sampling points in a sampling object.
///
/// This function returns the number of sampling points used in the specified
/// sampling object. This number is needed to allocate arrays of the correct size
/// when retrieving the actual sampling points.
///
/// # Arguments
///
/// * `s` - Pointer to the sampling object.
/// * `num_points` - Pointer to store the number of sampling points.
///
/// # Returns
///
/// A status code:
/// - `0` ([`SPIR_COMPUTATION_SUCCESS`]) on success
/// - A non-zero error code on failure
///
/// # See also
///
/// - [`spir_sampling_get_taus`]
/// - [`spir_sampling_get_matsus`]
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_get_npoints(
    s: *const spir_sampling,
    num_points: *mut libc::c_int,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || num_points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = unsafe { &*s };

        let n_points = match sampling_ref.inner() {
            SamplingType::TauFermionic(tau) => tau.n_sampling_points(),
            SamplingType::TauBosonic(tau) => tau.n_sampling_points(),
            SamplingType::MatsubaraFermionic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraBosonic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => matsu.n_sampling_points(),
        };

        unsafe {
            *num_points = n_points as libc::c_int;
        }
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the imaginary time (τ) sampling points used in the specified sampling object.
///
/// This function fills the provided array with the imaginary time (τ) sampling points used in the specified sampling object.
/// The array must be pre-allocated with sufficient size (use [`spir_sampling_get_npoints`] to determine the required size).
///
/// # Arguments
///
/// * `s` - Pointer to the sampling object.
/// * `points` - Pre-allocated array to store the τ sampling points.
///
/// # Returns
///
/// An integer status code:
/// - `0` ([`SPIR_COMPUTATION_SUCCESS`]) on success
/// - A non-zero error code on failure
///
/// # Notes
///
/// The array must be pre-allocated with size >= [`spir_sampling_get_npoints`](spir_sampling_get_npoints).
///
/// # See also
///
/// - [`spir_sampling_get_npoints`]
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_get_taus(s: *const spir_sampling, points: *mut f64) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = unsafe { &*s };

        match sampling_ref.inner() {
            SamplingType::TauFermionic(tau) => {
                let tau_points = tau.sampling_points();
                let out_slice = unsafe { std::slice::from_raw_parts_mut(points, tau_points.len()) };
                out_slice.copy_from_slice(tau_points);
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::TauBosonic(tau) => {
                let tau_points = tau.sampling_points();
                let out_slice = unsafe { std::slice::from_raw_parts_mut(points, tau_points.len()) };
                out_slice.copy_from_slice(tau_points);
                SPIR_COMPUTATION_SUCCESS
            }
            _ => SPIR_NOT_SUPPORTED,
        }
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the Matsubara frequency sampling points
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_get_matsus(
    s: *const spir_sampling,
    points: *mut i64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = unsafe { &*s };

        match sampling_ref.inner() {
            SamplingType::MatsubaraFermionic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice =
                    unsafe { std::slice::from_raw_parts_mut(points, matsu_freqs.len()) };
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice =
                    unsafe { std::slice::from_raw_parts_mut(points, matsu_freqs.len()) };
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice =
                    unsafe { std::slice::from_raw_parts_mut(points, matsu_freqs.len()) };
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice =
                    unsafe { std::slice::from_raw_parts_mut(points, matsu_freqs.len()) };
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            _ => SPIR_NOT_SUPPORTED,
        }
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the condition number of the sampling matrix.
///
/// This function returns the condition number of the sampling matrix used in the
/// specified sampling object. The condition number is a measure of how well-
/// conditioned the sampling matrix is.
///
/// # Parameters
/// - `s`: Pointer to the sampling object.
/// - `cond_num`: Pointer to store the condition number.
///
/// # Returns
/// An integer status code:
/// - 0 (`SPIR_COMPUTATION_SUCCESS`) on success
/// - Non-zero error code on failure
///
/// # Notes
/// - A large condition number indicates that the sampling matrix is ill-conditioned,
///   which may lead to numerical instability in transformations.
/// - The condition number is the ratio of the largest to smallest singular value
///   of the sampling matrix.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_get_cond_num(
    s: *const spir_sampling,
    cond_num: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || cond_num.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = unsafe { &*s };

        // Calculate condition number from SVD of the sampling matrix
        let condition_number = match sampling_ref.inner() {
            SamplingType::TauFermionic(tau) => {
                // For tau sampling, matrix is real
                let matrix = tau.matrix();
                compute_condition_number_real(matrix)
            }
            SamplingType::TauBosonic(tau) => {
                // For tau sampling, matrix is real
                let matrix = tau.matrix();
                compute_condition_number_real(matrix)
            }
            SamplingType::MatsubaraFermionic(matsu) => {
                // For Matsubara sampling, matrix is complex
                let matrix = matsu.matrix();
                compute_condition_number_complex(matrix)
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                let matrix = matsu.matrix();
                compute_condition_number_complex(matrix)
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                // For positive-only Matsubara, use the complex matrix
                // The fitter uses ComplexToRealFitter internally, but we can use the complex matrix
                let matrix = matsu.matrix();
                compute_condition_number_complex(matrix)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                let matrix = matsu.matrix();
                compute_condition_number_complex(matrix)
            }
        };

        unsafe {
            *cond_num = condition_number;
        }
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Compute condition number from real matrix using SVD
fn compute_condition_number_real(matrix: &mdarray::DTensor<f64, 2>) -> f64 {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    let mut matrix_copy = matrix.clone();
    let SVDDecomp { s, .. } = Faer.svd(&mut *matrix_copy).expect("SVD computation failed");

    let min_dim = s.shape().0.min(s.shape().1);
    if min_dim == 0 {
        return 1.0;
    }

    let max_sv = s[[0, 0]];
    let min_sv = s[[0, min_dim - 1]];

    if min_sv.abs() < 1e-15 {
        // Matrix is singular or nearly singular
        return f64::INFINITY;
    }

    max_sv / min_sv
}

/// Compute condition number from complex matrix using SVD
fn compute_condition_number_complex(matrix: &mdarray::DTensor<num_complex::Complex64, 2>) -> f64 {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    let mut matrix_copy = matrix.clone();
    let SVDDecomp { s, .. } = Faer.svd(&mut *matrix_copy).expect("SVD computation failed");

    let min_dim = s.shape().0.min(s.shape().1);
    if min_dim == 0 {
        return 1.0;
    }

    // Singular values are real (stored as Complex, but imaginary part is 0)
    let max_sv = s[[0, 0]].re;
    let min_sv = s[[0, min_dim - 1]].re;

    if min_sv.abs() < 1e-15 {
        // Matrix is singular or nearly singular
        return f64::INFINITY;
    }

    max_sv / min_sv
}

// ============================================================================
// Evaluation Functions (coefficients → sampling points)
// ============================================================================

/// Evaluates basis coefficients at sampling points (double to double version).
///
/// Transforms basis coefficients to values at sampling points, where both input
/// and output are real (double precision) values. The operation can be performed
/// along any dimension of a multidimensional array.
///
/// # Arguments
///
/// * `s` - Pointer to the sampling object
/// * `order` - Memory layout order (`SPIR_ORDER_ROW_MAJOR` or `SPIR_ORDER_COLUMN_MAJOR`)
/// * `ndim` - Number of dimensions in the input/output arrays
/// * `input_dims` - Array of dimension sizes
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array of basis coefficients
/// * `out` - Output array for the evaluated values at sampling points
///
/// # Returns
///
/// An integer status code:
/// - `0` (`SPIR_COMPUTATION_SUCCESS`) on success
/// - A non-zero error code on failure
///
/// # Notes
///
/// - For optimal performance, the target dimension should be either the
///   first (`0`) or the last (`ndim-1`) dimension to avoid large temporary array allocations
/// - The output array must be pre-allocated with the correct size
/// - The input and output arrays must be contiguous in memory
/// - The transformation is performed using a pre-computed sampling matrix
///   that is factorized using SVD for efficiency
///
/// # See also
/// - [`spir_sampling_eval_dz`]
/// - [`spir_sampling_eval_zz`]
/// # Note
/// Supports both row-major and column-major order. Zero-copy implementation.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_eval_dd(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        // For column-major, this reverses dims and adjusts target_dim
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches basis size
        let sampling_inner = sampling_ref.inner();
        let expected_basis_size = sampling_inner.basis_size();
        if row_major_dims[row_major_target_dim] != expected_basis_size {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions
        let n_points = sampling_inner.n_points();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, n_points);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (zero-copy: writes directly to output buffer)
        if !InplaceFitter::evaluate_nd_dd_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Evaluate basis coefficients at sampling points (double → complex)
///
/// For Matsubara sampling: transforms real IR coefficients to complex values.
/// Zero-copy implementation.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_eval_dz(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches basis size
        let sampling_inner = sampling_ref.inner();
        let expected_basis_size = sampling_inner.basis_size();
        if row_major_dims[row_major_target_dim] != expected_basis_size {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions
        let n_points = sampling_inner.n_points();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, n_points);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (dz: real → complex)
        if !InplaceFitter::evaluate_nd_dz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Evaluate basis coefficients at sampling points (complex → complex)
///
/// For Matsubara sampling: transforms complex coefficients to complex values.
/// Zero-copy implementation.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_eval_zz(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches basis size
        let sampling_inner = sampling_ref.inner();
        let expected_basis_size = sampling_inner.basis_size();
        if row_major_dims[row_major_target_dim] != expected_basis_size {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions
        let n_points = sampling_inner.n_points();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, n_points);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (zz: complex → complex)
        if !InplaceFitter::evaluate_nd_zz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

// ============================================================================
// Fitting Functions (sampling points → coefficients)
// ============================================================================

/// Fits values at sampling points to basis coefficients (double to double version).
///
/// Transforms values at sampling points back to basis coefficients, where both
/// input and output are real (double precision) values. The operation can be
/// performed along any dimension of a multidimensional array.
///
/// # Arguments
///
/// * `s` - Pointer to the sampling object
/// * `backend` - Pointer to the GEMM backend (can be null to use default)
/// * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `ndim` - Number of dimensions in the input/output arrays
/// * `input_dims` - Array of dimension sizes
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array of values at sampling points
/// * `out` - Output array for the fitted basis coefficients
///
/// # Returns
///
/// An integer status code:
/// * `0` (SPIR_COMPUTATION_SUCCESS) on success
/// * A non-zero error code on failure
///
/// # Notes
///
/// * The output array must be pre-allocated with the correct size
/// * This function performs the inverse operation of `spir_sampling_eval_dd`
/// * The transformation is performed using a pre-computed sampling matrix
///   that is factorized using SVD for efficiency
/// * Zero-copy implementation
///
/// # See also
///
/// * [`spir_sampling_eval_dd`]
/// * [`spir_sampling_fit_zz`]
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_fit_dd(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches n_points
        let sampling_inner = sampling_ref.inner();
        let expected_n_points = sampling_inner.n_points();
        if row_major_dims[row_major_target_dim] != expected_n_points {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions (replace n_points with basis_size)
        let basis_size = sampling_inner.basis_size();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, basis_size);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Fit using InplaceFitter (dd: real → real)
        if !InplaceFitter::fit_nd_dd_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Fits values at sampling points to basis coefficients (complex to complex version).
///
/// For more details, see [`spir_sampling_fit_dd`]
/// Zero-copy implementation for Tau and Matsubara (full).
/// MatsubaraPositiveOnly requires intermediate storage for real→complex conversion.
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_fit_zz(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches n_points
        let sampling_inner = sampling_ref.inner();
        let expected_n_points = sampling_inner.n_points();
        if row_major_dims[row_major_target_dim] != expected_n_points {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions (replace n_points with basis_size)
        let basis_size = sampling_inner.basis_size();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, basis_size);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Fit using InplaceFitter (zz: complex → complex)
        if !InplaceFitter::fit_nd_zz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Fit basis coefficients from Matsubara sampling points (complex input, real output)
///
/// This function fits basis coefficients from Matsubara sampling points
/// using complex input and real output.
///
/// # Supported Sampling Types
///
/// - **Matsubara (full)**: ✅ Supported (takes real part of fitted complex coefficients)
/// - **Matsubara (positive_only)**: ✅ Supported
/// - **Tau**: ❌ Not supported (use `spir_sampling_fit_dd` instead)
///
/// # Notes
///
/// For full-range Matsubara sampling, this function fits complex coefficients
/// internally and returns their real parts. This is physically correct for
/// Green's functions where IR coefficients are guaranteed to be real by symmetry.
///
/// Zero-copy implementation.
///
/// # Arguments
///
/// * `s` - Pointer to the sampling object (must be Matsubara)
/// * `backend` - Pointer to the GEMM backend (can be null to use default)
/// * `order` - Memory layout order (SPIR_ORDER_COLUMN_MAJOR or SPIR_ORDER_ROW_MAJOR)
/// * `ndim` - Number of dimensions in the input/output arrays
/// * `input_dims` - Array of dimension sizes
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array (complex)
/// * `out` - Output array (real)
///
/// # Returns
///
/// - `SPIR_COMPUTATION_SUCCESS` on success
/// - `SPIR_NOT_SUPPORTED` if the sampling type doesn't support this operation
/// - Other error codes on failure
///
/// # See also
///
/// * [`spir_sampling_fit_zz`]
/// * [`spir_sampling_fit_dd`]
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_fit_zd(
    s: *const spir_sampling,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = unsafe { &*s };
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Convert dimensions for row-major processing
        let (row_major_dims, row_major_target_dim) =
            convert_dims_for_row_major(&orig_dims, target_dim as usize, mem_order);

        // Create input view directly from buffer (zero-copy)
        let input_view = unsafe { create_dview_from_ptr(input, &row_major_dims) };

        // Validate that input dimension matches n_points
        let sampling_inner = sampling_ref.inner();
        let expected_n_points = sampling_inner.n_points();
        if row_major_dims[row_major_target_dim] != expected_n_points {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }

        // Build output dimensions (replace n_points with basis_size)
        let basis_size = sampling_inner.basis_size();
        let out_dims = build_output_dims(&row_major_dims, row_major_target_dim, basis_size);

        // Create output view directly from buffer (zero-copy)
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &out_dims) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Fit using InplaceFitter (zd: complex → real)
        // Note: For full-range Matsubara, this takes the real part of the fitted
        // complex coefficients. This is physically correct for Green's functions
        // where IR coefficients are guaranteed to be real by symmetry.
        if !InplaceFitter::fit_nd_zd_to(
            sampling_inner,
            backend_handle,
            &input_view,
            row_major_target_dim,
            &mut output_view,
        ) {
            return SPIR_NOT_SUPPORTED;
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_sampling_creation() {
        // Create a basis
        let mut status = 0;
        let kernel = crate::spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let sve = crate::spir_sve_result_new(kernel, 1e-6, -1, -1, -1, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Limit basis size to 5
        let basis = crate::spir_basis_new(1, 10.0, 1.0, 1e-6, kernel, sve, 5, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // Get actual basis size
        let mut actual_basis_size = 0;
        let ret = crate::spir_basis_get_size(basis, &mut actual_basis_size);
        assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);

        // Create tau sampling with enough points (at least basis_size)
        let tau_points: Vec<f64> = (0..actual_basis_size)
            .map(|i| (i as f64 + 1.0) * 10.0 / (actual_basis_size as f64 + 1.0))
            .collect();

        let sampling = spir_tau_sampling_new(
            basis,
            tau_points.len() as i32,
            tau_points.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!sampling.is_null());

        // Get number of points
        let mut n_points = 0;
        let ret = spir_sampling_get_npoints(sampling, &mut n_points);
        assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_points, actual_basis_size);

        // Get tau points back
        let mut retrieved_points = vec![0.0; actual_basis_size as usize];
        let ret = spir_sampling_get_taus(sampling, retrieved_points.as_mut_ptr());
        assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);

        // Check that retrieved points match
        for (i, (&retrieved, &original)) in
            retrieved_points.iter().zip(tau_points.iter()).enumerate()
        {
            assert!(
                (retrieved - original).abs() < 1e-10,
                "Point {} mismatch: {} vs {}",
                i,
                retrieved,
                original
            );
        }

        // Get condition number
        let mut cond = 0.0;
        let ret = spir_sampling_get_cond_num(sampling, &mut cond);
        assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
        assert!(cond >= 1.0); // Condition number >= 1

        // Clean up
        crate::spir_sampling_release(sampling);
        crate::spir_basis_release(basis);
        crate::spir_sve_result_release(sve);
        crate::spir_kernel_release(kernel);
    }
}
