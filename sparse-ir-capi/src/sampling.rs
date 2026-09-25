//! Sampling API for C
//!
//! This module provides the C API for sparse sampling in imaginary time (τ),
//! Matsubara frequency (iν), and real frequency (ω) domains.
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
use crate::types::{BasisType, SamplingType, is_in_domain, spir_basis, spir_sampling, tau_domain};
use crate::utils::{
    MemoryOrder, create_dview_from_ptr, create_dviewmut_from_ptr, read_tensor_nd, validate_dims,
    validate_transform_dims,
};
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, SPIR_STATISTICS_BOSONIC,
    SPIR_STATISTICS_FERMIONIC, StatusCode,
};
use sparse_ir::fitters::InplaceFitter;
use sparse_ir::freq::MatsubaraFreq;
use sparse_ir::traits::StatisticsType;
use sparse_ir::{Bosonic, Fermionic};

/// Converts caller-supplied Matsubara indices, rejecting an invalid index
/// before any sampling object is built.
///
/// Every index must have the parity of the statistics (odd for fermions, even
/// for bosons). With `positive_only`, every index must also be non-negative
/// (n = 0 is valid for bosons). Returns `SPIR_INVALID_ARGUMENT` otherwise.
fn matsubara_freqs<S: StatisticsType>(
    points: &[i64],
    positive_only: bool,
) -> Result<Vec<MatsubaraFreq<S>>, StatusCode> {
    points
        .iter()
        .map(|&n| {
            if positive_only && n < 0 {
                return Err(SPIR_INVALID_ARGUMENT);
            }
            MatsubaraFreq::new(n).map_err(|_| SPIR_INVALID_ARGUMENT)
        })
        .collect()
}

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

/// Check if the sampling pointer is non-null.
///
/// Note: This only performs a null check. It cannot detect dangling
/// pointers; dereferencing an arbitrary non-null pointer would be
/// undefined behaviour that `catch_unwind` cannot reliably catch.
///
/// # Returns
/// 1 if the pointer is non-null, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn spir_sampling_is_assigned(obj: *const spir_sampling) -> i32 {
    if obj.is_null() { 0 } else { 1 }
}

// ============================================================================
// Creation Functions
// ============================================================================

/// Creates a new tau sampling object for sparse sampling in imaginary time
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `num_points` - Number of sampling points
/// * `points` - Array of `num_points` sampling points τ ∈ [-β, β], with β the
///   inverse temperature of `b`; a negative τ is folded onto [0, β] as in
///   `spir_funcs_eval`
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails.
/// If `status` is non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `b` or `points` is NULL, `num_points` <= 0, or a
///   point is NaN, infinite or outside [-β, β]
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The points may be in any order, and the sampling object keeps it:
/// `spir_sampling_get_taus` returns `points` unchanged, and index i along the
/// sampling-point axis of the evaluate and fit functions refers to
/// `points[i]`.
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

        // Check every point before building the sampling object: the core
        // asserts τ ∈ [-β, β] (#266).
        let domain = tau_domain(basis_ref.beta());
        if !points_slice.iter().all(|&tau| is_in_domain(tau, domain)) {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

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
/// * `positive_only` - If true, only non-negative frequencies are used; the IR
///   coefficients are then real, i.e. G(-iν) = conj(G(iν))
/// * `num_points` - Number of sampling points
/// * `points` - Array of `num_points` reduced Matsubara frequencies n
///   (iν = iπn/β): odd for a fermionic basis, even for a bosonic basis, and
///   non-negative when `positive_only` is true
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails.
/// If `status` is non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `b` or `points` is NULL, `num_points` <= 0, an
///   index has the wrong parity for the statistics of `b`, or `positive_only`
///   is true and an index is negative
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The points may be in any order, and the sampling object keeps it: they
/// are not sorted, `spir_sampling_get_matsus` returns `points` unchanged, and
/// index i along the sampling-point axis of the evaluate and fit functions
/// refers to `points[i]`.
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

        // Helper macro to reduce duplication. The indices are validated against
        // the basis statistics before the sampling object is built (#247).
        macro_rules! create_matsu_sampling {
            ($basis:expr, Fermionic) => {{
                let matsu_freqs =
                    match matsubara_freqs::<Fermionic>(&matsu_points, positive_only) {
                        Ok(freqs) => freqs,
                        Err(code) => return (std::ptr::null_mut(), code),
                    };
                if positive_only {
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyFermionic(Arc::new(matsu_sampling))
                } else {
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraFermionic(Arc::new(matsu_sampling))
                }
            }};
            ($basis:expr, Bosonic) => {{
                let matsu_freqs =
                    match matsubara_freqs::<Bosonic>(&matsu_points, positive_only) {
                        Ok(freqs) => freqs,
                        Err(code) => return (std::ptr::null_mut(), code),
                    };
                if positive_only {
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
                } else {
                    let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
                }
            }};
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
/// * `order` - Memory layout of `matrix` (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size (the number of columns of `matrix`)
/// * `num_points` - Number of sampling points (the number of rows of `matrix`)
/// * `points` - Array of `num_points` finite sampling points in imaginary time
///   (τ), one per row of `matrix`. Without β the domain [-β, β] of
///   `spir_tau_sampling_new` cannot be checked here: any finite value is
///   accepted and reported back by `spir_sampling_get_taus`
/// * `matrix` - Pre-computed `num_points × basis_size` sampling matrix in
///   `order`, with finite entries
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails.
/// If `status` is non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `points` or `matrix` is NULL, `num_points` or
///   `basis_size` <= 0, `order` or `statistics` is not one of the constants
///   above, a point is NaN or infinite, or an entry of `matrix` is NaN or
///   infinite
/// - SPIR_INVALID_DIMENSION if the matrix is too large to be addressed
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The scalar arguments are validated before `points` or `matrix` is read.
///
/// The points may be in any order, and the sampling object keeps it:
/// `spir_sampling_get_taus` returns `points` unchanged, and index i along the
/// sampling-point axis of the evaluate and fit functions refers to
/// `points[i]`, the point of row i of `matrix`.
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
        let fermionic = match statistics {
            SPIR_STATISTICS_FERMIONIC => true,
            SPIR_STATISTICS_BOSONIC => false,
            _ => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };
        // The matrix must be addressable before `points` or `matrix` is read (#245).
        let dims = match validate_dims::<f64>(&[num_points, basis_size]) {
            Ok(dims) => dims,
            Err(code) => return (std::ptr::null_mut(), code),
        };

        // SAFETY: `points` is non-null and `dims[0] == num_points > 0` (checked
        // above); the caller guarantees that `points` holds `num_points` elements.
        let points_slice = unsafe { std::slice::from_raw_parts(points, dims[0]) };
        // Without β only finiteness can be checked, not τ ∈ [-β, β] (#266).
        if !points_slice.iter().all(|tau| tau.is_finite()) {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        let tau_points: Vec<f64> = points_slice.to_vec();

        // SAFETY: `matrix` is non-null and `validate_dims` proved that `dims`
        // passes `checked_len::<f64>`; the caller guarantees that `matrix` holds
        // `num_points * basis_size` elements.
        let dyn_tensor = unsafe { read_tensor_nd(matrix, &dims, mem_order) };
        // The fitter factorizes the matrix: reject NaN and infinities here
        // rather than in a panicking SVD at the first fit.
        if !dyn_tensor.iter().all(|x| x.is_finite()) {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

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
        let sampling_type = if fermionic {
            let tau_sampling = sparse_ir::sampling::TauSampling::<Fermionic>::from_matrix(
                tau_points,
                matrix_tensor,
            );
            SamplingType::TauFermionic(Arc::new(tau_sampling))
        } else {
            let tau_sampling =
                sparse_ir::sampling::TauSampling::<Bosonic>::from_matrix(tau_points, matrix_tensor);
            SamplingType::TauBosonic(Arc::new(tau_sampling))
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
/// * `order` - Memory layout of `matrix` (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size (the number of columns of `matrix`)
/// * `positive_only` - If true, only non-negative frequencies are used; the IR
///   coefficients are then real, i.e. G(-iν) = conj(G(iν))
/// * `num_points` - Number of sampling points (the number of rows of `matrix`)
/// * `points` - Array of `num_points` reduced Matsubara frequencies n
///   (iν = iπn/β): odd for fermionic, even for bosonic `statistics`, and
///   non-negative when `positive_only` is true
/// * `matrix` - Pre-computed complex `num_points × basis_size` sampling matrix
///   in `order`, with finite real and imaginary parts
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails.
/// If `status` is non-NULL, `*status` is set to:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if `points` or `matrix` is NULL, `num_points` or
///   `basis_size` <= 0, `order` or `statistics` is not one of the constants
///   above, an index has the wrong parity for `statistics`, `positive_only`
///   is true and an index is negative, or an entry of `matrix` has a NaN or
///   infinite part
/// - SPIR_INVALID_DIMENSION if the matrix is too large to be addressed
/// - SPIR_INTERNAL_ERROR if an internal error occurs
///
/// The scalar arguments are validated before `points` or `matrix` is read,
/// and the indices before `matrix` is read.
///
/// The points may be in any order, and the sampling object keeps it:
/// `spir_sampling_get_matsus` returns `points` unchanged, and index i along
/// the sampling-point axis of the evaluate and fit functions refers to
/// `points[i]`, the point of row i of `matrix`.
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
        if statistics != SPIR_STATISTICS_FERMIONIC && statistics != SPIR_STATISTICS_BOSONIC {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        // The matrix must be addressable before `points` or `matrix` is read (#245).
        let dims = match validate_dims::<Complex64>(&[num_points, basis_size]) {
            Ok(dims) => dims,
            Err(code) => return (std::ptr::null_mut(), code),
        };

        // Convert points to Vec<MatsubaraFreq>
        debug_println!("spir_matsu_sampling_new_with_matrix: creating points slice...");
        std::io::stderr().flush().ok();
        // SAFETY: `points` is non-null and `dims[0] == num_points > 0` (checked
        // above); the caller guarantees that `points` holds `num_points` elements.
        let points_slice = unsafe { std::slice::from_raw_parts(points, dims[0]) };
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

        // Validate the statistics and every index before the matrix is read or
        // anything is built (#247).
        enum MatsuFreqs {
            Fermionic(Vec<MatsubaraFreq<Fermionic>>),
            Bosonic(Vec<MatsubaraFreq<Bosonic>>),
        }
        let matsu_freqs = match statistics {
            SPIR_STATISTICS_FERMIONIC => {
                matsubara_freqs(&matsu_points, positive_only).map(MatsuFreqs::Fermionic)
            }
            SPIR_STATISTICS_BOSONIC => {
                matsubara_freqs(&matsu_points, positive_only).map(MatsuFreqs::Bosonic)
            }
            _ => Err(SPIR_INVALID_ARGUMENT),
        };
        let matsu_freqs = match matsu_freqs {
            Ok(freqs) => freqs,
            Err(code) => return (std::ptr::null_mut(), code),
        };

        // Convert matrix to Tensor using the new helper function
        debug_println!(
            "spir_matsu_sampling_new_with_matrix: dims = {:?}, mem_order = {:?}",
            dims,
            mem_order
        );
        std::io::stderr().flush().ok();

        debug_println!("spir_matsu_sampling_new_with_matrix: reading tensor from buffer...");
        std::io::stderr().flush().ok();
        // SAFETY: `matrix` is non-null and `validate_dims` proved that `dims`
        // passes `checked_len::<Complex64>`; the caller guarantees that `matrix`
        // holds `num_points * basis_size` elements.
        let dyn_tensor = unsafe { read_tensor_nd(matrix, &dims, mem_order) };
        // The fitter factorizes the matrix: reject NaN and infinities here
        // rather than in a panicking SVD at the first fit.
        if !dyn_tensor
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite())
        {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
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
        let sampling_type = match (matsu_freqs, positive_only) {
            (MatsuFreqs::Fermionic(matsu_freqs), true) => {
                debug_println!("spir_matsu_sampling_new_with_matrix: Fermionic, positive-only");
                std::io::stderr().flush().ok();
                // Fermionic, positive-only
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
            (MatsuFreqs::Fermionic(matsu_freqs), false) => {
                debug_println!("spir_matsu_sampling_new_with_matrix: Fermionic, full range");
                std::io::stderr().flush().ok();
                // Fermionic, full range
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
            (MatsuFreqs::Bosonic(matsu_freqs), true) => {
                // Bosonic, positive-only
                let matsu_sampling =
                    sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly::from_matrix(
                        matsu_freqs,
                        matrix_tensor.clone(),
                    );
                SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
            }
            (MatsuFreqs::Bosonic(matsu_freqs), false) => {
                // Bosonic, full range
                let matsu_sampling = sparse_ir::matsubara_sampling::MatsubaraSampling::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
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

/// Gets the condition number of the least-squares problem that fitting solves.
///
/// Stores in `*cond_num` the ratio σ_max / σ_min of the largest to the smallest
/// of the min(rows, columns) singular values of the matrix that the fit
/// functions (`spir_sampling_fit_dd`, `spir_sampling_fit_zz`,
/// `spir_sampling_fit_zd`) solve with. Let `A` be the `n_points × basis_size`
/// sampling matrix: `A[i, l]` is basis function `l` at sampling point `i`, or
/// `A` is the matrix passed to `spir_tau_sampling_new_with_matrix` or
/// `spir_matsu_sampling_new_with_matrix`. The matrix the fit solves with is:
///
/// - τ sampling: the real matrix `A`.
/// - Matsubara sampling with `positive_only = false`: the complex matrix `A`.
/// - Matsubara sampling with `positive_only = true`: the real
///   `2 n_points × basis_size` matrix `[Re A; Im A]` of the real least-squares
///   problem `[Re A; Im A] x = [Re g; Im g]` that the fit solves for real
///   coefficients `x`. This is not the condition number of the complex matrix
///   `A`: with `n_points ≈ basis_size / 2`, `A` is wide, and its condition
///   number can understate the error amplification of the fit by orders of
///   magnitude.
///
/// The value bounds how much fitting can amplify relative errors in the values.
///
/// # Parameters
/// - `s`: Pointer to the sampling object.
/// - `cond_num`: Pointer to store the condition number.
///
/// # Returns
/// An integer status code:
/// - 0 (`SPIR_COMPUTATION_SUCCESS`) on success
/// - `SPIR_INVALID_ARGUMENT` if `s` or `cond_num` is null; `*cond_num` is not
///   written
/// - `SPIR_INTERNAL_ERROR` if an internal error occurs
///
/// # Notes
/// - A large condition number indicates that the sampling problem is
///   ill-conditioned, which may lead to numerical instability in fitting.
/// - `+inf` is stored if the smallest singular value is below 1e-15
///   (numerically singular matrix).
/// - The singular value decomposition is the one the fit functions use: it is
///   computed once per sampling object (shared with its clones), by the first
///   call to this function or to a fit function, and then reused.
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

        // The core reports the condition number of the matrix its fitter
        // solves with, from the SVD it caches for fitting.
        let condition_number = match sampling_ref.inner() {
            SamplingType::TauFermionic(tau) => tau.condition_number(),
            SamplingType::TauBosonic(tau) => tau.condition_number(),
            SamplingType::MatsubaraFermionic(matsu) => matsu.condition_number(),
            SamplingType::MatsubaraBosonic(matsu) => matsu.condition_number(),
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => matsu.condition_number(),
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => matsu.condition_number(),
        };

        unsafe {
            *cond_num = condition_number;
        }
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
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
/// * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array of basis coefficients
/// * `out` - Output array for the evaluated values at sampling points
///
/// # Returns
///
/// An integer status code:
/// - `0` (`SPIR_COMPUTATION_SUCCESS`) on success
/// - `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// - `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// - `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the basis size
/// - `SPIR_NOT_SUPPORTED` if the sampling type does not support this operation
/// - `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<f64, f64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.basis_size(),
            sampling_inner.n_points(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (zero-copy: writes directly to output buffer)
        if !InplaceFitter::evaluate_nd_dd_to(
            sampling_inner,
            backend_handle,
            &input_view,
            dims.target_dim,
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
/// Zero-copy implementation. Arguments are as for [`spir_sampling_eval_dd`].
///
/// # Returns
///
/// - `SPIR_COMPUTATION_SUCCESS` on success
/// - `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// - `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// - `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the basis size
/// - `SPIR_NOT_SUPPORTED` if the sampling type does not support this operation
/// - `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<f64, Complex64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.basis_size(),
            sampling_inner.n_points(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (dz: real → complex)
        if !InplaceFitter::evaluate_nd_dz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            dims.target_dim,
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
/// Zero-copy implementation. Arguments are as for [`spir_sampling_eval_dd`].
///
/// # Returns
///
/// - `SPIR_COMPUTATION_SUCCESS` on success
/// - `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// - `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// - `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the basis size
/// - `SPIR_NOT_SUPPORTED` if the sampling type does not support this operation
/// - `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<Complex64, Complex64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.basis_size(),
            sampling_inner.n_points(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Evaluate using InplaceFitter (zz: complex → complex)
        if !InplaceFitter::evaluate_nd_zz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            dims.target_dim,
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
/// * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array of values at sampling points
/// * `out` - Output array for the fitted basis coefficients
///
/// # Returns
///
/// An integer status code:
/// * `0` (SPIR_COMPUTATION_SUCCESS) on success
/// * `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   number of sampling points
/// * `SPIR_NOT_SUPPORTED` if the sampling type does not support this operation
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<f64, f64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.n_points(),
            sampling_inner.basis_size(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Fit using InplaceFitter (dd: real → real)
        if !InplaceFitter::fit_nd_dd_to(
            sampling_inner,
            backend_handle,
            &input_view,
            dims.target_dim,
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
///
/// # Returns
///
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   number of sampling points
/// * `SPIR_NOT_SUPPORTED` if the sampling type does not support this operation
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<Complex64, Complex64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.n_points(),
            sampling_inner.basis_size(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Fit using InplaceFitter (zz: complex → complex)
        if !InplaceFitter::fit_nd_zz_to(
            sampling_inner,
            backend_handle,
            &input_view,
            dims.target_dim,
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
/// * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive
/// * `target_dim` - Target dimension for the transformation (0-based)
/// * `input` - Input array (complex)
/// * `out` - Output array (real)
///
/// # Returns
///
/// - `SPIR_COMPUTATION_SUCCESS` on success
/// - `SPIR_INVALID_ARGUMENT` if `s`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// - `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// - `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   number of sampling points
/// - `SPIR_NOT_SUPPORTED` if the sampling type doesn't support this operation
/// - `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// All shape arguments are validated before `input` or `out` is accessed.
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
        let sampling_inner = sampling_ref.inner();

        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before any view is built over `input` or `out`.
        // For column-major, this also reverses dims and adjusts target_dim.
        let dims = match validate_transform_dims::<Complex64, f64>(
            dims_slice,
            target_dim as usize,
            mem_order,
            sampling_inner.n_points(),
            sampling_inner.basis_size(),
        ) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // Create zero-copy views directly over the caller's buffers.
        // SAFETY: `validate_transform_dims` proved that both shapes have
        // addressable sizes; the caller guarantees that `input` and `out` hold
        // that many elements.
        let input_view = unsafe { create_dview_from_ptr(input, &dims.input) };
        let mut output_view = unsafe { create_dviewmut_from_ptr(out, &dims.output) };

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
            dims.target_dim,
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
