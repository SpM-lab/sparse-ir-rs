//! DLR (Discrete Lehmann Representation) API for C
//!
//! This module provides the C API for Discrete Lehmann Representation (DLR),
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.
//!
//! Functions:
//! - Creation: spir_dlr_new, spir_dlr_new_with_poles
//! - Introspection: spir_dlr_get_npoles, spir_dlr_get_poles
//! - Conversion: spir_ir2dlr_dd, spir_ir2dlr_zz, spir_dlr2ir_dd, spir_dlr2ir_zz

use num_complex::Complex64;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::gemm::{get_backend_handle, spir_gemm_backend};
use crate::status::status_from;
use crate::types::{BasisType, spir_basis};
use crate::utils::{
    MemoryOrder, copy_tensor_to_c_array, read_tensor_nd, transform_dims, validate_dims,
};
use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, StatusCode};
use sparse_ir::dlr::DiscreteLehmannRepresentation;

// ============================================================================
// Creation Functions
// ============================================================================

/// Creates a new DLR from an IR basis with default poles
///
/// The default poles are the default real-frequency sampling points of `b`
/// (see `spir_basis_get_default_ws`).
///
/// # Arguments
/// * `b` - Pointer to a finite temperature (IR) basis object
/// * `status` - Pointer to store the status code (may be NULL, in which case
///   no status is written)
///
/// # Returns
/// * Pointer to the newly created DLR basis object, or NULL on failure. The
///   caller owns it and must release it with `spir_basis_release`.
/// * Status code:
///   - `SPIR_COMPUTATION_SUCCESS` (0) on success
///   - `SPIR_INVALID_ARGUMENT` (-6) if `b` is NULL or already a DLR, or if
///     `b` has fewer default poles than basis functions
///     (`spir_basis_get_n_default_ws` < `spir_basis_get_size`). Root finding
///     can lose poles, e.g. for `RegularizedBoseKernel` at large lambda; pass
///     the poles explicitly with `spir_dlr_new_with_poles` instead.
///   - `SPIR_NOT_SUPPORTED` (-5) if the kernel of `b` does not support its
///     statistics (`RegularizedBoseKernel` with fermionic statistics; the
///     basis constructors already reject this combination), or the default
///     poles of `b` are not defined (see `spir_basis_get_default_ws`)
///   - `SPIR_INTERNAL_ERROR` (-7) if an internal panic occurs
///
/// # Safety
/// Caller must ensure `b` is a valid IR basis pointer
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr_new(b: *const spir_basis, status: *mut StatusCode) -> *mut spir_basis {
    let result = catch_unwind(AssertUnwindSafe(
        || -> Result<(*mut spir_basis, StatusCode), StatusCode> {
            // Validate inputs
            if b.is_null() {
                return Err(SPIR_INVALID_ARGUMENT);
            }

            let basis_ref = unsafe { &*b };

            // Create DLR based on basis type
            let dlr_type = match basis_ref.inner() {
                BasisType::LogisticFermionic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir(ir_basis.as_ref())
                        .map_err(|e| status_from(&e))?;
                    BasisType::DLRFermionic(Arc::new(dlr))
                }
                BasisType::LogisticBosonic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir(ir_basis.as_ref())
                        .map_err(|e| status_from(&e))?;
                    BasisType::DLRBosonic(Arc::new(dlr))
                }
                BasisType::RegularizedBoseFermionic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir(ir_basis.as_ref())
                        .map_err(|e| status_from(&e))?;
                    BasisType::DLRFermionic(Arc::new(dlr))
                }
                BasisType::RegularizedBoseBosonic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir(ir_basis.as_ref())
                        .map_err(|e| status_from(&e))?;
                    BasisType::DLRBosonic(Arc::new(dlr))
                }
                _ => {
                    // Already a DLR, return error
                    return Err(SPIR_INVALID_ARGUMENT);
                }
            };

            let dlr_basis = match dlr_type {
                BasisType::DLRFermionic(arc_dlr) => spir_basis::new_dlr_fermionic(arc_dlr),
                BasisType::DLRBosonic(arc_dlr) => spir_basis::new_dlr_bosonic(arc_dlr),
                _ => unreachable!(), // We know it's one of the DLR types
            };

            Ok((Box::into_raw(Box::new(dlr_basis)), SPIR_COMPUTATION_SUCCESS))
        },
    ));

    match result {
        Ok(Ok((ptr, code))) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            std::ptr::null_mut()
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

/// Creates a new DLR with custom poles
///
/// # Arguments
/// * `b` - Pointer to a finite temperature (IR) basis object
/// * `npoles` - Number of poles to use (must be > 0)
/// * `poles` - Array of `npoles` pole locations in [-omega_max, omega_max] of `b`
/// * `status` - Pointer to store the status code (may be NULL, in which case
///   no status is written)
///
/// # Returns
/// * Pointer to the newly created DLR basis object, or NULL on failure. The
///   caller owns it and must release it with `spir_basis_release`.
/// * Status code:
///   - `SPIR_COMPUTATION_SUCCESS` (0) on success
///   - `SPIR_INVALID_ARGUMENT` (-6) if `b` or `poles` is NULL, `npoles <= 0`,
///     or `b` is already a DLR, or a pole is outside [-omega_max, omega_max]
///     of `b` or not finite
///   - `SPIR_NOT_SUPPORTED` (-5) if the kernel of `b` does not support its
///     statistics (`RegularizedBoseKernel` with fermionic statistics). The
///     basis constructors already reject this combination.
///   - `SPIR_INTERNAL_ERROR` (-7) if an internal panic occurs
///
/// Duplicate poles are accepted. They make `spir_ir2dlr_dd` and
/// `spir_ir2dlr_zz` ill-conditioned: the coefficients of equal poles are not
/// unique, although the round trip through `spir_dlr2ir_dd` /
/// `spir_dlr2ir_zz` still recovers the IR coefficients.
///
/// # Safety
/// Caller must ensure `b` is valid and `poles` has `npoles` elements
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr_new_with_poles(
    b: *const spir_basis,
    npoles: libc::c_int,
    poles: *const f64,
    status: *mut StatusCode,
) -> *mut spir_basis {
    let result = catch_unwind(AssertUnwindSafe(
        || -> Result<(*mut spir_basis, StatusCode), StatusCode> {
            // Validate inputs
            if b.is_null() || poles.is_null() {
                return Err(SPIR_INVALID_ARGUMENT);
            }
            if npoles <= 0 {
                return Err(SPIR_INVALID_ARGUMENT);
            }

            let basis_ref = unsafe { &*b };
            let poles_slice = unsafe { std::slice::from_raw_parts(poles, npoles as usize) };
            let pole_vec: Vec<f64> = poles_slice.to_vec();

            // Create DLR based on basis type
            let dlr_type = match basis_ref.inner() {
                BasisType::LogisticFermionic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir_with_poles(
                        ir_basis.as_ref(),
                        pole_vec,
                    )
                    .map_err(|e| status_from(&e))?;
                    BasisType::DLRFermionic(Arc::new(dlr))
                }
                BasisType::LogisticBosonic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir_with_poles(
                        ir_basis.as_ref(),
                        pole_vec,
                    )
                    .map_err(|e| status_from(&e))?;
                    BasisType::DLRBosonic(Arc::new(dlr))
                }
                BasisType::RegularizedBoseFermionic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir_with_poles(
                        ir_basis.as_ref(),
                        pole_vec,
                    )
                    .map_err(|e| status_from(&e))?;
                    BasisType::DLRFermionic(Arc::new(dlr))
                }
                BasisType::RegularizedBoseBosonic(ir_basis) => {
                    let dlr = DiscreteLehmannRepresentation::from_ir_with_poles(
                        ir_basis.as_ref(),
                        pole_vec,
                    )
                    .map_err(|e| status_from(&e))?;
                    BasisType::DLRBosonic(Arc::new(dlr))
                }
                _ => {
                    // Already a DLR or invalid type
                    return Err(SPIR_INVALID_ARGUMENT);
                }
            };

            let dlr_basis = match dlr_type {
                BasisType::DLRFermionic(arc_dlr) => spir_basis::new_dlr_fermionic(arc_dlr),
                BasisType::DLRBosonic(arc_dlr) => spir_basis::new_dlr_bosonic(arc_dlr),
                _ => unreachable!(), // We know it's one of the DLR types
            };

            Ok((Box::into_raw(Box::new(dlr_basis)), SPIR_COMPUTATION_SUCCESS))
        },
    ));

    match result {
        Ok(Ok((ptr, code))) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            ptr
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                unsafe {
                    *status = code;
                }
            }
            std::ptr::null_mut()
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

/// Gets the number of poles in a DLR
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `num_poles` - Pointer to store the number of poles
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure `dlr` is a valid DLR basis pointer
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr_get_npoles(
    dlr: *const spir_basis,
    num_poles: *mut libc::c_int,
) -> StatusCode {
    if dlr.is_null() || num_poles.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dlr_ref = unsafe { &*dlr };

        // Get number of poles based on DLR type
        let npoles = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => dlr.poles().len(),
            BasisType::DLRBosonic(dlr) => dlr.poles().len(),
            _ => return SPIR_INVALID_ARGUMENT, // Not a DLR
        };

        unsafe {
            *num_poles = npoles as libc::c_int;
        }
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the pole locations in a DLR
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `poles` - Pre-allocated array to store pole locations
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure `dlr` is valid and `poles` has sufficient size
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr_get_poles(dlr: *const spir_basis, poles: *mut f64) -> StatusCode {
    if dlr.is_null() || poles.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dlr_ref = unsafe { &*dlr };

        // Get poles based on DLR type
        let pole_vec = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => dlr.poles(),
            BasisType::DLRBosonic(dlr) => dlr.poles(),
            _ => return SPIR_INVALID_ARGUMENT, // Not a DLR
        };

        // Copy poles to output array
        for (i, &pole) in pole_vec.iter().enumerate() {
            unsafe {
                *poles.add(i) = pole;
            }
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

// ============================================================================
// Conversion Functions
// ============================================================================

/// IR basis size and number of poles of a DLR, or `None` if `b` is not a DLR
/// built from an IR basis
fn dlr_sizes(b: &spir_basis) -> Option<(usize, usize)> {
    match b.inner() {
        BasisType::DLRFermionic(dlr) => Some((dlr.ir_basis_size()?, dlr.poles().len())),
        BasisType::DLRBosonic(dlr) => Some((dlr.ir_basis_size()?, dlr.poles().len())),
        _ => None,
    }
}

/// Convert IR coefficients to DLR (real-valued)
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order
/// * `ndim` - Number of dimensions
/// * `input_dims` - Array of `ndim` input dimensions, each of which must be positive
/// * `target_dim` - Dimension to transform
/// * `input` - IR coefficients
/// * `out` - Output DLR coefficients
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if `dlr`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   size of the IR basis of `dlr`
/// * `SPIR_NOT_SUPPORTED` if `dlr` is not a DLR basis
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// `input_dims` is validated before `input` or `out` is accessed.
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_ir2dlr_dd(
    dlr: *const spir_basis,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if dlr.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
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

        let dlr_ref = unsafe { &*dlr };
        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before `input` is read.
        let orig_dims = match validate_dims::<f64>(dims_slice) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // The target extent and the size of `out`, before `input` is read.
        // Only the check is used: the conversion below reads `input` in the
        // caller's order and axis numbering (read_tensor_nd), not in the
        // row-major frame of TransformDims.
        let Some((ir_size, n_poles)) = dlr_sizes(dlr_ref) else {
            return SPIR_NOT_SUPPORTED; // Not a DLR
        };
        if let Err(code) =
            transform_dims::<f64>(&orig_dims, target_dim as usize, mem_order, ir_size, n_poles)
        {
            return code;
        }

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        // SAFETY: `validate_dims` proved that `orig_dims` has an addressable size;
        // the caller guarantees that `input` holds that many elements.
        let input_tensor = match unsafe { read_tensor_nd(input, &orig_dims, mem_order) } {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Convert IR to DLR based on DLR type
        // target_dim is already correct since read_tensor_nd preserves orig_dims shape
        let result_tensor = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => {
                dlr.from_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            BasisType::DLRBosonic(dlr) => {
                dlr.from_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            _ => return SPIR_NOT_SUPPORTED, // Not a DLR
        };
        let result_tensor = match result_tensor {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Copy result to output with correct memory order
        if let Err(e) = unsafe { copy_tensor_to_c_array(result_tensor, out, mem_order) } {
            return status_from(&e);
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Convert IR coefficients to DLR (complex-valued)
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order
/// * `ndim` - Number of dimensions
/// * `input_dims` - Array of `ndim` input dimensions, each of which must be positive
/// * `target_dim` - Dimension to transform
/// * `input` - Complex IR coefficients
/// * `out` - Output complex DLR coefficients
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if `dlr`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   size of the IR basis of `dlr`
/// * `SPIR_NOT_SUPPORTED` if `dlr` is not a DLR basis
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// `input_dims` is validated before `input` or `out` is accessed.
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_ir2dlr_zz(
    dlr: *const spir_basis,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if dlr.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
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

        let dlr_ref = unsafe { &*dlr };
        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before `input` is read.
        let orig_dims = match validate_dims::<Complex64>(dims_slice) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // The target extent and the size of `out`, before `input` is read.
        // Only the check is used: the conversion below reads `input` in the
        // caller's order and axis numbering (read_tensor_nd), not in the
        // row-major frame of TransformDims.
        let Some((ir_size, n_poles)) = dlr_sizes(dlr_ref) else {
            return SPIR_NOT_SUPPORTED; // Not a DLR
        };
        if let Err(code) = transform_dims::<Complex64>(
            &orig_dims,
            target_dim as usize,
            mem_order,
            ir_size,
            n_poles,
        ) {
            return code;
        }

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        // SAFETY: `validate_dims` proved that `orig_dims` has an addressable size;
        // the caller guarantees that `input` holds that many elements.
        let input_tensor = match unsafe { read_tensor_nd(input, &orig_dims, mem_order) } {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Convert IR to DLR based on DLR type
        // target_dim is already correct since read_tensor_nd preserves orig_dims shape
        let result_tensor = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => {
                dlr.from_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            BasisType::DLRBosonic(dlr) => {
                dlr.from_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            _ => return SPIR_NOT_SUPPORTED, // Not a DLR
        };
        let result_tensor = match result_tensor {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Copy result to output with correct memory order
        if let Err(e) = unsafe { copy_tensor_to_c_array(result_tensor, out, mem_order) } {
            return status_from(&e);
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Convert DLR coefficients to IR (real-valued)
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order
/// * `ndim` - Number of dimensions
/// * `input_dims` - Array of `ndim` input dimensions, each of which must be positive
/// * `target_dim` - Dimension to transform
/// * `input` - DLR coefficients
/// * `out` - Output IR coefficients
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if `dlr`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   number of poles of `dlr` (`spir_dlr_get_npoles`)
/// * `SPIR_NOT_SUPPORTED` if `dlr` is not a DLR basis
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// `input_dims` is validated before `input` or `out` is accessed.
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr2ir_dd(
    dlr: *const spir_basis,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if dlr.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
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

        let dlr_ref = unsafe { &*dlr };
        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before `input` is read.
        let orig_dims = match validate_dims::<f64>(dims_slice) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // The target extent and the size of `out`, before `input` is read.
        // Only the check is used: the conversion below reads `input` in the
        // caller's order and axis numbering (read_tensor_nd), not in the
        // row-major frame of TransformDims.
        let Some((ir_size, n_poles)) = dlr_sizes(dlr_ref) else {
            return SPIR_NOT_SUPPORTED; // Not a DLR
        };
        if let Err(code) =
            transform_dims::<f64>(&orig_dims, target_dim as usize, mem_order, n_poles, ir_size)
        {
            return code;
        }

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        // SAFETY: `validate_dims` proved that `orig_dims` has an addressable size;
        // the caller guarantees that `input` holds that many elements.
        let input_tensor = match unsafe { read_tensor_nd(input, &orig_dims, mem_order) } {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Convert DLR to IR based on DLR type
        // target_dim is already correct since read_tensor_nd preserves orig_dims shape
        let result_tensor = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => {
                dlr.to_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            BasisType::DLRBosonic(dlr) => {
                dlr.to_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            _ => return SPIR_NOT_SUPPORTED, // Not a DLR
        };
        let result_tensor = match result_tensor {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Copy result to output with correct memory order
        if let Err(e) = unsafe { copy_tensor_to_c_array(result_tensor, out, mem_order) } {
            return status_from(&e);
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Convert DLR coefficients to IR (complex-valued)
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order
/// * `ndim` - Number of dimensions
/// * `input_dims` - Array of `ndim` input dimensions, each of which must be positive
/// * `target_dim` - Dimension to transform
/// * `input` - Complex DLR coefficients
/// * `out` - Output complex IR coefficients
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if `dlr`, `input_dims`, `input` or `out` is null,
///   `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)`
/// * `SPIR_INVALID_DIMENSION` if an element of `input_dims` is zero or negative,
///   or the input or output array is too large to be addressed
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim]` is not the
///   number of poles of `dlr` (`spir_dlr_get_npoles`)
/// * `SPIR_NOT_SUPPORTED` if `dlr` is not a DLR basis
/// * `SPIR_INTERNAL_ERROR` if an internal panic occurs
///
/// `input_dims` is validated before `input` or `out` is accessed.
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub extern "C" fn spir_dlr2ir_zz(
    dlr: *const spir_basis,
    backend: *const spir_gemm_backend,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if dlr.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
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

        let dlr_ref = unsafe { &*dlr };
        // SAFETY: `input_dims` is non-null and `ndim > 0` (checked above); the
        // caller guarantees that it points to `ndim` readable elements.
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        // Validate every extent before `input` is read.
        let orig_dims = match validate_dims::<Complex64>(dims_slice) {
            Ok(dims) => dims,
            Err(code) => return code,
        };

        // The target extent and the size of `out`, before `input` is read.
        // Only the check is used: the conversion below reads `input` in the
        // caller's order and axis numbering (read_tensor_nd), not in the
        // row-major frame of TransformDims.
        let Some((ir_size, n_poles)) = dlr_sizes(dlr_ref) else {
            return SPIR_NOT_SUPPORTED; // Not a DLR
        };
        if let Err(code) = transform_dims::<Complex64>(
            &orig_dims,
            target_dim as usize,
            mem_order,
            n_poles,
            ir_size,
        ) {
            return code;
        }

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        // SAFETY: `validate_dims` proved that `orig_dims` has an addressable size;
        // the caller guarantees that `input` holds that many elements.
        let input_tensor = match unsafe { read_tensor_nd(input, &orig_dims, mem_order) } {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Get backend handle (NULL means use default)
        let backend_handle = unsafe { get_backend_handle(backend) };

        // Convert DLR to IR based on DLR type
        // target_dim is already correct since read_tensor_nd preserves orig_dims shape
        let result_tensor = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => {
                dlr.to_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            BasisType::DLRBosonic(dlr) => {
                dlr.to_ir_nd(backend_handle, &input_tensor, target_dim as usize)
            }
            _ => return SPIR_NOT_SUPPORTED, // Not a DLR
        };
        let result_tensor = match result_tensor {
            Ok(tensor) => tensor,
            Err(e) => return status_from(&e),
        };

        // Copy result to output with correct memory order
        if let Err(e) = unsafe { copy_tensor_to_c_array(result_tensor, out, mem_order) } {
            return status_from(&e);
        }

        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SPIR_COMPUTATION_SUCCESS;
    use crate::basis::spir_basis_new;
    use crate::kernel::spir_logistic_kernel_new;
    use crate::sve::spir_sve_result_new;

    #[test]
    fn test_dlr_creation() {
        // Ensure clean state before test (BLAS backend should be default Faer)
        sparse_ir::gemm::clear_blas_backend();
        unsafe {
            // Create kernel
            let mut kernel_status = crate::SPIR_INTERNAL_ERROR;
            let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
            assert_eq!(kernel_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!kernel.is_null());

            // Create SVE result
            let mut sve_status = crate::SPIR_INTERNAL_ERROR;
            let sve = spir_sve_result_new(kernel, 1e-6, -1, -1, 0, &mut sve_status);
            assert_eq!(sve_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!sve.is_null());

            // Create IR basis (Fermionic)
            let mut basis_status = crate::SPIR_INTERNAL_ERROR;
            let basis = spir_basis_new(1, 10.0, 1.0, 1e-6, kernel, sve, -1, &mut basis_status);
            assert_eq!(basis_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!basis.is_null());

            // Create DLR
            let mut dlr_status = crate::SPIR_INTERNAL_ERROR;
            let dlr = spir_dlr_new(basis, &mut dlr_status);
            assert_eq!(dlr_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!dlr.is_null());

            // Get number of poles
            let mut npoles = 0;
            let status = spir_dlr_get_npoles(dlr, &mut npoles);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert!(npoles > 0);
            debug_println!("DLR has {} poles", npoles);

            // Get poles
            let mut poles = vec![0.0; npoles as usize];
            let status = spir_dlr_get_poles(dlr, poles.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            debug_println!("First 3 poles: {:?}", &poles[0..3.min(npoles as usize)]);

            // Test get_u funcs from DLR
            let mut u_status = crate::SPIR_INTERNAL_ERROR;
            let u_funcs = crate::basis::spir_basis_get_u(dlr, &mut u_status);
            assert_eq!(u_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!u_funcs.is_null());
            debug_println!("✓ Got u funcs from DLR");

            // Evaluate u at tau=0.5
            let tau = 0.5;
            let mut u_values = vec![0.0; npoles as usize];
            let status = crate::funcs::spir_funcs_eval(u_funcs, tau, u_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            println!(
                "✓ Evaluated u at τ={}: {:?}",
                tau,
                &u_values[0..3.min(npoles as usize)]
            );

            // Test get_uhat funcs from DLR
            let mut uhat_status = crate::SPIR_INTERNAL_ERROR;
            let uhat_funcs = crate::basis::spir_basis_get_uhat(dlr, &mut uhat_status);
            assert_eq!(uhat_status, SPIR_COMPUTATION_SUCCESS);
            assert!(!uhat_funcs.is_null());
            debug_println!("✓ Got uhat funcs from DLR");

            // Evaluate uhat at Matsubara frequency n=1
            let n_matsu = 1i64;
            let mut uhat_values = vec![num_complex::Complex64::new(0.0, 0.0); npoles as usize];
            let status =
                crate::funcs::spir_funcs_eval_matsu(uhat_funcs, n_matsu, uhat_values.as_mut_ptr());
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            println!(
                "✓ Evaluated uhat at n={}: |uhat|={:?}",
                n_matsu,
                &uhat_values[0..3.min(npoles as usize)]
                    .iter()
                    .map(|v| v.norm())
                    .collect::<Vec<_>>()
            );

            // Cleanup
            crate::funcs::spir_funcs_release(uhat_funcs);
            crate::funcs::spir_funcs_release(u_funcs);
            crate::basis::spir_basis_release(dlr);
            crate::basis::spir_basis_release(basis);
            crate::sve::spir_sve_result_release(sve);
            crate::kernel::spir_kernel_release(kernel);
        }
    }

    // ------------------------------------------------------------------------
    // Status codes of the DLR constructors (#237)
    // ------------------------------------------------------------------------

    use crate::basis::{spir_basis_get_n_default_ws, spir_basis_get_size, spir_basis_release};
    use crate::kernel::spir_kernel_release;
    use crate::{SPIR_INTERNAL_ERROR, SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_FERMIONIC};
    use sparse_ir::basis::FiniteTempBasis;
    use sparse_ir::kernel::{
        CentrosymmKernel, KernelProperties, LogisticKernel, RegularizedBoseKernel,
    };
    use sparse_ir::poly::PiecewiseLegendrePolyVector;
    use sparse_ir::sve::{SVEResult, TworkType, compute_sve};
    use sparse_ir::traits::{Bosonic, Fermionic};
    use std::ptr;

    /// Basis size of the bases built from [`sve_with_too_few_default_poles`].
    const TRUNCATED_BASIS_SIZE: usize = 6;

    /// The SVE result of `kernel`, truncated to `TRUNCATED_BASIS_SIZE`
    /// singular values and left singular functions, with every right singular
    /// function replaced by v_0.
    ///
    /// A basis built from it uses all of its functions, so its default
    /// real-frequency sampling points (the default DLR poles) are the extrema
    /// of v_0 plus two outer points: 3 instead of `TRUNCATED_BASIS_SIZE`. The
    /// `RegularizedBoseKernel` case reported in #114 (lambda = 1e4,
    /// epsilon = 1e-12) no longer loses poles, and no other kernel reachable
    /// through the C API is known to, hence the synthetic SVE result.
    fn sve_with_too_few_default_poles<K>(kernel: K) -> SVEResult
    where
        K: CentrosymmKernel + KernelProperties + Clone + 'static,
    {
        let sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Float64).unwrap();
        assert!(sve.s().len() > TRUNCATED_BASIS_SIZE);
        let v0 = sve.v().get_polys()[0].clone();
        SVEResult::new(
            PiecewiseLegendrePolyVector::new(sve.u().get_polys()[..TRUNCATED_BASIS_SIZE].to_vec())
                .unwrap(),
            sve.s()[..TRUNCATED_BASIS_SIZE].to_vec(),
            PiecewiseLegendrePolyVector::new(vec![v0; TRUNCATED_BASIS_SIZE]).unwrap(),
            sve.epsilon(),
        )
        .unwrap()
    }

    /// `spir_dlr_new` on a basis with fewer default poles than basis functions
    /// must return `SPIR_INVALID_ARGUMENT` and NULL, not panic.
    fn check_dlr_new_insufficient_default_poles(basis: *mut spir_basis, case: &str) {
        // The trigger, observed through the C API.
        let mut size = -1;
        assert_eq!(
            spir_basis_get_size(basis, &mut size),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(size, TRUNCATED_BASIS_SIZE as libc::c_int, "{}", case);
        let mut n_default_ws = -1;
        assert_eq!(
            spir_basis_get_n_default_ws(basis, &mut n_default_ws),
            SPIR_COMPUTATION_SUCCESS
        );
        assert!(
            (0..size).contains(&n_default_ws),
            "{}: expected fewer default poles than basis functions, got {} for size {}",
            case,
            n_default_ws,
            size
        );

        let mut status = SPIR_COMPUTATION_SUCCESS;
        let dlr = spir_dlr_new(basis, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT, "{}", case);
        assert!(dlr.is_null(), "{}", case);

        // Without a status pointer the failure is still reported by NULL.
        assert!(spir_dlr_new(basis, ptr::null_mut()).is_null(), "{}", case);

        // Only the default poles are missing: explicit poles still work.
        let poles: Vec<f64> = (0..size)
            .map(|i| -0.9 + 1.8 * i as f64 / (size - 1) as f64)
            .collect();
        let mut status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new_with_poles(basis, size, poles.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{}", case);
        assert!(!dlr.is_null(), "{}", case);
        let mut npoles = -1;
        assert_eq!(
            spir_dlr_get_npoles(dlr, &mut npoles),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(npoles, size, "{}", case);
        spir_basis_release(dlr);
    }

    #[test]
    fn test_dlr_new_insufficient_default_poles_status() {
        let beta = 10.0;
        let lambda = 10.0; // wmax = 1

        let kernel = LogisticKernel::new(lambda).unwrap();
        let sve = sve_with_too_few_default_poles(kernel);
        let fermionic: FiniteTempBasis<LogisticKernel, Fermionic> =
            FiniteTempBasis::from_sve_result(kernel, beta, sve.clone(), Some(1e-6), None).unwrap();
        let bosonic: FiniteTempBasis<LogisticKernel, Bosonic> =
            FiniteTempBasis::from_sve_result(kernel, beta, sve, Some(1e-6), None).unwrap();

        let kernel = RegularizedBoseKernel::new(lambda).unwrap();
        let sve = sve_with_too_few_default_poles(kernel);
        let reg_bose: FiniteTempBasis<RegularizedBoseKernel, Bosonic> =
            FiniteTempBasis::from_sve_result(kernel, beta, sve, Some(1e-6), None).unwrap();

        let handles = [
            (
                "LogisticKernel, fermionic",
                spir_basis::new_logistic_fermionic(fermionic),
            ),
            (
                "LogisticKernel, bosonic",
                spir_basis::new_logistic_bosonic(bosonic),
            ),
            (
                "RegularizedBoseKernel, bosonic",
                spir_basis::new_regularized_bose_bosonic(reg_bose),
            ),
        ];
        for (case, basis) in handles {
            let basis = Box::into_raw(Box::new(basis));
            check_dlr_new_insufficient_default_poles(basis, case);
            spir_basis_release(basis);
        }
    }

    /// A `RegularizedBoseKernel` basis with fermionic statistics must be
    /// rejected with `SPIR_NOT_SUPPORTED` by both DLR constructors, not panic
    /// in the fermionic regularizer.
    #[test]
    fn test_dlr_rejects_fermionic_regularized_bose_status() {
        let beta = 1.0;
        let lambda = 10.0;
        let ir_basis: FiniteTempBasis<RegularizedBoseKernel, Fermionic> = FiniteTempBasis::new(
            RegularizedBoseKernel::new(lambda).unwrap(),
            beta,
            Some(1e-6),
            None,
        )
        .unwrap();
        // The C basis constructors reject this combination (#241), so build
        // the handle directly to reach the defensive dispatch arms.
        let basis = Box::into_raw(Box::new(spir_basis::new_regularized_bose_fermionic(
            ir_basis,
        )));

        let mut status = SPIR_COMPUTATION_SUCCESS;
        let dlr = spir_dlr_new(basis, &mut status);
        assert_eq!(status, SPIR_NOT_SUPPORTED);
        assert!(dlr.is_null());

        let poles = [-2.0, 0.5, 3.0];
        let mut status = SPIR_COMPUTATION_SUCCESS;
        let dlr = spir_dlr_new_with_poles(
            basis,
            poles.len() as libc::c_int,
            poles.as_ptr(),
            &mut status,
        );
        assert_eq!(status, SPIR_NOT_SUPPORTED);
        assert!(dlr.is_null());

        spir_basis_release(basis);
    }

    /// The argument checks documented for both DLR constructors.
    #[test]
    fn test_dlr_constructors_reject_invalid_arguments() {
        let poles = [-0.5, 0.5];
        let expect_invalid = |dlr: *mut spir_basis, status: StatusCode, case: &str| {
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "{}", case);
            assert!(dlr.is_null(), "{}", case);
        };

        let mut status = SPIR_COMPUTATION_SUCCESS;
        let dlr = spir_dlr_new(ptr::null(), &mut status);
        expect_invalid(dlr, status, "spir_dlr_new with NULL basis");

        let mut status = SPIR_COMPUTATION_SUCCESS;
        let dlr = spir_dlr_new_with_poles(ptr::null(), 2, poles.as_ptr(), &mut status);
        expect_invalid(dlr, status, "spir_dlr_new_with_poles with NULL basis");

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        for statistics in [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC] {
            let mut status = SPIR_INTERNAL_ERROR;
            let basis = spir_basis_new(
                statistics,
                10.0,
                1.0,
                1e-6,
                kernel,
                ptr::null(),
                -1,
                &mut status,
            );
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            let mut status = SPIR_COMPUTATION_SUCCESS;
            let dlr = spir_dlr_new_with_poles(basis, 2, ptr::null(), &mut status);
            expect_invalid(dlr, status, "spir_dlr_new_with_poles with NULL poles");

            for npoles in [0, -1] {
                let mut status = SPIR_COMPUTATION_SUCCESS;
                let dlr = spir_dlr_new_with_poles(basis, npoles, poles.as_ptr(), &mut status);
                expect_invalid(
                    dlr,
                    status,
                    &format!("spir_dlr_new_with_poles with npoles = {}", npoles),
                );
            }

            // A DLR is not an IR basis.
            let mut status = SPIR_INTERNAL_ERROR;
            let dlr = spir_dlr_new(basis, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert!(!dlr.is_null());

            let mut status = SPIR_COMPUTATION_SUCCESS;
            let dlr_of_dlr = spir_dlr_new(dlr, &mut status);
            expect_invalid(dlr_of_dlr, status, "spir_dlr_new on a DLR");

            let mut status = SPIR_COMPUTATION_SUCCESS;
            let dlr_of_dlr = spir_dlr_new_with_poles(dlr, 2, poles.as_ptr(), &mut status);
            expect_invalid(dlr_of_dlr, status, "spir_dlr_new_with_poles on a DLR");

            spir_basis_release(dlr);
            spir_basis_release(basis);
        }
        spir_kernel_release(kernel);
    }

    /// A pole outside [-omega_max, omega_max] of the basis, NaN or an
    /// infinity made the core evaluate v outside its domain and panic
    /// (SPIR_INTERNAL_ERROR). It is an invalid argument now; ±omega_max are
    /// valid poles.
    #[test]
    fn test_dlr_new_with_poles_rejects_poles_outside_the_frequency_domain() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        for statistics in [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC] {
            let mut status = SPIR_INTERNAL_ERROR;
            let basis = spir_basis_new(
                statistics,
                10.0,
                1.0,
                1e-6,
                kernel,
                ptr::null(),
                -1,
                &mut status,
            );
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            for pole in [2.0, -1.5, f64::NAN, f64::INFINITY] {
                let poles = [0.5, pole];
                let mut status = SPIR_COMPUTATION_SUCCESS;
                let dlr = spir_dlr_new_with_poles(basis, 2, poles.as_ptr(), &mut status);
                assert_eq!(status, SPIR_INVALID_ARGUMENT, "pole = {pole}");
                assert!(dlr.is_null());
            }
            let poles = [-1.0, 1.0];
            let mut status = SPIR_INTERNAL_ERROR;
            let dlr = spir_dlr_new_with_poles(basis, 2, poles.as_ptr(), &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            spir_basis_release(dlr);
            spir_basis_release(basis);
        }
        spir_kernel_release(kernel);
    }

    /// A wrong input_dims[target_dim] made the core assert and panic
    /// (SPIR_INTERNAL_ERROR). It is SPIR_INPUT_DIMENSION_MISMATCH now, in both
    /// memory orders and along either axis, and `out` is not written. (The
    /// input buffers hold every element that input_dims describes.)
    #[test]
    fn test_dlr_transforms_reject_a_wrong_target_extent() {
        use crate::{SPIR_INPUT_DIMENSION_MISMATCH, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR};

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut l = 0;
        assert_eq!(spir_basis_get_size(basis, &mut l), SPIR_COMPUTATION_SUCCESS);
        let mut n_poles = 0;
        assert_eq!(
            spir_dlr_get_npoles(dlr, &mut n_poles),
            SPIR_COMPUTATION_SUCCESS
        );

        const SENTINEL: f64 = -12345.0;
        let z_sentinel = Complex64::new(SENTINEL, -SENTINEL);
        let n_max = 3 * (l.max(n_poles) as usize + 1);
        let input_d = vec![1.0; n_max];
        let input_z = vec![Complex64::new(1.0, 0.5); n_max];
        for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
            for target_dim in [0, 1] {
                for (name, n_in) in [("ir2dlr", l), ("dlr2ir", n_poles)] {
                    let mut dims = [3, 3];
                    dims[target_dim as usize] = n_in + 1;
                    let mut out_d = vec![SENTINEL; n_max];
                    let mut out_z = vec![z_sentinel; n_max];
                    let (status_d, status_z) = if name == "ir2dlr" {
                        (
                            spir_ir2dlr_dd(
                                dlr,
                                ptr::null(),
                                order,
                                2,
                                dims.as_ptr(),
                                target_dim,
                                input_d.as_ptr(),
                                out_d.as_mut_ptr(),
                            ),
                            spir_ir2dlr_zz(
                                dlr,
                                ptr::null(),
                                order,
                                2,
                                dims.as_ptr(),
                                target_dim,
                                input_z.as_ptr(),
                                out_z.as_mut_ptr(),
                            ),
                        )
                    } else {
                        (
                            spir_dlr2ir_dd(
                                dlr,
                                ptr::null(),
                                order,
                                2,
                                dims.as_ptr(),
                                target_dim,
                                input_d.as_ptr(),
                                out_d.as_mut_ptr(),
                            ),
                            spir_dlr2ir_zz(
                                dlr,
                                ptr::null(),
                                order,
                                2,
                                dims.as_ptr(),
                                target_dim,
                                input_z.as_ptr(),
                                out_z.as_mut_ptr(),
                            ),
                        )
                    };
                    let case =
                        format!("{name}: order={order}, target_dim={target_dim}, dims={dims:?}");
                    assert_eq!(status_d, SPIR_INPUT_DIMENSION_MISMATCH, "{case} (dd)");
                    assert_eq!(status_z, SPIR_INPUT_DIMENSION_MISMATCH, "{case} (zz)");
                    assert!(
                        out_d.iter().all(|&x| x == SENTINEL),
                        "{case}: out written (dd)"
                    );
                    assert!(
                        out_z.iter().all(|&x| x == z_sentinel),
                        "{case}: out written (zz)"
                    );
                }
            }
        }
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    /// spir_ir2dlr_* maps L IR coefficients to n_poles DLR coefficients, and
    /// n_poles may exceed L, so the output can be too large to address when
    /// the input is not. That is SPIR_INVALID_DIMENSION, checked before
    /// `input` is read (the buffers here are tiny) and `out` written. Before,
    /// the conversion read input_dims worth of elements.
    #[test]
    fn test_ir2dlr_rejects_an_output_too_large_to_address() {
        use crate::{SPIR_INVALID_DIMENSION, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR};

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            10.0,
            1.0,
            1e-6,
            kernel,
            ptr::null(),
            -1,
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let mut l = 0;
        assert_eq!(spir_basis_get_size(basis, &mut l), SPIR_COMPUTATION_SUCCESS);

        // 16 L distinct poles in [-omega_max, omega_max] = [-1, 1].
        let n_poles = 16 * l as usize;
        let poles: Vec<f64> = (0..n_poles)
            .map(|i| -1.0 + 2.0 * i as f64 / (n_poles - 1) as f64)
            .collect();
        let mut status = SPIR_INTERNAL_ERROR;
        let dlr = spir_dlr_new_with_poles(basis, n_poles as i32, poles.as_ptr(), &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        // The input has l * 2^30 * (2^28 / l) <= 2^58 elements (at most 2^62
        // bytes of complex); the output 16 times as many, more than
        // isize::MAX bytes.
        let dims = [l, 1 << 30, (1 << 28) / l];
        const SENTINEL: f64 = -12345.0;
        let z_sentinel = Complex64::new(SENTINEL, -SENTINEL);
        let input_d = [1.0; 4];
        let input_z = [Complex64::new(1.0, 0.5); 4];
        for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
            let mut out_d = [SENTINEL; 4];
            let mut out_z = [z_sentinel; 4];
            let status_d = spir_ir2dlr_dd(
                dlr,
                ptr::null(),
                order,
                3,
                dims.as_ptr(),
                0,
                input_d.as_ptr(),
                out_d.as_mut_ptr(),
            );
            let status_z = spir_ir2dlr_zz(
                dlr,
                ptr::null(),
                order,
                3,
                dims.as_ptr(),
                0,
                input_z.as_ptr(),
                out_z.as_mut_ptr(),
            );
            assert_eq!(status_d, SPIR_INVALID_DIMENSION, "order={order} (dd)");
            assert_eq!(status_z, SPIR_INVALID_DIMENSION, "order={order} (zz)");
            assert!(out_d.iter().all(|&x| x == SENTINEL));
            assert!(out_z.iter().all(|&x| x == z_sentinel));
        }
        spir_basis_release(dlr);
        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }
}
