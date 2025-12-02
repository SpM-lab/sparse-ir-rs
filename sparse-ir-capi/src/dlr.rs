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
use crate::types::{BasisType, spir_basis};
use crate::utils::{MemoryOrder, copy_tensor_to_c_array, read_tensor_nd};
use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, StatusCode};
use sparse_ir::dlr::DiscreteLehmannRepresentation;

// ============================================================================
// Creation Functions
// ============================================================================

/// Creates a new DLR from an IR basis with default poles
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created DLR basis object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `b` is a valid IR basis pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_dlr_new(
    b: *const spir_basis,
    status: *mut StatusCode,
) -> *mut spir_basis {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if b.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = unsafe { &*b };

        // Create DLR based on basis type
        let dlr_type = match basis_ref.inner() {
            BasisType::LogisticFermionic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::new(ir_basis.as_ref());
                BasisType::DLRFermionic(Arc::new(dlr))
            }
            BasisType::LogisticBosonic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::new(ir_basis.as_ref());
                BasisType::DLRBosonic(Arc::new(dlr))
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::new(ir_basis.as_ref());
                BasisType::DLRFermionic(Arc::new(dlr))
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::new(ir_basis.as_ref());
                BasisType::DLRBosonic(Arc::new(dlr))
            }
            _ => {
                // Already a DLR, return error
                return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
            }
        };

        let dlr_basis = match dlr_type {
            BasisType::DLRFermionic(arc_dlr) => spir_basis::new_dlr_fermionic(arc_dlr),
            BasisType::DLRBosonic(arc_dlr) => spir_basis::new_dlr_bosonic(arc_dlr),
            _ => unreachable!(), // We know it's one of the DLR types
        };

        (Box::into_raw(Box::new(dlr_basis)), SPIR_COMPUTATION_SUCCESS)
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

/// Creates a new DLR with custom poles
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `npoles` - Number of poles to use
/// * `poles` - Array of pole locations on the real-frequency axis
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created DLR basis object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `b` is valid and `poles` has `npoles` elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_dlr_new_with_poles(
    b: *const spir_basis,
    npoles: libc::c_int,
    poles: *const f64,
    status: *mut StatusCode,
) -> *mut spir_basis {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if b.is_null() || poles.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if npoles <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = unsafe { &*b };
        let poles_slice = unsafe { std::slice::from_raw_parts(poles, npoles as usize) };
        let pole_vec: Vec<f64> = poles_slice.to_vec();

        // Create DLR based on basis type
        let dlr_type = match basis_ref.inner() {
            BasisType::LogisticFermionic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::with_poles(ir_basis.as_ref(), pole_vec);
                BasisType::DLRFermionic(Arc::new(dlr))
            }
            BasisType::LogisticBosonic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::with_poles(ir_basis.as_ref(), pole_vec);
                BasisType::DLRBosonic(Arc::new(dlr))
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::with_poles(ir_basis.as_ref(), pole_vec);
                BasisType::DLRFermionic(Arc::new(dlr))
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                let dlr = DiscreteLehmannRepresentation::with_poles(ir_basis.as_ref(), pole_vec);
                BasisType::DLRBosonic(Arc::new(dlr))
            }
            _ => {
                // Already a DLR or invalid type
                return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
            }
        };

        let dlr_basis = match dlr_type {
            BasisType::DLRFermionic(arc_dlr) => spir_basis::new_dlr_fermionic(arc_dlr),
            BasisType::DLRBosonic(arc_dlr) => spir_basis::new_dlr_bosonic(arc_dlr),
            _ => unreachable!(), // We know it's one of the DLR types
        };

        (Box::into_raw(Box::new(dlr_basis)), SPIR_COMPUTATION_SUCCESS)
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
pub unsafe extern "C" fn spir_dlr_get_npoles(
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
            BasisType::DLRFermionic(dlr) => dlr.poles.len(),
            BasisType::DLRBosonic(dlr) => dlr.poles.len(),
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
pub unsafe extern "C" fn spir_dlr_get_poles(dlr: *const spir_basis, poles: *mut f64) -> StatusCode {
    if dlr.is_null() || poles.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dlr_ref = unsafe { &*dlr };

        // Get poles based on DLR type
        let pole_vec = match dlr_ref.inner() {
            BasisType::DLRFermionic(dlr) => &dlr.poles,
            BasisType::DLRBosonic(dlr) => &dlr.poles,
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

/// Convert IR coefficients to DLR (real-valued)
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order
/// * `ndim` - Number of dimensions
/// * `input_dims` - Array of input dimensions
/// * `target_dim` - Dimension to transform
/// * `input` - IR coefficients
/// * `out` - Output DLR coefficients
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_ir2dlr_dd(
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
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        let input_tensor = unsafe { read_tensor_nd(input, &orig_dims, mem_order) };

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

        // Copy result to output with correct memory order
        unsafe {
            copy_tensor_to_c_array(result_tensor, out, mem_order);
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
/// * `input_dims` - Array of input dimensions
/// * `target_dim` - Dimension to transform
/// * `input` - Complex IR coefficients
/// * `out` - Output complex DLR coefficients
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_ir2dlr_zz(
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
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        let input_tensor = unsafe { read_tensor_nd(input, &orig_dims, mem_order) };

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

        // Copy result to output with correct memory order
        unsafe {
            copy_tensor_to_c_array(result_tensor, out, mem_order);
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
/// * `input_dims` - Array of input dimensions
/// * `target_dim` - Dimension to transform
/// * `input` - DLR coefficients
/// * `out` - Output IR coefficients
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_dlr2ir_dd(
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
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        let input_tensor = unsafe { read_tensor_nd(input, &orig_dims, mem_order) };

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

        // Copy result to output with correct memory order
        unsafe {
            copy_tensor_to_c_array(result_tensor, out, mem_order);
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
/// * `input_dims` - Array of input dimensions
/// * `target_dim` - Dimension to transform
/// * `input` - Complex DLR coefficients
/// * `out` - Output complex IR coefficients
///
/// # Returns
/// Status code
///
/// # Safety
/// Caller must ensure pointers are valid and arrays have correct sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spir_dlr2ir_zz(
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
        let dims_slice = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        // Read input tensor using the unified helper function
        // read_tensor_nd handles memory order internally and returns tensor with orig_dims shape
        let input_tensor = unsafe { read_tensor_nd(input, &orig_dims, mem_order) };

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

        // Copy result to output with correct memory order
        unsafe {
            copy_tensor_to_c_array(result_tensor, out, mem_order);
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
}
