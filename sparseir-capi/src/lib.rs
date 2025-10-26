//! C API for SparseIR Rust implementation
//!
//! This crate provides a C-compatible interface to the SparseIR library,
//! enabling usage from languages like Julia, Python, Fortran, and C++.

// C API requires unsafe operations with raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[macro_use]
mod macros;

mod basis;
mod dlr;
mod funcs;
mod gemm;
mod kernel;
mod sampling;
mod sve;
mod types;
mod utils;

pub use basis::*;
pub use dlr::*;
pub use funcs::*;
pub use gemm::*;
pub use kernel::*;
pub use sampling::*;
pub use sve::*;
pub use types::*;

/// Complex number type for C API (compatible with C's double complex)
/// 
/// This type is compatible with C99's `double complex` and C++'s `std::complex<double>`.
/// Layout: `{double re; double im;}` with standard alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

/// Error codes for C API (compatible with libsparseir)
pub type StatusCode = libc::c_int;

// Status codes matching libsparseir
pub const SPIR_COMPUTATION_SUCCESS: StatusCode = 0;
pub const SPIR_GET_IMPL_FAILED: StatusCode = -1;
pub const SPIR_INVALID_DIMENSION: StatusCode = -2;
pub const SPIR_INPUT_DIMENSION_MISMATCH: StatusCode = -3;
pub const SPIR_OUTPUT_DIMENSION_MISMATCH: StatusCode = -4;
pub const SPIR_NOT_SUPPORTED: StatusCode = -5;
pub const SPIR_INVALID_ARGUMENT: StatusCode = -6;
pub const SPIR_INTERNAL_ERROR: StatusCode = -7;

// Order type constants (matching libsparseir)
pub const SPIR_ORDER_ROW_MAJOR: libc::c_int = 0;
pub const SPIR_ORDER_COLUMN_MAJOR: libc::c_int = 1;

// Statistics type constants (matching libsparseir)
pub const SPIR_STATISTICS_BOSONIC: libc::c_int = 0;
pub const SPIR_STATISTICS_FERMIONIC: libc::c_int = 1;
