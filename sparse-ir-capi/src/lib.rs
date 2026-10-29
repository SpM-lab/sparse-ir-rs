//! C API for SparseIR Rust implementation
//!
//! This crate provides a C-compatible interface to the SparseIR library,
//! enabling usage from languages like Julia, Python, Fortran, and C++.

// C API requires unsafe operations with raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[macro_use]
mod macros;

mod basis;
mod cpu;
mod dlr;
mod funcs;
mod gemm;
mod kernel;
mod minipole;
mod sampling;
mod status;
mod sve;
mod types;
mod utils;

pub use basis::*;
pub use cpu::*;
pub use dlr::*;
pub use funcs::*;
pub use gemm::*;
pub use kernel::*;
pub use minipole::*;
pub use sampling::*;
pub use sve::*;
pub use types::*;
pub use utils::*;

/// `catch_unwind` of the body of a C API call, which runs every parallel
/// region it reaches on the calling thread
///
/// The C library is built without the `parallel` feature of sparse-ir, so
/// the parallel regions of sparse-ir (the SVE) use no rayon pool
/// (fork-after-pool deadlocks with Python `multiprocessing`,
/// oversubscription inside OpenMP regions). Should the feature be enabled
/// anyway (Cargo unifies the features of a build, so another crate of the
/// same build can enable it), this keeps them on the calling thread. It does
/// not reach the SVDs that the fitters and MiniPole take through tenferro's
/// `CpuBackend`, which builds a pool of its own (#344).
#[doc(hidden)] // public only for the expansion of `impl_opaque_type_common`
pub fn ffi_call<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> std::thread::Result<R> {
    sparse_ir::parallel::sequential(|| std::panic::catch_unwind(f))
}

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

// Twork type constants (matching libsparseir)
pub const SPIR_TWORK_FLOAT64: libc::c_int = 0;
pub const SPIR_TWORK_FLOAT64X2: libc::c_int = 1;
pub const SPIR_TWORK_AUTO: libc::c_int = -1;

// SVD strategy constants (matching libsparseir)
// Note: Currently not used in Rust implementation, but included for API compatibility

#[cfg(test)]
mod tests {
    /// A C API call runs its parallel regions on the calling thread, also if
    /// the `parallel` feature is enabled (as in a workspace build)
    #[test]
    fn ffi_call_is_sequential() {
        let caller = std::thread::current().id();
        let threads = super::ffi_call(|| {
            sparse_ir::parallel::map_on_pool(64, usize::MAX, |_| std::thread::current().id())
        })
        .unwrap();
        assert!(threads.iter().all(|id| *id == caller));
        assert_eq!(super::ffi_call(sparse_ir::parallel::n_threads).unwrap(), 1);
    }
}
