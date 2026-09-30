//! # sparse-ir-core
//!
//! The representation-independent layer of sparse-ir: statistics and
//! Matsubara frequencies, the error type, GEMM dispatch, the least-squares
//! fitters, the [`Basis`] trait, and sparse sampling in imaginary time and
//! Matsubara frequency for any [`Basis`].
//!
//! Most users depend on the `sparse-ir` crate, which re-exports this crate
//! together with the IR basis, the DLR and MiniPole.

/// Print a warning to stderr only when the `SPARSEIR_DEBUG` environment
/// variable enables debug diagnostics (see [`is_debug_enabled`] for the
/// accepted values).
///
/// Library code must not write to stderr unconditionally; these are advisory
/// diagnostics (e.g. ill-conditioned sampling) for debugging, not error reports.
#[macro_export]
macro_rules! debug_warn {
    ($($arg:tt)*) => {
        if $crate::is_debug_enabled() {
            eprintln!("[SPARSEIR WARN] {}", format!($($arg)*));
        }
    };
}

pub mod basis_trait; // Common trait for basis representations
mod debug; // SPARSEIR_DEBUG switch for debug diagnostics
pub mod error; // Error type of the public API
pub mod fitters; // Least-squares fitters (real/complex matrices)
pub mod fpu_check; // FPU state checking for Intel Fortran compatibility
pub mod freq;
pub mod gemm; // Matrix multiplication utilities (Faer backend)
pub mod matrix; // Column-major dense containers for internal numerics
pub mod matsubara_sampling; // Sparse sampling in Matsubara frequencies
pub mod sampling; // Sparse sampling in imaginary time
pub mod taufuncs; // Imaginary time τ normalization utilities
pub mod traits;

pub use basis_trait::Basis;
pub use debug::is_debug_enabled;
pub use error::{ArrayRole, Error, ErrorKind, Result};
pub use fitters::InplaceFitter;
pub use freq::{BosonicFreq, FermionicFreq, MatsubaraFreq};
pub use matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
pub use sampling::TauSampling;
pub use traits::{Bosonic, Fermionic, Statistics, StatisticsMarker, StatisticsType};

// Re-export external dependencies for convenience
pub use tenferro_tensor::{
    DynRank, Rank, TensorScalar, TypedTensor, TypedTensorView, TypedTensorViewMut,
};

/// Dense column-major host matrix used by the public API.
pub type Matrix<T> = tenferro_tensor::TypedTensor<T, tenferro_tensor::Rank<2>>;
