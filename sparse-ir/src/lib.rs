//! # sparse-ir: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

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

pub mod basis;
pub mod basis_trait; // Common trait for basis representations
pub(crate) mod col_piv_qr; // Column-pivoted QR decomposition using nalgebra
mod debug; // SPARSEIR_DEBUG switch for debug diagnostics
pub mod dlr; // Discrete Lehmann Representation utilities
mod dlr_id; // Interpolative-decomposition construction of the DLR
pub mod error; // Error type of the public API
pub mod esprit; // Exponential-sum estimation (ESPRIT)
pub mod fitters; // Least-squares fitters (real/complex matrices)
pub mod fpu_check; // FPU state checking for Intel Fortran compatibility
pub mod freq;
pub mod gauss;
pub mod gemm; // Matrix multiplication utilities (Faer backend)
pub(crate) mod interpolation1d;
pub mod kernel;
pub mod kernelmatrix;
pub mod matrix; // Column-major dense containers for internal numerics
pub mod matsubara_sampling; // Sparse sampling in Matsubara frequencies
pub mod numeric;
pub mod poly;
pub mod polyfourier;
pub mod sampling; // Sparse sampling in imaginary time
pub mod special_functions;
pub mod sve;
pub mod taufuncs;
pub mod traits;
pub mod tsvd; // High-precision truncated SVD using nalgebra // Imaginary time τ normalization utilities

// Re-export commonly used types and traits
pub use basis::{BosonicBasis, FermionicBasis, FiniteTempBasis};
pub use basis_trait::Basis;
pub use debug::is_debug_enabled;
pub use dlr::{
    DiscreteLehmannRepresentation, DlrBuilder, IrDlrTransform, bosonic_single_pole,
    fermionic_single_pole, giwn_single_pole, gtau_single_pole,
};
pub use error::{ArrayRole, Error, ErrorKind, Result};
pub use fitters::InplaceFitter;
pub use freq::{BosonicFreq, FermionicFreq, MatsubaraFreq};
pub use gauss::{Rule, legendre, legendre_custom, legendre_twofloat};
pub use kernel::{
    AbstractKernel, CentrosymmKernel, KernelProperties, LogisticKernel, LogisticSVEHints,
    RegularizedBoseKernel, RegularizedBoseSVEHints, SVEHints, SymmetryType,
    compute_logistic_kernel,
};
pub use kernelmatrix::{
    DiscretizedKernel, matrix_from_gauss, matrix_from_gauss_noncentrosymmetric,
    matrix_from_gauss_with_segments,
};
pub use matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
pub use numeric::CustomNumeric;
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use polyfourier::{
    BosonicPiecewiseLegendreFT, BosonicPiecewiseLegendreFTVector, FermionicPiecewiseLegendreFT,
    FermionicPiecewiseLegendreFTVector, PiecewiseLegendreFT, PiecewiseLegendreFTVector, PowerModel,
};
pub use sampling::TauSampling;
pub use sve::{
    CentrosymmSVE, SVDStrategy, SVEResult, SVEStrategy, SamplingSVE, TworkType, compute_sve,
};
pub use traits::{Bosonic, Fermionic, Statistics, StatisticsMarker, StatisticsType};
pub use tsvd::{
    SVDResult, TSVDConfig, svd_decompose, tsvd, tsvd_df64, tsvd_df64_from_f64, tsvd_f64,
};

// Re-export external dependencies for convenience
pub use tenferro_tensor::{
    DynRank, Rank, TensorScalar, TypedTensor, TypedTensorView, TypedTensorViewMut,
};

/// Dense column-major host matrix used by the public API.
pub type Matrix<T> = tenferro_tensor::TypedTensor<T, tenferro_tensor::Rank<2>>;
pub use xprec::Df64;

// Test utilities (only available in test mode)
#[cfg(test)]
pub mod test_utils;
