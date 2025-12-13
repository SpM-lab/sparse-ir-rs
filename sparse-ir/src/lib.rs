//! # sparse-ir: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

pub mod basis;
pub mod basis_trait; // Common trait for basis representations
pub mod col_piv_qr; // Column-pivoted QR decomposition using nalgebra
pub mod dlr; // Discrete Lehmann Representation utilities
pub mod fitters; // Least-squares fitters (real/complex matrices)
pub mod fpu_check; // FPU state checking for Intel Fortran compatibility
pub mod freq;
pub mod gauss;
pub mod gemm; // Matrix multiplication utilities (Faer backend)
pub mod interpolation1d;
pub mod interpolation2d;
pub mod kernel;
pub mod kernelmatrix;
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
pub mod working_buffer; // Reusable working buffer for in-place operations

// Re-export commonly used types and traits
pub use basis::{BosonicBasis, FermionicBasis, FiniteTempBasis};
pub use basis_trait::Basis;
pub use dlr::{
    DiscreteLehmannRepresentation, bosonic_single_pole, fermionic_single_pole, giwn_single_pole,
    gtau_single_pole,
};
pub use freq::{BosonicFreq, FermionicFreq, MatsubaraFreq};
pub use gauss::{Rule, legendre, legendre_custom, legendre_twofloat};
pub use interpolation1d::Interpolate1D;
pub use interpolation2d::Interpolate2D;
pub use kernel::{
    AbstractKernel, CentrosymmKernel, KernelProperties, LogisticKernel, LogisticSVEHints,
    RegularizedBoseKernel, RegularizedBoseSVEHints, SVEHints, SymmetryType,
    compute_logistic_kernel,
};
pub use kernelmatrix::{
    DiscretizedKernel, InterpolatedKernel, matrix_from_gauss, matrix_from_gauss_noncentrosymmetric,
    matrix_from_gauss_with_segments,
};
pub use matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
pub use numeric::CustomNumeric;
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use polyfourier::{
    BosonicPiecewiseLegendreFT, BosonicPiecewiseLegendreFTVector, FermionicPiecewiseLegendreFT,
    FermionicPiecewiseLegendreFTVector, PiecewiseLegendreFT, PiecewiseLegendreFTVector, PowerModel,
};
pub use fitters::InplaceFitter;
pub use sampling::TauSampling;
pub use sve::{
    CentrosymmSVE, SVDStrategy, SVEResult, SVEStrategy, SamplingSVE, TworkType, compute_sve,
    truncate,
};
pub use traits::{Bosonic, Fermionic, Statistics, StatisticsMarker, StatisticsType};
pub use tsvd::{
    SVDResult, TSVDConfig, TSVDError, svd_decompose, tsvd, tsvd_df64, tsvd_df64_from_f64, tsvd_f64,
};

// Re-export external dependencies for convenience
pub use mdarray::{DTensor, DynRank, Tensor};
pub use xprec::Df64;

// Test utilities (only available in test mode)
#[cfg(test)]
pub mod test_utils;
