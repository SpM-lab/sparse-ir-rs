//! # sparse-ir-basis
//!
//! The intermediate representation (IR) basis: the kernels and their physical
//! properties ([`KernelProperties`]), the Fourier transforms of the singular
//! functions, and [`FiniteTempBasis`]. Also the DLR built from an IR basis
//! ([`DlrFromIr`]) and the IR <-> DLR transform. The singular value expansion
//! and the piecewise Legendre polynomials come from `sparse-ir-sve`,
//! re-exported here under the same paths.
//!
//! Most users depend on the `sparse-ir` crate, which re-exports this crate.

#[macro_use]
extern crate sparse_ir_core;

// Modules and items of the core and the DLR, under the paths the code uses
#[allow(unused_imports)]
use sparse_ir_core::{
    ArrayRole, Matrix, MatsubaraFreq, StatisticsType, TypedTensor, basis_trait, error, fitters,
    fpu_check, freq, matrix, matsubara_sampling, parallel, sampling, taufuncs, traits,
};
#[allow(unused_imports)]
use sparse_ir_dlr::{dlr, fermionic_single_pole, giwn_single_pole, gtau_single_pole};
// Names the tests of this crate use at the crate root
#[cfg(test)]
#[allow(unused_imports)]
use sparse_ir_core::{
    Basis, Bosonic, Error, Fermionic, MatsubaraSampling, MatsubaraSamplingPositiveOnly, Statistics,
    TauSampling,
};
#[cfg(test)]
#[allow(unused_imports)]
use sparse_ir_dlr::{
    DiscreteLehmannRepresentation, DlrBuilder, IrDlrTransform, bosonic_single_pole,
};

pub mod basis;
pub mod ir_dlr; // DLR built from an IR basis, and the IR <-> DLR transform
pub mod kernel;
pub(crate) mod pair_select; // Matsubara sampling-point selection by pairs of rows
pub mod polyfourier;
pub mod special_functions;

// The SVE and the numerics it is built on, under the paths of this crate
pub use sparse_ir_sve::unstable::{kernelmatrix, tsvd};
pub use sparse_ir_sve::{gauss, numeric, poly};

/// Singular value expansion (SVE) of the kernels, from `sparse-ir-sve`
pub mod sve {
    pub use sparse_ir_sve::sve::*;
    pub use sparse_ir_sve::unstable::sve::*;
}

pub use basis::{BosonicBasis, FermionicBasis, FiniteTempBasis, MatsubaraPointMethod};
pub use gauss::{Rule, legendre, legendre_custom, legendre_twofloat};
pub use ir_dlr::{DlrFromIr, IrBasis};
pub use kernel::{
    AbstractKernel, BasisKernel, CentrosymmKernel, ExpandableKernel, KernelProperties,
    LogisticKernel, LogisticKernelWindowed, LogisticSVEHints, LogisticWindowedSVEHints,
    ReducedKernel, RegularizedBoseKernel, RegularizedBoseSVEHints, SVEHints, SVEKernel,
    SymmetryType, compute_logistic_kernel,
};
pub use kernelmatrix::{
    DiscretizedKernel, matrix_from_gauss, matrix_from_gauss_noncentrosymmetric,
    matrix_from_gauss_with_segments,
};
pub use numeric::CustomNumeric;
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use polyfourier::{
    BosonicPiecewiseLegendreFT, BosonicPiecewiseLegendreFTVector, FermionicPiecewiseLegendreFT,
    FermionicPiecewiseLegendreFTVector, PiecewiseLegendreFT, PiecewiseLegendreFTVector, PowerModel,
};
pub use sparse_ir_sve::Df64;
pub use sve::{
    CentrosymmSVE, KernelBlock, SVEResult, SVEStrategy, SamplingSVE, TworkType, compute_sve,
};
pub use tsvd::{SVDResult, TSVDConfig, svd_decompose, tsvd_df64, tsvd_df64_from_f64, tsvd_f64};

// Test utilities (only available in test mode)
#[cfg(test)]
pub mod test_utils;

// Tests of the core samplings, the Basis trait and the DLR on an IR basis
#[cfg(test)]
mod basis_trait_tests;
#[cfg(test)]
mod dlr_independent_tests;
#[cfg(test)]
mod dlr_tests;
#[cfg(test)]
mod matsubara_sampling_tests;
#[cfg(test)]
mod tau_sampling_tests;
// Tests of the SVE of the kernels of this crate
#[cfg(test)]
mod sve_tests;
