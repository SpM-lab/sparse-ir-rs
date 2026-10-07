#![doc = include_str!("../README.md")]

//! ## Where to go next
//!
//! - The [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/) walks
//!   through sparse sampling, IR/DLR transforms, MiniPole and applied
//!   calculations, with runnable examples.
//! - [`FiniteTempBasis`], [`TauSampling`] and [`MatsubaraSampling`] are the
//!   entry points for the IR basis; [`DiscreteLehmannRepresentation`] and
//!   [`DlrBuilder`] for the DLR; [`minipole`] and [`esprit`] for pole
//!   reconstruction.

pub use sparse_ir_core::debug_warn;
pub use sparse_ir_core::{
    ArrayRole, Basis, Bosonic, BosonicFreq, DynRank, Error, ErrorKind, Fermionic, FermionicFreq,
    InplaceFitter, Matrix, MatsubaraFreq, MatsubaraSampling, MatsubaraSamplingPositiveOnly, Rank,
    Result, Statistics, StatisticsMarker, StatisticsType, TauSampling, TensorScalar, TypedTensor,
    TypedTensorView, TypedTensorViewMut, is_debug_enabled,
};
pub use sparse_ir_core::{
    basis_trait, error, fitters, fpu_check, freq, gemm, matrix, matsubara_sampling, sampling,
    taufuncs, traits,
};

pub use sparse_ir_dlr::dlr;
pub use sparse_ir_dlr::{
    DiscreteLehmannRepresentation, DlrBuilder, IrDlrTransform, bosonic_single_pole,
    fermionic_single_pole, giwn_single_pole, gtau_single_pole,
};

pub use sparse_ir_minipole::{esprit, minipole};

pub use sparse_ir_basis::{
    AbstractKernel, BosonicBasis, BosonicPiecewiseLegendreFT, BosonicPiecewiseLegendreFTVector,
    CentrosymmKernel, CentrosymmSVE, CustomNumeric, Df64, DiscretizedKernel, DlrFromIr,
    FermionicBasis, FermionicPiecewiseLegendreFT, FermionicPiecewiseLegendreFTVector,
    FiniteTempBasis, IrBasis, KernelProperties, LogisticKernel, LogisticSVEHints,
    PiecewiseLegendreFT, PiecewiseLegendreFTVector, PiecewiseLegendrePoly,
    PiecewiseLegendrePolyVector, PowerModel, RegularizedBoseKernel, RegularizedBoseSVEHints, Rule,
    SVDResult, SVDStrategy, SVEHints, SVEResult, SVEStrategy, SamplingSVE, SymmetryType,
    TSVDConfig, TworkType, compute_logistic_kernel, compute_sve, legendre, legendre_custom,
    legendre_twofloat, matrix_from_gauss, matrix_from_gauss_noncentrosymmetric,
    matrix_from_gauss_with_segments, svd_decompose, tsvd_df64, tsvd_df64_from_f64, tsvd_f64,
};
pub use sparse_ir_basis::{
    basis, gauss, ir_dlr, kernel, kernelmatrix, numeric, poly, polyfourier, special_functions, sve,
    tsvd,
};
