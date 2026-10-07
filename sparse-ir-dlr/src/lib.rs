//! # sparse-ir-dlr
//!
//! The discrete Lehmann representation (DLR), built independently of an IR
//! basis by an interpolative decomposition of the logistic kernel (Kaye,
//! Chen, Parcollet, PRB 105, 235115), with its interpolation nodes and the
//! single-pole Green's functions.
//!
//! A DLR built from an IR basis (`DlrFromIr`) and the IR <-> DLR transform
//! live in `sparse-ir-basis`. Most users depend on the `sparse-ir` crate.

// Modules of the core, under the paths the code uses
#[allow(unused_imports)]
use sparse_ir_core::{
    Matrix, MatsubaraSampling, TauSampling, basis_trait, error, fitters, freq, gemm, matrix,
    taufuncs, traits,
};

pub mod dlr; // Discrete Lehmann Representation utilities
mod dlr_id; // Interpolative-decomposition construction of the DLR

pub use dlr::{
    DiscreteLehmannRepresentation, DlrBuilder, IrDlrTransform, bosonic_single_pole,
    fermionic_single_pole, giwn_single_pole, gtau_single_pole,
};
