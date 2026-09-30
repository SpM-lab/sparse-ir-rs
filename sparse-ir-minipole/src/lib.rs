//! # sparse-ir-minipole
//!
//! Minimal pole representations (Zhang and Gull, PRB 110, 035154) from DLR
//! coefficients or Matsubara data, and the ESPRIT algorithm they rest on.
//! Most users depend on the `sparse-ir` crate.

// Modules of the core and the DLR, under the paths the code uses
#[allow(unused_imports)]
use sparse_ir_core::{MatsubaraSampling, basis_trait, error, fitters, freq, traits};
#[allow(unused_imports)]
use sparse_ir_dlr::dlr;

pub mod esprit; // Exponential-sum estimation (ESPRIT)
pub mod minipole; // Minimal pole representation via ESPRIT
