//! # sparse-ir-minipole
//!
//! Minimal pole representations from DLR coefficients or Matsubara data, and
//! the ESPRIT algorithm they rest on: L. Zhang and E. Gull, Phys. Rev. B 110,
//! 035154 (2024); L. Zhang, Y. Yu and E. Gull, Phys. Rev. B 110, 235131
//! (2024). Most users depend on the `sparse-ir` crate.
//!
//! The implementation is a port of the reference implementation
//! Green-Phys/MiniPole (MIT License, Copyright (c) 2024 lzphy; see
//! `LICENSE-THIRD-PARTY`), checked against its outputs.

// Modules of the core and the DLR, under the paths the code uses
#[allow(unused_imports)]
use sparse_ir_core::{basis_trait, error, fitters, traits};
#[allow(unused_imports)]
use sparse_ir_dlr::dlr;

pub mod esprit; // Matrix ESPRIT (port of esprit.py)
mod linalg; // Dense linear algebra in NumPy conventions
pub mod minipole; // Minimal pole method (port of mini_pole*.py)
