//! Singular Value Expansion (SVE) module
//!
//! This module provides functionality for computing the singular value expansion
//! of integral kernels, which is the core of the SparseIR algorithm.
//!
//! # Design Principles
//!
//! 1. **Separation of Concerns**: Clear separation between general SVE processing
//!    and symmetry-specific logic
//! 2. **Type-driven Precision**: Automatic selection of working precision based on
//!    required accuracy
//! 3. **Symmetry Exploitation**: Efficient computation for centrosymmetric kernels
//!    via even/odd decomposition
//! 4. **Composability**: Modular design allowing easy extension and testing

mod compute;
mod result;
mod strategy;
mod types;
pub mod utils; // Public for testing

// Re-export public API
pub use compute::{compute_sve, compute_sve_general, truncate};
pub use result::SVEResult;
pub use strategy::{CentrosymmSVE, NonCentrosymmSVE, SVEStrategy, SamplingSVE};
pub use types::{SVDStrategy, TworkType, safe_epsilon};

#[cfg(test)]
mod tests;
