//! Singular Value Expansion (SVE) module
//!
//! This module provides functionality for computing the singular value expansion
//! of integral kernels.
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

pub(crate) mod compute;
mod residual;
mod result;
pub(crate) mod strategy;
mod types;
pub(crate) mod utils;

// Re-export public API
pub use compute::{BASIS_CUTOFF_FACTOR, compute_sve, compute_sve_for_basis, compute_sve_general};
pub use residual::sve_residual;
pub use result::SVEResult;
pub use types::TworkType;

#[cfg(test)]
mod tests;
