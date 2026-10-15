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

mod compute;
mod result;
mod strategy;
mod types;
pub(crate) mod utils;

// Re-export public API
pub use compute::{
    BASIS_CUTOFF_FACTOR, compute_sve, compute_sve_dense, compute_sve_for_basis, compute_sve_general,
};
pub use result::SVEResult;
pub use strategy::{CentrosymmSVE, KernelBlock, NonCentrosymmSVE, SVEStrategy, SamplingSVE};
pub use types::TworkType;
pub use utils::mirror_segments_to_full_domain;

#[cfg(test)]
mod tests;
