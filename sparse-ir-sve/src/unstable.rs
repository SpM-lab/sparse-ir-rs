//! Items the other `sparse-ir` crates build on, outside the semver
//! guarantees of this crate
//!
//! These are the machinery of the SVE (its strategies, the discretized
//! kernel matrices, the truncated SVD), the FPU guard and the thread pool.
//! Any release may change or remove them; a program that only computes
//! SVEs does not need them.

/// The strategies [`compute_sve`](crate::compute_sve) decomposes a kernel
/// with, and the dense reference path
pub mod sve {
    pub use crate::sve::compute::compute_sve_dense;
    pub use crate::sve::strategy::{
        CentrosymmSVE, KernelBlock, NonCentrosymmSVE, SVEStrategy, SamplingSVE,
    };
    pub use crate::sve::utils::{half_domain_segments, mirror_segments_to_full_domain};
}

/// Kernel matrices on the Gauss points of the SVE
pub mod kernelmatrix {
    pub use crate::kernelmatrix::*;
}

/// Truncated SVD in `f64` and double-double, by nalgebra
pub mod tsvd {
    pub use crate::tsvd::*;
}

/// FPU state checking for Intel Fortran compatibility
pub mod fpu_check {
    pub use crate::fpu_check::*;
}

/// Parallel execution on the rayon pool
pub mod parallel {
    pub use crate::parallel::*;
}

pub use crate::checks::*;
