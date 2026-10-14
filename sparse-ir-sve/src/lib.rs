//! # sparse-ir-sve
//!
//! The singular value expansion (SVE) of integral kernels K(x, y), computed
//! in extended precision, together with the numerics it is built on: dense
//! column-major containers, the error type, the FPU guard, and the parallel
//! execution on the rayon pool.
//!
//! The crate knows nothing about statistics, temperatures or Matsubara
//! frequencies; the `sparse-ir` crates build the IR basis on top of it.

pub mod error; // Error type of the public API
pub mod fpu_check; // FPU state checking for Intel Fortran compatibility
pub mod matrix; // Column-major dense containers for internal numerics
#[doc(hidden)]
pub mod parallel; // Parallel execution on the rayon pool (internal)

pub use error::{Error, Result};
pub use matrix::{Mat, Mat3};
