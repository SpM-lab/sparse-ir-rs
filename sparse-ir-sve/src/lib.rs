//! # sparse-ir-sve
//!
//! The singular value expansion (SVE) of integral kernels K(x, y), computed
//! in extended precision with an ACA-accelerated truncated SVD, and the
//! numerics it is built on: Gauss–Legendre quadrature, piecewise Legendre
//! polynomials, dense column-major containers, the error type, the FPU guard,
//! and the parallel execution on the rayon pool.
//!
//! A kernel implements [`AbstractKernel`] (its values) and [`SVEKernel`] (its
//! domain and the discretization hints of its SVE), and either
//! [`ReducedKernel`] (a centrosymmetric kernel, through its even and odd
//! reduced kernels) or [`SVEStrategyKernel`] (any other kernel, no methods);
//! [`compute_sve`] then returns its singular values and singular functions
//! ([`SVEResult`]).
//!
//! In `f64`, [`compute_sve`] decomposes the discretized kernel by a dense
//! column-pivoted QR; in double-double (`Df64`), where that is costly, by
//! adaptive cross approximation (ACA), which reads only a few rows and
//! columns of it. [`unstable::sve::compute_sve_dense`] takes the dense path in both,
//! to check the result on a new kernel.
//!
//! The crate is mathematics only: it defines no concrete kernel. The
//! `sparse-ir` crates define the kernels of the IR basis and build the basis
//! on top of it.
//!
//! # Example
//!
//! The SVE of K(x, y) = exp(xy) on [-1, 1] × [-1, 1]. The kernel is
//! centrosymmetric, K(-x, -y) = K(x, y), so the SVE is computed in even and
//! odd blocks: the even and odd reduced kernels K(x, y) ± K(x, -y) are
//! 2 cosh(xy) and 2 sinh(xy). [`SegmentHints`] discretizes it on segments of
//! equal length.
//!
//! ```
//! use sparse_ir_sve::{
//!     AbstractKernel, CustomNumeric, ReducedKernel, SVEKernel, SegmentHints, SymmetryType,
//!     TworkType, compute_sve, sve_residual,
//! };
//! use std::fmt::Debug;
//!
//! #[derive(Debug, Clone, Copy)]
//! struct ExpKernel;
//!
//! impl AbstractKernel for ExpKernel {
//!     fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
//!         (x * y).exp()
//!     }
//! }
//!
//! impl ReducedKernel for ExpKernel {
//!     fn compute_reduced<T: CustomNumeric + Copy + Debug>(
//!         &self,
//!         x: T,
//!         y: T,
//!         symmetry: SymmetryType,
//!     ) -> T {
//!         let two = T::from_f64_unchecked(2.0);
//!         match symmetry {
//!             SymmetryType::Even => two * (x * y).cosh(),
//!             SymmetryType::Odd => two * (x * y).sinh(),
//!         }
//!     }
//! }
//!
//! impl SVEKernel for ExpKernel {
//!     type SVEHintsType<T>
//!         = SegmentHints
//!     where
//!         T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
//!
//!     fn xmax(&self) -> f64 {
//!         1.0
//!     }
//!
//!     fn ymax(&self) -> f64 {
//!         1.0
//!     }
//!
//!     fn sve_hints<T>(&self, _epsilon: f64) -> SegmentHints
//!     where
//!         T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
//!     {
//!         // Four segments on each axis, 16 Gauss points on each segment
//!         SegmentHints::uniform(self, 4, 4, 16).unwrap()
//!     }
//! }
//!
//! let sve = compute_sve(ExpKernel, Some(1e-10), None, None, TworkType::Auto)?;
//! let s = sve.s();
//! assert!(s.windows(2).all(|w| w[0] >= w[1]));
//!
//! // K(x, y) = Σ_l s_l u_l(x) v_l(y), to the requested accuracy
//! let (x, y) = (0.3, -0.7);
//! let k: f64 = (0..s.len())
//!     .map(|l| s[l] * sve.u()[l].evaluate(x) * sve.v()[l].evaluate(y))
//!     .sum();
//! assert!((k - f64::exp(x * y)).abs() < 1e-8);
//!
//! // The segments resolve the kernel: the residual is far below 1e-10
//! assert!(sve_residual(&ExpKernel, &sve) < 1e-12);
//! # Ok::<(), sparse_ir_sve::Error>(())
//! ```

pub(crate) mod aca; // ACA-accelerated truncated SVD for the SVE kernel blocks
mod checks; // Parameter checks, re-exported from `unstable`
pub(crate) mod col_piv_qr; // Column-pivoted QR decomposition using nalgebra
pub(crate) mod cost; // Work estimates for the threading decisions
pub(crate) mod ddgemm; // Double-double matrix products
pub mod error; // Error type of the public API
mod fpu_check; // FPU state checking for Intel Fortran compatibility
pub mod gauss;
pub mod hints;
pub(crate) mod interpolation1d;
pub mod kernel;
mod kernelmatrix; // Kernel matrices on Gauss points
pub mod matrix; // Column-major dense containers for internal numerics
pub mod numeric;
mod parallel; // Parallel execution on the rayon pool
pub mod poly;
pub mod sve;
mod tsvd; // High-precision truncated SVD using nalgebra
pub mod unstable; // Items for the other sparse-ir crates, outside semver

// Kernels for the tests of this crate
#[cfg(test)]
mod test_kernels;

pub use error::{Error, Result};
pub use gauss::{Rule, legendre, legendre_custom, legendre_twofloat};
pub use hints::SegmentHints;
pub use kernel::{
    AbstractKernel, ReducedKernel, SVEHints, SVEKernel, SVEStrategyKernel, SymmetryType,
};
pub use matrix::{Mat, Mat3};
pub use numeric::CustomNumeric;
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use sve::{SVEResult, TworkType, compute_sve, sve_residual};
pub use xprec::Df64;
