//! Fitters for least-squares problems with various matrix types
//!
//! This module provides fitters for solving min ||A * coeffs - values||^2
//! where the matrix A and value types can vary:
//!
//! - [`RealMatrixFitter`]: Real matrix A ∈ R^{n×m}
//! - [`ComplexToRealFitter`]: Complex matrix A ∈ C^{n×m}, real coefficients
//! - [`ComplexMatrixFitter`]: Complex matrix A ∈ C^{n×m}, complex coefficients

pub(crate) mod common;
mod complex;
mod complex_to_real;
mod real;

pub(crate) use complex::ComplexMatrixFitter;
pub(crate) use complex_to_real::ComplexToRealFitter;
pub(crate) use real::RealMatrixFitter;

#[cfg(test)]
mod tests;
