//! 1D interpolation functionality for SparseIR
//!
//! This module provides the Legendre collocation matrix the SVE uses to turn
//! the singular vectors on the Gauss grid into piecewise Legendre polynomials.

use crate::gauss::{Rule, legendre_vandermonde};
use crate::numeric::CustomNumeric;
use mdarray::DTensor;

/// Create Legendre collocation matrix (inverse of Vandermonde matrix)
///
/// This function creates a matrix C such that V * C ≈ I, where V is the
/// Legendre Vandermonde matrix. This avoids solving linear systems during
/// interpolation by pre-computing the inverse.
///
/// # Arguments
/// * `gauss_rule` - Gauss quadrature rule containing the grid points
///
/// # Returns
/// Collocation matrix C where V * C ≈ I
pub fn legendre_collocation_matrix<T: CustomNumeric>(gauss_rule: &Rule<T>) -> DTensor<T, 2> {
    let n = gauss_rule.x.len();

    // Create Legendre Vandermonde matrix
    let v = legendre_vandermonde(&gauss_rule.x, n - 1);

    // Create normalization factors: range(0.5; length=n) in Julia
    let invnorm: Vec<T> = (0..n)
        .map(|i| T::from_f64_unchecked(0.5 + i as f64))
        .collect();

    // Compute: res = permutedims(V .* w) .* invnorm
    // This is equivalent to: result[i,j] = V[j,i] * w[j] * invnorm[i]
    DTensor::<T, 2>::from_fn([n, n], |idx| {
        let (i, j) = (idx[0], idx[1]);
        v[[j, i]] * gauss_rule.w[j] * invnorm[i]
    })
}

#[cfg(test)]
#[path = "interpolation1d_tests.rs"]
mod tests;
