//! 1D interpolation functionality for SparseIR
//!
//! This module provides efficient 1D interpolation using Legendre polynomials
//! with pre-computed collocation matrices.

use crate::gauss::{Rule, legendre_vandermonde};
use crate::numeric::CustomNumeric;
use mdarray::DTensor;
use std::fmt::Debug;

/// 1D interpolator with pre-computed Legendre polynomial coefficients
///
/// This struct stores pre-computed interpolation coefficients and domain information
/// for efficient 1D interpolation using Legendre polynomials.
#[derive(Debug, Clone)]
pub struct Interpolate1D<T> {
    /// Left boundary of interpolation domain
    pub x_min: T,
    /// Right boundary of interpolation domain  
    pub x_max: T,
    /// Pre-computed Legendre polynomial coefficients
    pub coeffs: Vec<T>,
    /// Gauss quadrature rule used for interpolation
    pub gauss_rule: Rule<T>,
}

impl<T: CustomNumeric + Debug + Clone + 'static> Interpolate1D<T> {
    /// Create a new 1D interpolator from function values and Gauss rule
    ///
    /// # Arguments
    /// * `values` - Function values at Gauss points
    /// * `gauss_rule` - Gauss quadrature rule
    ///
    /// # Returns
    /// New Interpolate1D instance with pre-computed coefficients
    pub fn new(values: &[T], gauss_rule: &Rule<T>) -> Self {
        assert!(
            !values.is_empty(),
            "Cannot create interpolation from empty values"
        );
        assert_eq!(
            values.len(),
            gauss_rule.x.len(),
            "Values length must match Gauss points"
        );

        let coeffs = interpolate_1d_legendre(values, gauss_rule);

        Interpolate1D {
            x_min: gauss_rule.a,
            x_max: gauss_rule.b,
            coeffs,
            gauss_rule: gauss_rule.clone(),
        }
    }

    /// Evaluate the interpolated function at a given point
    ///
    /// # Arguments
    /// * `x` - Point to evaluate at (must be within [x_min, x_max])
    ///
    /// # Returns
    /// Interpolated value at x
    ///
    /// # Panics
    /// Panics if x is outside the interpolation domain [x_min, x_max]
    pub fn evaluate(&self, x: T) -> T {
        assert!(
            x >= self.x_min && x <= self.x_max,
            "Point x={:?} is outside interpolation domain [{:?}, {:?}]",
            x,
            self.x_min,
            self.x_max
        );

        evaluate_interpolated_polynomial(x, &self.coeffs)
    }

    /// Get the domain boundaries
    pub fn domain(&self) -> (T, T) {
        (self.x_min, self.x_max)
    }

    /// Get the number of interpolation points
    pub fn n_points(&self) -> usize {
        self.coeffs.len()
    }
}

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
    let invnorm: Vec<T> = (0..n).map(|i| T::from_f64_unchecked(0.5 + i as f64)).collect();

    // Compute: res = permutedims(V .* w) .* invnorm
    // This is equivalent to: result[i,j] = V[j,i] * w[j] * invnorm[i]
    DTensor::<T, 2>::from_fn([n, n], |idx| {
        let (i, j) = (idx[0], idx[1]);
        v[[j, i]] * gauss_rule.w[j] * invnorm[i]
    })
}

/// 1D polynomial interpolation using Legendre collocation matrix
///
/// This function uses the pre-computed collocation matrix to avoid
/// solving linear systems during interpolation.
///
/// # Arguments
/// * `values` - Function values at grid points
/// * `gauss_rule` - Gauss quadrature rule containing the grid points
///
/// # Returns
/// Coefficient vector for the interpolating polynomial
pub fn interpolate_1d_legendre<T: CustomNumeric>(values: &[T], gauss_rule: &Rule<T>) -> Vec<T> {
    let n = values.len();
    assert_eq!(
        n,
        gauss_rule.x.len(),
        "Values length must match grid points"
    );

    // Get collocation matrix (pre-computed inverse of Vandermonde matrix)
    let collocation_matrix = legendre_collocation_matrix(gauss_rule);

    // Compute coefficients: coeffs = C * values
    let mut coeffs = vec![T::zero(); n];
    for i in 0..n {
        for j in 0..n {
            coeffs[i] = coeffs[i] + collocation_matrix[[i, j]] * values[j];
        }
    }

    coeffs
}

/// Evaluate interpolated polynomial at point x using coefficient vector
///
/// # Arguments
/// * `x` - Point to evaluate at
/// * `coeffs` - Coefficient vector from interpolate_1d_legendre
///
/// # Returns
/// Interpolated value at x
pub fn evaluate_interpolated_polynomial<T: CustomNumeric>(x: T, coeffs: &[T]) -> T {
    let n = coeffs.len();
    let mut result = T::zero();

    for i in 0..n {
        result = result + coeffs[i] * evaluate_legendre_polynomial(x, i);
    }

    result
}

/// Evaluate Legendre polynomial P_n(x) using recurrence relation
///
/// # Arguments
/// * `x` - Point to evaluate at
/// * `n` - Polynomial degree
///
/// # Returns
/// P_n(x)
fn evaluate_legendre_polynomial<T: CustomNumeric>(x: T, n: usize) -> T {
    evaluate_legendre_basis(x, n + 1)[n]
}

/// Evaluate Legendre polynomial basis functions at point x
///
/// Returns a vector of Legendre polynomial values [P_0(x), P_1(x), ..., P_{n-1}(x)]
pub fn evaluate_legendre_basis<T: CustomNumeric>(x: T, n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }

    let mut p = Vec::with_capacity(n);

    // P_0(x) = 1
    p.push(T::from_f64_unchecked(1.0));

    if n > 1 {
        // P_1(x) = x
        p.push(x);
    }

    // Recurrence relation: (n+1) * P_{n+1}(x) = (2n+1) * x * P_n(x) - n * P_{n-1}(x)
    for i in 1..n - 1 {
        let i_f64 = i as f64;
        let next_p = (T::from_f64_unchecked(2.0 * i_f64 + 1.0) * x * p[i] - T::from_f64_unchecked(i_f64) * p[i - 1])
            / T::from_f64_unchecked(i_f64 + 1.0);
        p.push(next_p);
    }

    p
}

#[cfg(test)]
#[path = "interpolation1d_tests.rs"]
mod tests;
