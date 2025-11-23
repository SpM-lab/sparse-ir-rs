//! 2D interpolation functionality for SparseIR
//!
//! This module provides efficient 2D interpolation using Legendre polynomials
//! within individual grid cells.

use crate::gauss::Rule;
use crate::interpolation1d::{evaluate_legendre_basis, legendre_collocation_matrix};
use crate::numeric::CustomNumeric;
use mdarray::DTensor;
use std::fmt::Debug;

/// 2D interpolation object for a single grid cell
///
/// This structure stores pre-computed polynomial coefficients for efficient
/// interpolation within a single grid cell.
#[derive(Debug, Clone)]
pub struct Interpolate2D<T> {
    /// Cell boundaries
    pub x_min: T,
    pub x_max: T,
    pub y_min: T,
    pub y_max: T,

    /// Pre-computed polynomial coefficients
    pub coeffs: DTensor<T, 2>,

    /// Grid points (for validation)
    pub gauss_x: Rule<T>,
    pub gauss_y: Rule<T>,
}

impl<T: CustomNumeric + Debug + 'static> Interpolate2D<T> {
    /// Create a new Interpolate2D object from grid values
    ///
    /// # Arguments
    /// * `values` - Function values at grid points
    /// * `gauss_x` - x-direction grid points (can be in any range)
    /// * `gauss_y` - y-direction grid points (can be in any range)
    ///
    /// # Panics
    /// Panics if dimensions don't match or if grid is empty
    pub fn new(values: &DTensor<T, 2>, gauss_x: &Rule<T>, gauss_y: &Rule<T>) -> Self {
        let shape = *values.shape();
        assert!(
            shape.0 > 0 && shape.1 > 0,
            "Cannot create interpolation from empty grid"
        );
        assert_eq!(
            shape.0,
            gauss_x.x.len(),
            "Values height must match gauss_x length"
        );
        assert_eq!(
            shape.1,
            gauss_y.x.len(),
            "Values width must match gauss_y length"
        );

        // Create normalized Gauss rules for coefficient computation
        // interpolate_2d_legendre expects Gauss points in [-1, 1] range
        let normalized_gauss_x =
            gauss_x.reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));
        let normalized_gauss_y =
            gauss_y.reseat(T::from_f64_unchecked(-1.0), T::from_f64_unchecked(1.0));

        let coeffs = interpolate_2d_legendre(values, &normalized_gauss_x, &normalized_gauss_y);

        Self {
            x_min: gauss_x.a,
            x_max: gauss_x.b,
            y_min: gauss_y.a,
            y_max: gauss_y.b,
            coeffs,
            gauss_x: gauss_x.clone(),
            gauss_y: gauss_y.clone(),
        }
    }

    /// Interpolate the function at point (x, y)
    ///
    /// # Arguments
    /// * `x` - x-coordinate
    /// * `y` - y-coordinate
    ///
    /// # Panics
    /// Panics if (x, y) is outside the cell boundaries
    pub fn interpolate(&self, x: T, y: T) -> T {
        assert!(
            x >= self.x_min && x <= self.x_max,
            "x={} is outside cell bounds [{}, {}]",
            x,
            self.x_min,
            self.x_max
        );
        assert!(
            y >= self.y_min && y <= self.y_max,
            "y={} is outside cell bounds [{}, {}]",
            y,
            self.y_min,
            self.y_max
        );

        evaluate_2d_legendre_polynomial(x, y, &self.coeffs, &self.gauss_x, &self.gauss_y)
    }

    /// Get the coefficient matrix
    pub fn coefficients(&self) -> &DTensor<T, 2> {
        &self.coeffs
    }

    /// Get cell boundaries
    pub fn bounds(&self) -> (T, T, T, T) {
        (self.x_min, self.x_max, self.y_min, self.y_max)
    }

    /// Get domain boundaries (alias for bounds)
    pub fn domain(&self) -> (T, T, T, T) {
        self.bounds()
    }

    /// Get the number of interpolation points in x direction
    pub fn n_points_x(&self) -> usize {
        self.coeffs.shape().0
    }

    /// Get the number of interpolation points in y direction
    pub fn n_points_y(&self) -> usize {
        self.coeffs.shape().1
    }

    /// Evaluate the interpolated function at a given point (alias for interpolate)
    pub fn evaluate(&self, x: T, y: T) -> T {
        self.interpolate(x, y)
    }
}

/// 2D Legendre polynomial interpolation using collocation matrices
///
/// This function computes coefficients for a 2D Legendre polynomial that
/// interpolates the given function values at the Gauss points using the
/// efficient collocation matrix approach.
///
/// # Arguments
/// * `values` - Function values at grid points (n_x x n_y)
/// * `gauss_x` - x-direction Gauss quadrature rule
/// * `gauss_y` - y-direction Gauss quadrature rule
///
/// # Returns
/// Coefficient matrix (n_x x n_y) for the interpolating polynomial
pub fn interpolate_2d_legendre<T: CustomNumeric + 'static>(
    values: &DTensor<T, 2>,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
) -> DTensor<T, 2> {
    let n_x = gauss_x.x.len();
    let n_y = gauss_y.x.len();

    let shape = *values.shape();
    assert_eq!(shape.0, n_x, "Values matrix rows must match x grid points");
    assert_eq!(shape.1, n_y, "Values matrix cols must match y grid points");

    // Get collocation matrices (pre-computed inverses of Vandermonde matrices)
    let collocation_x = legendre_collocation_matrix(gauss_x);
    let collocation_y = legendre_collocation_matrix(gauss_y);

    // Compute coefficients using tensor product approach
    // coeffs = C_x * values * C_y^T
    let mut temp = DTensor::<T, 2>::from_elem([n_x, n_y], T::zero());
    for i in 0..n_x {
        for j in 0..n_y {
            for k in 0..n_x {
                temp[[i, j]] = temp[[i, j]] + collocation_x[[i, k]] * values[[k, j]];
            }
        }
    }

    let mut coeffs = DTensor::<T, 2>::from_elem([n_x, n_y], T::zero());
    for i in 0..n_x {
        for j in 0..n_y {
            for k in 0..n_y {
                coeffs[[i, j]] = coeffs[[i, j]] + temp[[i, k]] * collocation_y[[j, k]];
            }
        }
    }

    coeffs
}

/// Evaluate 2D Legendre polynomial at point (x, y) using coefficient matrix
pub fn evaluate_2d_legendre_polynomial<T: CustomNumeric>(
    x: T,
    y: T,
    coeffs: &DTensor<T, 2>,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
) -> T {
    let shape = *coeffs.shape();
    let n_x = shape.0;
    let n_y = shape.1;

    // Normalize coordinates from [a,b] to [-1,1] where [a,b] is the cell domain
    let x_norm = T::from_f64_unchecked(2.0) * (x - gauss_x.a) / (gauss_x.b - gauss_x.a)
        - T::from_f64_unchecked(1.0);
    let y_norm = T::from_f64_unchecked(2.0) * (y - gauss_y.a) / (gauss_y.b - gauss_y.a)
        - T::from_f64_unchecked(1.0);

    // Evaluate Legendre polynomials at normalized coordinates
    let p_x = evaluate_legendre_basis(x_norm, n_x);
    let p_y = evaluate_legendre_basis(y_norm, n_y);

    // Compute tensor product sum: sum_{i,j} coeffs[i,j] * P_i(x) * P_j(y)
    let mut result = T::zero();
    for i in 0..n_x {
        for j in 0..n_y {
            result = result + coeffs[[i, j]] * p_x[i] * p_y[j];
        }
    }

    result
}

#[cfg(test)]
#[path = "interpolation2d_tests.rs"]
mod tests;
