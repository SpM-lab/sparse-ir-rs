//! Piecewise Legendre polynomials
//!
//! This module provides high-performance piecewise Legendre polynomial
//! functionality compatible with the C++ implementation.

use crate::error::Error;
use crate::matrix::{Mat, Mat3};

/// A single piecewise Legendre polynomial
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePoly {
    /// Polynomial order (degree of Legendre polynomials in each segment)
    pub(crate) polyorder: usize,
    /// Minimum x value of the domain
    pub(crate) xmin: f64,
    /// Maximum x value of the domain
    pub(crate) xmax: f64,
    /// Knot points defining the segments
    pub(crate) knots: Vec<f64>,
    /// Segment widths (for numerical stability)
    pub(crate) delta_x: Vec<f64>,
    /// Coefficient matrix: [degree][segment_index]
    pub(crate) data: Mat<f64>,
    /// Symmetry parameter
    pub(crate) symm: i32,
    /// Index of this function in the sequence of singular functions it belongs
    /// to (0-based, in order of non-increasing singular value)
    ///
    /// SVE results set it to the position of the function in the result, and
    /// `PiecewiseLegendrePolyVector::from_3d_data` to the position in the
    /// vector.
    pub(crate) l: i32,
    /// Segment midpoints
    pub(crate) xm: Vec<f64>,
    /// Inverse segment widths
    pub(crate) inv_xs: Vec<f64>,
    /// Normalization factors
    pub(crate) norms: Vec<f64>,
}

/// `Ok` if there are `nsegments + 1` finite knots and every segment length is
/// a positive normal double (so that `2 / length` is finite)
fn check_knots(knots: &[f64], nsegments: usize) -> Result<(), Error> {
    if knots.len() != nsegments + 1 {
        return Err(Error::InvalidParameter {
            name: "knots",
            value: format!("{} knots", knots.len()),
            reason: format!(
                "must have {} entries, one more than the segments of data",
                nsegments + 1
            ),
        });
    }
    if let Some((i, k)) = knots.iter().enumerate().find(|(_, k)| !k.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "knots",
            value: format!("{k:?} at index {i}"),
            reason: "must be finite".to_string(),
        });
    }
    for i in 1..knots.len() {
        let length = knots[i] - knots[i - 1];
        if !(length > 0.0 && length.is_normal()) {
            return Err(Error::InvalidParameter {
                name: "knots",
                value: format!("{:?} after {:?} at index {i}", knots[i], knots[i - 1]),
                reason: "must be strictly increasing, with each segment length a normal double"
                    .to_string(),
            });
        }
    }
    Ok(())
}

/// `Ok` if `delta_x` has one entry per segment, each equal to the knot
/// spacing `e = knots[i + 1] - knots[i]` within
/// `max(1e-10 * |e|, 8 * f64::EPSILON * max(|knots[i]|, |knots[i + 1]|))`
/// (NaN is rejected)
///
/// The first term is a relative tolerance on the segment length. The second
/// covers the rounding of the knots, which grows with their magnitude: knots
/// and widths scaled separately (as when the functions of an SVE on [-1, 1]
/// are mapped onto a longer interval) differ by a few units in the last place
/// of the knots, which next to large knots can exceed 1e-10 times a narrow
/// segment.
fn check_delta_x(delta_x: &[f64], knots: &[f64]) -> Result<(), Error> {
    let nsegments = knots.len() - 1;
    if delta_x.len() != nsegments {
        return Err(Error::InvalidParameter {
            name: "delta_x",
            value: format!("{} entries", delta_x.len()),
            reason: format!("must have one entry per segment ({nsegments})"),
        });
    }
    for (i, &d) in delta_x.iter().enumerate() {
        let expected = knots[i + 1] - knots[i];
        let rounding = 8.0 * f64::EPSILON * knots[i].abs().max(knots[i + 1].abs());
        let tolerance = (1e-10 * expected.abs()).max(rounding);
        if !((d - expected).abs() <= tolerance) {
            return Err(Error::InvalidParameter {
                name: "delta_x",
                value: format!("{d:?} at index {i}"),
                reason: format!(
                    "must equal the knot spacing {expected:?} to a relative 1e-10, or to 8 machine epsilons times the magnitude of the knots"
                ),
            });
        }
    }
    Ok(())
}

impl PiecewiseLegendrePoly {
    /// Create a new PiecewiseLegendrePoly from data and knots
    ///
    /// `data` holds the Legendre coefficients, one column per segment; the
    /// `nsegments + 1` knots bound the segments. `delta_x` (the segment
    /// widths) is computed from the knots when `None`. `symm` is the parity of
    /// the polynomial: 1 (even), -1 (odd) or 0 (no definite parity).
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `data` has no row or no column
    /// * [`Error::InvalidParameter`] if `knots` does not have one entry more
    ///   than `data` has columns, a knot is not finite, or a segment length
    ///   `knots[i] - knots[i - 1]` is not a positive normal double (this
    ///   includes decreasing knots, NaN, lengths that overflow and subnormal
    ///   lengths); or if `delta_x` does not have one entry per segment or
    ///   differs from the knot spacing `e` by more than
    ///   `max(1e-10 * |e|, 8 * f64::EPSILON * max(|knots[i]|, |knots[i + 1]|))`
    /// * [`Error::InvalidParameter`] if `symm` is not -1, 0 or 1
    pub fn new(
        data: Mat<f64>,
        knots: Vec<f64>,
        l: i32,
        delta_x: Option<Vec<f64>>,
        symm: i32,
    ) -> Result<Self, Error> {
        let polyorder = data.shape().0;
        let nsegments = data.shape().1;
        if polyorder == 0 || nsegments == 0 {
            return Err(Error::EmptyInput { name: "data" });
        }
        check_knots(&knots, nsegments)?;

        // Compute delta_x if not provided
        let delta_x =
            delta_x.unwrap_or_else(|| (1..knots.len()).map(|i| knots[i] - knots[i - 1]).collect());
        check_delta_x(&delta_x, &knots)?;
        if !matches!(symm, -1..=1) {
            return Err(Error::InvalidParameter {
                name: "symm",
                value: symm.to_string(),
                reason: "must be -1, 0 or 1".to_string(),
            });
        }

        // Compute segment midpoints
        let xm: Vec<f64> = (0..nsegments)
            .map(|i| 0.5 * (knots[i] + knots[i + 1]))
            .collect();

        // Compute inverse segment widths
        let inv_xs: Vec<f64> = delta_x.iter().map(|&dx| 2.0 / dx).collect();

        // Compute normalization factors
        let norms: Vec<f64> = inv_xs.iter().map(|&inv_x| inv_x.sqrt()).collect();

        Ok(Self {
            polyorder,
            xmin: knots[0],
            xmax: knots[knots.len() - 1],
            knots,
            delta_x,
            data,
            symm,
            l,
            xm,
            inv_xs,
            norms,
        })
    }

    /// Create a new PiecewiseLegendrePoly with new data but same structure
    ///
    /// Crate-internal: `new_data` is not checked against the knots, so a
    /// caller could break the invariants of the type.
    pub(crate) fn with_data(&self, new_data: Mat<f64>) -> Self {
        Self {
            data: new_data,
            ..self.clone()
        }
    }

    /// Get the symmetry parameter
    pub fn symm(&self) -> i32 {
        self.symm
    }

    /// The polynomial with every coefficient negated
    ///
    /// Knots, widths and normalizations are those of `self`, so this equals
    /// `new` on the negated data without repeating its checks.
    pub(crate) fn negated(&self) -> Self {
        self.with_data(Mat::<f64>::from_fn(self.data.dims(), |idx| -self.data[idx]))
    }

    /// Rescale domain: create a new polynomial with the same data but different knots
    ///
    /// This is useful for transforming from one domain to another, e.g.,
    /// from [-1, 1] to [0, b].
    ///
    /// # Arguments
    ///
    /// * `new_knots` - New knot points
    /// * `new_delta_x` - Optional new segment widths (computed from knots if None)
    /// * `new_symm` - Optional new symmetry parameter (keeps old if None)
    ///
    /// # Returns
    ///
    /// New polynomial with rescaled domain
    ///
    /// # Errors
    ///
    /// The errors of [`new`](Self::new) for the new knots and widths
    pub fn rescale_domain(
        &self,
        new_knots: Vec<f64>,
        new_delta_x: Option<Vec<f64>>,
        new_symm: Option<i32>,
    ) -> Result<Self, Error> {
        Self::new(
            self.data.clone(),
            new_knots,
            self.l,
            new_delta_x,
            new_symm.unwrap_or(self.symm),
        )
    }

    /// Scale all data values by a constant factor
    ///
    /// This is useful for normalizations, e.g., multiplying by √c to keep
    /// the functions normalized after the domain is scaled by c.
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor to multiply all data by
    ///
    /// # Returns
    ///
    /// New polynomial with scaled data
    pub fn scale_data(&self, factor: f64) -> Self {
        Self::with_data(
            self,
            Mat::<f64>::from_fn(self.data.dims(), |idx| self.data[idx] * factor),
        )
    }

    /// `Ok` if `x` lies in [xmin, xmax] (NaN does not)
    fn check_in_domain(&self, name: &'static str, x: f64) -> Result<(), Error> {
        if x >= self.xmin && x <= self.xmax {
            Ok(())
        } else {
            Err(Error::OutOfDomain {
                name,
                value: x,
                domain: (self.xmin, self.xmax),
            })
        }
    }

    /// Evaluate the polynomial at a given point
    ///
    /// # Panics
    ///
    /// Panics if `x` is outside [xmin, xmax] or NaN; see [`Self::try_evaluate`].
    pub fn evaluate(&self, x: f64) -> f64 {
        self.try_evaluate(x).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::evaluate`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] if `x` is outside [xmin, xmax] or NaN
    pub fn try_evaluate(&self, x: f64) -> Result<f64, Error> {
        self.check_in_domain("x", x)?;
        Ok(self.evaluate_in_domain(x))
    }

    /// [`Self::evaluate`] for an `x` already checked to lie in the domain
    fn evaluate_in_domain(&self, x: f64) -> f64 {
        let (i, x_tilde) = self.split_in_domain(x);
        // Extract column i into a Vec
        let coeffs: Vec<f64> = (0..self.data.shape().0)
            .map(|row| self.data[[row, i]])
            .collect();
        let value = legendre_series(x_tilde, &coeffs);
        value * self.norms[i]
    }

    /// Evaluate the polynomial at multiple points
    ///
    /// # Panics
    ///
    /// Panics if a point is outside [xmin, xmax] or NaN; see
    /// [`Self::try_evaluate_many`].
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        self.try_evaluate_many(xs).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::evaluate_many`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] for the first point of `xs` that is outside
    /// [xmin, xmax] or NaN; no point is evaluated then
    pub fn try_evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, Error> {
        for &x in xs {
            self.check_in_domain("xs", x)?;
        }
        Ok(xs.iter().map(|&x| self.evaluate_in_domain(x)).collect())
    }

    /// Split x into segment index and normalized x
    ///
    /// # Panics
    ///
    /// Panics if `x` is outside [xmin, xmax] or NaN; see [`Self::try_split`].
    pub fn split(&self, x: f64) -> (usize, f64) {
        self.try_split(x).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::split`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] if `x` is outside [xmin, xmax] or NaN
    pub fn try_split(&self, x: f64) -> Result<(usize, f64), Error> {
        self.check_in_domain("x", x)?;
        Ok(self.split_in_domain(x))
    }

    /// [`Self::split`] for an `x` already checked to lie in the domain
    fn split_in_domain(&self, x: f64) -> (usize, f64) {
        // Find the segment containing x
        for i in 0..self.knots.len() - 1 {
            if x >= self.knots[i] && x <= self.knots[i + 1] {
                // Transform x to [-1, 1] for Legendre polynomials
                let x_tilde = 2.0 * (x - self.xm[i]) / self.delta_x[i];
                return (i, x_tilde);
            }
        }

        // Handle edge case: x exactly at the last knot
        let last_idx = self.knots.len() - 2;
        let x_tilde = 2.0 * (x - self.xm[last_idx]) / self.delta_x[last_idx];
        (last_idx, x_tilde)
    }

    /// Compute derivative of the polynomial
    ///
    /// The result has `polyorder` equal to the number of its coefficient rows
    /// (at least 1).
    pub fn deriv(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }

        // Compute derivative coefficients
        let mut ddata = self.data.clone();
        for _ in 0..n {
            ddata = self.compute_derivative_coefficients(&ddata);
        }

        // Apply scaling factors (C++: ddata.col(i) *= std::pow(inv_xs[i], n))
        let ddata_shape = *ddata.shape();
        for i in 0..ddata_shape.1 {
            let inv_x_power = self.inv_xs[i].powi(n as i32);
            for j in 0..ddata_shape.0 {
                ddata[[j, i]] *= inv_x_power;
            }
        }

        // Update symmetry: C++: int new_symm = std::pow(-1, n) * symm;
        let new_symm = if n % 2 == 0 { self.symm } else { -self.symm };

        Self {
            polyorder: ddata.shape().0,
            data: ddata,
            symm: new_symm,
            ..self.clone()
        }
    }

    /// Compute derivative coefficients using the same algorithm as C++ legder function
    fn compute_derivative_coefficients(&self, coeffs: &Mat<f64>) -> Mat<f64> {
        let (nrows, ncols) = *coeffs.shape();
        let ders: Vec<Vec<f64>> = (0..ncols)
            .map(|col| legder(&(0..nrows).map(|row| coeffs[[row, col]]).collect::<Vec<_>>()))
            .collect();
        let der_rows = nrows.saturating_sub(1).max(1);
        Mat::<f64>::from_fn([der_rows, ncols], |idx| ders[idx[1]][idx[0]])
    }

    /// Compute derivatives at a point x
    ///
    /// Returns the values of the derivatives of order 0 to `polyorder - 1`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is outside [xmin, xmax] or NaN; see [`Self::try_derivs`].
    pub fn derivs(&self, x: f64) -> Vec<f64> {
        self.try_derivs(x).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::derivs`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] if `x` is outside [xmin, xmax] or NaN
    pub fn try_derivs(&self, x: f64) -> Result<Vec<f64>, Error> {
        let (i, x_tilde) = self.try_split(x)?;
        // Differentiate only the segment containing x, one order at a time,
        // and scale the coefficients (not the sum) as `deriv` does.
        let mut coeffs: Vec<f64> = (0..self.data.shape().0)
            .map(|row| self.data[[row, i]])
            .collect();
        let mut results = Vec::with_capacity(self.polyorder);
        for n in 0..self.polyorder {
            if n > 0 {
                coeffs = legder(&coeffs);
            }
            let inv_x_power = self.inv_xs[i].powi(n as i32);
            let scaled: Vec<f64> = coeffs.iter().map(|&c| c * inv_x_power).collect();
            results.push(legendre_series(x_tilde, &scaled) * self.norms[i]);
        }
        Ok(results)
    }

    /// Compute overlap integral with a function
    pub fn overlap<F>(&self, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut integral = 0.0;

        for i in 0..self.knots.len() - 1 {
            let segment_integral =
                self.gauss_legendre_quadrature(self.knots[i], self.knots[i + 1], |x| {
                    self.evaluate(x) * f(x)
                });
            integral += segment_integral;
        }

        integral
    }

    /// Gauss-Legendre quadrature over [a, b]
    fn gauss_legendre_quadrature<F>(&self, a: f64, b: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        // 5-point Gauss-Legendre quadrature
        const XG: [f64; 5] = [
            -0.906179845938664,
            -0.538469310105683,
            0.0,
            0.538469310105683,
            0.906179845938664,
        ];
        const WG: [f64; 5] = [
            0.236926885056189,
            0.478628670499366,
            0.568888888888889,
            0.478628670499366,
            0.236926885056189,
        ];

        let c1 = (b - a) / 2.0;
        let c2 = (b + a) / 2.0;

        let mut integral = 0.0;
        for j in 0..5 {
            let x = c1 * XG[j] + c2;
            integral += WG[j] * f(x);
        }

        integral * c1
    }

    /// Find roots of the polynomial using C++ compatible algorithm
    pub fn roots(&self) -> Vec<f64> {
        let xmid = (self.xmax + self.xmin) / 2.0;

        // Exploit symmetry: only search the right half, then mirror.
        // This matches the Python 1.x / Julia v1 algorithm and guarantees
        // exactly symmetric root positions.
        let grid = if self.symm != 0 {
            let nsegments = self.knots.len() - 1;
            let mid_idx = nsegments / 2;
            if (self.knots[mid_idx] - xmid).abs() < 1e-15 {
                self.knots[mid_idx..].to_vec()
            } else {
                let mut g = vec![xmid];
                g.extend(self.knots.iter().filter(|&&x| x > xmid));
                g
            }
        } else {
            self.knots.clone()
        };

        let refined_grid = self.refine_grid(&grid, 4);
        let roots_half = self.find_all_roots(&refined_grid);

        if self.symm == 1 {
            // Even symmetry: roots on right half, mirror to left
            let mut all_roots: Vec<f64> = roots_half
                .iter()
                .rev()
                .map(|&r| (self.xmax + self.xmin) - r)
                .collect();
            all_roots.extend_from_slice(&roots_half);
            all_roots
        } else if self.symm == -1 {
            // Odd symmetry: there must be a zero at xmid
            let mut right = roots_half;
            if !right.is_empty() {
                // Remove the root at xmid if found (may be slightly off),
                // or if f(xmid) and f'(xmid) have opposite signs (spurious zero)
                let f_mid = self.evaluate(xmid);
                let f_deriv_mid = self.deriv(1).evaluate(xmid);
                if (right[0] - xmid).abs() < 1e-13 || f_mid * f_deriv_mid < 0.0 {
                    right.remove(0);
                }
            }
            let mut all_roots: Vec<f64> = right
                .iter()
                .rev()
                .map(|&r| (self.xmax + self.xmin) - r)
                .collect();
            all_roots.push(xmid);
            all_roots.extend_from_slice(&right);
            all_roots
        } else {
            // No symmetry: search the full domain
            let full_grid = self.refine_grid(&self.knots, 4);
            self.find_all_roots(&full_grid)
        }
    }

    /// Refine grid by factor alpha (C++ compatible)
    fn refine_grid(&self, grid: &[f64], alpha: usize) -> Vec<f64> {
        let mut refined = Vec::new();

        for i in 0..grid.len() - 1 {
            let start = grid[i];
            let step = (grid[i + 1] - grid[i]) / (alpha as f64);
            for j in 0..alpha {
                refined.push(start + (j as f64) * step);
            }
        }
        refined.push(grid[grid.len() - 1]);
        refined
    }

    /// Find all roots using refined grid (C++ compatible)
    fn find_all_roots(&self, xgrid: &[f64]) -> Vec<f64> {
        if xgrid.is_empty() {
            return Vec::new();
        }

        // Evaluate function at all grid points
        let fx: Vec<f64> = xgrid.iter().map(|&x| self.evaluate(x)).collect();

        // Find exact zeros (direct hits)
        let mut x_hit = Vec::new();
        for i in 0..fx.len() {
            if fx[i] == 0.0 {
                x_hit.push(xgrid[i]);
            }
        }

        // Find sign changes
        let mut sign_change = Vec::new();
        for i in 0..fx.len() - 1 {
            let has_sign_change = fx[i].signum() != fx[i + 1].signum();
            let not_hit = fx[i] != 0.0 && fx[i + 1] != 0.0;
            let sc = has_sign_change && not_hit;
            sign_change.push(sc);
        }

        // If no sign changes, return only direct hits
        if sign_change.iter().all(|&sc| !sc) {
            x_hit.sort_by(|a, b| a.partial_cmp(b).unwrap());
            return x_hit;
        }

        // Find intervals with sign changes
        let mut a_intervals = Vec::new();
        let mut b_intervals = Vec::new();
        let mut fa_values = Vec::new();

        for i in 0..sign_change.len() {
            if sign_change[i] {
                a_intervals.push(xgrid[i]);
                b_intervals.push(xgrid[i + 1]);
                fa_values.push(fx[i]);
            }
        }

        // Calculate epsilon for convergence
        let max_elm = xgrid.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let epsilon_x = f64::EPSILON * max_elm;

        // Use bisection for each interval with sign change
        for i in 0..a_intervals.len() {
            let root = self.bisect(a_intervals[i], b_intervals[i], fa_values[i], epsilon_x);
            x_hit.push(root);
        }

        // Sort and return
        x_hit.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x_hit
    }

    /// Bisection method to find root (C++ compatible)
    fn bisect(&self, a: f64, b: f64, fa: f64, eps: f64) -> f64 {
        let mut a = a;
        let mut b = b;
        let mut fa = fa;

        loop {
            let mid = (a + b) / 2.0;
            if self.close_enough(a, mid, eps) {
                return mid;
            }

            let fmid = self.evaluate(mid);
            if fa.signum() != fmid.signum() {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
    }

    /// Check if two values are close enough (C++ compatible)
    fn close_enough(&self, a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    /// Lower end of the domain
    pub fn xmin(&self) -> f64 {
        self.xmin
    }
    /// Upper end of the domain
    pub fn xmax(&self) -> f64 {
        self.xmax
    }
    /// Index of the function in its sequence of singular functions
    /// (0-based, in order of non-increasing singular value)
    pub fn l(&self) -> i32 {
        self.l
    }
    /// The domain `(xmin, xmax)`
    pub fn domain(&self) -> (f64, f64) {
        (self.xmin, self.xmax)
    }
    /// Segment boundaries, strictly increasing from `xmin` to `xmax`
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }
    /// Segment lengths
    pub fn delta_x(&self) -> &[f64] {
        &self.delta_x
    }
    /// Legendre coefficients, `polyorder × nsegments` (one column per
    /// segment)
    pub fn data(&self) -> &Mat<f64> {
        &self.data
    }
    /// Normalization of the Legendre polynomials on each segment
    pub fn norms(&self) -> &[f64] {
        &self.norms
    }
    /// Number of Legendre coefficients per segment (degree + 1)
    pub fn polyorder(&self) -> usize {
        self.polyorder
    }
}

/// The Legendre series Σ_n coeffs[n] P_n(x), by the three-term recurrence
pub(crate) fn legendre_series(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    let mut result = 0.0;
    let mut p_prev = 1.0; // P_0(x) = 1
    let mut p_curr = x; // P_1(x) = x

    // Add first two terms
    if !coeffs.is_empty() {
        result += coeffs[0] * p_prev;
    }
    if coeffs.len() > 1 {
        result += coeffs[1] * p_curr;
    }

    // Use recurrence relation: P_{n+1}(x) = ((2n+1)x*P_n(x) - n*P_{n-1}(x))/(n+1)
    for n in 1..coeffs.len() - 1 {
        let p_next =
            ((2.0 * (n as f64) + 1.0) * x * p_curr - (n as f64) * p_prev) / ((n + 1) as f64);
        result += coeffs[n + 1] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }

    result
}

/// Coefficients of the derivative of the Legendre series `c`
/// (the same algorithm as the C++ legder function with cnt = 1)
fn legder(c: &[f64]) -> Vec<f64> {
    if c.len() <= 1 {
        return vec![0.0];
    }
    let mut c = c.to_vec();
    let n = c.len() - 1;
    let mut der = vec![0.0; n];
    for j in (2..=n).rev() {
        der[j - 1] = (2.0 * (j as f64) - 1.0) * c[j];
        c[j - 2] += c[j];
    }
    if n > 1 {
        der[1] = 3.0 * c[2];
    }
    der[0] = c[1];
    der
}

/// Former names of the accessors
impl PiecewiseLegendrePoly {
    /// The Legendre series Σ_n coeffs[n] P_n(x); it does not depend on the
    /// polynomial
    #[deprecated(note = "an internal helper, to be removed")]
    #[doc(hidden)]
    pub fn evaluate_legendre_polynomial(&self, x: f64, coeffs: &[f64]) -> f64 {
        legendre_series(x, coeffs)
    }

    /// [`Self::xmin`]
    #[deprecated(note = "renamed xmin")]
    #[doc(hidden)]
    pub fn get_xmin(&self) -> f64 {
        self.xmin()
    }

    /// [`Self::xmax`]
    #[deprecated(note = "renamed xmax")]
    #[doc(hidden)]
    pub fn get_xmax(&self) -> f64 {
        self.xmax()
    }

    /// [`Self::l`]
    #[deprecated(note = "renamed l")]
    #[doc(hidden)]
    pub fn get_l(&self) -> i32 {
        self.l()
    }

    /// [`Self::domain`]
    #[deprecated(note = "renamed domain")]
    #[doc(hidden)]
    pub fn get_domain(&self) -> (f64, f64) {
        self.domain()
    }

    /// [`Self::knots`]
    #[deprecated(note = "renamed knots")]
    #[doc(hidden)]
    pub fn get_knots(&self) -> &[f64] {
        self.knots()
    }

    /// [`Self::delta_x`]
    #[deprecated(note = "renamed delta_x")]
    #[doc(hidden)]
    pub fn get_delta_x(&self) -> &[f64] {
        self.delta_x()
    }

    /// [`Self::symm`]
    #[deprecated(note = "renamed symm")]
    #[doc(hidden)]
    pub fn get_symm(&self) -> i32 {
        self.symm()
    }

    /// [`Self::data`]
    #[deprecated(note = "renamed data")]
    #[doc(hidden)]
    pub fn get_data(&self) -> &Mat<f64> {
        self.data()
    }

    /// [`Self::norms`]
    #[deprecated(note = "renamed norms")]
    #[doc(hidden)]
    pub fn get_norms(&self) -> &[f64] {
        self.norms()
    }

    /// [`Self::polyorder`]
    #[deprecated(note = "renamed polyorder")]
    #[doc(hidden)]
    pub fn get_polyorder(&self) -> usize {
        self.polyorder()
    }
}

/// Vector of piecewise Legendre polynomials
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePolyVector {
    /// Individual polynomials
    pub(crate) polyvec: Vec<PiecewiseLegendrePoly>,
}

impl PiecewiseLegendrePolyVector {
    /// Constructor with a vector of PiecewiseLegendrePoly
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `polyvec` is empty
    /// * [`Error::InvalidParameter`] if a polynomial has other knots or
    ///   another data shape than the first: the accessors of the vector
    ///   (`get_knots`, `get_data`, ...) describe all of them by the first
    pub fn new(polyvec: Vec<PiecewiseLegendrePoly>) -> Result<Self, Error> {
        let Some(first) = polyvec.first() else {
            return Err(Error::EmptyInput { name: "polyvec" });
        };
        if let Some(i) = polyvec
            .iter()
            .position(|p| p.knots != first.knots || p.data.shape() != first.data.shape())
        {
            return Err(Error::InvalidParameter {
                name: "polyvec",
                value: format!("polynomial {i}"),
                reason: "must have the knots and the data shape of polynomial 0".to_string(),
            });
        }
        Ok(Self { polyvec })
    }

    /// Constructor that skips the checks of [`Self::new`]
    ///
    /// For the SVE, which builds every polynomial of a vector from the same
    /// knots and the same data shape and may legitimately build an empty
    /// vector (a symmetrized half of an expansion with no singular value of
    /// that parity).
    pub(crate) fn from_polys_unchecked(polyvec: Vec<PiecewiseLegendrePoly>) -> Self {
        Self { polyvec }
    }

    /// The polynomials
    pub fn polys(&self) -> &[PiecewiseLegendrePoly] {
        &self.polyvec
    }

    /// Constructor with a 3D array, knots, and symmetry vector
    ///
    /// `data3d` has the shape `(polyorder, nsegments, npolys)`; polynomial `i`
    /// gets `l = i` and the symmetry `symm[i]` (0 if `symm` is `None`).
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `data3d` has no polynomial
    /// * [`Error::InvalidParameter`] if `symm` does not have one entry per
    ///   polynomial
    /// * The errors of [`PiecewiseLegendrePoly::new`] for the data and knots
    pub fn from_3d_data(
        data3d: Mat3<f64>,
        knots: Vec<f64>,
        symm: Option<Vec<i32>>,
    ) -> Result<Self, Error> {
        let npolys = data3d.shape().2;
        if npolys == 0 {
            return Err(Error::EmptyInput { name: "data3d" });
        }
        let mut polyvec = Vec::with_capacity(npolys);

        if let Some(ref symm_vec) = symm {
            if symm_vec.len() != npolys {
                return Err(Error::InvalidParameter {
                    name: "symm",
                    value: format!("{} entries", symm_vec.len()),
                    reason: format!("must have one entry per polynomial ({npolys})"),
                });
            }
        }

        // Compute delta_x from knots
        let delta_x: Vec<f64> = (1..knots.len()).map(|i| knots[i] - knots[i - 1]).collect();

        for i in 0..npolys {
            // Extract 2D data for this polynomial
            let data3d_shape = data3d.shape();
            let mut data = Mat::<f64>::from_elem([data3d_shape.0, data3d_shape.1], 0.0);
            for j in 0..data3d_shape.0 {
                for k in 0..data3d_shape.1 {
                    data[[j, k]] = data3d[[j, k, i]];
                }
            }

            let poly = PiecewiseLegendrePoly::new(
                data,
                knots.clone(),
                i as i32,
                Some(delta_x.clone()),
                symm.as_ref().map_or(0, |s| s[i]),
            )?;

            polyvec.push(poly);
        }

        Ok(Self { polyvec })
    }

    /// Number of polynomials
    pub fn len(&self) -> usize {
        self.polyvec.len()
    }

    /// Whether there is no polynomial
    pub fn is_empty(&self) -> bool {
        self.polyvec.is_empty()
    }

    /// Rescale domain for all polynomials in the vector
    ///
    /// Creates a new PiecewiseLegendrePolyVector where each polynomial has
    /// the same data but new knots and delta_x.
    ///
    /// # Arguments
    ///
    /// * `new_knots` - New knot points (same for all polynomials)
    /// * `new_delta_x` - Optional new segment widths
    /// * `new_symm` - Optional vector of new symmetry parameters (one per polynomial)
    ///
    /// # Returns
    ///
    /// New vector with rescaled domains
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if `new_symm` does not have one entry per
    ///   polynomial
    /// * The errors of [`PiecewiseLegendrePoly::rescale_domain`]
    pub fn rescale_domain(
        &self,
        new_knots: Vec<f64>,
        new_delta_x: Option<Vec<f64>>,
        new_symm: Option<Vec<i32>>,
    ) -> Result<Self, Error> {
        if let Some(symm) = &new_symm {
            if symm.len() != self.polyvec.len() {
                return Err(Error::InvalidParameter {
                    name: "new_symm",
                    value: format!("{} entries", symm.len()),
                    reason: format!(
                        "must have one entry per polynomial ({})",
                        self.polyvec.len()
                    ),
                });
            }
        }
        let polyvec = self
            .polyvec
            .iter()
            .enumerate()
            .map(|(i, poly)| {
                let symm = new_symm.as_ref().map(|s| s[i]);
                poly.rescale_domain(new_knots.clone(), new_delta_x.clone(), symm)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { polyvec })
    }

    /// Scale all data values by a constant factor
    ///
    /// Multiplies the data of all polynomials by the same factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor to multiply all data by
    ///
    /// # Returns
    ///
    /// New vector with scaled data
    pub fn scale_data(&self, factor: f64) -> Self {
        let polyvec = self
            .polyvec
            .iter()
            .map(|poly| poly.scale_data(factor))
            .collect();

        Self { polyvec }
    }

    /// Get polynomial by index (immutable)
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendrePoly> {
        self.polyvec.get(index)
    }

    /// Get polynomial by index (mutable) - deprecated, use immutable design instead
    #[deprecated(
        note = "PiecewiseLegendrePolyVector is designed to be immutable. Use get() and create new instances for modifications."
    )]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PiecewiseLegendrePoly> {
        self.polyvec.get_mut(index)
    }

    /// Extract a single polynomial as a vector
    pub fn slice_single(&self, index: usize) -> Option<Self> {
        self.polyvec.get(index).map(|poly| Self {
            polyvec: vec![poly.clone()],
        })
    }

    /// Extract multiple polynomials by indices, in the order of `indices`
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `indices` is empty
    /// * [`Error::InvalidParameter`] for the first index that is out of range
    ///   or repeats an earlier one
    pub fn slice_multi(&self, indices: &[usize]) -> Result<Self, Error> {
        if indices.is_empty() {
            return Err(Error::EmptyInput { name: "indices" });
        }
        let len = self.polyvec.len();
        let mut seen = vec![false; len];
        for &idx in indices {
            if idx >= len {
                return Err(Error::InvalidParameter {
                    name: "indices",
                    value: format!("{idx}"),
                    reason: format!("must be less than the size {len}"),
                });
            }
            if seen[idx] {
                return Err(Error::InvalidParameter {
                    name: "indices",
                    value: format!("{idx}"),
                    reason: "must not repeat".to_string(),
                });
            }
            seen[idx] = true;
        }

        let new_polyvec: Vec<_> = indices
            .iter()
            .map(|&idx| self.polyvec[idx].clone())
            .collect();

        Ok(Self {
            polyvec: new_polyvec,
        })
    }

    /// Evaluate all polynomials at a single point
    ///
    /// # Panics
    ///
    /// Panics if `x` is outside the domain of a polynomial or NaN; see
    /// [`Self::try_evaluate_at`].
    pub fn evaluate_at(&self, x: f64) -> Vec<f64> {
        self.try_evaluate_at(x).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::evaluate_at`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] if `x` is outside the domain of a polynomial
    /// or NaN
    pub fn try_evaluate_at(&self, x: f64) -> Result<Vec<f64>, Error> {
        self.polyvec
            .iter()
            .map(|poly| poly.try_evaluate(x))
            .collect()
    }

    /// Evaluate all polynomials at multiple points
    ///
    /// The result has the shape `(size, xs.len())`.
    ///
    /// # Panics
    ///
    /// Panics if a point is outside the domain of a polynomial or NaN; see
    /// [`Self::try_evaluate_at_many`].
    pub fn evaluate_at_many(&self, xs: &[f64]) -> Mat<f64> {
        self.try_evaluate_at_many(xs)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::evaluate_at_many`] returning an error instead of panicking
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] for the first point of `xs` that is outside
    /// the domain of a polynomial or NaN; every point is checked before any
    /// is evaluated
    pub fn try_evaluate_at_many(&self, xs: &[f64]) -> Result<Mat<f64>, Error> {
        // `new` gives every polynomial the knots of the first, so checking the
        // points against the first checks them against all of them.
        if let Some(first) = self.polyvec.first() {
            for &x in xs {
                first.check_in_domain("xs", x)?;
            }
        }
        let n_funcs = self.polyvec.len();
        let n_points = xs.len();
        let mut results = Mat::<f64>::from_elem([n_funcs, n_points], 0.0);

        for (i, poly) in self.polyvec.iter().enumerate() {
            for (j, &x) in xs.iter().enumerate() {
                results[[i, j]] = poly.evaluate_in_domain(x);
            }
        }

        Ok(results)
    }

    pub fn xmin(&self) -> f64 {
        if self.polyvec.is_empty() {
            panic!("Cannot get xmin from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].xmin
    }

    pub fn xmax(&self) -> f64 {
        if self.polyvec.is_empty() {
            panic!("Cannot get xmax from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].xmax
    }

    /// The knots of all the polynomials, merged where closer than
    /// `tolerance` (1e-10 by default)
    pub fn knots(&self, tolerance: Option<f64>) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get knots from empty PiecewiseLegendrePolyVector");
        }
        const DEFAULT_TOLERANCE: f64 = 1e-10;
        let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);

        // Collect all knots from all polynomials
        let mut all_knots = Vec::new();
        for poly in &self.polyvec {
            for &knot in &poly.knots {
                all_knots.push(knot);
            }
        }

        // Sort and remove duplicates
        {
            all_knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_knots.dedup_by(|a, b| (*a - *b).abs() < tolerance);
        }
        all_knots
    }

    /// Segment lengths of the first polynomial
    pub fn delta_x(&self) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get delta_x from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].delta_x.clone()
    }

    /// Number of Legendre coefficients per segment (degree + 1)
    pub fn polyorder(&self) -> usize {
        if self.polyvec.is_empty() {
            panic!("Cannot get polyorder from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].polyorder
    }

    /// Normalization of the Legendre polynomials on each segment
    pub fn norms(&self) -> &[f64] {
        if self.polyvec.is_empty() {
            panic!("Cannot get norms from empty PiecewiseLegendrePolyVector");
        }
        &self.polyvec[0].norms
    }

    /// Parity of each polynomial: 1 even, -1 odd, 0 none
    pub fn symm(&self) -> Vec<i32> {
        if self.polyvec.is_empty() {
            panic!("Cannot get symm from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec.iter().map(|poly| poly.symm).collect()
    }

    /// Legendre coefficients as a `polyorder × nsegments × npolys` array
    pub fn data(&self) -> Mat3<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get data from empty PiecewiseLegendrePolyVector");
        }

        let nsegments = self.polyvec[0].data.shape().1;
        let polyorder = self.polyvec[0].polyorder;
        let npolys = self.polyvec.len();

        let mut data = Mat3::<f64>::from_elem([nsegments, polyorder, npolys], 0.0);

        for (poly_idx, poly) in self.polyvec.iter().enumerate() {
            for segment in 0..nsegments {
                for degree in 0..polyorder {
                    data[[segment, degree, poly_idx]] = poly.data[[degree, segment]];
                }
            }
        }

        data
    }

    /// Find roots of all polynomials
    pub fn roots(&self, tolerance: Option<f64>) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get roots from empty PiecewiseLegendrePolyVector");
        }
        const DEFAULT_TOLERANCE: f64 = 1e-10;
        let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);
        let mut all_roots = Vec::new();

        for poly in &self.polyvec {
            let poly_roots = poly.roots();
            for root in poly_roots {
                all_roots.push(root);
            }
        }

        // Sort in descending order and remove duplicates (like C++ implementation)
        {
            all_roots.sort_by(|a, b| b.partial_cmp(a).unwrap());
            all_roots.dedup_by(|a, b| (*a - *b).abs() < tolerance);
        }
        all_roots
    }

    /// Get reference to last polynomial
    ///
    /// C++ equivalent: u.polyvec.back()
    pub fn last(&self) -> &PiecewiseLegendrePoly {
        self.polyvec
            .last()
            .expect("Cannot get last from empty PiecewiseLegendrePolyVector")
    }

    /// Get the number of roots
    pub fn nroots(&self, tolerance: Option<f64>) -> usize {
        if self.polyvec.is_empty() {
            panic!("Cannot get nroots from empty PiecewiseLegendrePolyVector");
        }
        self.roots(tolerance).len()
    }
}

/// Former names of the accessors
impl PiecewiseLegendrePolyVector {
    /// [`Self::len`]
    #[deprecated(note = "renamed len")]
    #[doc(hidden)]
    pub fn size(&self) -> usize {
        self.len()
    }

    /// [`Self::polys`]
    #[deprecated(note = "renamed polys")]
    #[doc(hidden)]
    pub fn get_polys(&self) -> &[PiecewiseLegendrePoly] {
        self.polys()
    }

    /// [`Self::knots`]
    #[deprecated(note = "renamed knots")]
    #[doc(hidden)]
    pub fn get_knots(&self, tolerance: Option<f64>) -> Vec<f64> {
        self.knots(tolerance)
    }

    /// [`Self::delta_x`]
    #[deprecated(note = "renamed delta_x")]
    #[doc(hidden)]
    pub fn get_delta_x(&self) -> Vec<f64> {
        self.delta_x()
    }

    /// [`Self::polyorder`]
    #[deprecated(note = "renamed polyorder")]
    #[doc(hidden)]
    pub fn get_polyorder(&self) -> usize {
        self.polyorder()
    }

    /// [`Self::norms`]
    #[deprecated(note = "renamed norms")]
    #[doc(hidden)]
    pub fn get_norms(&self) -> &[f64] {
        self.norms()
    }

    /// [`Self::symm`]
    #[deprecated(note = "renamed symm")]
    #[doc(hidden)]
    pub fn get_symm(&self) -> Vec<i32> {
        self.symm()
    }

    /// [`Self::data`]
    #[deprecated(note = "renamed data")]
    #[doc(hidden)]
    pub fn get_data(&self) -> Mat3<f64> {
        self.data()
    }
}

impl std::ops::Index<usize> for PiecewiseLegendrePolyVector {
    type Output = PiecewiseLegendrePoly;

    fn index(&self, index: usize) -> &Self::Output {
        &self.polyvec[index]
    }
}

#[cfg(test)]
#[path = "poly_tests.rs"]
mod poly_tests;
