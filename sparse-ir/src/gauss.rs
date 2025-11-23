//! Gauss quadrature rules for numerical integration
//!
//! This module provides quadrature rules for approximating integrals by weighted sums.
//!
//! The integral of f(x) * omega(x) is approximated by a weighted sum:
//!
//! sum(f(xi) * wi for (xi, wi) in zip(x, w))
//!
//! where we generally have superexponential convergence for smooth f(x)
//! with the number of quadrature points.

use crate::numeric::CustomNumeric;
use simba::scalar::ComplexField;
use std::fmt::Debug;

/// Quadrature rule for numerical integration.
///
/// Represents an approximation of an integral by a weighted sum over discrete points.
/// The rule contains quadrature points `x`, weights `w`, and auxiliary arrays
/// `x_forward` and `x_backward` for efficient computation.
#[derive(Debug, Clone)]
pub struct Rule<T> {
    /// Quadrature points
    pub x: Vec<T>, //COMMENT: ADD CHECK CODE TO MAKE SURE x is in non-decreasing order
    /// Quadrature weights
    pub w: Vec<T>,
    /// Distance from left endpoint: x - a
    pub x_forward: Vec<T>,
    /// Distance from right endpoint: b - x
    pub x_backward: Vec<T>,
    /// Left endpoint of integration interval
    pub a: T,
    /// Right endpoint of integration interval
    pub b: T,
}

impl<T> Rule<T>
where
    T: CustomNumeric,
{
    /// Create a new quadrature rule from points and weights.
    ///
    /// # Arguments
    /// * `x` - Quadrature points
    /// * `w` - Quadrature weights
    /// * `a` - Left endpoint (default: -1.0)
    /// * `b` - Right endpoint (default: 1.0)
    ///
    /// # Panics
    /// Panics if x and w have different lengths.
    pub fn new(x: Vec<T>, w: Vec<T>, a: T, b: T) -> Self {
        assert_eq!(x.len(), w.len(), "x and w must have the same length");

        let x_forward: Vec<T> = x.iter().map(|&xi| xi - a).collect();
        let x_backward: Vec<T> = x.iter().map(|&xi| b - xi).collect();

        Self {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }

    /// Create a new quadrature rule from vectors.
    pub fn from_vectors(x: Vec<T>, w: Vec<T>, a: T, b: T) -> Self {
        Self::new(x, w, a, b)
    }

    /// Create a default rule with empty arrays.
    pub fn empty() -> Self {
        Self {
            x: vec![],
            w: vec![],
            x_forward: vec![],
            x_backward: vec![],
            a: <T as CustomNumeric>::from_f64_unchecked(-1.0),
            b: <T as CustomNumeric>::from_f64_unchecked(1.0),
        }
    }

    /// Reseat the rule to a new interval [a, b].
    ///
    /// Scales and translates the quadrature points and weights to the new interval.
    pub fn reseat(&self, a: T, b: T) -> Self {
        let scaling = (b - a) / (self.b - self.a);
        let midpoint_old = (self.b + self.a) * <T as CustomNumeric>::from_f64_unchecked(0.5);
        let midpoint_new = (b + a) * <T as CustomNumeric>::from_f64_unchecked(0.5);

        // Transform x: scaling * (xi - midpoint_old) + midpoint_new
        let new_x: Vec<T> = self
            .x
            .iter()
            .map(|&xi| scaling * (xi - midpoint_old) + midpoint_new)
            .collect();
        let new_w: Vec<T> = self.w.iter().map(|&wi| wi * scaling).collect();
        let new_x_forward: Vec<T> = self.x_forward.iter().map(|&xi| xi * scaling).collect();
        let new_x_backward: Vec<T> = self.x_backward.iter().map(|&xi| xi * scaling).collect();

        Self {
            x: new_x,
            w: new_w,
            x_forward: new_x_forward,
            x_backward: new_x_backward,
            a,
            b,
        }
    }

    /// Scale the weights by a factor.
    pub fn scale(&self, factor: T) -> Self {
        Self {
            x: self.x.clone(),
            w: self.w.iter().map(|&wi| wi * factor).collect(),
            x_forward: self.x_forward.clone(),
            x_backward: self.x_backward.clone(),
            a: self.a,
            b: self.b,
        }
    }

    /// Create a piecewise rule over multiple segments.
    ///
    /// # Arguments
    /// * `edges` - Segment boundaries (must be sorted in ascending order)
    ///
    /// # Panics
    /// Panics if edges are not sorted or have less than 2 elements.
    pub fn piecewise(&self, edges: &[T]) -> Self {
        if edges.len() < 2 {
            panic!("edges must have at least 2 elements");
        }

        // Check if edges are sorted
        for i in 1..edges.len() {
            if edges[i] <= edges[i - 1] {
                panic!("edges must be sorted in ascending order");
            }
        }

        let mut rules = Vec::new();
        for i in 0..edges.len() - 1 {
            let rule = self.reseat(edges[i], edges[i + 1]);
            rules.push(rule);
        }

        Self::join(&rules)
    }

    /// Join multiple rules into a single rule.
    ///
    /// # Arguments
    /// * `rules` - Vector of rules to join (must be contiguous and sorted)
    ///
    /// # Panics
    /// Panics if rules are empty, not contiguous, or not sorted.
    pub fn join(rules: &[Self]) -> Self {
        if rules.is_empty() {
            return Self::empty();
        }

        let a = rules[0].a;
        let b = rules[rules.len() - 1].b;

        // Check that rules are contiguous
        for i in 1..rules.len() {
            if (rules[i].a - rules[i - 1].b).abs_as_same_type() > T::epsilon() {
                panic!("rules must be contiguous");
            }
        }

        // Concatenate all arrays
        let mut x_vec = Vec::new();
        let mut w_vec = Vec::new();
        let mut x_forward_vec = Vec::new();
        let mut x_backward_vec = Vec::new();

        for rule in rules {
            // Adjust x_forward and x_backward for global coordinates
            let x_forward_adj: Vec<T> =
                rule.x_forward.iter().map(|&xi| xi + (rule.a - a)).collect();
            let x_backward_adj: Vec<T> = rule
                .x_backward
                .iter()
                .map(|&xi| xi + (b - rule.b))
                .collect();

            x_vec.extend(rule.x.iter().cloned());
            w_vec.extend(rule.w.iter().cloned());
            x_forward_vec.extend(x_forward_adj.iter().cloned());
            x_backward_vec.extend(x_backward_adj.iter().cloned());
        }

        // Sort by x values to maintain order
        let mut indices: Vec<usize> = (0..x_vec.len()).collect();
        indices.sort_by(|&a, &b| x_vec[a].partial_cmp(&x_vec[b]).unwrap());

        let sorted_x: Vec<T> = indices.iter().map(|&i| x_vec[i]).collect();
        let sorted_w: Vec<T> = indices.iter().map(|&i| w_vec[i]).collect();

        // Recalculate x_forward and x_backward after sorting
        let sorted_x_forward: Vec<T> = sorted_x.iter().map(|&xi| xi - a).collect();
        let sorted_x_backward: Vec<T> = sorted_x.iter().map(|&xi| b - xi).collect();

        Self {
            x: sorted_x,
            w: sorted_w,
            x_forward: sorted_x_forward,
            x_backward: sorted_x_backward,
            a,
            b,
        }
    }

    /// Convert the rule to a different numeric type.
    pub fn convert<U>(&self) -> Rule<U>
    where
        U: CustomNumeric + Copy + Debug + std::fmt::Display,
    {
        let x: Vec<U> = self
            .x
            .iter()
            .map(|&xi| <U as CustomNumeric>::from_f64_unchecked(xi.to_f64()))
            .collect();
        let w: Vec<U> = self
            .w
            .iter()
            .map(|&wi| <U as CustomNumeric>::from_f64_unchecked(wi.to_f64()))
            .collect();
        let x_forward: Vec<U> = self
            .x_forward
            .iter()
            .map(|&xi| <U as CustomNumeric>::from_f64_unchecked(xi.to_f64()))
            .collect();
        let x_backward: Vec<U> = self
            .x_backward
            .iter()
            .map(|&xi| <U as CustomNumeric>::from_f64_unchecked(xi.to_f64()))
            .collect();
        let a = <U as CustomNumeric>::from_f64_unchecked(self.a.to_f64());
        let b = <U as CustomNumeric>::from_f64_unchecked(self.b.to_f64());

        Rule {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }

    /// Validate the rule for consistency.
    ///
    /// # Returns
    /// `true` if the rule is valid, `false` otherwise.
    pub fn validate(&self) -> bool {
        // Check interval validity
        if self.a >= self.b {
            return false;
        }

        // Check array lengths
        if self.x.len() != self.w.len() {
            return false;
        }

        if self.x.len() != self.x_forward.len() || self.x.len() != self.x_backward.len() {
            return false;
        }

        // Check that all points are within [a, b]
        for &xi in self.x.iter() {
            if xi < self.a || xi > self.b {
                return false;
            }
        }

        // Check that points are sorted
        for i in 1..self.x.len() {
            if self.x[i] <= self.x[i - 1] {
                return false;
            }
        }

        // Check x_forward and x_backward consistency
        for i in 0..self.x.len() {
            let expected_forward = self.x[i] - self.a;
            let expected_backward = self.b - self.x[i];

            if (self.x_forward[i] - expected_forward).abs_as_same_type() > T::epsilon() {
                return false;
            }
            if (self.x_backward[i] - expected_backward).abs_as_same_type() > T::epsilon() {
                return false;
            }
        }

        true
    }
}

/// CustomNumeric-based implementation for f64 and Df64 support
impl<T> Rule<T>
where
    T: CustomNumeric,
{
    /// Create a new quadrature rule from points and weights (CustomNumeric version).
    pub fn new_custom(x: Vec<T>, w: Vec<T>, a: T, b: T) -> Self {
        assert_eq!(x.len(), w.len(), "x and w must have the same length");

        let x_forward: Vec<T> = x.iter().map(|&xi| xi - a).collect();
        let x_backward: Vec<T> = x.iter().map(|&xi| b - xi).collect();

        Self {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }

    /// Create a new quadrature rule from vectors (CustomNumeric version).
    pub fn from_vectors_custom(x: Vec<T>, w: Vec<T>, a: T, b: T) -> Self {
        Self::new_custom(x, w, a, b)
    }

    /// Reseat the rule to a new interval [a, b] (CustomNumeric version).
    pub fn reseat_custom(&self, a: T, b: T) -> Self {
        let scaling = (b - a) / (self.b - self.a);
        let midpoint_old = (self.b + self.a) * <T as CustomNumeric>::from_f64_unchecked(0.5);
        let midpoint_new = (b + a) * <T as CustomNumeric>::from_f64_unchecked(0.5);

        // Transform x: scaling * (xi - midpoint_old) + midpoint_new
        let new_x: Vec<T> = self
            .x
            .iter()
            .map(|&xi| scaling * (xi - midpoint_old) + midpoint_new)
            .collect();
        let new_w: Vec<T> = self.w.iter().map(|&wi| wi * scaling).collect();
        let new_x_forward: Vec<T> = self.x_forward.iter().map(|&xi| xi * scaling).collect();
        let new_x_backward: Vec<T> = self.x_backward.iter().map(|&xi| xi * scaling).collect();

        Self {
            x: new_x,
            w: new_w,
            x_forward: new_x_forward,
            x_backward: new_x_backward,
            a,
            b,
        }
    }

    /// Scale the weights by a factor (CustomNumeric version).
    pub fn scale_custom(&self, factor: T) -> Self {
        Self {
            x: self.x.clone(),
            w: self.w.iter().map(|&wi| wi * factor).collect(),
            x_forward: self.x_forward.clone(),
            x_backward: self.x_backward.clone(),
            a: self.a,
            b: self.b,
        }
    }

    /// Validate the rule for consistency (CustomNumeric version).
    pub fn validate_custom(&self) -> bool {
        // Check interval validity
        if self.a >= self.b {
            return false;
        }

        // Check array lengths
        if self.x.len() != self.w.len() {
            return false;
        }

        if self.x.len() != self.x_forward.len() || self.x.len() != self.x_backward.len() {
            return false;
        }

        // Check that all points are within [a, b]
        for &xi in self.x.iter() {
            if xi < self.a || xi > self.b {
                return false;
            }
        }

        // Check that points are sorted
        for i in 1..self.x.len() {
            if self.x[i] <= self.x[i - 1] {
                return false;
            }
        }

        // Check x_forward and x_backward consistency
        for i in 0..self.x.len() {
            let expected_forward = self.x[i] - self.a;
            let expected_backward = self.b - self.x[i];

            if (self.x_forward[i] - expected_forward).abs_as_same_type() > T::epsilon() {
                return false;
            }
            if (self.x_backward[i] - expected_backward).abs_as_same_type() > T::epsilon() {
                return false;
            }
        }

        true
    }
}

/// Df64-specific implementation without ScalarOperand requirement
impl Rule<crate::Df64> {
    /// Create a new quadrature rule from points and weights (Df64 version).
    pub fn new_twofloat(
        x: Vec<crate::Df64>,
        w: Vec<crate::Df64>,
        a: crate::Df64,
        b: crate::Df64,
    ) -> Self {
        assert_eq!(x.len(), w.len(), "x and w must have the same length");

        let x_forward: Vec<crate::Df64> = x.iter().map(|&xi| xi - a).collect();
        let x_backward: Vec<crate::Df64> = x.iter().map(|&xi| b - xi).collect();

        Self {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }

    /// Create a new quadrature rule from vectors (Df64 version).
    pub fn from_vectors_twofloat(
        x: Vec<crate::Df64>,
        w: Vec<crate::Df64>,
        a: crate::Df64,
        b: crate::Df64,
    ) -> Self {
        Self::new_twofloat(x, w, a, b)
    }

    /// Reseat the rule to a new interval [a, b] (Df64 version).
    pub fn reseat_twofloat(&self, a: crate::Df64, b: crate::Df64) -> Self {
        let scaling = (b - a) / (self.b - self.a);
        let midpoint_old =
            (self.b + self.a) * <crate::Df64 as CustomNumeric>::from_f64_unchecked(0.5);
        let midpoint_new = (b + a) * <crate::Df64 as CustomNumeric>::from_f64_unchecked(0.5);

        // Transform x: scaling * (xi - midpoint_old) + midpoint_new
        let new_x: Vec<crate::Df64> = self
            .x
            .iter()
            .map(|&xi| scaling * (xi - midpoint_old) + midpoint_new)
            .collect();
        let new_w: Vec<crate::Df64> = self.w.iter().map(|&wi| wi * scaling).collect();
        let new_x_forward: Vec<crate::Df64> =
            self.x_forward.iter().map(|&xi| xi * scaling).collect();
        let new_x_backward: Vec<crate::Df64> =
            self.x_backward.iter().map(|&xi| xi * scaling).collect();

        Self {
            x: new_x,
            w: new_w,
            x_forward: new_x_forward,
            x_backward: new_x_backward,
            a,
            b,
        }
    }

    /// Scale the weights by a factor (Df64 version).
    pub fn scale_twofloat(&self, factor: crate::Df64) -> Self {
        Self {
            x: self.x.clone(),
            w: self.w.iter().map(|&wi| wi * factor).collect(),
            x_forward: self.x_forward.clone(),
            x_backward: self.x_backward.clone(),
            a: self.a,
            b: self.b,
        }
    }

    /// Validate the rule for consistency (Df64 version).
    pub fn validate_twofloat(&self) -> bool {
        // Check interval validity
        if self.a >= self.b {
            return false;
        }

        // Check array lengths
        if self.x.len() != self.w.len() {
            return false;
        }

        if self.x.len() != self.x_forward.len() || self.x.len() != self.x_backward.len() {
            return false;
        }

        // Check that all points are within [a, b]
        for &xi in self.x.iter() {
            if xi < self.a || xi > self.b {
                return false;
            }
        }

        // Check that points are sorted
        for i in 1..self.x.len() {
            if self.x[i] <= self.x[i - 1] {
                return false;
            }
        }

        // Check x_forward and x_backward consistency
        for i in 0..self.x.len() {
            let expected_forward = self.x[i] - self.a;
            let expected_backward = self.b - self.x[i];

            if (self.x_forward[i] - expected_forward).abs() > crate::Df64::epsilon() {
                return false;
            }
            if (self.x_backward[i] - expected_backward).abs() > crate::Df64::epsilon() {
                return false;
            }
        }

        true
    }
}

/// Compute Gauss-Legendre quadrature nodes and weights using Newton's method.
///
/// This is a simplified implementation of the Gauss-Legendre quadrature rule.
/// For production use, a more sophisticated algorithm would be preferred.
fn gauss_legendre_nodes_weights<T>(n: usize) -> (Vec<T>, Vec<T>)
where
    T: CustomNumeric + Copy + Debug + std::fmt::Display + 'static,
{
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    if n == 1 {
        return (
            vec![<T as CustomNumeric>::from_f64_unchecked(0.0)],
            vec![<T as CustomNumeric>::from_f64_unchecked(2.0)],
        );
    }

    let mut x = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);

    // Use Newton's method to find roots of Legendre polynomial
    let m = n.div_ceil(2);

    // Use high-precision constants via CustomNumeric trait
    let pi = T::pi();

    for i in 0..m {
        // Convert integers directly to avoid f64 intermediate
        let i_val = <T as CustomNumeric>::from_f64_unchecked(i as f64);
        let n_val = <T as CustomNumeric>::from_f64_unchecked(n as f64);
        let three_quarters = <T as CustomNumeric>::from_f64_unchecked(0.75);
        let half = <T as CustomNumeric>::from_f64_unchecked(0.5);

        // Initial guess using Chebyshev nodes
        let mut z = (pi * (i_val + three_quarters) / (n_val + half)).cos();

        // Newton's method to refine the root
        for _ in 0..10 {
            let (p0, p1) = legendre_polynomial_and_derivative(n, z);
            if p0.abs_as_same_type() < T::epsilon() {
                break;
            }
            z = z - p0 / p1;
        }

        // Compute weight using high-precision constants
        let two = <T as CustomNumeric>::from_f64_unchecked(2.0);
        let one = <T as CustomNumeric>::from_f64_unchecked(1.0);
        let (_, p1) = legendre_polynomial_and_derivative(n, z);
        let weight = two / ((one - z * z) * p1 * p1);

        x.push(-z);
        w.push(weight);

        if i != n - 1 - i {
            x.push(z);
            w.push(weight);
        }
    }

    // Sort by x values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let sorted_x: Vec<T> = indices.iter().map(|&i| x[i]).collect();
    let sorted_w: Vec<T> = indices.iter().map(|&i| w[i]).collect();

    (sorted_x, sorted_w)
}

/// Compute Legendre polynomial P_n(x) and its derivative using recurrence relation.
fn legendre_polynomial_and_derivative<T>(n: usize, x: T) -> (T, T)
where
    T: CustomNumeric + Copy + Debug + std::fmt::Display + 'static,
{
    if n == 0 {
        return (
            <T as CustomNumeric>::from_f64_unchecked(1.0),
            <T as CustomNumeric>::from_f64_unchecked(0.0),
        );
    }

    if n == 1 {
        return (x, <T as CustomNumeric>::from_f64_unchecked(1.0));
    }

    let mut p0 = <T as CustomNumeric>::from_f64_unchecked(1.0);
    let mut p1 = x;
    let mut dp0 = <T as CustomNumeric>::from_f64_unchecked(0.0);
    let mut dp1 = <T as CustomNumeric>::from_f64_unchecked(1.0);

    for k in 2..=n {
        let k_f = <T as CustomNumeric>::from_f64_unchecked(k as f64);
        let k1_f = <T as CustomNumeric>::from_f64_unchecked((k - 1) as f64);
        let _k2_f = <T as CustomNumeric>::from_f64_unchecked((k - 2) as f64);

        let p2 = ((<T as CustomNumeric>::from_f64_unchecked(2.0) * k1_f
            + <T as CustomNumeric>::from_f64_unchecked(1.0))
            * x
            * p1
            - k1_f * p0)
            / k_f;
        let dp2 = ((<T as CustomNumeric>::from_f64_unchecked(2.0) * k1_f
            + <T as CustomNumeric>::from_f64_unchecked(1.0))
            * (p1 + x * dp1)
            - k1_f * dp0)
            / k_f;

        p0 = p1;
        p1 = p2;
        dp0 = dp1;
        dp1 = dp2;
    }

    (p1, dp1)
}

/// Create a Gauss-Legendre quadrature rule with n points on [-1, 1].
///
/// # Arguments
/// * `n` - Number of quadrature points
///
/// # Returns
/// A Gauss-Legendre quadrature rule
pub fn legendre<T>(n: usize) -> Rule<T>
where
    T: CustomNumeric + Copy + Debug + std::fmt::Display + 'static,
{
    if n == 0 {
        return Rule::empty();
    }

    let (x, w) = gauss_legendre_nodes_weights(n);

    Rule::from_vectors(
        x,
        w,
        <T as CustomNumeric>::from_f64_unchecked(-1.0),
        <T as CustomNumeric>::from_f64_unchecked(1.0),
    )
}

/// Compute Gauss-Legendre quadrature nodes and weights using CustomNumeric
fn gauss_legendre_nodes_weights_custom<T>(n: usize) -> (Vec<T>, Vec<T>)
where
    T: CustomNumeric,
{
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    if n == 1 {
        return (
            vec![<T as CustomNumeric>::from_f64_unchecked(0.0)],
            vec![<T as CustomNumeric>::from_f64_unchecked(2.0)],
        );
    }

    let mut x = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);

    // Use Newton's method to find roots of Legendre polynomial
    let m = n.div_ceil(2);
    let pi = <T as CustomNumeric>::from_f64_unchecked(std::f64::consts::PI);

    for i in 0..m {
        // Initial guess using Chebyshev nodes
        // Note: Df64's cos() has only f64-level precision (~15-16 digits), not the full
        // theoretical 30-digit precision. This limits Df64 interpolation accuracy to ~1e-16,
        // not the 1e-30 that might be theoretically possible with perfect double-double arithmetic.
        let mut z = (pi * <T as CustomNumeric>::from_f64_unchecked(i as f64 + 0.75)
            / <T as CustomNumeric>::from_f64_unchecked(n as f64 + 0.5))
        .cos();

        // Newton's method to refine the root
        for _ in 0..10 {
            let (p0, p1) = legendre_polynomial_and_derivative_custom(n, z);
            if p0.abs_as_same_type() < T::epsilon() {
                break;
            }
            z = z - p0 / p1;
        }

        // Compute weight
        let (_, p1) = legendre_polynomial_and_derivative_custom(n, z);
        let weight = <T as CustomNumeric>::from_f64_unchecked(2.0)
            / ((<T as CustomNumeric>::from_f64_unchecked(1.0) - z * z) * p1 * p1);

        x.push(-z);
        w.push(weight);

        if i != n - 1 - i {
            x.push(z);
            w.push(weight);
        }
    }

    // Sort by x values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let sorted_x: Vec<T> = indices.iter().map(|&i| x[i]).collect();
    let sorted_w: Vec<T> = indices.iter().map(|&i| w[i]).collect();

    (sorted_x, sorted_w)
}

/// Compute Legendre polynomial P_n(x) and its derivative using CustomNumeric
fn legendre_polynomial_and_derivative_custom<T>(n: usize, x: T) -> (T, T)
where
    T: CustomNumeric,
{
    if n == 0 {
        return (
            <T as CustomNumeric>::from_f64_unchecked(1.0),
            <T as CustomNumeric>::from_f64_unchecked(0.0),
        );
    }

    if n == 1 {
        return (x, <T as CustomNumeric>::from_f64_unchecked(1.0));
    }

    let mut p0 = <T as CustomNumeric>::from_f64_unchecked(1.0);
    let mut p1 = x;
    let mut dp0 = <T as CustomNumeric>::from_f64_unchecked(0.0);
    let mut dp1 = <T as CustomNumeric>::from_f64_unchecked(1.0);

    for k in 2..=n {
        let k_f = <T as CustomNumeric>::from_f64_unchecked(k as f64);
        let k1_f = <T as CustomNumeric>::from_f64_unchecked((k - 1) as f64);
        let _k2_f = <T as CustomNumeric>::from_f64_unchecked((k - 2) as f64);

        let two = <T as CustomNumeric>::from_f64_unchecked(2.0);
        let one = <T as CustomNumeric>::from_f64_unchecked(1.0);

        let p2 = ((two * k1_f + one) * x * p1 - k1_f * p0) / k_f;
        let dp2 = ((two * k1_f + one) * (p1 + x * dp1) - k1_f * dp0) / k_f;

        p0 = p1;
        p1 = p2;
        dp0 = dp1;
        dp1 = dp2;
    }

    (p1, dp1)
}

/// Create a Gauss-Legendre quadrature rule with n points on [-1, 1] (CustomNumeric version).
pub fn legendre_custom<T>(n: usize) -> Rule<T>
where
    T: CustomNumeric,
{
    if n == 0 {
        return Rule::new_custom(
            vec![],
            vec![],
            <T as CustomNumeric>::from_f64_unchecked(-1.0),
            <T as CustomNumeric>::from_f64_unchecked(1.0),
        );
    }

    let (x, w) = gauss_legendre_nodes_weights_custom(n);

    Rule::from_vectors_custom(
        x,
        w,
        <T as CustomNumeric>::from_f64_unchecked(-1.0),
        <T as CustomNumeric>::from_f64_unchecked(1.0),
    )
}

/// Create a Gauss-Legendre quadrature rule with n points on [-1, 1] (Df64 version).
pub fn legendre_twofloat(n: usize) -> Rule<crate::Df64> {
    if n == 0 {
        return Rule::new_twofloat(
            vec![],
            vec![],
            <crate::Df64 as CustomNumeric>::from_f64_unchecked(-1.0),
            <crate::Df64 as CustomNumeric>::from_f64_unchecked(1.0),
        );
    }

    let mut x: Vec<crate::Df64> = vec![crate::Df64::ZERO; n];
    let mut w: Vec<crate::Df64> = vec![crate::Df64::ZERO; n];
    xprec::gauss::gauss_legendre(&mut x, &mut w);

    Rule::from_vectors_twofloat(
        x,
        w,
        <crate::Df64 as CustomNumeric>::from_f64_unchecked(-1.0),
        <crate::Df64 as CustomNumeric>::from_f64_unchecked(1.0),
    )
}

/// Create Legendre Vandermonde matrix for polynomial interpolation
///
/// # Arguments
/// * `x` - Points where polynomials are evaluated
/// * `degree` - Maximum degree of Legendre polynomials
///
/// # Returns
/// Matrix V where V[i,j] = P_j(x_i), with P_j being the j-th Legendre polynomial
pub fn legendre_vandermonde<T: CustomNumeric>(x: &[T], degree: usize) -> mdarray::DTensor<T, 2> {
    use mdarray::DTensor;

    let n = x.len();
    let mut v = DTensor::<T, 2>::from_elem([n, degree + 1], T::zero());

    // First column is all ones (P_0(x) = 1)
    for i in 0..n {
        v[[i, 0]] = T::from_f64_unchecked(1.0);
    }

    // Second column is x (P_1(x) = x)
    if degree > 0 {
        for i in 0..n {
            v[[i, 1]] = x[i];
        }
    }

    // Recurrence relation: P_n(x) = ((2n-1)x*P_{n-1}(x) - (n-1)*P_{n-2}(x)) / n
    for j in 2..=degree {
        for i in 0..n {
            let n_f64 = j as f64;
            let term1 = T::from_f64_unchecked(2.0 * n_f64 - 1.0) * x[i] * v[[i, j - 1]];
            let term2 = T::from_f64_unchecked(n_f64 - 1.0) * v[[i, j - 2]];
            v[[i, j]] = (term1 - term2) / T::from_f64_unchecked(n_f64);
        }
    }

    v
}

/// Generic Legendre Gauss quadrature rule for CustomNumeric types
pub fn legendre_generic<T: CustomNumeric + 'static>(n: usize) -> Rule<T> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // For f64, use the existing legendre function
        let rule_f64 = legendre::<f64>(n);
        Rule::new(
            rule_f64
                .x
                .iter()
                .map(|&x| T::from_f64_unchecked(x))
                .collect(),
            rule_f64
                .w
                .iter()
                .map(|&w| T::from_f64_unchecked(w))
                .collect(),
            T::from_f64_unchecked(rule_f64.a),
            T::from_f64_unchecked(rule_f64.b),
        )
    } else {
        // For Df64, use legendre_twofloat
        let rule_tf = legendre_twofloat(n);
        Rule::new(
            rule_tf.x.iter().map(|&x| T::convert_from(x)).collect(),
            rule_tf.w.iter().map(|&w| T::convert_from(w)).collect(),
            T::from_f64_unchecked(rule_tf.a.into()),
            T::from_f64_unchecked(rule_tf.b.into()),
        )
    }
}

#[cfg(test)]
#[path = "gauss_tests.rs"]
mod tests;
