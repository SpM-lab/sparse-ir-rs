//! Gauss-Legendre quadrature over a set of segments.
//!
//! Several of the tutorials need an integral that the library itself has no
//! reason to expose: the overlap of a basis function with a spectral function
//! the reader has chosen. A basis function is a polynomial on each of its
//! segments, so a Gauss-Legendre rule of high enough order is exact on every
//! segment as long as the *other* factor is smooth there too. Making the other
//! factor smooth is the caller's job — see the substitution the sparse
//! sampling example uses to get rid of the semicircle's square-root edge.

use sparse_ir::legendre;

/// Integrates `f` over `[edges[0], edges[last]]` with an `order`-point
/// Gauss-Legendre rule on each segment between consecutive edges.
///
/// Panics if fewer than two edges are given, if they are not increasing, or if
/// `order` is zero — each of which would silently return a meaningless number.
pub fn integrate_segments<F>(mut f: F, edges: &[f64], order: usize) -> f64
where
    F: FnMut(f64) -> f64,
{
    assert!(edges.len() >= 2, "need at least two edges, got {edges:?}");
    assert!(order > 0, "a quadrature rule needs at least one point");
    assert!(
        edges.windows(2).all(|pair| pair[0] < pair[1]),
        "the edges must be strictly increasing, got {edges:?}"
    );

    let rule = legendre::<f64>(order);
    let mut total = 0.0;
    for pair in edges.windows(2) {
        let segment = rule.reseat(pair[0], pair[1]);
        for (&x, &w) in segment.x().iter().zip(segment.w()) {
            total += w * f(x);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Gauss-Legendre rule of order `n` is exact for polynomials of degree
    /// up to `2n − 1`, segments or no segments.
    #[test]
    fn a_polynomial_is_integrated_exactly() {
        // ∫₀² x⁵ dx = 2⁶/6 = 32/3
        let value = integrate_segments(|x| x.powi(5), &[0.0, 0.7, 1.3, 2.0], 4);
        assert!((value - 32.0 / 3.0).abs() < 1e-13, "{value}");
    }

    #[test]
    fn a_smooth_function_converges_with_the_order() {
        // ∫₋₁¹ exp(x) dx = e − 1/e
        let exact = std::f64::consts::E - std::f64::consts::E.recip();
        let coarse = integrate_segments(f64::exp, &[-1.0, 1.0], 4);
        let fine = integrate_segments(f64::exp, &[-1.0, 1.0], 12);
        assert!((fine - exact).abs() < (coarse - exact).abs());
        assert!((fine - exact).abs() < 1e-15, "{fine}");
    }
}
