//! The semicircular spectral function, and its IR coefficients.
//!
//! Two of the examples use the same model — the semicircle of full bandwidth
//! 2 — so the quadrature that turns it into `ρₗ` lives here rather than in
//! either of them.

use sparse_ir::{CentrosymmKernel, FiniteTempBasis, KernelProperties, StatisticsType};
use std::f64::consts::PI;

use crate::integrate_segments;

/// `ρ(ω) = (2/π) √(1 − ω²)` for `|ω| < 1`, and zero outside.
pub fn semicircle(omega: f64) -> f64 {
    if omega.abs() < 1.0 {
        (2.0 / PI) * (1.0 - omega * omega).sqrt()
    } else {
        0.0
    }
}

/// `ρₗ = ∫ dω vₗ(ω) ρ(ω)` for the semicircular `ρ`, to machine precision.
///
/// `ρ` has a square-root edge at `ω = ±1`, where plain quadrature converges
/// slowly. The substitution `ω = sin θ` removes it: `ρ(ω) dω` becomes
/// `(2/π) cos²θ dθ`, so on every segment the integrand is a polynomial in
/// `sin θ` times `cos²θ`, which a Gauss-Legendre rule of modest order
/// integrates to the last bit.
pub fn semicircle_overlaps<K, S>(basis: &FiniteTempBasis<K, S>) -> Vec<f64>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let v = basis.v();

    // Only the part of the basis' ω range where ρ is nonzero contributes.
    let mut edges: Vec<f64> = v
        .get_knots(None)
        .into_iter()
        .filter(|&omega| omega.abs() < 1.0)
        .collect();
    edges.insert(0, -1.0);
    edges.push(1.0);
    let theta_edges: Vec<f64> = edges.iter().map(|&omega| omega.asin()).collect();

    // The polynomials have `polyorder` coefficients per segment, and cos²θ
    // adds two more degrees; an order well above half of that is exact.
    let order = v.get_polyorder() + 8;

    (0..basis.size())
        .map(|l| {
            let poly = &v[l];
            integrate_segments(
                |theta| {
                    let cos = theta.cos();
                    poly.evaluate(theta.sin()) * (2.0 / PI) * cos * cos
                },
                &theta_edges,
                order,
            )
        })
        .collect()
}

/// `Gₗ = −sₗ ρₗ` for the semicircular `ρ`.
pub fn semicircle_coefficients<K, S>(basis: &FiniteTempBasis<K, S>) -> (Vec<f64>, Vec<f64>)
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let rho_l = semicircle_overlaps(basis);
    let g_l = basis
        .s()
        .iter()
        .zip(&rho_l)
        .map(|(s, rho)| -s * rho)
        .collect();
    (rho_l, g_l)
}
