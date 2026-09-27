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
    shifted_semicircle(omega, 0.0, 1.0, 1.0)
}

/// The same semicircle of total weight `weight`, centred at `center`, with
/// half-bandwidth `half_width` — the model [`shifted_semicircle_overlaps`]
/// integrates against the basis.
pub fn shifted_semicircle(omega: f64, center: f64, half_width: f64, weight: f64) -> f64 {
    let x = (omega - center) / half_width;
    if x.abs() < 1.0 {
        weight * (2.0 / (PI * half_width)) * (1.0 - x * x).sqrt()
    } else {
        0.0
    }
}

/// `ρₗ = ∫ dω vₗ(ω) ρ(ω)` for the semicircular `ρ`, to machine precision.
pub fn semicircle_overlaps<K, S>(basis: &FiniteTempBasis<K, S>) -> Vec<f64>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    shifted_semicircle_overlaps(basis, 0.0, 1.0, 1.0)
}

/// `ρₗ = ∫ dω vₗ(ω) ρ(ω)` for a semicircle of total weight `weight` centred at
/// `center` with half-bandwidth `half_width`,
///
/// ```text
/// ρ(ω) = weight · (2 / (π · half_width)) · √(1 − ((ω − center)/half_width)²),
/// ```
///
/// to machine precision.
///
/// `ρ` has a square-root edge at each end, where plain quadrature converges
/// slowly. The substitution `ω = center + half_width · sin θ` removes it:
/// `ρ(ω) dω` becomes `weight · (2/π) cos²θ dθ`, with the half-width cancelling,
/// so on every segment the integrand is a polynomial in `sin θ` times `cos²θ`,
/// which a Gauss-Legendre rule of modest order integrates to the last bit.
///
/// A spectral function built from several semicircles — a band split into two
/// by a gap, say — is the sum of the results of one call per piece.
pub fn shifted_semicircle_overlaps<K, S>(
    basis: &FiniteTempBasis<K, S>,
    center: f64,
    half_width: f64,
    weight: f64,
) -> Vec<f64>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    assert!(
        half_width > 0.0,
        "a semicircle needs a positive half-width, got {half_width}"
    );
    let v = basis.v();

    // Only the part of the basis' ω range where ρ is nonzero contributes.
    let mut edges: Vec<f64> = v
        .get_knots(None)
        .into_iter()
        .filter(|&omega| (omega - center).abs() < half_width)
        .collect();
    edges.insert(0, center - half_width);
    edges.push(center + half_width);
    let theta_edges: Vec<f64> = edges
        .iter()
        .map(|&omega| ((omega - center) / half_width).asin())
        .collect();

    // The polynomials have `polyorder` coefficients per segment, and cos²θ
    // adds two more degrees; an order well above half of that is exact.
    let order = v.get_polyorder() + 8;

    (0..basis.size())
        .map(|l| {
            let poly = &v[l];
            integrate_segments(
                |theta| {
                    let cos = theta.cos();
                    poly.evaluate(center + half_width * theta.sin())
                        * weight
                        * (2.0 / PI)
                        * cos
                        * cos
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
