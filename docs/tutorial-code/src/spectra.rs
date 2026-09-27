//! Model spectral functions shared by more than one example.

/// The three-Gaussian spectral function used by the transformation and
/// sparse-modeling examples,
///
/// ```text
/// ρ(ω) = 0.2 g(ω; 0, 0.15) + 0.4 g(ω; 1, 0.8) + 0.4 g(ω; −1, 0.8),
/// g(ω; μ, σ) = exp(−((ω − μ)/σ)²) / (√π σ).
/// ```
///
/// Each Gaussian integrates to one, so the weights are the spectral weights
/// and `∫ dω ρ(ω) = 1`. The narrow peak at `ω = 0` is what makes the
/// continuation problem hard: it is the feature that disappears first when the
/// data are noisy.
pub fn three_gaussians(omega: f64) -> f64 {
    let gaussian = |mu: f64, sigma: f64| {
        (-((omega - mu) / sigma).powi(2)).exp() / (std::f64::consts::PI.sqrt() * sigma)
    };
    0.2 * gaussian(0.0, 0.15) + 0.4 * gaussian(1.0, 0.8) + 0.4 * gaussian(-1.0, 0.8)
}
