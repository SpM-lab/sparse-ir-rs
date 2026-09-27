//! Complete elliptic integrals of the first and second kind.
//!
//! The orbital magnetic susceptibility of the square lattice has a closed form
//! at zero temperature that the tutorial compares its numbers against, and
//! that form is written in `K` and `E`. Two functions is not worth a
//! dependency, and the arithmetic-geometric mean gives both to the last few
//! bits in a dozen iterations.
//!
//! The argument is the *parameter* `m = k²`, matching `scipy.special.ellipk`
//! and `scipy.special.ellipe` — not the modulus `k`, which some references
//! and some libraries use instead.

use std::f64::consts::FRAC_PI_2;

/// Complete elliptic integral of the first kind,
/// `K(m) = ∫₀^{π/2} dθ / √(1 − m sin²θ)`.
///
/// `K(1)` is infinite and comes back as `f64::INFINITY`. Panics outside
/// `[0, 1]`, where the AGM below does not apply.
pub fn ellipk(m: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&m),
        "the parameter of a complete elliptic integral must lie in [0, 1], got {m}"
    );
    if m == 1.0 {
        return f64::INFINITY;
    }
    FRAC_PI_2 / agm(1.0, (1.0 - m).sqrt())
}

/// Complete elliptic integral of the second kind,
/// `E(m) = ∫₀^{π/2} √(1 − m sin²θ) dθ`.
///
/// Panics outside `[0, 1]`.
pub fn ellipe(m: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&m),
        "the parameter of a complete elliptic integral must lie in [0, 1], got {m}"
    );
    if m == 1.0 {
        return 1.0;
    }

    // Legendre's relation in AGM form: with a₀ = 1, b₀ = √(1 − m) and
    // cₙ₊₁ = (aₙ − bₙ)/2, one has E = K (1 − Σₙ 2ⁿ⁻¹ cₙ²), the sum starting
    // at c₀ = √m.
    let (mut a, mut b) = (1.0, (1.0 - m).sqrt());
    let mut sum = 0.5 * m;
    let mut weight = 1.0;
    for _ in 0..MAX_ITERATIONS {
        if (a - b).abs() <= f64::EPSILON * a {
            break;
        }
        let c = 0.5 * (a - b);
        let next_b = (a * b).sqrt();
        a = 0.5 * (a + b);
        b = next_b;
        sum += weight * c * c;
        weight *= 2.0;
    }
    (FRAC_PI_2 / a) * (1.0 - sum)
}

const MAX_ITERATIONS: usize = 60;

/// Arithmetic-geometric mean, which converges quadratically.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..MAX_ITERATIONS {
        if (a - b).abs() <= f64::EPSILON * a {
            break;
        }
        let next_b = (a * b).sqrt();
        a = 0.5 * (a + b);
        b = next_b;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.special.ellipk` and `ellipe`, printed with `repr(float(...))`.
    /// `m = 0` is left to `the_endpoints_are_the_known_ones`, where the value
    /// can be written as `FRAC_PI_2` rather than as its decimal expansion.
    const REFERENCE: [(f64, f64, f64); 7] = [
        (0.1, 1.6124413487202192, 1.5307576368977631),
        (0.25, 1.685750354812596, 1.4674622093394272),
        (0.5, 1.8540746773013719, 1.3506438810476755),
        (0.75, 2.156515647499643, 1.2110560275684594),
        (0.9, 2.5780921133481733, 1.1047747327040733),
        (0.99, 3.6956373629898747, 1.015993545025224),
        (0.999999, 8.294051463601061, 1.0000038970261722),
    ];

    #[test]
    fn both_kinds_match_scipy() {
        for (m, k, e) in REFERENCE {
            assert!(
                (ellipk(m) - k).abs() <= 1e-14 * k,
                "K({m}) = {} but scipy says {k}",
                ellipk(m)
            );
            assert!(
                (ellipe(m) - e).abs() <= 1e-14 * e,
                "E({m}) = {} but scipy says {e}",
                ellipe(m)
            );
        }
    }

    #[test]
    fn the_endpoints_are_the_known_ones() {
        assert_eq!(ellipk(1.0), f64::INFINITY);
        assert_eq!(ellipe(1.0), 1.0);
        assert_eq!(ellipk(0.0), FRAC_PI_2);
        assert_eq!(ellipe(0.0), FRAC_PI_2);
    }

    #[test]
    #[should_panic(expected = "must lie in [0, 1]")]
    fn a_negative_parameter_is_rejected() {
        ellipk(-0.1);
    }
}
