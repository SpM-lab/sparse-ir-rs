//! Oscillatory quadrature for the contour integrals.
//!
//! The reference evaluates `∫_a^b f(θ) sin(ωθ) dθ` and the cosine
//! counterpart with QUADPACK's QAWO (`scipy.integrate.quad` with
//! `weight="sin"/"cos"`) to `epsabs = epsrel = err`. Here `f` is smooth
//! (an exponential sum, or a sum of poles off the contour), and composite
//! Gauss–Legendre quadrature with doubling of the number of panels until two
//! successive results agree to that tolerance is used instead.

use num_complex::Complex;
use std::sync::OnceLock;

type C64 = Complex<f64>;

const ORDER: usize = 32;
const MAX_PANELS: usize = 1 << 16;

/// Gauss–Legendre nodes and weights of order [`ORDER`] on `[-1, 1]`.
fn gauss_legendre() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| {
        let n = ORDER;
        let mut x = vec![0.0; n];
        let mut w = vec![0.0; n];
        for i in 0..n.div_ceil(2) {
            let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, z);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
                let dz = p1 / dp;
                z -= dz;
                if dz.abs() < 1e-16 {
                    break;
                }
            }
            x[i] = -z;
            x[n - 1 - i] = z;
            let wi = 2.0 / ((1.0 - z * z) * dp * dp);
            w[i] = wi;
            w[n - 1 - i] = wi;
        }
        (x, w)
    })
}

fn composite(f: &dyn Fn(f64) -> C64, a: f64, b: f64, panels: usize) -> C64 {
    let (x, w) = gauss_legendre();
    let h = (b - a) / panels as f64;
    let mut sum = C64::new(0.0, 0.0);
    for p in 0..panels {
        let mid = a + (p as f64 + 0.5) * h;
        let mut s = C64::new(0.0, 0.0);
        for (xi, wi) in x.iter().zip(w) {
            s += f(mid + 0.5 * h * xi) * *wi;
        }
        sum += s * (0.5 * h);
    }
    sum
}

/// `∫_a^b f(θ) sin(ωθ) dθ` (`sine = true`) or `∫_a^b f(θ) cos(ωθ) dθ`, to
/// the absolute or relative tolerance `eps`.
pub(crate) fn oscillatory(
    f: &dyn Fn(f64) -> C64,
    a: f64,
    b: f64,
    omega: f64,
    sine: bool,
    eps: f64,
) -> C64 {
    let g = |t: f64| {
        let wt = if sine {
            (omega * t).sin()
        } else {
            (omega * t).cos()
        };
        f(t) * wt
    };
    let mut panels = ((omega * (b - a) / std::f64::consts::PI).ceil() as usize).max(1);
    let mut prev = composite(&g, a, b, panels);
    while panels < MAX_PANELS {
        panels *= 2;
        let next = composite(&g, a, b, panels);
        if (next - prev).norm() <= eps.max(eps * next.norm()) {
            return next;
        }
        prev = next;
    }
    prev
}
