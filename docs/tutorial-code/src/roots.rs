//! One-dimensional root finding.
//!
//! The tutorial needs exactly one thing here: given a continuous function that
//! changes sign on a bracket, find where it crosses zero. Brent's method does
//! that with the reliability of bisection and the speed of interpolation, and
//! it is short enough to keep in the tutorial rather than pull a crate in for.

/// Why a root search stopped without a root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootError {
    /// `f(a)` and `f(b)` have the same sign, so the bracket may hold no root
    /// or an even number of them. Either way there is nothing to bisect.
    NotBracketed,
    /// The iteration limit was reached before the bracket became small enough.
    NotConverged { iterations: usize },
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotBracketed => write!(f, "f(a) and f(b) have the same sign"),
            Self::NotConverged { iterations } => {
                write!(f, "no root after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for RootError {}

/// Bisection: halve the bracket until it is narrower than `xtol`.
///
/// Slow but unconditionally reliable; the tutorial uses it to cross-check
/// [`brent`] in tests.
pub fn bisect<F>(
    mut f: F,
    mut a: f64,
    mut b: f64,
    xtol: f64,
    max_iter: usize,
) -> Result<f64, RootError>
where
    F: FnMut(f64) -> f64,
{
    let mut fa = f(a);
    let fb = f(b);
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(RootError::NotBracketed);
    }

    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        if (b - a).abs() <= xtol {
            return Ok(m);
        }
        let fm = f(m);
        if fm == 0.0 {
            return Ok(m);
        }
        if fm.signum() == fa.signum() {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }
    Err(RootError::NotConverged {
        iterations: max_iter,
    })
}

/// Brent's method on the bracket `[a, b]`, where `f(a)` and `f(b)` must have
/// opposite signs.
///
/// Inverse quadratic interpolation where it helps, a secant step where it does
/// not, and a bisection step whenever either would leave the bracket or stop
/// shrinking it — so the bracket always at least halves every two iterations.
pub fn brent<F>(mut f: F, a: f64, b: f64, xtol: f64, max_iter: usize) -> Result<f64, RootError>
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut fa) = (a, f(a));
    let (mut b, mut fb) = (b, f(b));
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(RootError::NotBracketed);
    }

    // `b` is kept as the better of the two endpoints, `a` as the contrapoint.
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    // The previous contrapoint, and the length of the step before last, which
    // together decide whether an interpolated step is making enough progress.
    let (mut c, mut fc) = (a, fa);
    let mut previous_step = b - a;
    let mut step_before_last = previous_step;

    for _ in 0..max_iter {
        if fc.abs() < fb.abs() {
            // The contrapoint became the better guess; rotate.
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tolerance = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let bisection_step = 0.5 * (c - b);
        if bisection_step.abs() <= tolerance || fb == 0.0 {
            return Ok(b);
        }

        let mut step = bisection_step;
        if step_before_last.abs() >= tolerance && fa.abs() > fb.abs() {
            // An interpolated step is worth trying.
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                // Only two distinct points: secant.
                (2.0 * bisection_step * s, 1.0 - s)
            } else {
                // Three distinct points: inverse quadratic interpolation.
                let (q0, r) = (fa / fc, fb / fc);
                (
                    s * (2.0 * bisection_step * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            // Accept it only if it stays inside the bracket and shrinks the
            // step by at least a third; otherwise fall back to bisection.
            let within_bracket = 2.0 * p < (3.0 * bisection_step * q - (tolerance * q).abs());
            let shrinking = p < (0.5 * step_before_last * q).abs();
            if within_bracket && shrinking {
                step = p / q;
            }
        }

        step_before_last = previous_step;
        previous_step = step;

        a = b;
        fa = fb;
        b += if step.abs() > tolerance {
            step
        } else {
            tolerance.copysign(bisection_step)
        };
        fb = f(b);

        if (fb > 0.0) == (fc > 0.0) {
            // `b` moved to the side `c` is on, so the old `a` becomes the
            // contrapoint and the bracket stays valid.
            c = a;
            fc = fa;
            step_before_last = b - a;
            previous_step = step_before_last;
        }
    }

    Err(RootError::NotConverged {
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_and_bisect_agree_on_a_smooth_root() {
        // cos(x) - x has its only root near 0.7390851332151607.
        let f = |x: f64| x.cos() - x;
        let by_brent = brent(f, 0.0, 2.0, 1e-14, 200).expect("the root is bracketed");
        let by_bisection = bisect(f, 0.0, 2.0, 1e-14, 200).expect("the root is bracketed");
        assert!(
            (by_brent - by_bisection).abs() < 1e-12,
            "{by_brent} {by_bisection}"
        );
        assert!(f(by_brent).abs() < 1e-13);
    }

    #[test]
    fn a_steep_root_is_still_found() {
        // x^3 has a triple root at zero, where interpolation is useless and
        // the method has to fall back to bisection.
        let f = |x: f64| x * x * x;
        let root = brent(f, -1.0, 2.0, 1e-12, 200).expect("the root is bracketed");
        assert!(root.abs() < 1e-4, "{root}");
    }

    #[test]
    fn an_endpoint_that_is_already_a_root_is_returned_as_is() {
        assert_eq!(brent(|x: f64| x - 1.0, 1.0, 3.0, 1e-12, 100), Ok(1.0));
    }

    #[test]
    fn a_bracket_without_a_sign_change_is_refused() {
        let f = |x: f64| x * x + 1.0;
        assert_eq!(
            brent(f, -1.0, 1.0, 1e-12, 100),
            Err(RootError::NotBracketed)
        );
        assert_eq!(
            bisect(f, -1.0, 1.0, 1e-12, 100),
            Err(RootError::NotBracketed)
        );
    }
}
