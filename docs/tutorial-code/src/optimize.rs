//! FISTA, the proximal-gradient method the sparse-modeling example uses.
//!
//! FISTA minimises `f(x) + g(x)` where `f` is smooth with an `L`-Lipschitz
//! gradient and `g` is anything whose proximal operator is cheap. The sparse
//! modeling example takes `f(x) = ½‖Ax − b‖²` and `g(x) = λ‖x‖₁`, whose
//! proximal operator is [`soft_threshold`].
//!
//! The iteration is deterministic: from the same starting point it visits the
//! same iterates and stops at the same step, which is what lets the Rust and
//! the Python version of the example be compared to each other rather than
//! only to a loose tolerance.

/// How a [`fista`] run ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FistaReport {
    pub iterations: usize,
    /// Whether the last step moved the solution by less than the tolerance.
    pub converged: bool,
    /// `‖xₖ − xₖ₋₁‖∞` at the last step, relative to `‖xₖ‖∞`.
    pub last_relative_change: f64,
}

/// Minimises `f + g` from `x0`, in place.
///
/// * `lipschitz` — an upper bound on the Lipschitz constant of `∇f`. For
///   `½‖Ax − b‖²` that is the largest eigenvalue of `AᵀA`, i.e. the square of
///   the largest singular value of `A`. Too large only slows convergence; too
///   small makes the iteration diverge.
/// * `gradient` — writes `∇f(x)` of its first argument into its second.
/// * `prox` — applies the proximal operator of `step · g` to its first
///   argument in place, where `step` is `1 / lipschitz`.
///
/// Stops when the relative change of the iterate falls below `tol`, or after
/// `max_iter` iterations. The caller decides whether a run that did not
/// converge is a failure; the tutorial's examples assert that it did.
pub fn fista<G, P>(
    x: &mut [f64],
    lipschitz: f64,
    mut gradient: G,
    mut prox: P,
    max_iter: usize,
    tol: f64,
) -> FistaReport
where
    G: FnMut(&[f64], &mut [f64]),
    P: FnMut(&mut [f64], f64),
{
    assert!(
        lipschitz > 0.0 && lipschitz.is_finite(),
        "the Lipschitz bound must be a positive finite number, got {lipschitz}"
    );
    let step = 1.0 / lipschitz;

    let mut previous = x.to_vec();
    let mut momentum_point = x.to_vec();
    let mut grad = vec![0.0; x.len()];
    let mut t = 1.0_f64;

    let mut report = FistaReport {
        iterations: 0,
        converged: false,
        last_relative_change: f64::INFINITY,
    };

    for iteration in 1..=max_iter {
        gradient(&momentum_point, &mut grad);
        for (xi, (yi, gi)) in x.iter_mut().zip(momentum_point.iter().zip(&grad)) {
            *xi = yi - step * gi;
        }
        prox(x, step);

        let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let momentum = (t - 1.0) / t_next;
        for (yi, (xi, pi)) in momentum_point.iter_mut().zip(x.iter().zip(&previous)) {
            *yi = xi + momentum * (xi - pi);
        }
        t = t_next;

        let change = x
            .iter()
            .zip(&previous)
            .fold(0.0_f64, |acc, (xi, pi)| acc.max((xi - pi).abs()));
        let scale = x.iter().fold(1.0_f64, |acc, xi| acc.max(xi.abs()));
        report.iterations = iteration;
        report.last_relative_change = change / scale;

        previous.copy_from_slice(x);

        if report.last_relative_change <= tol {
            report.converged = true;
            break;
        }
    }

    report
}

/// The proximal operator of `threshold · ‖·‖₁`:
/// `xᵢ ↦ sign(xᵢ) · max(|xᵢ| − threshold, 0)`.
///
/// Shrinking every component towards zero by a fixed amount, and clipping the
/// ones that would cross it, is what makes an L1 penalty produce a solution
/// with exact zeros rather than merely small numbers.
pub fn soft_threshold(x: &mut [f64], threshold: f64) {
    for xi in x {
        *xi = xi.signum() * (xi.abs() - threshold).max(0.0);
    }
}

/// The proximal operator of `threshold · ‖·‖₁` restricted to `x ≥ 0`:
/// `xᵢ ↦ max(xᵢ − threshold, 0)`.
pub fn soft_threshold_nonneg(x: &mut [f64], threshold: f64) {
    for xi in x {
        *xi = (*xi - threshold).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `½‖x − b‖²  +  λ‖x‖₁` with `x ≥ 0` has the closed-form minimiser
    /// `max(bᵢ − λ, 0)`, so FISTA has a known answer to reach.
    #[test]
    fn fista_reaches_the_closed_form_minimiser() {
        let b = [3.0, 0.5, -2.0, 1.25];
        let lambda = 1.0;
        let mut x = vec![0.0; b.len()];

        let report = fista(
            &mut x,
            1.0,
            |x, grad| {
                for (g, (xi, bi)) in grad.iter_mut().zip(x.iter().zip(&b)) {
                    *g = xi - bi;
                }
            },
            |x, step| soft_threshold_nonneg(x, step * lambda),
            10_000,
            1e-14,
        );

        assert!(report.converged, "{report:?}");
        for (xi, bi) in x.iter().zip(&b) {
            assert!((xi - (bi - lambda).max(0.0)).abs() < 1e-10, "{x:?}");
        }
    }

    /// A Lipschitz bound far above the true curvature makes every step tiny,
    /// which is the usual reason a real run hits the iteration limit.
    #[test]
    fn a_run_that_runs_out_of_iterations_says_so() {
        let mut x = vec![0.0; 2];
        let report = fista(
            &mut x,
            1.0,
            |x, grad| {
                for (g, xi) in grad.iter_mut().zip(x) {
                    *g = 1e-3 * (xi - 1.0);
                }
            },
            |_, _| {},
            2,
            1e-16,
        );
        assert!(!report.converged, "{report:?}");
        assert_eq!(report.iterations, 2);
        assert!(report.last_relative_change > 0.0, "{report:?}");
    }

    #[test]
    fn the_soft_threshold_shrinks_towards_zero_from_both_sides() {
        let mut x = [2.0, 0.5, -1.0, -2.5];
        soft_threshold(&mut x, 1.0);
        assert_eq!(x, [1.0, 0.0, 0.0, -1.5]);
    }

    #[test]
    fn the_nonnegative_soft_threshold_clips_at_zero() {
        let mut x = [2.0, 0.5, -1.0];
        soft_threshold_nonneg(&mut x, 1.0);
        assert_eq!(x, [1.0, 0.0, 0.0]);
    }
}
