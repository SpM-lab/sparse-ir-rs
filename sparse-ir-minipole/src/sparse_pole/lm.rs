//! Levenberg–Marquardt on the Gauss–Newton model of `Φ`, with the damping
//! update of H. B. Nielsen, "Damping parameter in Marquardt's method",
//! IMM-REP-1999-05 (Technical University of Denmark, 1999).

use super::model::{Eval, Problem, Workspace};
use crate::error::Result;
use tenferro_linalg::TypedTensorLinalgExt;
use tenferro_tensor::{BackendSession, TypedTensor};

/// Stopping tolerances of one minimization.
#[derive(Debug, Clone, Copy)]
pub(super) struct LmOptions {
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
}

/// A minimizer of `Φ` for one multiplier.
pub(super) struct LmOutcome {
    pub theta: Vec<f64>,
    pub eval: Eval,
    pub iterations: usize,
}

/// Solve `(gn + mu diag(d)) h = -g`; `None` if the system is singular.
fn damped_step(
    gn: &[f64],
    d: &[f64],
    g: &[f64],
    mu: f64,
    session: &mut dyn BackendSession,
) -> Result<Option<Vec<f64>>> {
    let p = g.len();
    let mut a = gn.to_vec();
    for i in 0..p {
        a[i + p * i] += mu * d[i];
    }
    let a = TypedTensor::<f64>::from_vec_col_major(vec![p, p], a)?;
    let b = TypedTensor::<f64>::from_vec_col_major(vec![p, 1], g.iter().map(|v| -v).collect())?;
    let Ok(h) = a.solve(&b, session) else {
        return Ok(None);
    };
    let h = h.host_data()?.to_vec();
    Ok(h.iter().all(|v| v.is_finite()).then_some(h))
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Minimize `Φ` from `theta` for the multiplier `alpha`.
pub(super) fn minimize(
    ws: &mut Workspace,
    pb: &Problem,
    mut theta: Vec<f64>,
    alpha: f64,
    opts: &LmOptions,
    session: &mut dyn BackendSession,
) -> Result<LmOutcome> {
    let p = theta.len();
    let mut cur = ws.eval(pb, &theta, alpha, true, session)?;
    let mut scale: Vec<f64> = (0..p).map(|i| cur.gn[i + p * i]).collect();
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let mut iterations = 0;
    while iterations < opts.max_iter {
        let gmax = cur.grad.iter().fold(0.0f64, |a, g| a.max(g.abs()));
        if gmax <= opts.gtol * cur.phi.max(1.0) {
            break;
        }
        iterations += 1;
        // Running maximum of the Gauss–Newton diagonal, floored relative to
        // its largest entry so that no direction is left undamped.
        let floor = f64::EPSILON * scale.iter().fold(0.0f64, |a, &s| a.max(s));
        let d: Vec<f64> = scale
            .iter()
            .map(|&s| s.max(floor).max(f64::MIN_POSITIVE))
            .collect();
        let accepted = loop {
            if mu > 1e30 {
                break None;
            }
            let Some(h) = damped_step(&cur.gn, &d, &cur.grad, mu, session)? else {
                mu *= nu;
                nu *= 2.0;
                continue;
            };
            let pred: f64 = 0.5
                * h.iter()
                    .zip(d.iter().zip(&cur.grad))
                    .map(|(&hi, (&di, &gi))| hi * (mu * di * hi - gi))
                    .sum::<f64>();
            let trial: Vec<f64> = theta.iter().zip(&h).map(|(t, h)| t + h).collect();
            let next = ws.eval(pb, &trial, alpha, false, session)?;
            let rho = if pred > 0.0 && next.phi.is_finite() {
                (cur.phi - next.phi) / pred
            } else {
                -1.0
            };
            if rho > 0.0 {
                mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                break Some((trial, h, next.phi));
            }
            mu *= nu;
            nu *= 2.0;
        };
        let Some((trial, h, phi)) = accepted else {
            break;
        };
        let small_step = norm(&h) <= opts.xtol * (norm(&theta) + opts.xtol);
        let small_decrease = cur.phi - phi <= opts.ftol * cur.phi;
        theta = trial;
        cur = ws.eval(pb, &theta, alpha, true, session)?;
        for (s, i) in scale.iter_mut().zip(0..p) {
            *s = s.max(cur.gn[i + p * i]);
        }
        if small_step || small_decrease {
            break;
        }
    }
    // Leave the workspace at `theta`.
    let eval = ws.eval(pb, &theta, alpha, false, session)?;
    Ok(LmOutcome {
        theta,
        eval,
        iterations,
    })
}
