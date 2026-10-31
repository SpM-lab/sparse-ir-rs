//! Sparse causal pole representations.
//!
//! Pole fits of Matsubara data such as ESPRIT, MiniPole or AAA return
//! `g(z) = Σ_i A_i / (z - p_i)` with complex `p_i` and complex (matrix)
//! weights `A_i`. Nothing makes such a fit causal: poles may lie in the upper
//! half plane and the spectral function `-(g(ω+i0) - g(ω+i0)†) / (2πi)` may
//! fail to be positive semidefinite. [`refine`] replaces it by a nearby
//! causal function with as few poles as the data require, independently of
//! how the input was obtained. A discrete bath `Σ_b v_b v_b† / (z - ε_b)`
//! with real levels is a valid input as well.
//!
//! # Parametrization
//!
//! The refined function is
//!
//! ```text
//! Σ(z) = Σ_j W_j / (z - ξ_j),   W_j = -2i Σ_k R_j R_k† / (ξ_j - ξ_k*),
//! ```
//!
//! with `M` poles `ξ_j = x_j - i γ_j`, `γ_j = e^{s_j}`, in the lower half
//! plane and `N x K` factors `R_j = √γ_j V_j`. It is the Cauchy transform of
//! the positive semidefinite matrix measure `ρ(ω) = F(ω) F(ω)† / π` with
//! `F(ω) = Σ_j R_j / (ω - ξ_j)`, so every parameter value is causal
//! (`Im Σ(ω + i0) ≤ 0` for scalar data); conversely, by the matrix
//! Fejér–Riesz theorem, every rational positive semidefinite spectral function
//! of this degree arises this way. The unconstrained parameters `(x, s, V)`
//! make causality structural rather than a constraint.
//!
//! As `γ_j → 0` the cross terms of pole `j` vanish and its weight tends to
//! `V_j V_j†`: a bath level on the real axis, as in the bath fitting of
//! H. Shinaoka and Y. Nagai, "Sparse modeling of large-scale quantum impurity
//! models with low symmetries", arXiv:2007.03955. The parametrization thus contains both real-axis
//! (discrete bath) and broadened poles.
//!
//! # Objective and the choice of the multiplier
//!
//! Distances between functions are measured in the Hardy space H² on the
//! line `Im z = η`, `η = π / β` the height of the lowest fermionic Matsubara
//! frequency: `‖f‖² = Σ_x ∫ |f_x(ω + iη)|² dω`, in closed form for pole sums
//! and finite for poles on the real axis. The refinement solves
//!
//! ```text
//! min ‖Σ_θ - g‖²_H²   subject to   rms_n |Σ_θ(z_n) - d_n| ≤ ε,
//! ```
//!
//! "among the causal functions that explain the data as well as the input
//! does, the one closest to the input". The multiplier `α` of the Lagrangian
//! `‖Σ_θ - g‖² / ‖g‖² + α (rms / ε)²` is found by the discrepancy principle
//! (V. A. Morozov, Soviet Math. Dokl. 7, 414 (1966)): `α = 0` (the plain H²
//! projection) if that already meets the bound, otherwise the `α` for which
//! `rms ∈ [(1 - τ) ε, ε]`, by bracketing and bisection in `log α`. Each `α` is
//! a nonlinear least-squares problem, solved by Levenberg–Marquardt on its
//! Gauss–Newton model (exact for the H² term, which is quadratic in the
//! weights).
//!
//! The target `ε` defaults to the residual of the input representation on the
//! given data, so the method has no free regularization parameter: the result
//! fits the data no worse than the input (within `τ`) and is causal. A smaller
//! `ε` may be unreachable; then the status is
//! [`Status::DiscrepancyNotReached`] and the solution with the
//! smallest residual is returned.
//!
//! # Pruning
//!
//! With [`Params::prune`] (off by default) the number of poles is then
//! reduced, with the same `ε` and no further threshold: the pole of least
//! spectral weight `‖V_j‖²` is removed and the multiplier search is repeated
//! from the remaining parameters, as long as the bound `rms ≤ ε` remains
//! reachable. The result is the H²-closest causal function among those with
//! the fewest poles found this way; superfluous poles, which otherwise only
//! fade slowly (`V_j → 0`) and stall the minimization, are removed. This
//! replaces the group-LASSO penalty and the residual tolerance of the bath
//! fitting above by the discrepancy bound. With a loose `ε` the data may be
//! explained by fewer poles than the input has, at the cost of detail that
//! the data do not constrain; pruning is therefore opt-in.
//!
//! # Fitting from real-axis levels
//!
//! [`fit`] starts instead from poles on the real axis at given levels with
//! equal weights, as the bath fitting above. It first minimizes the residual
//! alone (residual `r₀`), then selects the causal function of least H² norm
//! with `rms ≤ ε`, `ε = (1 + τ) r₀` by default (the analogue of the residual
//! tolerance of the bath fitting), by the same multiplier search.
//!
//! The algorithm (the parametrization combined with the H²-distance
//! objective, the discrepancy principle and the pruning) is original to this
//! crate; it is not a port.
//!
//! # Data
//!
//! `values` must be `g(z_n)` without a constant (high-frequency) term:
//! subtract it before calling, and add it back to [`CausalPoles::evaluate`].

mod lm;
mod model;

use crate::error::{ArrayRole, Error, Result};
use lm::{LmOptions, minimize};
use model::{Problem, Workspace, gram};
use num_complex::Complex;
use sparse_ir_core::fpu_check::FpuGuard;
use std::f64::consts::PI;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TypedTensorLinalgExt;
use tenferro_tensor::{BackendSession, BackendSessionHost, TypedTensor};

type C64 = Complex<f64>;

/// Multipliers below this count as zero: the bisection towards `α = 0` stops.
const ALPHA_MIN: f64 = 1e-12;

/// Smallest initial pole width relative to `η`, for input poles on or near
/// the real axis. It only sets the starting point of the minimization.
const INITIAL_WIDTH: f64 = 1e-2;

/// Spectral weight `‖V_j‖²_F = ‖R_j‖²_F / γ_j` of the diagonal term of pole
/// `j`, the pruning order.
fn pole_weight(theta: &[f64], m: usize, nk: usize, j: usize) -> f64 {
    let r = 2 * m + nk * j;
    (r..r + nk)
        .map(|i| theta[i] * theta[i] + theta[i + nk * m] * theta[i + nk * m])
        .sum()
}

/// `theta` without pole `j`.
fn remove_pole(theta: &[f64], m: usize, nk: usize, j: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(theta.len() - 2 - 2 * nk);
    for block in [&theta[..m], &theta[m..2 * m]] {
        out.extend(
            block
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .map(|(_, v)| v),
        );
    }
    for part in 0..2 {
        let base = 2 * m + part * nk * m;
        out.extend(&theta[base..base + nk * j]);
        out.extend(&theta[base + nk * (j + 1)..base + nk * m]);
    }
    out
}

/// Parameters of [`refine`] and [`fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Rank `K` of the factors `R_j`, `1 ≤ K ≤ N`; `None` for `N`.
    pub rank: Option<usize>,
    /// Discrepancy target `ε` (rms over all data entries); `None` for the
    /// residual of the input representation ([`refine`]) or `1 + τ` times the
    /// residual of the unregularized least-squares fit ([`fit`]).
    pub eps: Option<f64>,
    /// Relative width `τ` of the accepted band `[(1 - τ) ε, ε]`.
    pub discrepancy_rtol: f64,
    /// Maximum number of minimizations (values of the multiplier) per search
    /// of the multiplier.
    pub max_solves: usize,
    /// Maximum number of Levenberg–Marquardt iterations per minimization.
    pub max_iter: usize,
    /// Stop when an accepted step lowers `Φ` by at most `ftol Φ`.
    pub ftol: f64,
    /// Stop when an accepted step `h` has `‖h‖ ≤ xtol (‖θ‖ + xtol)`.
    pub xtol: f64,
    /// Stop when `‖∇Φ‖_∞ ≤ gtol max(1, Φ)`.
    pub gtol: f64,
    /// Remove poles while `rms ≤ ε` remains reachable (see the module
    /// documentation); off by default.
    pub prune: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            rank: None,
            eps: None,
            discrepancy_rtol: 0.1,
            max_solves: 40,
            max_iter: 500,
            ftol: 1e-12,
            xtol: 1e-12,
            gtol: 1e-10,
            prune: false,
        }
    }
}

/// How the multiplier was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The H² projection (`α = 0`) already meets the discrepancy bound
    /// (for [`fit`]: the zero function does).
    Projection,
    /// The residual lies in `[(1 - τ) ε, ε]` (or the bisection bracket
    /// closed with the residual at most `ε`).
    Discrepancy,
    /// No multiplier reached `rms ≤ ε`; the result has the smallest residual
    /// found.
    DiscrepancyNotReached,
}

/// Diagnostics of [`refine`] and [`fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    /// How the multiplier was determined
    pub status: Status,
    /// Rms residual of the result on the data
    pub residual_rms: f64,
    /// The target `ε`
    pub target_rms: f64,
    /// Rms residual on the data of the input representation ([`refine`]) or
    /// of the unregularized least-squares fit ([`fit`])
    pub input_residual_rms: f64,
    /// Relative H² distance `‖Σ - g‖ / ‖g‖` of the result from the input
    /// ([`refine`]), or the H² norm of the result relative to that of the
    /// least-squares fit ([`fit`])
    pub h2_distance: f64,
    /// The multiplier `α`
    pub multiplier: f64,
    /// Levenberg–Marquardt iterations over all minimizations
    pub lm_iterations: usize,
    /// Number of minimizations, including those of the pruning
    pub solves: usize,
    /// Number of poles removed by the pruning
    pub pruned: usize,
}

/// A causal pole representation `Σ(z) = Σ_j W_j / (z - ξ_j)`.
#[derive(Debug)]
pub struct CausalPoles {
    /// Poles `ξ_j`, `Im ξ_j < 0`, sorted by real part; at most as many as
    /// the input poles or levels.
    pub poles: Vec<C64>,
    /// Factors `R_j`, shape `[M, N, K]`; see the module documentation.
    pub factors: TypedTensor<C64>,
    /// Weights `W_j`, shape `[M]` for scalar data, else `[M, N, N]`.
    pub weights: TypedTensor<C64>,
}

impl CausalPoles {
    /// Evaluate `Σ(z)`; returns shape `[len(z), ...]` (the channel shape of
    /// [`CausalPoles::weights`]).
    ///
    /// # Errors
    /// Propagates tensor construction errors.
    pub fn evaluate(&self, z: &[C64]) -> Result<TypedTensor<C64>> {
        let shape = self.weights.shape().to_vec();
        let m = self.poles.len();
        let d: usize = shape[1..].iter().product();
        let w = self.weights.host_data()?;
        let mut out = vec![C64::new(0.0, 0.0); z.len() * d];
        for (i, &zi) in z.iter().enumerate() {
            for (j, &p) in self.poles.iter().enumerate() {
                let f = (zi - p).inv();
                for c in 0..d {
                    out[i + z.len() * c] += w[j + m * c] * f;
                }
            }
        }
        let mut out_shape = shape;
        out_shape[0] = z.len();
        Ok(TypedTensor::from_vec_col_major(out_shape, out)?)
    }
}

fn check_finite(name: &'static str, v: &[C64], shape: &[usize]) -> Result<()> {
    if let Some(i) = v
        .iter()
        .position(|c| !(c.re.is_finite() && c.im.is_finite()))
    {
        let mut index = Vec::with_capacity(shape.len());
        let mut rest = i;
        for &n in shape {
            index.push(rest % n);
            rest /= n;
        }
        let c = v[i];
        let value = if c.re.is_finite() { c.im } else { c.re };
        return Err(Error::NonFiniteInput { name, index, value });
    }
    Ok(())
}

/// Matrix size `N` of a weight or data tensor `[len, ...]`.
fn channel_size(t: &TypedTensor<C64>, len: usize) -> Result<usize> {
    let shape = t.shape();
    match shape {
        [l] if *l == len => Ok(1),
        [l, a, b] if *l == len && a == b && *a > 0 => Ok(*a),
        _ => {
            let n = if shape.len() == 3 { shape[1] } else { 1 };
            let mut expected = vec![len];
            if shape.len() == 3 {
                expected.extend([n, n]);
            }
            Err(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected,
                actual: shape.to_vec(),
            })
        }
    }
}

/// Rms of `Σ_i w_i / (z_n - p_i) - d_n` over all entries.
fn pole_residual_rms(poles: &[C64], w: &[C64], z: &[C64], d: &[C64], x: usize) -> f64 {
    let (m, l) = (poles.len(), z.len());
    let mut sumsq = 0.0;
    for ch in 0..x {
        for (i, &zi) in z.iter().enumerate() {
            let mut acc = -d[i + l * ch];
            for (j, &p) in poles.iter().enumerate() {
                acc += w[j + m * ch] / (zi - p);
            }
            sumsq += acc.norm_sqr();
        }
    }
    (sumsq / (l * x) as f64).sqrt()
}

/// Initial parameters: each reference pole reflected into the lower half
/// plane, `ξ = Re p - i γ` with `γ = max(|Im p|, INITIAL_WIDTH η)`, and
/// `R R† ≈ γ |A|` from the eigendecomposition of the Hermitian part of `γ A`
/// (absolute eigenvalues, the `K` largest), the exact factorization for a
/// causal isolated pole. The floor moves poles on the real axis (a bath
/// representation) just below it.
fn initial_theta(
    poles: &[C64],
    w: &[C64],
    n: usize,
    k: usize,
    eta: f64,
    session: &mut dyn BackendSession,
) -> Result<Vec<f64>> {
    let m = poles.len();
    let nrk = n * k * m;
    let mut theta = vec![0.0; 2 * m + 2 * nrk];
    let mut eig: Vec<(Vec<f64>, Vec<C64>)> = Vec::with_capacity(m);
    let mut gammas = Vec::with_capacity(m);
    for (j, &p) in poles.iter().enumerate() {
        let gamma = p.im.abs().max(INITIAL_WIDTH * eta);
        theta[j] = p.re;
        theta[m + j] = gamma.ln();
        gammas.push(gamma);
        let mut herm = vec![C64::new(0.0, 0.0); n * n];
        for q in 0..n {
            for pp in 0..n {
                let a = w[j + m * (pp + n * q)];
                let b = w[j + m * (q + n * pp)].conj();
                herm[pp + n * q] = 0.5 * gamma * (a + b);
            }
        }
        let t = TypedTensor::<C64>::from_vec_col_major(vec![n, n], herm)?;
        let (vals, vecs) = t.eigh(session).map_err(|e| Error::DecompositionFailed {
            reason: format!("the eigendecomposition of the weight of pole {j} failed: {e}"),
        })?;
        let ld = vecs.shape()[0];
        let vecs = vecs.host_data()?;
        let mut u = vec![C64::new(0.0, 0.0); n * n];
        for c in 0..n {
            for pp in 0..n {
                u[pp + n * c] = vecs[pp + ld * c];
            }
        }
        eig.push((vals.host_data()?.to_vec(), u));
    }
    let emax = eig
        .iter()
        .flat_map(|(e, _)| e.iter())
        .fold(0.0f64, |a, e| a.max(e.abs()));
    let floor = 1e-6 * emax;
    for (j, (vals, u)) in eig.iter().enumerate() {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| vals[b].abs().total_cmp(&vals[a].abs()));
        for (c, &e) in order.iter().take(k).enumerate() {
            // V = R / √γ
            let sq = (vals[e].abs().max(floor) / gammas[j]).sqrt();
            for pp in 0..n {
                let v = u[pp + n * e] * sq;
                let idx = pp + n * (c + k * j);
                theta[2 * m + idx] = v.re;
                theta[2 * m + nrk + idx] = v.im;
            }
        }
    }
    Ok(theta)
}

/// The result of one minimization: `(α, θ, rms, h2)`.
type Sol = (f64, Vec<f64>, f64, f64);

/// The result of [`optimize`].
struct Outcome {
    xi: Vec<C64>,
    r: Vec<C64>,
    sol: Sol,
    status: Status,
    pruned: usize,
    solves: usize,
    iterations: usize,
}

/// Minimizations of one problem with shared settings.
struct Solver<'p> {
    params: &'p Params,
    opts: LmOptions,
    eps: f64,
    solves: usize,
    iterations: usize,
}

impl Solver<'_> {
    fn solve(
        &mut self,
        ws: &mut Workspace,
        pb: &Problem,
        alpha: f64,
        theta: Vec<f64>,
        session: &mut dyn BackendSession,
    ) -> Result<Sol> {
        let out = minimize(ws, pb, theta, alpha, &self.opts, session)?;
        self.solves += 1;
        self.iterations += out.iterations;
        Ok((alpha, out.theta, out.eval.rms, out.eval.h2))
    }

    /// Discrepancy search for `α` starting at `alpha0` from `theta`: down by
    /// factors of 10 while `rms ≤ ε` (to `α = 0` below [`ALPHA_MIN`]), up
    /// while `rms > ε`, then bisection in `log α`.
    fn search(
        &mut self,
        ws: &mut Workspace,
        pb: &Problem,
        theta: Vec<f64>,
        alpha0: f64,
        session: &mut dyn BackendSession,
    ) -> Result<(Sol, Status)> {
        let (eps, tau) = (self.eps, self.params.discrepancy_rtol);
        let budget = self.solves + self.params.max_solves;
        let first = self.solve(ws, pb, alpha0, theta, session)?;
        let (mut lo, mut hi) = if first.2 <= eps {
            if alpha0 == 0.0 {
                return Ok((first, Status::Projection));
            }
            let mut hi = first;
            loop {
                if hi.2 >= (1.0 - tau) * eps || self.solves >= budget {
                    return Ok((hi, Status::Discrepancy));
                }
                let a = if hi.0 / 10.0 < ALPHA_MIN {
                    0.0
                } else {
                    hi.0 / 10.0
                };
                let t = self.solve(ws, pb, a, hi.1.clone(), session)?;
                if t.2 > eps {
                    break (t, hi);
                }
                hi = t;
                if a == 0.0 {
                    return Ok((hi, Status::Projection));
                }
            }
        } else {
            let mut best = first.clone();
            let mut lo = first;
            let mut alpha = if lo.0 == 0.0 {
                (lo.3 * (eps / lo.2).powi(2)).max(ALPHA_MIN)
            } else {
                10.0 * lo.0
            };
            loop {
                if self.solves >= budget || alpha > 1e12 {
                    return Ok((best, Status::DiscrepancyNotReached));
                }
                let t = self.solve(ws, pb, alpha, lo.1.clone(), session)?;
                if t.2 < best.2 {
                    best = t.clone();
                }
                if t.2 <= eps {
                    break (lo, t);
                }
                lo = t;
                alpha *= 10.0;
            }
        };
        // Bracket: `lo` violates the bound, `hi` meets it; warm start from
        // the satisfying side.
        while hi.2 < (1.0 - tau) * eps
            && self.solves < budget
            && !(lo.0 > 0.0 && hi.0 / lo.0 < 1.05)
            && !(lo.0 == 0.0 && hi.0 <= ALPHA_MIN)
        {
            let mid = if lo.0 > 0.0 {
                (lo.0 * hi.0).sqrt()
            } else {
                hi.0 / 10.0
            };
            let t = self.solve(ws, pb, mid, hi.1.clone(), session)?;
            if t.2 > eps {
                lo = t;
            } else {
                hi = t;
            }
        }
        Ok((hi, Status::Discrepancy))
    }
}

/// The multiplier search from `alpha0`, then the optional pruning.
fn optimize(
    pb: Problem,
    theta0: Vec<f64>,
    alpha0: f64,
    params: &Params,
    session: &mut dyn BackendSession,
) -> Result<Outcome> {
    let mut solver = Solver {
        params,
        opts: lm_options(params),
        eps: pb.eps,
        solves: 0,
        iterations: 0,
    };
    let mut ws = Workspace::new(&pb)?;
    let (mut sol, mut status) = solver.search(&mut ws, &pb, theta0, alpha0, session)?;
    // Pruning: remove the pole of least spectral weight and repeat the
    // search from the current multiplier while the bound remains reachable.
    let nk = pb.n * pb.k;
    let mut pb = pb;
    let mut pruned = 0;
    while params.prune && status != Status::DiscrepancyNotReached && pb.m > 1 {
        let j = (0..pb.m)
            .min_by(|&a, &b| {
                pole_weight(&sol.1, pb.m, nk, a).total_cmp(&pole_weight(&sol.1, pb.m, nk, b))
            })
            .unwrap_or(0);
        let sub = Problem {
            m: pb.m - 1,
            ..pb.clone()
        };
        let mut sws = Workspace::new(&sub)?;
        let start = remove_pole(&sol.1, pb.m, nk, j);
        let (s, st) = solver.search(&mut sws, &sub, start, sol.0, session)?;
        if st == Status::DiscrepancyNotReached {
            break;
        }
        (pb, ws, sol, status) = (sub, sws, s, st);
        pruned += 1;
    }
    // Leave the workspace at the solution to read ξ and R.
    ws.eval(&pb, &sol.1, sol.0, false, session)?;
    Ok(Outcome {
        xi: ws.poles().to_vec(),
        r: ws.factors().to_vec(),
        sol,
        status,
        pruned,
        solves: solver.solves,
        iterations: solver.iterations,
    })
}

fn lm_options(params: &Params) -> LmOptions {
    LmOptions {
        max_iter: params.max_iter,
        ftol: params.ftol,
        xtol: params.xtol,
        gtol: params.gtol,
    }
}

/// The result in output order (by `Re ξ`), `W_j` recomputed from `ξ`, `R`.
fn assemble(xi: &[C64], r: &[C64], n: usize, k: usize, scalar: bool) -> Result<CausalPoles> {
    let m = xi.len();
    let x = n * n;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| xi[a].re.total_cmp(&xi[b].re));
    let mut wout = vec![C64::new(0.0, 0.0); m * x];
    let mut fout = vec![C64::new(0.0, 0.0); m * n * k];
    for (jn, &j) in order.iter().enumerate() {
        for q in 0..n {
            for pp in 0..n {
                let mut acc = C64::new(0.0, 0.0);
                for kk in 0..m {
                    let ckj = C64::new(0.0, -2.0) / (xi[j] - xi[kk].conj());
                    for c in 0..k {
                        acc += ckj * r[pp + n * (c + k * j)] * r[q + n * (c + k * kk)].conj();
                    }
                }
                wout[jn + m * (pp + n * q)] = acc;
            }
        }
        for c in 0..k {
            for pp in 0..n {
                fout[jn + m * (pp + n * c)] = r[pp + n * (c + k * j)];
            }
        }
    }
    let wshape = if scalar { vec![m] } else { vec![m, n, n] };
    Ok(CausalPoles {
        poles: order.iter().map(|&j| xi[j]).collect(),
        factors: TypedTensor::from_vec_col_major(vec![m, n, k], fout)?,
        weights: TypedTensor::from_vec_col_major(wshape, wout)?,
    })
}

/// Validate the points, the data and `β`; returns `(N, η)`.
fn check_data(z: &[C64], values: &TypedTensor<C64>, beta: f64) -> Result<(usize, f64)> {
    if z.is_empty() {
        return Err(Error::EmptyInput { name: "z" });
    }
    let n = channel_size(values, z.len())?;
    check_finite("z", z, &[z.len()])?;
    check_finite("values", values.host_data()?, values.shape())?;
    if let Some(zi) = z.iter().find(|z| z.im <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "z",
            value: format!("{zi}"),
            reason: "must lie in the upper half plane".to_string(),
        });
    }
    if !(beta > 0.0 && beta.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "beta",
            value: format!("{beta:?}"),
            reason: "must be positive and finite".to_string(),
        });
    }
    Ok((n, PI / beta))
}

/// Validate `params` for matrix size `n`; returns the rank `K`.
fn check_params(params: &Params, n: usize) -> Result<usize> {
    let k = params.rank.unwrap_or(n);
    if k == 0 || k > n {
        return Err(Error::InvalidParameter {
            name: "rank",
            value: format!("{k}"),
            reason: format!("must be in 1..={n}"),
        });
    }
    let tau = params.discrepancy_rtol;
    if !(tau > 0.0 && tau < 1.0) {
        return Err(Error::InvalidParameter {
            name: "discrepancy_rtol",
            value: format!("{tau:?}"),
            reason: "must be in (0, 1)".to_string(),
        });
    }
    for (name, v) in [
        ("max_solves", params.max_solves),
        ("max_iter", params.max_iter),
    ] {
        if v == 0 {
            return Err(Error::InvalidParameter {
                name,
                value: "0".to_string(),
                reason: "must be positive".to_string(),
            });
        }
    }
    for (name, v) in [
        ("ftol", params.ftol),
        ("xtol", params.xtol),
        ("gtol", params.gtol),
    ] {
        if !(v >= 0.0 && v.is_finite()) {
            return Err(Error::InvalidParameter {
                name,
                value: format!("{v:?}"),
                reason: "must be non-negative and finite".to_string(),
            });
        }
    }
    if let Some(e) = params.eps
        && !(e > 0.0 && e.is_finite())
    {
        return Err(Error::InvalidParameter {
            name: "eps",
            value: format!("{e:?}"),
            reason: "must be positive and finite".to_string(),
        });
    }
    Ok(k)
}

/// Rms of the data entries.
fn data_scale(d: &[C64]) -> f64 {
    (d.iter().map(|v| v.norm_sqr()).sum::<f64>() / d.len() as f64).sqrt()
}

/// Replace a complex-weight pole representation by a nearby causal one whose
/// residual on the data does not exceed `ε`.
///
/// `poles` (`[M]`) and `weights` (`[M]`, or `[M, N, N]` for matrix data) are
/// the input representation `g(z) = Σ_i A_i / (z - p_i)`, e.g. from ESPRIT or
/// MiniPole. `values` (`[L]` or `[L, N, N]`) are the data at the points `z`
/// (`Im z > 0`, e.g. `i ω_n`), without a constant term. H² distances are
/// measured on the line `Im z = π / β`, the height of the lowest fermionic
/// Matsubara frequency; input poles may lie anywhere off that line, including
/// on the real axis. The result has `M` poles (at most `M` with pruning), all
/// in the lower half plane. See the module documentation for the method.
///
/// # Errors
/// [`Error::EmptyInput`] for no poles or no points;
/// [`Error::ShapeMismatch`] for inconsistent shapes;
/// [`Error::NonFiniteInput`] for non-finite inputs;
/// [`Error::InvalidParameter`] for a non-positive `β`, a pole on the line
/// `Im z = π / β`, a point with `Im z ≤ 0`, all weights zero, a rank outside
/// `1..=N`, a non-positive `ε`, `τ` outside `(0, 1)`, or zero
/// `max_solves`/`max_iter`;
/// [`Error::DecompositionFailed`] if an eigendecomposition fails.
pub fn refine(
    poles: &[C64],
    weights: &TypedTensor<C64>,
    z: &[C64],
    values: &TypedTensor<C64>,
    beta: f64,
    params: &Params,
) -> Result<(CausalPoles, Info)> {
    let m = poles.len();
    if m == 0 {
        return Err(Error::EmptyInput { name: "poles" });
    }
    let (n, eta) = check_data(z, values, beta)?;
    if channel_size(weights, m)? != n || weights.shape().len() != values.shape().len() {
        let mut expected = vec![m];
        expected.extend(&values.shape()[1..]);
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected,
            actual: weights.shape().to_vec(),
        });
    }
    let k = check_params(params, n)?;
    let x = n * n;
    let w = weights.host_data()?;
    let d = values.host_data()?;
    check_finite("poles", poles, &[m])?;
    check_finite("weights", w, weights.shape())?;
    let shift = C64::new(0.0, eta);
    if let Some(p) = poles.iter().find(|p| p.im == eta) {
        return Err(Error::InvalidParameter {
            name: "poles",
            value: format!("{p}"),
            reason: "must not lie on the line Im z = π / β (infinite H² norm)".to_string(),
        });
    }

    // ‖g‖² in H² on the line Im z = η over all poles; only the poles below
    // the line couple to the refined function.
    let mut ref_norm = 0.0;
    for ch in 0..x {
        for (i, &pi) in poles.iter().enumerate() {
            for (j, &pj) in poles.iter().enumerate() {
                ref_norm +=
                    (w[i + m * ch].conj() * gram(pi - shift, 1, pj - shift, 1) * w[j + m * ch]).re;
            }
        }
    }
    if ref_norm.is_nan() || ref_norm <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "weights",
            value: format!("H² norm {ref_norm:?}"),
            reason: "must not all vanish".to_string(),
        });
    }
    let lower: Vec<usize> = (0..m).filter(|&i| poles[i].im < eta).collect();
    let mr = lower.len();
    let mut ref_weights = vec![C64::new(0.0, 0.0); mr * x];
    for ch in 0..x {
        for (ii, &i) in lower.iter().enumerate() {
            ref_weights[ii + mr * ch] = w[i + m * ch];
        }
    }

    let input_rms = pole_residual_rms(poles, w, z, d, x);
    if !input_rms.is_finite() {
        return Err(Error::InvalidParameter {
            name: "poles",
            value: format!("residual {input_rms:?}"),
            reason: "the input representation must be finite at z".to_string(),
        });
    }
    // Exact input: aim at the rounding level of the data instead.
    let eps = params.eps.unwrap_or_else(|| {
        input_rms
            .max(f64::EPSILON * data_scale(d))
            .max(f64::MIN_POSITIVE)
    });

    let pb = Problem {
        m,
        n,
        k,
        z,
        d,
        eta,
        ref_poles: lower.iter().map(|&i| poles[i] - shift).collect(),
        ref_weights,
        ref_norm,
        h2_weight: 1.0 / ref_norm,
        eps,
    };
    let _guard = FpuGuard::new_protect_computation();
    let out = CpuBackend::new().with_backend_session(|session| -> Result<_> {
        let theta0 = initial_theta(poles, w, n, k, eta, session)?;
        optimize(pb, theta0, 0.0, params, session)
    })?;
    let result = assemble(&out.xi, &out.r, n, k, values.shape().len() == 1)?;
    let info = Info {
        status: out.status,
        residual_rms: out.sol.2,
        target_rms: eps,
        input_residual_rms: input_rms,
        h2_distance: out.sol.3.sqrt(),
        multiplier: out.sol.0,
        lm_iterations: out.iterations,
        solves: out.solves,
        pruned: out.pruned,
    };
    Ok((result, info))
}

/// Fit a causal pole representation to data, starting from poles on the
/// real axis at `levels` (a discrete bath).
///
/// `values` (`[L]` or `[L, N, N]`) are the data at the points `z`
/// (`Im z > 0`, e.g. `i ω_n`), without a constant term. `levels` are the
/// initial pole positions, e.g. the default real-frequency sampling points of
/// an IR basis (`default_omega_sampling_points`) or DLR poles, as in the bath
/// fitting of arXiv:2007.03955; all levels start with equal weights from the
/// high-frequency tail `z g(z) → Σ_j W_j` at the point of largest `|z|`.
///
/// First the residual alone is minimized (the unregularized least-squares
/// fit, residual `r₀`). Among the causal functions with `rms ≤ ε`,
/// `ε = (1 + τ) r₀` by default, the one of least H² norm on the line
/// `Im z = π / β` is then selected by the discrepancy principle, which
/// suppresses the spurious structure of the least-squares fit. With
/// [`Params::prune`] the levels the data do not need are removed. See the
/// module documentation for the method.
///
/// # Errors
/// [`Error::EmptyInput`] for no levels or no points;
/// [`Error::ShapeMismatch`] for a malformed `values`;
/// [`Error::NonFiniteInput`] for non-finite inputs;
/// [`Error::InvalidParameter`] for a non-positive `β`, a point with
/// `Im z ≤ 0`, data whose tail has no weight, a vanishing least-squares fit, a
/// rank outside `1..=N`, a non-positive `ε`, `τ` outside `(0, 1)`, or zero
/// `max_solves`/`max_iter`;
/// [`Error::DecompositionFailed`] if an eigendecomposition fails.
pub fn fit(
    levels: &[f64],
    z: &[C64],
    values: &TypedTensor<C64>,
    beta: f64,
    params: &Params,
) -> Result<(CausalPoles, Info)> {
    let m = levels.len();
    if m == 0 {
        return Err(Error::EmptyInput { name: "levels" });
    }
    let (n, eta) = check_data(z, values, beta)?;
    let k = check_params(params, n)?;
    if let Some(i) = levels.iter().position(|e| !e.is_finite()) {
        return Err(Error::NonFiniteInput {
            name: "levels",
            index: vec![i],
            value: levels[i],
        });
    }
    let (l, x) = (z.len(), n * n);
    let d = values.host_data()?;

    // Equal initial weights from the tail: W_tot ≈ z g(z) at the largest |z|.
    let far = (0..l)
        .max_by(|&a, &b| z[a].norm_sqr().total_cmp(&z[b].norm_sqr()))
        .unwrap_or(0);
    let mut start_w = vec![C64::new(0.0, 0.0); m * x];
    for ch in 0..x {
        let wt = z[far] * d[far + l * ch] / m as f64;
        for j in 0..m {
            start_w[j + m * ch] = wt;
        }
    }
    if start_w.iter().all(|w| w.norm_sqr() == 0.0) {
        return Err(Error::InvalidParameter {
            name: "values",
            value: format!("{}", d[far]),
            reason: "the high-frequency tail z g(z) must not vanish".to_string(),
        });
    }
    let start_p: Vec<C64> = levels.iter().map(|&e| C64::new(e, 0.0)).collect();

    let scale = data_scale(d).max(f64::MIN_POSITIVE);
    let base = Problem {
        m,
        n,
        k,
        z,
        d,
        eta,
        ref_poles: Vec::new(),
        ref_weights: Vec::new(),
        ref_norm: 0.0,
        h2_weight: 0.0,
        eps: scale,
    };
    let tau = params.discrepancy_rtol;
    let _guard = FpuGuard::new_protect_computation();
    let (out, r0, eps) = CpuBackend::new().with_backend_session(|session| -> Result<_> {
        let theta0 = initial_theta(&start_p, &start_w, n, k, eta, session)?;
        // Unregularized least squares (no H² term).
        let mut ws = Workspace::new(&base)?;
        let ls = minimize(&mut ws, &base, theta0, 1.0, &lm_options(params), session)?;
        let r0 = ls.eval.rms;
        // ‖Σ_LS‖² normalizes the H² term.
        let norm_pb = Problem {
            h2_weight: 1.0,
            ..base.clone()
        };
        let norm = ws.eval(&norm_pb, &ls.theta, 0.0, false, session)?.h2;
        if !(norm > 0.0 && norm.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: format!("least-squares H² norm {norm:?}"),
                reason: "the least-squares fit must not vanish".to_string(),
            });
        }
        let eps = params.eps.unwrap_or_else(|| {
            ((1.0 + tau) * r0)
                .max(f64::EPSILON * scale)
                .max(f64::MIN_POSITIVE)
        });
        let pb = Problem {
            h2_weight: 1.0 / norm,
            eps,
            ..base.clone()
        };
        // At the least-squares fit both terms of Φ are of order one for α = 1.
        let mut out = optimize(pb, ls.theta, 1.0, params, session)?;
        out.solves += 1;
        out.iterations += ls.iterations;
        Ok((out, r0, eps))
    })?;
    let result = assemble(&out.xi, &out.r, n, k, values.shape().len() == 1)?;
    let info = Info {
        status: out.status,
        residual_rms: out.sol.2,
        target_rms: eps,
        input_residual_rms: r0,
        h2_distance: out.sol.3.sqrt(),
        multiplier: out.sol.0,
        lm_iterations: out.iterations,
        solves: out.solves,
        pruned: out.pruned,
    };
    Ok((result, info))
}

#[cfg(test)]
mod tests;
