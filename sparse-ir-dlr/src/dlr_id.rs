//! Interpolative-decomposition (ID) construction of the DLR.
//!
//! Implements the construction of Kaye, Chen and Parcollet,
//! "Discrete Lehmann representation of imaginary time Green's functions",
//! Phys. Rev. B 105, 235115 (2022):
//!
//! 1. discretize the dimensionless logistic kernel
//!    `K(τ, ω) = e^{-τω} / (1 + e^{-ω})`, `τ ∈ [0, 1]`, `ω ∈ [-Λ, Λ]`, on
//!    composite Chebyshev grids refined dyadically towards `τ ∈ {0, 1}` and
//!    `ω = 0`;
//! 2. select real-frequency poles by a column-pivoted Gram–Schmidt (rank
//!    revealing) factorization of the discretized kernel;
//! 3. select imaginary-time and Matsubara nodes by the same procedure on the
//!    rows of the kernel restricted to the selected poles.
//!
//! The grids follow the parameter choices described in the paper (24-point
//! Chebyshev panels, `⌈log2 Λ⌉ - 2` dyadic levels in τ and `⌈log2 Λ⌉` in ω).
//! Everything here is statistics independent except the Matsubara candidates.

use num_complex::Complex;

/// Chebyshev nodes (first kind) per panel.
const PANEL_ORDER: usize = 24;

/// Matsubara indices `|k| <= DENSE_K` are all candidates; beyond that the
/// candidates are geometrically spaced.
const DENSE_K: i64 = 128;

/// Geometric candidates per octave beyond `DENSE_K`.
const POINTS_PER_OCTAVE: f64 = 32.0;

/// Imaginary-time candidate `τ/β`, stored as the distance `t` to the nearer
/// endpoint so that the kernel keeps full relative accuracy near `τ = β`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TauCandidate {
    /// Distance to the nearer endpoint, `t ∈ (0, 1/2]`.
    pub t: f64,
    /// `true` when `τ/β = 1 - t`, `false` when `τ/β = t`.
    pub from_end: bool,
}

impl TauCandidate {
    /// Dimensionless `τ/β`.
    pub fn value(self) -> f64 {
        if self.from_end { 1.0 - self.t } else { self.t }
    }
}

fn dyadic_levels(lambda: f64, offset: i64) -> usize {
    let l = lambda.max(1.0).log2().ceil() as i64 - offset;
    l.max(1) as usize
}

/// Chebyshev nodes of the first kind on each panel `[breaks[i], breaks[i+1]]`.
fn composite_chebyshev(breaks: &[f64]) -> Vec<f64> {
    let p = PANEL_ORDER;
    let mut x = Vec::with_capacity(p * (breaks.len() - 1));
    for w in breaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        for k in (0..p).rev() {
            let c = (std::f64::consts::PI * (2 * k + 1) as f64 / (2 * p) as f64).cos();
            x.push(0.5 * (a + b) + 0.5 * (b - a) * c);
        }
    }
    x
}

/// Imaginary-time candidates on `[0, 1]`: dyadic panels towards both ends.
pub(crate) fn tau_candidates(lambda: f64) -> Vec<TauCandidate> {
    let npt = dyadic_levels(lambda, 2);
    // Breaks 0, 2^-npt, ..., 1/4, 1/2 on the half interval [0, 1/2].
    let mut breaks = vec![0.0];
    breaks.extend((0..npt).map(|i| 0.5 * 2f64.powi(i as i32 - npt as i32 + 1)));
    let half = composite_chebyshev(&breaks);
    let mut out: Vec<TauCandidate> = half
        .iter()
        .map(|&t| TauCandidate { t, from_end: false })
        .collect();
    out.extend(
        half.iter()
            .rev()
            .map(|&t| TauCandidate { t, from_end: true }),
    );
    out
}

/// Real-frequency candidates on `[-Λ, Λ]`: dyadic panels towards `ω = 0`.
pub(crate) fn omega_candidates(lambda: f64) -> Vec<f64> {
    let npo = dyadic_levels(lambda, 0);
    let mut breaks = vec![0.0];
    breaks.extend((0..npo).map(|i| lambda * 2f64.powi(i as i32 - npo as i32 + 1)));
    let pos = composite_chebyshev(&breaks);
    let mut out: Vec<f64> = pos.iter().rev().map(|&w| -w).collect();
    out.extend(pos);
    out
}

/// Dimensionless logistic kernel `e^{-τω} / (1 + e^{-ω})`, evaluated in the
/// branch that avoids overflow and cancellation.
pub(crate) fn logistic_kernel(tau: TauCandidate, omega: f64) -> f64 {
    let (t, from_end) = (tau.t, tau.from_end);
    if omega >= 0.0 {
        // e^{-τω} / (1 + e^{-ω}), with τ = t or 1 - t.
        let expo = if from_end {
            t * omega - omega
        } else {
            -t * omega
        };
        expo.exp() / (1.0 + (-omega).exp())
    } else {
        // e^{(1-τ)ω} / (1 + e^{ω}), with 1 - τ = 1 - t or t.
        let expo = if from_end {
            t * omega
        } else {
            omega - t * omega
        };
        expo.exp() / (1.0 + omega.exp())
    }
}

/// Matsubara index candidates `n` (with `ν = nπ/β`) of parity `zeta`.
///
/// Every allowed index with `|n| <= 2 DENSE_K + zeta` is included; above that
/// indices are geometrically spaced up to `n_max`. With `positive_only`, only
/// `n >= 0` is returned.
pub(crate) fn matsubara_candidates(lambda: f64, zeta: i64, positive_only: bool) -> Vec<i64> {
    let to_parity = |x: f64| -> i64 {
        let k = ((x - zeta as f64) / 2.0).round() as i64;
        2 * k + zeta
    };
    let n_dense = 2 * DENSE_K + zeta;
    let n_max = to_parity((8.0 * lambda).max(n_dense as f64));
    let mut pos: Vec<i64> = (0..=DENSE_K).map(|k| 2 * k + zeta).collect();
    let ratio = 2f64.powf(1.0 / POINTS_PER_OCTAVE);
    let mut x = n_dense as f64;
    while (x as i64) < n_max {
        x *= ratio;
        let n = to_parity(x.min(n_max as f64));
        if n > *pos.last().unwrap() {
            pos.push(n);
        }
    }
    if positive_only {
        return pos;
    }
    let mut all: Vec<i64> = pos.iter().rev().filter(|&&n| n > 0).map(|&n| -n).collect();
    all.extend(pos);
    all
}

/// Scalars accepted by [`pivoted_gram_schmidt`].
pub(crate) trait GsScalar:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + num_traits::Zero
{
    fn conj(self) -> Self;
    fn norm_sqr(self) -> f64;
    fn scale(self, s: f64) -> Self;
}

impl GsScalar for f64 {
    fn conj(self) -> Self {
        self
    }
    fn norm_sqr(self) -> f64 {
        self * self
    }
    fn scale(self, s: f64) -> Self {
        self * s
    }
}

impl GsScalar for Complex<f64> {
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
    fn norm_sqr(self) -> f64 {
        Complex::norm_sqr(&self)
    }
    fn scale(self, s: f64) -> Self {
        self * s
    }
}

fn dot<T: GsScalar>(q: &[T], v: &[T]) -> T {
    q.iter()
        .zip(v)
        .fold(T::zero(), |acc, (&qi, &vi)| acc + qi.conj() * vi)
}

fn axpy_neg<T: GsScalar>(c: T, q: &[T], v: &mut [T]) {
    for (vi, &qi) in v.iter_mut().zip(q) {
        *vi = *vi - c * qi;
    }
}

/// Column-pivoted Gram–Schmidt on the `n` columns (each of length `m`) of the
/// column-major matrix `a`, which is overwritten.
///
/// At each step the column with the largest residual norm is selected and
/// the remaining columns are orthogonalized against it (with one step of
/// re-orthogonalization). Selection stops when the largest residual norm
/// drops to `rtol` times the largest initial column norm, or after
/// `max_rank` columns.
///
/// Returns the selected column indices in selection order.
pub(crate) fn pivoted_gram_schmidt<T: GsScalar>(
    a: &mut [T],
    m: usize,
    n: usize,
    rtol: f64,
    max_rank: usize,
) -> Vec<usize> {
    assert_eq!(a.len(), m * n);
    let col_norm2 = |a: &[T], j: usize| a[j * m..(j + 1) * m].iter().map(|v| v.norm_sqr()).sum();
    let mut norms2: Vec<f64> = (0..n).map(|j| col_norm2(a, j)).collect();
    let nrm0 = norms2.iter().cloned().fold(0.0, f64::max).sqrt();
    let mut active = vec![true; n];
    let mut q: Vec<Vec<T>> = Vec::new();
    let mut piv = Vec::new();
    let max_rank = max_rank.min(m).min(n);
    while piv.len() < max_rank {
        let Some((jstar, &best)) = norms2
            .iter()
            .enumerate()
            .filter(|(j, _)| active[*j])
            .max_by(|x, y| x.1.total_cmp(y.1))
        else {
            break;
        };
        if nrm0 == 0.0 || best.sqrt() <= rtol * nrm0 {
            break;
        }
        active[jstar] = false;
        let mut v = a[jstar * m..(jstar + 1) * m].to_vec();
        for qk in &q {
            let c = dot(qk, &v);
            axpy_neg(c, qk, &mut v);
        }
        let nv = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if nv == 0.0 {
            break;
        }
        v.iter_mut().for_each(|x| *x = x.scale(1.0 / nv));
        for j in (0..n).filter(|&j| active[j]) {
            let col = &mut a[j * m..(j + 1) * m];
            let c = dot(&v, col);
            axpy_neg(c, &v, col);
            norms2[j] = col.iter().map(|x| x.norm_sqr()).sum();
        }
        q.push(v);
        piv.push(jstar);
    }
    piv
}

/// Select dimensionless DLR poles `ω ∈ [-Λ, Λ]` for accuracy `eps`
/// (sorted ascending), optionally capped at `max_rank`.
pub(crate) fn select_poles(lambda: f64, eps: f64, max_rank: Option<usize>) -> Vec<f64> {
    let taus = tau_candidates(lambda);
    let omegas = omega_candidates(lambda);
    let m = taus.len();
    let n = omegas.len();
    let mut a = Vec::with_capacity(m * n);
    for &w in &omegas {
        a.extend(taus.iter().map(|&t| logistic_kernel(t, w)));
    }
    let piv = pivoted_gram_schmidt(&mut a, m, n, eps, max_rank.unwrap_or(usize::MAX));
    let mut poles: Vec<f64> = piv.iter().map(|&j| omegas[j]).collect();
    poles.sort_by(f64::total_cmp);
    poles
}

/// Select `rank` rows of the column-major `rows x cols` matrix `mat` (rows
/// are treated as vectors) and return their indices sorted ascending.
pub(crate) fn select_rows<T: GsScalar>(
    mat: &[T],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Vec<usize> {
    assert_eq!(mat.len(), rows * cols);
    // Transpose so that each row becomes a contiguous vector.
    let mut t = vec![T::zero(); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            t[i * cols + j] = mat[i + rows * j];
        }
    }
    let mut piv = pivoted_gram_schmidt(&mut t, cols, rows, 0.0, rank);
    piv.sort_unstable();
    piv
}

#[cfg(test)]
#[path = "dlr_id_tests.rs"]
mod tests;
