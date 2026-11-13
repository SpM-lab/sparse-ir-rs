//! ACA-accelerated truncated SVD for the SVE kernel matrices.
//!
//! Partial-pivot Adaptive Cross Approximation (ACA) produces a low-rank
//! factorization `K̃ ≈ U_a V_aᵀ` in `O(k²(m+n))`, replacing the `O(r·m·n)`
//! ColPivQR front-end of [`crate::tsvd::tsvd`] for the (smooth, exponentially
//! rank-decaying) SVE kernel matrices. The factors are recompressed to a true
//! SVD by two thin QRs plus a small `k×k` SVD.

use nalgebra::{ComplexField, DMatrix, RealField, SVD};
use num_traits::{One, ToPrimitive, Zero};

use crate::cost::KERNEL_ENTRY;
use crate::error::Error;
use crate::numeric::CustomNumeric;
use crate::parallel::for_each_chunk;
use crate::tsvd::{
    SVDResult, TSVDConfig, check_finite_entries, check_tsvd_input, svd_max_sweeps,
    try_svd_decompose, tsvd,
};

/// Oversampling window: the number of consecutive pivots that must pass the
/// norm-based stopping test, *inclusive of the trigger iteration*. With
/// `OVERSAMPLE = 3` the loop stops after the triggering pivot and 2 further
/// pivots that also pass; a pivot that fails the test restarts the count.
/// (The trigger counts as the 1st of `OVERSAMPLE` because
/// the residual past the true rank settles near machine epsilon but not
/// bit-exact zero, so the degenerate-pivot guard does not bound the window on
/// its own — the counter does.) Oversampling defers the final rank decision to
/// the downstream small-SVD + rtol truncation in [`tsvd_aca`], rather than
/// letting a single marginal ACA stop-step drop a real singular value.
const OVERSAMPLE: usize = 3;

/// What the threading decisions and the matrix products of the ACA need to
/// know of its element type
pub(crate) trait AcaScalar: ComplexField + ToPrimitive + Copy {
    /// Cost of one operation, in `f64` operations (see [`crate::cost`])
    const OP_COST: usize = 1;

    /// `a * b`
    fn matmul(a: &DMatrix<Self>, b: &DMatrix<Self>) -> DMatrix<Self> {
        a * b
    }
}

impl AcaScalar for f64 {}

impl AcaScalar for num_complex::Complex<f64> {}

/// Double-double products go to [`crate::ddgemm::matmul_df64`].
impl AcaScalar for crate::Df64 {
    const OP_COST: usize = crate::cost::DF64_OP;

    fn matmul(a: &DMatrix<Self>, b: &DMatrix<Self>) -> DMatrix<Self> {
        crate::ddgemm::matmul_df64(a, b)
    }
}

/// The work of `ops` operations on `T`, in `f64` operations, for the threading
/// decision of [`for_each_chunk`]
fn work_of<T: AcaScalar>(ops: usize) -> usize {
    ops * T::OP_COST
}

/// How ACA estimates `‖A_k‖_F` of its approximation `A_k = Σ_t u_t v_tᴴ`
/// for the stopping test `‖u_t‖‖v_t‖ ≤ rtol·‖A_k‖_F`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AcaNorm {
    /// `‖A_k‖_F² = Σ_{s,t} (u_sᴴu_t)(v_sᴴv_t)`, updated with the cross terms
    /// of each new pivot against all previous ones (Bebendorf & Rjasanow,
    /// Computing 70, 2003): as many operations as the residual updates.
    Exact,
    /// `Σ_t ‖u_t‖²‖v_t‖²`, which drops the cross terms (as in Börm,
    /// Grasedyck & Hackbusch, "Hierarchical matrices", lecture notes 21/2003,
    /// MPI MiS). The factors are not orthogonal, so it can be off by an O(1)
    /// factor; the caller checks the stop against the exact norm afterwards.
    SumOfSquares,
}

/// ACA factors `A_k = U Vᴴ`.
struct AcaFactors<T: ComplexField> {
    u: DMatrix<T>,
    v: DMatrix<T>,
    /// Largest `‖u_t‖‖v_t‖` of the last `OVERSAMPLE` pivots, all of which
    /// passed the stopping test, or `None` if ACA stopped otherwise (exact
    /// rank or no row left).
    trigger: Option<T::RealField>,
}

/// `out[k] = entry(index(k))` for each `k`, on the thread pool when large;
/// [`Error::NonFiniteInput`] for the first entry that is NaN or infinite
fn fill_entries<T, F>(
    out: &mut [T],
    entry: &F,
    index: impl Fn(usize) -> (usize, usize) + Sync,
) -> Result<(), Error>
where
    T: AcaScalar + Send + Sync,
    F: Fn(usize, usize) -> T + Sync,
{
    for_each_chunk(out, work_of::<T>(out.len() * KERNEL_ENTRY), |k0, chunk| {
        for (dk, x) in chunk.iter_mut().enumerate() {
            let (i, j) = index(k0 + dk);
            *x = entry(i, j);
        }
    });
    check_finite_entries(out, index)
}

/// One row or column of the residual `A - Σ_s x_s y_sᵀ`, of length `len`:
/// `out[k] = entry(index(k)) - Σ_s mul(x_s[fixed], y_s[k])`, where `fixed` is
/// the row or column index that `index` holds fixed and `mul` the product
/// `u_s[i] v_s[j]` in its operand order (a double-double product is not
/// bitwise commutative)
///
/// # Errors
/// The errors of [`fill_entries`]
fn residual<T, F>(
    len: usize,
    entry: &F,
    index: impl Fn(usize) -> (usize, usize) + Sync,
    x: &[Vec<T>],
    fixed: usize,
    y: &[Vec<T>],
    mul: impl Fn(T, T) -> T + Sync,
) -> Result<Vec<T>, Error>
where
    T: AcaScalar + Send + Sync,
    F: Fn(usize, usize) -> T + Sync,
{
    let mut out = vec![T::zero(); len];
    fill_entries(&mut out, entry, index)?;
    for_each_chunk(&mut out, work_of::<T>(len * y.len()), |k0, out| {
        for (x_s, y_s) in x.iter().zip(y) {
            let coeff = x_s[fixed];
            for (o, &y) in out.iter_mut().zip(&y_s[k0..]) {
                *o -= mul(coeff, y);
            }
        }
    });
    Ok(out)
}

/// Partial-pivot ACA with the given norm estimate for its stopping test, or
/// `None` if it does not converge.
///
/// # Errors
/// [`Error::NonFiniteInput`] if an entry that ACA reads is NaN or infinite.
/// Only its pivot rows and columns are read, so a non-finite entry elsewhere
/// goes unnoticed.
fn aca<T, F>(
    m: usize,
    n: usize,
    entry: F,
    rtol: T::RealField,
    norm: AcaNorm,
) -> Result<Option<AcaFactors<T>>, Error>
where
    T: AcaScalar + Send + Sync,
    F: Fn(usize, usize) -> T + Sync,
{
    // Running factors, stored column-wise as Vec<column>.
    let mut u_cols: Vec<Vec<T>> = Vec::new(); // each length m
    let mut v_cols: Vec<Vec<T>> = Vec::new(); // each length n

    let mut used_rows = vec![false; m];
    let mut i_star = 0usize; // initial pivot row
    let mut approx_norm_sq = T::RealField::zero();
    // Largest ‖u_t‖‖v_t‖ of the pivots that passed the stopping test in a row
    let mut trigger = None;
    // Number of those pivots
    let mut since_trigger = 0;

    // Whether ACA stopped (by its stopping test, at the exact rank, or with
    // no row left) rather than running into the full rank
    let converged = loop {
        if u_cols.len() == m.min(n) {
            break trigger.is_some();
        }
        used_rows[i_star] = true;

        // Row residual a[j] = A[i*, j] - Σ_s u_s[i*] * v_s[j]
        let a = residual(
            n,
            &entry,
            |j| (i_star, j),
            &u_cols,
            i_star,
            &v_cols,
            |u, v| u * v,
        )?;

        // Column pivot j* = argmax_j |a[j]|.
        let mut j_star = 0usize;
        let mut best = T::RealField::zero();
        for (j, x) in a.iter().enumerate() {
            let m_abs = x.modulus();
            if m_abs > best {
                best = m_abs;
                j_star = j;
            }
        }
        let delta = a[j_star];

        // Column residual b[i] = A[i, j*] - Σ_s u_s[i] * v_s[j*]
        let b = residual(
            m,
            &entry,
            |i| (i, j_star),
            &v_cols,
            j_star,
            &u_cols,
            |v, u| u * v,
        )?;

        // A row whose residual vanishes adds nothing: move on to another row
        // (a zero row says nothing about the others), and conclude only when
        // every row has been tried.
        if delta.modulus().is_zero() {
            if trigger.is_none()
                && let Some(next) = next_pivot_row(&used_rows, u_cols.last())
            {
                i_star = next;
                continue;
            }
            break true;
        }

        // New rank-1 factor: u_t = b, v_t = a / delta  (so u_t v_tᵀ matches the
        // residual at row i* and column j*).
        let inv = T::one() / delta;
        let v_t: Vec<T> = a.iter().map(|&x| x * inv).collect();
        let u_t = b;

        // Incremental Frobenius norm of the approximation:
        // ‖A_k‖² = ‖A_{k-1}‖² + ‖u_t‖²‖v_t‖² + 2 Re Σ_{s<t}(u_sᴴu_t)(v_sᴴv_t)
        let u_t_nrm2 = u_t
            .iter()
            .fold(T::RealField::zero(), |acc, &x| acc + x.modulus_squared());
        let v_t_nrm2 = v_t
            .iter()
            .fold(T::RealField::zero(), |acc, &x| acc + x.modulus_squared());
        let self_term = u_t_nrm2 * v_t_nrm2;

        // (u_sᴴu_t)(v_sᴴv_t) for each s < t, in parallel over s, for the
        // exact norm only
        let n_cross = if norm == AcaNorm::Exact {
            u_cols.len()
        } else {
            0
        };
        let mut prods: Vec<T> = vec![T::zero(); n_cross];
        for_each_chunk(&mut prods, work_of::<T>((m + n) * n_cross), |s0, prods| {
            for (ds, prod) in prods.iter_mut().enumerate() {
                let s = s0 + ds;
                *prod = dot_conj(&u_cols[s], &u_t) * dot_conj(&v_cols[s], &v_t);
            }
        });
        let mut cross = T::RealField::zero();
        for prod in prods {
            cross += prod.real();
        }
        approx_norm_sq += self_term.clone() + (T::RealField::one() + T::RealField::one()) * cross;

        u_cols.push(u_t);
        v_cols.push(v_t);

        // Stopping: ‖u_t‖‖v_t‖ = sqrt(self_term) ≤ rtol · ‖A_k‖_F for
        // OVERSAMPLE pivots in a row. A larger pivot clears the trigger: a
        // small pivot of partial-pivot ACA does not mean the residual is
        // small, its row may just miss the rest of the matrix.
        let term = self_term.sqrt();
        if term <= rtol.clone() * approx_norm_sq.clone().sqrt() {
            trigger = Some(match trigger {
                Some(t) if t > term => t,
                _ => term,
            });
            since_trigger += 1;
            if since_trigger >= OVERSAMPLE {
                break true;
            }
        } else {
            trigger = None;
            since_trigger = 0;
        }

        // Next pivot row: largest |b[i]| among unused rows. No unused row
        // remains once every row has been pivoted on.
        match next_pivot_row(&used_rows, u_cols.last()) {
            Some(next) => i_star = next,
            None => break true,
        }
    };

    if !converged {
        return Ok(None);
    }

    let k = u_cols.len();
    Ok(Some(AcaFactors {
        u: DMatrix::from_fn(m, k, |i, t| u_cols[t][i]),
        v: DMatrix::from_fn(n, k, |j, t| v_cols[t][j]),
        trigger,
    }))
}

/// `xᴴy`
fn dot_conj<T: ComplexField + Copy>(x: &[T], y: &[T]) -> T {
    x.iter()
        .zip(y)
        .fold(T::zero(), |acc, (&x, &y)| acc + x.conjugate() * y)
}

/// The unused row with the largest `|u[i]|`, the first unused row if there is
/// no `u` yet or it vanishes on every unused row, or `None` if every row is
/// used.
fn next_pivot_row<T: ComplexField>(used: &[bool], u: Option<&Vec<T>>) -> Option<usize> {
    let mut best: Option<(usize, T::RealField)> = None;
    for (i, _) in used.iter().enumerate().filter(|&(_, &used)| !used) {
        let size = u.map_or_else(T::RealField::zero, |u| u[i].clone().modulus());
        if best.as_ref().is_none_or(|(_, b)| size > *b) {
            best = Some((i, size));
        }
    }
    best.map(|(i, _)| i)
}

/// Rank-revealing truncated SVD via ACA + dual-QR recompression of the
/// `m × n` matrix whose entries are `entry(i, j)`.
///
/// `K̃ ≈ U_a V_aᵀ = Q_u (R_u R_vᵀ) Q_vᵀ = Q_u M Q_vᵀ`; SVD of the small `k×k`
/// `M = Û Σ V̂ᵀ` gives `K̃ ≈ (Q_u Û) Σ (Q_v V̂)ᵀ`. ACA evaluates only its pivot
/// rows and columns; the whole matrix is formed only if ACA fails to converge
/// and the result is taken from the CPQR-based [`tsvd`] instead.
///
/// `config.rtol` is also the ACA stopping tolerance. Partial-pivot ACA has
/// no a-priori error bound, and its stopping test (even with the exact norm)
/// can underestimate the residual, so singular values close to `rtol` are not
/// reliable: callers must take `rtol` well below the accuracy they need, as
/// the SVE does with its default `2ε` and, in
/// [`crate::sve::SVEOptions::cutoff_from_epsilon`], through
/// [`crate::sve::ACCURACY_CUTOFF_FACTOR`], whose value was validated with this
/// stopping rule; changing either (the norm estimate, the redo check,
/// `OVERSAMPLE`) requires checking the SVE accuracy again.
///
/// # Errors
/// * The input errors of [`tsvd`]: [`Error::EmptyInput`] and
///   [`Error::InvalidParameter`]
/// * [`Error::NonFiniteInput`] if an entry that ACA reads is NaN or infinite
///   (or any entry, on the CPQR fallback)
/// * [`Error::DecompositionFailed`] if an SVD does not converge
pub(crate) fn tsvd_aca<T, F>(
    m: usize,
    n: usize,
    entry: F,
    config: TSVDConfig<T>,
) -> Result<SVDResult<T>, Error>
where
    T: AcaScalar + RealField + std::fmt::Debug + ToPrimitive + CustomNumeric + Send + Sync,
    F: Fn(usize, usize) -> T + Sync,
{
    check_tsvd_input(m, n, config.rtol)?;

    // Step 1: ACA low-rank factorization on the weighted matrix, stopped with
    // the cheap norm estimate. For a RealField T, T::RealField == T, so
    // config.rtol is a valid rtol.
    let rtol_f64 = CustomNumeric::to_f64(config.rtol);
    let decompose = |norm| -> Result<Option<_>, Error> {
        let Some(factors) = aca(m, n, &entry, config.rtol, norm)? else {
            return Ok(None);
        };
        let svd = recompress(factors.u, factors.v, rtol_f64)?;
        Ok(Some((svd, factors.trigger)))
    };
    let svd = match decompose(AcaNorm::SumOfSquares)? {
        // The stop must also hold for the exact norm ‖A_k‖_F = ‖Σ‖_F (up to
        // the singular values below rtol, dropped in the recompression);
        // otherwise ACA stopped early and is redone with the exact norm.
        Some((svd, Some(trigger))) if trigger > config.rtol * svd.s.norm() => {
            decompose(AcaNorm::Exact)?.map(|(svd, _)| svd)
        }
        result => result.map(|(svd, _)| svd),
    };
    match svd {
        Some(svd) => Ok(svd),
        // Fallback: CPQR path on the full matrix if ACA did not converge.
        None => {
            // Column-major: element k is (k % m, k / m)
            let mut data = vec![T::zero(); m * n];
            fill_entries(&mut data, &entry, |k| (k % m, k / m))?;
            tsvd(&DMatrix::from_vec(m, n, data), config)
        }
    }
}

/// SVD of `U_a V_aᵀ` (k ≥ 0 columns): orthonormalize the factors,
/// `U_a = Q_u R_u` and `V_a = Q_v R_v`, so that `U_a V_aᵀ = Q_u M Q_vᵀ` with
/// `M = R_u R_vᵀ`; take the SVD of the small `M` and lift its singular
/// vectors back.
///
/// # Errors
/// The errors of [`try_svd_decompose`] for the small SVD
fn recompress<T>(u_a: DMatrix<T>, v_a: DMatrix<T>, rtol: f64) -> Result<SVDResult<T>, Error>
where
    T: AcaScalar + RealField + ToPrimitive + CustomNumeric + Send + Sync,
{
    if u_a.ncols() == 0 {
        return Ok(SVDResult {
            u: DMatrix::zeros(u_a.nrows(), 0),
            s: nalgebra::DVector::zeros(0),
            v: DMatrix::zeros(v_a.nrows(), 0),
            rank: 0,
        });
    }
    match recompress_cholesky(&u_a, &v_a, rtol)? {
        Some(svd) => Ok(svd),
        None => recompress_householder(u_a, v_a, rtol),
    }
}

/// Largest loss of orthogonality, bounded by `κ²·ε` for a factor with
/// condition number `κ` (after column scaling) in working precision `ε`,
/// accepted from Cholesky QR in [`recompress_cholesky`]. Far below the
/// precision (`f64`) in which the SVE results are returned.
const MAX_CHOLESKY_QR_LOSS: f64 = 1e-20;

/// SVD of `U_a V_aᵀ` by Cholesky QR of the factors, or `None` where Cholesky QR
/// would not keep orthogonality to [`MAX_CHOLESKY_QR_LOSS`].
///
/// With the column scalings `U_a = Ũ D_u` and `V_a = Ṽ D_v` and the Cholesky
/// factors `ŨᵀŨ = R_uᵀR_u` and `ṼᵀṼ = R_vᵀR_v`, `Q_u = Ũ R_u⁻¹` and `Q_v`
/// have orthonormal columns up to `κ(Ũ)²·ε` (Yamamoto et al., ETNA 44, 2015),
/// and `U_a V_aᵀ = Q_u (R_u D_u D_v R_vᵀ) Q_vᵀ`. From the SVD `Û Σ V̂ᵀ` of the
/// middle `k×k` matrix, `U = Ũ (R_u⁻¹ Û)` and `V = Ṽ (R_v⁻¹ V̂)`: two Gram
/// products and two lifts, all matrix products, instead of two Householder
/// QRs, the formation of their `Q`, and the lifts. The ACA factors are well
/// conditioned once their columns are normalized (κ(Ũ) ≲ 10⁵, κ(Ṽ) ≲ 10² for
/// the SVE blocks of e^{-Λy(x+1)/2} / (1 + e^{-Λy}) up to Λ = 10⁶), so this
/// is taken in double-double precision when `κ²·ε` is small enough, and
/// Householder QR otherwise; in `f64`, `κ²·ε` always exceeds the bound.
///
/// # Errors
/// The errors of [`try_svd_decompose`] for the small SVD
fn recompress_cholesky<T>(
    u_a: &DMatrix<T>,
    v_a: &DMatrix<T>,
    rtol: f64,
) -> Result<Option<SVDResult<T>>, Error>
where
    T: AcaScalar + RealField + ToPrimitive + CustomNumeric + Send + Sync,
{
    let eps = CustomNumeric::to_f64(<T as CustomNumeric>::epsilon());
    if eps > MAX_CHOLESKY_QR_LOSS {
        return Ok(None);
    }
    // Ũ = U_a D⁻¹ and the Cholesky factor R of ŨᵀŨ, if κ(R)²·ε is small.
    let factor = |a: &DMatrix<T>| -> Option<(DMatrix<T>, Vec<T>, DMatrix<T>)> {
        let d: Vec<T> = a.column_iter().map(|c| c.norm()).collect();
        if d.iter().any(|x| x.is_zero()) {
            return None;
        }
        let mut scaled = a.clone();
        for (mut col, &dj) in scaled.column_iter_mut().zip(&d) {
            col /= dj;
        }
        let r = T::matmul(&scaled.transpose(), &scaled)
            .cholesky()?
            .l()
            .transpose();
        let r_f64 = r.map(|x| CustomNumeric::to_f64(x));
        // Where ŨᵀŨ is not numerically positive definite, the double-double
        // Cholesky can return NaN instead of failing; an SVD of NaN never
        // converges.
        if !r_f64.iter().all(|x| x.is_finite()) {
            return None;
        }
        // nalgebra's singular_values() with an iteration limit
        let n = r_f64.nrows();
        let sv = SVD::try_new_unordered(
            r_f64,
            false,
            false,
            5.0 * f64::EPSILON,
            svd_max_sweeps(n, n),
        )?
        .singular_values;
        let kappa = sv.max() / sv.min();
        (kappa * kappa * eps <= MAX_CHOLESKY_QR_LOSS).then_some((scaled, d, r))
    };
    let (Some((u_t, d_u, r_u)), Some((v_t, d_v, r_v))) = (factor(u_a), factor(v_a)) else {
        return Ok(None);
    };

    let mut r_v_t = r_v.transpose();
    for (t, mut row) in r_v_t.row_iter_mut().enumerate() {
        row *= d_u[t] * d_v[t];
    }
    let svd = try_svd_decompose(&T::matmul(&r_u, &r_v_t), rtol)?;
    // R_u and R_v are Cholesky factors of positive definite matrices, so
    // they are nonsingular; should a solve fail anyway, take the Householder
    // path.
    let (Some(x_u), Some(x_v)) = (
        r_u.solve_upper_triangular(&svd.u),
        r_v.solve_upper_triangular(&svd.v),
    ) else {
        return Ok(None);
    };
    Ok(Some(SVDResult {
        u: T::matmul(&u_t, &x_u),
        s: svd.s,
        v: T::matmul(&v_t, &x_v),
        rank: svd.rank,
    }))
}

/// SVD of `U_a V_aᵀ` by Householder QR of the factors.
///
/// # Errors
/// The errors of [`try_svd_decompose`] for the small SVD
fn recompress_householder<T>(
    u_a: DMatrix<T>,
    v_a: DMatrix<T>,
    rtol: f64,
) -> Result<SVDResult<T>, Error>
where
    T: AcaScalar + RealField + ToPrimitive + CustomNumeric + Send + Sync,
{
    // Thin QR of both factors (economy form: Q is m×k / n×k, R is k×k).
    let thin_qr = |x: DMatrix<T>| {
        let qr = x.qr();
        (qr.q(), qr.r())
    };
    let (q_u, r_u) = thin_qr(u_a);
    let (q_v, r_v) = thin_qr(v_a);

    // Small SVD of M = R_u · R_vᵀ (k×k).
    let svd = try_svd_decompose(&(r_u * r_v.transpose()), rtol)?;

    // Lift back: U = Q_u Û, V = Q_v V̂.
    Ok(SVDResult {
        u: T::matmul(&q_u, &svd.u),
        s: svd.s,
        v: T::matmul(&q_v, &svd.v),
        rank: svd.rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Df64;
    use crate::tsvd::{TSVDConfig, tsvd};
    use nalgebra::DMatrix;

    /// ACA with the exact norm estimate on the formed matrix `a`.
    fn aca_low_rank<T>(a: &DMatrix<T>, rtol: T::RealField) -> Option<(DMatrix<T>, DMatrix<T>)>
    where
        T: AcaScalar + Send + Sync,
    {
        aca(a.nrows(), a.ncols(), |i, j| a[(i, j)], rtol, AcaNorm::Exact)
            .unwrap()
            .map(|f| (f.u, f.v))
    }

    /// [`tsvd_aca`] on the formed matrix `a`.
    fn tsvd_aca_dense(a: &DMatrix<f64>, config: TSVDConfig<f64>) -> Result<SVDResult<f64>, Error> {
        tsvd_aca(a.nrows(), a.ncols(), |i, j| a[(i, j)], config)
    }

    /// A rank-3 matrix built as an explicit sum of 3 outer products must be
    /// recovered by ACA to `rtol·σ₀`, using at most a few extra pivots.
    #[test]
    fn recovers_known_rank3_f64() {
        let m = 40;
        let n = 30;
        // Three smooth, linearly independent outer products.
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..3)
            .map(|t| {
                let u: Vec<f64> = (0..m)
                    .map(|i| ((i as f64 + 1.0) * (t as f64 + 1.0)).sin())
                    .collect();
                let v: Vec<f64> = (0..n)
                    .map(|j| ((j as f64 + 1.0) / (t as f64 + 2.0)).cos())
                    .collect();
                (u, v)
            })
            .collect();
        let a = DMatrix::<f64>::from_fn(m, n, |i, j| cols.iter().map(|(u, v)| u[i] * v[j]).sum());

        let (u_a, v_a) = aca_low_rank(&a, 1e-12).expect("ACA should converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < 1e-10,
            "relative reconstruction error too large: {err:e}"
        );
        assert!(u_a.ncols() <= 6, "rank overshoot: {}", u_a.ncols());
    }

    /// Same recovery in extended precision (Df64) — the SVE work type for small eps.
    ///
    /// The matrix is built as an explicit sum of 3 outer products whose factor
    /// vectors are ratios of small integers, computed entirely in `Df64`
    /// arithmetic (no `f64` transcendental functions in the construction).
    /// Such ratios are not exactly representable in binary, so the resulting
    /// matrix is a genuine rank-3 matrix down to Df64 precision (~1e-32), not
    /// merely to f64 precision (~1e-16) as an `f64`-built-then-converted
    /// matrix would be. This lets rtol/error bounds far tighter than 1e-16
    /// actually exercise the extended-precision path instead of being
    /// trivially satisfied by f64-level agreement.
    #[test]
    fn recovers_known_rank3_df64() {
        let m = 24;
        let n = 20;
        // Three outer products u_t[i] = (i+1)/(t+3+offset), v_t[j] = (j+2)/(t+5+offset),
        // built directly in Df64 from small integer ratios.
        let cols: Vec<(Vec<Df64>, Vec<Df64>)> = (0..3)
            .map(|t| {
                let u: Vec<Df64> = (0..m)
                    .map(|i| Df64::from(i as f64 + 1.0) / Df64::from(t as f64 + 3.0))
                    .collect();
                let v: Vec<Df64> = (0..n)
                    .map(|j| Df64::from(j as f64 + 2.0) / Df64::from(t as f64 + 5.0))
                    .collect();
                (u, v)
            })
            .collect();
        let a = DMatrix::<Df64>::from_fn(m, n, |i, j| {
            cols.iter()
                .fold(Df64::from(0.0), |acc, (u, v)| acc + u[i] * v[j])
        });
        let (u_a, v_a) = aca_low_rank(&a, Df64::from(1e-28)).expect("ACA should converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < Df64::from(1e-24),
            "df64 reconstruction error too large: achieved {err:?}"
        );
    }

    /// ACA itself is generic over the scalar: it also works for `Complex<f64>`.
    #[test]
    fn works_for_complex() {
        use num_complex::Complex64;
        let m = 16;
        let n = 12;
        let a = DMatrix::<Complex64>::from_fn(m, n, |i, j| {
            (0..2)
                .map(|t| {
                    Complex64::new(((i + t) as f64).cos(), ((j + t) as f64).sin())
                        * Complex64::new(((j * (t + 1)) as f64).sin(), 0.3)
                })
                .sum()
        });
        let (u_a, v_a) = aca_low_rank(&a, 1e-12).expect("converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < 1e-9,
            "complex reconstruction error too large: {err:e}"
        );
    }

    /// Build a smooth, exponentially rank-decaying matrix (a stand-in for the
    /// weighted SVE kernel) and check tsvd_aca's singular values match the
    /// CPQR tsvd's to rtol order, with orthonormal U,V.
    fn smooth_matrix(m: usize, n: usize) -> DMatrix<f64> {
        DMatrix::from_fn(m, n, |i, j| {
            let x = i as f64 / m as f64;
            let y = j as f64 / n as f64;
            (-(x - y).powi(2) * 8.0).exp() + 0.5 * (3.0 * x + 2.0 * y).cos()
        })
    }

    /// Every row of a zero matrix has a vanishing residual: ACA tries them
    /// all and stops with no factor, which the recompression returns as
    /// rank 0.
    #[test]
    fn tsvd_aca_of_a_zero_matrix_has_rank_zero() {
        let svd = tsvd_aca(5, 4, |_, _| 0.0, TSVDConfig::new(1e-10)).unwrap();
        assert_eq!(svd.rank, 0);
        assert_eq!(svd.s.len(), 0);
        assert_eq!(svd.u.shape(), (5, 0));
        assert_eq!(svd.v.shape(), (4, 0));
    }

    #[test]
    fn tsvd_aca_matches_cpqr_f64() {
        let a = smooth_matrix(60, 50);
        let rtol = 2.0 * f64::EPSILON;
        let cpqr = tsvd(&a, TSVDConfig::new(rtol)).unwrap();
        let aca = tsvd_aca_dense(&a, TSVDConfig::new(rtol)).unwrap();

        let k = cpqr.rank.min(aca.rank);
        assert!(k > 0);
        for i in 0..k {
            let rel = (cpqr.s[i] - aca.s[i]).abs() / cpqr.s[0];
            // rtol requested above is 2·EPSILON (~4.4e-16); 1e-10 is a loose
            // bound well above that, to absorb recompression round-off
            // (dual-QR + small-SVD lift), not a tight rtol-order check.
            assert!(
                rel < 1e-10,
                "sv {i} differs: cpqr={} aca={} rel={rel:e}",
                cpqr.s[i],
                aca.s[i]
            );
        }
        // Orthonormal columns: UᵀU ≈ I.
        let utu = aca.u.transpose() * &aca.u;
        let ident = DMatrix::<f64>::identity(aca.u.ncols(), aca.u.ncols());
        assert!((utu - ident).norm() < 1e-9, "U not orthonormal");
        // Reconstruction error ≲ rtol·σ₀ (with margin).
        let recon = &aca.u * DMatrix::from_diagonal(&aca.s) * aca.v.transpose();
        assert!(
            (&a - &recon).norm() / aca.s[0] < 1e-8,
            "reconstruction too large"
        );
    }

    /// Cholesky-QR recompression of the ACA factors of SVE kernel blocks
    /// (double-double) agrees with Householder recompression: same singular
    /// values and functions to far below `f64` precision, and orthonormal
    /// singular vectors. In `f64` it declines.
    #[test]
    fn cholesky_recompression_matches_householder_df64() {
        use crate::test_kernels::ExpKernel;
        use crate::unstable::sve::{CentrosymmSVE, SVEStrategy};

        let kernel = ExpKernel { a: 10.0, nseg: 8 };
        let sve = CentrosymmSVE::<Df64, _>::new(kernel, 1e-12).unwrap();
        let rtol = 1e-18;
        for block in sve.blocks() {
            let (m, n) = (block.nrows(), block.ncols());
            let (u_a, v_a) = aca(
                m,
                n,
                |i, j| block.entry(i, j),
                Df64::from(rtol),
                AcaNorm::Exact,
            )
            .unwrap()
            .map(|f| (f.u, f.v))
            .unwrap();
            let u_f64 = u_a.map(|x| x.to_f64());
            let v_f64 = v_a.map(|x| x.to_f64());
            assert!(recompress_cholesky(&u_f64, &v_f64, rtol).unwrap().is_none());

            let chol = recompress_cholesky(&u_a, &v_a, rtol)
                .unwrap()
                .expect("applies in Df64");
            let hh = recompress_householder(u_a, v_a, rtol).unwrap();
            assert_eq!(chol.rank, hh.rank);
            let s0 = hh.s[0];
            let mut ds = 0.0f64;
            let mut du = 0.0f64;
            for l in 0..hh.rank {
                ds = ds.max(((chol.s[l] - hh.s[l]) / s0).to_f64().abs());
                // Singular vectors up to sign, weighted by s_l / s_0.
                let sign = if (chol.u.column(l).dot(&hh.u.column(l))) < Df64::from(0.0) {
                    Df64::from(-1.0)
                } else {
                    Df64::from(1.0)
                };
                let w = (hh.s[l] / s0).to_f64();
                for (a, b) in chol.u.column(l).iter().zip(hh.u.column(l).iter()) {
                    du = du.max((*a * sign - *b).to_f64().abs() * w);
                }
            }
            let gram = chol.u.transpose() * &chol.u;
            let ortho = (gram - DMatrix::<Df64>::identity(chol.rank, chol.rank))
                .iter()
                .map(|x| CustomNumeric::to_f64(*x).abs())
                .fold(0.0f64, f64::max);
            assert!(ds < 1e-28, "singular values differ: {ds:e}");
            assert!(du < 1e-26, "singular vectors differ: {du:e}");
            assert!(ortho < 1e-20, "U not orthonormal: {ortho:e}");
        }
    }

    /// When the sum-of-squares estimate overstates ‖A_k‖_F so much that the
    /// stopping test fails against the exact norm, `tsvd_aca` redoes ACA
    /// with the exact norm. The small integer matrix below was found by
    /// search: its ACA terms cancel strongly, so the estimate is too large.
    #[test]
    fn tsvd_aca_redoes_aca_when_estimate_stops_too_early() {
        let rows: [[f64; 6]; 6] = [
            [0.0, 1.0, -3.0, 3.0, -3.0, -3.0],
            [-3.0, 0.0, -3.0, -1.0, 2.0, 3.0],
            [0.0, 0.0, -1.0, 3.0, 0.0, -3.0],
            [2.0, 3.0, -2.0, -3.0, -1.0, -3.0],
            [-60.0, -2.0, 90.0, -2.0, 0.0, 3.0],
            [-2.0, 0.0, -2.0, -1.0, 3.0, -3.0],
        ];
        let (m, n) = (6, 6);
        let a = DMatrix::<f64>::from_fn(m, n, |i, j| rows[i][j]);
        let rtol = 0.05;
        let entry = |i: usize, j: usize| a[(i, j)];

        // The estimate stops at a pivot the exact norm would not accept,
        // missing a singular value above rtol ...
        let fast = aca(m, n, entry, rtol, AcaNorm::SumOfSquares)
            .unwrap()
            .unwrap();
        let trigger = fast.trigger.expect("stopped by the norm test");
        let fast = recompress(fast.u, fast.v, rtol).unwrap();
        assert!(
            trigger > rtol * fast.s.norm(),
            "the case no longer needs a redo"
        );

        // ... so the result is that of ACA with the exact norm, which has it.
        let exact = aca(m, n, entry, rtol, AcaNorm::Exact).unwrap().unwrap();
        assert!(exact.trigger.is_some(), "exact-norm ACA converges");
        let exact = recompress(exact.u, exact.v, rtol).unwrap();
        assert!(exact.rank > fast.rank);
        let result = tsvd_aca(m, n, entry, TSVDConfig::new(rtol)).unwrap();
        assert_eq!(result.rank, exact.rank);
        assert_eq!(result.s, exact.s);
        assert_eq!(result.u, exact.u);
        assert_eq!(result.v, exact.v);
    }

    /// Stopping ACA with the sum-of-squares norm estimate keeps the exact
    /// stopping test satisfied and the singular values of the exact-norm ACA,
    /// on SVE blocks of a smooth kernel. The estimate is not a bound (for
    /// exp(10 x y) on these segments it stops too early, and `tsvd_aca` redoes
    /// ACA, see `tsvd_aca_redoes_aca_when_estimate_stops_too_early`); the
    /// kernel here is one for which it holds.
    #[test]
    fn sum_of_squares_stop_matches_exact_norm() {
        use crate::test_kernels::ExpKernel;
        use crate::unstable::sve::{CentrosymmSVE, SVEStrategy};

        let kernel = ExpKernel { a: 20.0, nseg: 8 };
        let sve = CentrosymmSVE::<Df64, _>::new(kernel, 1e-12).unwrap();
        let rtol = 1e-18;
        for block in sve.blocks() {
            let (m, n) = (block.nrows(), block.ncols());
            let entry = |i, j| block.entry(i, j);
            let rtol_t = Df64::from(rtol);
            let exact = aca(m, n, entry, rtol_t, AcaNorm::Exact).unwrap().unwrap();
            let fast = aca(m, n, entry, rtol_t, AcaNorm::SumOfSquares)
                .unwrap()
                .unwrap();
            let exact = recompress(exact.u, exact.v, rtol).unwrap();
            let trigger = fast.trigger.expect("stopped by the norm test");
            let fast = recompress(fast.u, fast.v, rtol).unwrap();
            assert!(trigger <= rtol_t * fast.s.norm());
            let ds = (0..exact.rank.min(fast.rank))
                .map(|l| ((exact.s[l] - fast.s[l]) / exact.s[0]).to_f64().abs())
                .fold(0.0f64, f64::max);
            assert!(ds < 10.0 * rtol, "singular values differ by {ds:e}");
            assert!(exact.rank.abs_diff(fast.rank) <= 1);
        }
    }

    /// On a lazily evaluated matrix, `tsvd_aca` gives the same result as
    /// [`tsvd_aca_dense`] on the formed matrix while evaluating only the pivot rows
    /// and columns, `(m + n)` entries per ACA step.
    #[test]
    fn tsvd_aca_evaluates_few_entries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (m, n) = (300, 400);
        let a = smooth_matrix(m, n);
        let rtol = 1e-12;
        let calls = AtomicUsize::new(0);
        let lazy = tsvd_aca(
            m,
            n,
            |i, j| {
                calls.fetch_add(1, Ordering::Relaxed);
                a[(i, j)]
            },
            TSVDConfig::new(rtol),
        )
        .unwrap();
        let dense = tsvd_aca_dense(&a, TSVDConfig::new(rtol)).unwrap();
        assert_eq!(lazy.s, dense.s);
        assert_eq!(lazy.u, dense.u);
        assert_eq!(lazy.v, dense.v);

        let (u_a, _) = aca_low_rank(&a, rtol).unwrap();
        let steps = u_a.ncols();
        assert_eq!(calls.load(Ordering::Relaxed), steps * (m + n));
        assert!(steps * (m + n) < m * n / 4, "ACA took {steps} steps");
    }

    #[test]
    fn tsvd_aca_rejects_empty_matrices_and_bad_tolerances() {
        let entry = |_: usize, _: usize| 1.0;
        assert!(matches!(
            tsvd_aca(0, 3, entry, TSVDConfig::new(1e-10)),
            Err(Error::EmptyInput { name: "matrix" })
        ));
        for rtol in [0.0, 1.0, -1e-3, f64::NAN] {
            assert!(matches!(
                tsvd_aca(3, 3, entry, TSVDConfig::new(rtol)),
                Err(Error::InvalidParameter { name: "rtol", .. })
            ));
        }
    }

    /// A NaN or infinite entry that ACA reads is reported as by `tsvd`, at
    /// its index (it would otherwise propagate into the factors)
    #[test]
    fn tsvd_aca_reports_non_finite_entries() {
        let a = smooth_matrix(20, 15);
        for bad in [f64::NAN, f64::INFINITY] {
            // Row 0 is the first pivot row, read in full
            let entry = |i: usize, j: usize| if (i, j) == (0, 7) { bad } else { a[(i, j)] };
            let err = tsvd_aca(20, 15, entry, TSVDConfig::new(1e-12)).unwrap_err();
            assert!(
                matches!(&err, Error::NonFiniteInput { name: "matrix", index, .. } if index == &vec![0, 7]),
                "{err:?}"
            );
        }
    }

    /// When ACA does not converge, `tsvd_aca` returns the CPQR `tsvd` of the
    /// formed matrix.
    ///
    /// ACA fails only if it exhausts all `min(m, n)` pivots without its
    /// stopping test ever holding, i.e. for a full-rank matrix whose singular
    /// values do not decay below `rtol·‖A‖`. The matrix below is dense (no
    /// degenerate pivot can stop ACA early), diagonally dominant (genuinely
    /// full rank) and has a flat spectrum.
    #[test]
    fn tsvd_aca_falls_back_to_cpqr() {
        let m = 100;
        let n = 90;
        let a = DMatrix::from_fn(m, n, |i, j| {
            if i == j {
                3.0
            } else {
                1.0 / (2.0 + (i as f64 - j as f64).abs())
            }
        });
        let rtol = 1e-12;

        assert!(
            aca_low_rank(&a, rtol).is_none(),
            "test must exercise the None fallback"
        );

        let cpqr = tsvd(&a, TSVDConfig::new(rtol)).unwrap();
        let aca = tsvd_aca_dense(&a, TSVDConfig::new(rtol)).unwrap();

        // The fallback forms the matrix from the entries (on the thread pool
        // at this size) and must hand the CPQR exactly `a`.
        assert!(work_of::<f64>(m * n * KERNEL_ENTRY) >= crate::parallel::PARALLEL_MIN_WORK);
        assert_eq!(cpqr.rank, aca.rank, "fallback must reproduce CPQR's rank");
        assert_eq!(cpqr.s, aca.s);
        assert_eq!(cpqr.u, aca.u);
        assert_eq!(cpqr.v, aca.v);
    }

    /// Cholesky QR declines factors that it cannot orthogonalize to
    /// [`MAX_CHOLESKY_QR_LOSS`] (a zero column, a singular Gram matrix, or
    /// κ²·ε above the bound), on either side, and the recompression then
    /// takes the Householder path.
    #[test]
    fn cholesky_recompression_declines_bad_factors() {
        let d = Df64::from;
        let m = 6;
        let rtol = 1e-30;
        let good = DMatrix::from_fn(m, 2, |i, j| d(((i + 1) as f64 * (j as f64 + 0.5)).sin()));
        assert!(recompress_cholesky(&good, &good, rtol).unwrap().is_some());

        let mut zero_column = good.clone();
        zero_column.column_mut(1).fill(d(0.0));
        // Two equal unit columns: the Gram matrix is exactly [[1, 1], [1, 1]].
        let singular = DMatrix::from_fn(m, 2, |i, _| d(if i == 0 { 1.0 } else { 0.0 }));
        // κ ≈ 2·10⁸ after column scaling, so κ²·ε ≈ 10⁻¹⁵ > 10⁻²⁰.
        let ill_conditioned = DMatrix::from_fn(m, 2, |i, j| match (i, j) {
            (0, _) => d(1.0),
            (1, 1) => d(1e-8),
            _ => d(0.0),
        });
        for (name, bad) in [
            ("zero column", zero_column),
            ("singular Gram matrix", singular),
            ("ill-conditioned", ill_conditioned),
        ] {
            for (u_a, v_a) in [(bad.clone(), good.clone()), (good.clone(), bad.clone())] {
                assert!(
                    recompress_cholesky(&u_a, &v_a, rtol).unwrap().is_none(),
                    "{name}"
                );
                let svd = recompress(u_a.clone(), v_a.clone(), rtol).unwrap();
                let hh = recompress_householder(u_a, v_a, rtol).unwrap();
                assert_eq!(svd.s, hh.s, "{name}");
                assert_eq!(svd.u, hh.u, "{name}");
                assert_eq!(svd.v, hh.v, "{name}");
            }
        }
    }

    /// Rows whose residual vanishes do not stop ACA: here the first pivot
    /// row is zero, and so is one in the middle.
    #[test]
    fn tsvd_aca_skips_zero_rows() {
        let (m, n) = (30, 25);
        let a = DMatrix::from_fn(m, n, |i, j| {
            if i == 0 || i == 17 {
                0.0
            } else {
                (0..3)
                    .map(|t| {
                        ((i * (t + 1)) as f64 * 0.3).sin()
                            * (((j + 1) * (t + 1)) as f64 * 0.2).cos()
                    })
                    .sum()
            }
        });
        let rtol = 1e-12;
        let cpqr = tsvd(&a, TSVDConfig::new(rtol)).unwrap();
        let aca = tsvd_aca_dense(&a, TSVDConfig::new(rtol)).unwrap();
        assert_eq!(cpqr.rank, 3);
        assert_eq!(aca.rank, cpqr.rank);
        for l in 0..3 {
            assert!((aca.s[l] - cpqr.s[l]).abs() < 1e-12 * cpqr.s[0]);
        }
    }

    /// A small pivot followed by large ones does not stop ACA: the stopping
    /// test held at the second pivot (1e-20 on the diagonal), but the next
    /// rows hold three more unit singular values. With the trigger never
    /// cleared, ACA stopped two pivots later and missed the last one.
    #[test]
    fn tsvd_aca_continues_after_a_small_pivot() {
        let diag = [1.0, 1e-20, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let a = DMatrix::from_fn(8, 8, |i, j| if i == j { diag[i] } else { 0.0 });
        let rtol = 1e-15;
        let aca = tsvd_aca_dense(&a, TSVDConfig::new(rtol)).unwrap();
        assert_eq!(aca.rank, 4, "s = {:?}", aca.s.as_slice());
        for l in 0..4 {
            assert!((aca.s[l] - 1.0).abs() < 1e-15, "s = {:?}", aca.s.as_slice());
        }
    }

    #[test]
    fn work_counts_double_double_operations_as_f64_ones() {
        assert_eq!(work_of::<f64>(10), 10);
        assert_eq!(work_of::<Df64>(10), 160);
    }
}
