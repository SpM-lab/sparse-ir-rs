//! Sampling-point selection by pairs of rows.
//!
//! Selects pairs of rows of a real matrix `A` (pair `i` = rows `2i`, `2i + 1`)
//! so that the rows of the first `n_cols` columns, together with some fixed
//! rows, form a well-conditioned square matrix `F_S`. Used for Matsubara
//! frequencies, where a pair is `(Re û(n), Im û(n))` (see
//! [`crate::polyfourier::select_sign_changes`]).
//!
//! The selection minimizes `‖F_S⁻¹ T_S‖² + ‖F_S⁻¹‖²` (Frobenius norms), where
//! `T` are the remaining columns of `A` (the next functions, beyond those
//! fitted). The first term is the aliasing of these functions into the
//! fitted coefficients: sampling where they are small mimics Gauss
//! quadrature, whose nodes are the zeros of the next orthogonal function.
//! The second term is the amplification of noise in the data. Pairs are
//! chosen greedily by volume, then exchanged one at a time while the
//! objective decreases.

use nalgebra::{DMatrix, DMatrixView, DVector, Matrix2};

/// An exchange is made only if it lowers the objective by more than this
/// fraction, which ends the exchanges on near-ties instead of cycling.
const EXCHANGE_TOL: f64 = 1e-2;

/// Upper bound on the number of exchanges, per selected pair.
const EXCHANGES_PER_PAIR: usize = 2;

/// Rows whose residual norm (relative to the largest row norm) is below this
/// are linearly dependent on those already selected.
const RANK_TOL: f64 = 1e-12;

/// Select pairs of rows of `a` (pair `i` = rows `2i`, `2i + 1`).
///
/// Columns `..n_cols` of `a` and `fixed` are fitted; the remaining columns of
/// `a` (and `fixed`), if any, enter the aliasing term. `fixed` rows are always
/// part of the selection. Pairs `..n_square` are added until there are at
/// least `n_cols` rows; if this makes the selection square, it is improved by
/// exchanges. Then `n_extra` more pairs, among all those of `a`, are added
/// (oversampling), each maximizing the growth of `det(F_Sᵀ F_S)`.
///
/// Returns pair indices in selection order. Fewer pairs are returned if the
/// rows of the first `n_square` pairs do not have rank `n_cols`.
pub(crate) fn select_pairs(
    a: &DMatrix<f64>,
    n_square: usize,
    fixed: &DMatrix<f64>,
    n_cols: usize,
    n_extra: usize,
) -> Vec<usize> {
    assert_eq!(a.nrows() % 2, 0, "rows of `a` come in pairs");
    assert!(2 * n_square <= a.nrows());
    assert_eq!(a.ncols(), fixed.ncols());
    let n_cols = n_cols.min(a.ncols());
    if n_cols == 0 {
        return Vec::new();
    }
    let square = a.rows(0, 2 * n_square).into_owned();
    let mut pairs = greedy_pairs(
        &square.columns(0, n_cols),
        &fixed.columns(0, n_cols),
        n_cols,
    );
    if fixed.nrows() + 2 * pairs.len() == n_cols {
        exchange_pairs(&square, fixed, n_cols, &mut pairs);
    }
    add_pairs_oversampling(
        &a.columns(0, n_cols),
        &fixed.columns(0, n_cols),
        &mut pairs,
        n_extra,
    );
    pairs
}

/// Rows of the selection: `fixed` rows, then the selected pairs of `a`
fn selected_rows(a: &DMatrixView<f64>, fixed: &DMatrixView<f64>, pairs: &[usize]) -> DMatrix<f64> {
    let nf = fixed.nrows();
    DMatrix::from_fn(nf + 2 * pairs.len(), a.ncols(), |i, j| {
        if i < nf {
            fixed[(i, j)]
        } else {
            let p = pairs[(i - nf) / 2];
            a[(2 * p + (i - nf) % 2, j)]
        }
    })
}

/// Residuals of the rows of a matrix after projecting out a growing set of
/// rows, kept with an orthonormal basis of that set.
struct Residuals {
    res: DMatrix<f64>,
    basis: Vec<DVector<f64>>,
    tol: f64,
}

impl Residuals {
    fn new(a: &DMatrixView<f64>) -> Self {
        let scale = a.row_iter().map(|r| r.norm()).fold(0.0, f64::max);
        Self {
            res: a.clone_owned(),
            basis: Vec::new(),
            tol: RANK_TOL * scale,
        }
    }

    /// Project `row` out of the residuals (no-op if it is in the span already)
    fn add(&mut self, row: DVector<f64>) {
        let mut v = row;
        for _ in 0..2 {
            for q in &self.basis {
                let c = q.dot(&v);
                v.axpy(-c, q, 1.0);
            }
        }
        let norm = v.norm();
        if norm <= self.tol {
            return;
        }
        let q = v / norm;
        let coef = &self.res * &q;
        self.res.ger(-1.0, &coef, &q, 1.0);
        self.basis.push(q);
    }
}

/// Gram matrix `X[I, :] X[I, :]ᵀ` of each pair of rows `I` of `x`
fn pair_grams(x: &DMatrix<f64>) -> Vec<Matrix2<f64>> {
    let mut grams = vec![Matrix2::zeros(); x.nrows() / 2];
    for col in x.column_iter() {
        for (g, v) in grams.iter_mut().zip(col.as_slice().as_chunks::<2>().0) {
            g[(0, 0)] += v[0] * v[0];
            g[(0, 1)] += v[0] * v[1];
            g[(1, 1)] += v[1] * v[1];
        }
    }
    for g in &mut grams {
        g[(1, 0)] = g[(0, 1)];
    }
    grams
}

/// Greedy selection by volume: repeatedly add the pair whose rows, projected
/// onto the complement of the rows already selected, span the largest area,
/// until at least `n_rows` rows are selected or the rank is exhausted.
fn greedy_pairs(a: &DMatrixView<f64>, fixed: &DMatrixView<f64>, n_rows: usize) -> Vec<usize> {
    let n_pairs = a.nrows() / 2;
    let mut res = Residuals::new(a);
    for i in 0..fixed.nrows() {
        res.add(fixed.row(i).transpose());
    }
    let mut selected = vec![false; n_pairs];
    let mut pairs = Vec::new();
    while fixed.nrows() + 2 * pairs.len() < n_rows {
        let grams = pair_grams(&res.res);
        let Some(best) = (0..n_pairs)
            .filter(|&p| !selected[p])
            .max_by(|&p, &q| grams[p].determinant().total_cmp(&grams[q].determinant()))
        else {
            break;
        };
        if grams[best].diagonal().max() <= res.tol * res.tol {
            break; // rank exhausted
        }
        selected[best] = true;
        pairs.push(best);
        for k in 0..2 {
            res.add(a.row(2 * best + k).transpose());
        }
    }
    pairs
}

/// 2×2 block of `x` at rows `i, i + 1` and columns `j, j + 1`
fn block(x: &DMatrix<f64>, i: usize, j: usize) -> Matrix2<f64> {
    Matrix2::new(x[(i, j)], x[(i, j + 1)], x[(i + 1, j)], x[(i + 1, j + 1)])
}

/// The matrices from which exchanges are evaluated, for the current square
/// selection `F_S` (rows of `F`) and the tail `T_S` (rows of `T`).
///
/// Replacing rows `J` of `F_S` by rows `I` of `F` is a rank-2 update: with
/// `M = B[I, J]`, `X = F_S⁻¹[:, J]`, `D = B[I, :] − E_J` and `K = M⁻¹ D`,
/// `F_S⁻¹ ← F_S⁻¹ − X K`, `B ← B − B[:, J] K`,
/// `C ← C + X M⁻¹ R[I, :]` and `R ← R − B[:, J] M⁻¹ R[I, :]`; `Q` and
/// `B Q` follow. Each exchange then costs O(rows · n_cols) instead of the
/// O(rows · n_cols²) of computing these matrices afresh.
#[derive(Debug, Clone)]
struct ExchangeState {
    /// `F_S⁻¹`
    finv: DMatrix<f64>,
    /// `B = F F_S⁻¹`: all rows in the basis of the selected ones
    b: DMatrix<f64>,
    /// `C = F_S⁻¹ T_S`: aliasing of the tail
    c: DMatrix<f64>,
    /// `R = T − B T_S`: residual of the tail
    r: DMatrix<f64>,
    /// `Q = F_S⁻ᵀ F_S⁻¹`
    q: DMatrix<f64>,
    /// `B Q`
    bq: DMatrix<f64>,
}

impl ExchangeState {
    /// State for the selected `rows` (fitted columns, then tail columns) of
    /// `[F T]`, or `None` if they are singular
    fn new(f: &DMatrix<f64>, t: &DMatrixView<f64>, rows: &DMatrix<f64>) -> Option<Self> {
        let n_cols = f.ncols();
        let finv = rows.columns(0, n_cols).into_owned().try_inverse()?;
        let ts = rows.columns(n_cols, t.ncols());
        let b = f * &finv;
        let c = &finv * ts;
        let r = t - &b * ts;
        let q = finv.transpose() * &finv;
        let bq = &b * &q;
        Some(Self {
            finv,
            b,
            c,
            r,
            q,
            bq,
        })
    }

    /// `‖F_S⁻¹ T_S‖² + ‖F_S⁻¹‖²`
    fn objective(&self) -> f64 {
        self.c.norm_squared() + self.finv.norm_squared()
    }

    /// Replace rows `j, j + 1` of the selection by rows `i, i + 1` of `F`,
    /// given `minv = B[I, J]⁻¹`.
    fn swap(&mut self, i: usize, j: usize, minv: &Matrix2<f64>) {
        let minv = DMatrix::from_column_slice(2, 2, minv.as_slice());
        let mut d = self.b.rows(i, 2).into_owned();
        d[(0, j)] -= 1.0;
        d[(1, j + 1)] -= 1.0;
        let k = &minv * d;
        let mr = &minv * self.r.rows(i, 2);
        let x = self.finv.columns(j, 2).into_owned();
        let bj = self.b.columns(j, 2).into_owned();
        let qj = self.q.columns(j, 2).into_owned();
        let qjj_k = self.q.view((j, j), (2, 2)) * &k;
        self.c += &x * &mr;
        self.r -= &bj * &mr;
        self.finv -= &x * &k;
        self.b -= &bj * &k;
        // Q ← Q + ΔQ with ΔQ = −Q[:, J] K − Kᵀ Q[J, :] + Kᵀ Q[J, J] K, and
        // B Q ← (B Q − B[:, J] K Q) + B ΔQ (B already updated)
        self.bq -= &bj * (&k * &self.q);
        let bqj = self.bq.columns(j, 2).into_owned();
        let bk = &self.b * k.transpose();
        self.bq -= &bqj * &k + &bk * qj.transpose() - &bk * &qjj_k;
        let qj_k = &qj * &k;
        self.q += k.transpose() * &qjj_k - &qj_k - qj_k.transpose();
    }
}

/// Exchange selected pairs for unselected ones while this lowers
/// `‖F_S⁻¹ T_S‖² + ‖F_S⁻¹‖²` (see the module documentation). `F_S` must be
/// square.
///
/// The change of the objective by each exchange is evaluated from 2×2
/// blocks of the matrices of [`ExchangeState`], following its rank-2
/// update.
fn exchange_pairs(a: &DMatrix<f64>, fixed: &DMatrix<f64>, n_cols: usize, pairs: &mut [usize]) {
    let n_pairs = a.nrows() / 2;
    let nf = fixed.nrows();
    let f = a.columns(0, n_cols).into_owned();
    let t = a.columns(n_cols, a.ncols() - n_cols);
    let rows = selected_rows(&a.as_view(), &fixed.as_view(), pairs);
    let Some(mut state) = ExchangeState::new(&f, &t, &rows) else {
        return; // singular selection: keep it
    };
    let mut selected = vec![false; n_pairs];
    for &p in pairs.iter() {
        selected[p] = true;
    }
    for _ in 0..EXCHANGES_PER_PAIR * pairs.len() {
        let ExchangeState {
            finv,
            b,
            c,
            r,
            q,
            bq,
        } = &state;
        let rc = r * (c.transpose() * finv);
        let objective = state.objective();
        let (r_grams, b_grams) = (pair_grams(r), pair_grams(b));
        let mut best = (objective * (1.0 - EXCHANGE_TOL), None);
        for p in (0..n_pairs).filter(|&p| !selected[p]) {
            let i = 2 * p;
            let (rr, bb) = (r_grams[p], b_grams[p]);
            for (k, j) in (nf..n_cols).step_by(2).enumerate() {
                let m = block(b, i, j);
                let Some(minv) = m.try_inverse() else {
                    continue;
                };
                let qjj = block(q, j, j);
                let aliasing = 2.0 * (minv * block(&rc, i, j)).trace()
                    + (minv * rr * minv.transpose() * qjj).trace();
                let dd = bb - m - m.transpose() + Matrix2::identity();
                let noise = -2.0 * (minv * (block(bq, i, j) - qjj)).trace()
                    + (minv * dd * minv.transpose() * qjj).trace();
                let value = objective + aliasing + noise;
                if value < best.0 {
                    best = (value, Some((k, p, minv)));
                }
            }
        }
        let Some((k, p, minv)) = best.1 else { return };
        state.swap(2 * p, nf + 2 * k, &minv);
        selected[pairs[k]] = false;
        selected[p] = true;
        pairs[k] = p;
    }
}

/// Add `n_extra` pairs, each maximizing `det(I + A_I G⁻¹ A_Iᵀ)` with
/// `G = F_Sᵀ F_S`, i.e. the growth of `det(G)`.
fn add_pairs_oversampling(
    a: &DMatrixView<f64>,
    fixed: &DMatrixView<f64>,
    pairs: &mut Vec<usize>,
    n_extra: usize,
) {
    let n_pairs = a.nrows() / 2;
    let mut selected = vec![false; n_pairs];
    for &p in pairs.iter() {
        selected[p] = true;
    }
    for _ in 0..n_extra {
        let rows = selected_rows(a, fixed, pairs);
        let Some(gram_inv) = (rows.transpose() * &rows).try_inverse() else {
            return;
        };
        let ag = a * gram_inv;
        let growth: Vec<f64> = (0..n_pairs)
            .map(|p| {
                let (i, j) = (2 * p, 2 * p + 1);
                let h = Matrix2::new(
                    ag.row(i).dot(&a.row(i)),
                    ag.row(i).dot(&a.row(j)),
                    ag.row(j).dot(&a.row(i)),
                    ag.row(j).dot(&a.row(j)),
                );
                (Matrix2::identity() + h).determinant()
            })
            .collect();
        let Some(best) = (0..n_pairs)
            .filter(|&p| !selected[p])
            .max_by(|&p, &q| growth[p].total_cmp(&growth[q]))
        else {
            return;
        };
        selected[best] = true;
        pairs.push(best);
    }
}

#[cfg(test)]
#[path = "pair_select_tests.rs"]
mod pair_select_tests;
