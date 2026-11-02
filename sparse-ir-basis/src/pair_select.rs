//! Sampling-point selection by pairs of rows.
//!
//! Selects pairs of rows of a real matrix `A` (pair `i` = rows `2i`, `2i + 1`)
//! so that they, together with some fixed rows, form a well-conditioned
//! sampling matrix `F_S`. Used for Matsubara frequencies, where a pair is
//! `(Re û(n), Im û(n))`, equivalent to the frequencies `±n` (see
//! [`crate::polyfourier::pair_sampling_points`]).
//!
//! The square selection is an interpolative decomposition (ID) of the rows:
//! the greedy, column-pivoted QR (Businger and Golub, Numer. Math. 7, 269
//! (1965)) applied to the rows, which repeatedly takes the row with the
//! largest residual after projecting out those already taken, i.e. the
//! largest growth of the volume of the selection. This is how Kaye, Chen
//! and Parcollet (Phys. Rev. B 105, 235115 (2022), Secs. IV E and V B) choose
//! the sampling points of the DLR. Here the pivots are pairs of rows, the
//! one with the largest projected area, so that the frequencies come in
//! `±n` pairs; this block form is our choice. The pivoted Gram–Schmidt is
//! that of the DLR construction (`sparse_ir_dlr::dlr_id`), which takes
//! single rows for the nodes of the DLR and pairs here.
//!
//! Oversampling adds pairs one at a time, each the one that most decreases
//! `trace((F_SᵀF_S)⁻¹)`, the variance of the fitted coefficients under
//! white noise of unit variance (A-optimal design, see e.g. Atkinson, Donev
//! and Tobias, *Optimum Experimental Designs, with SAS*, Oxford University
//! Press (2007), Ch. 10). Continuing the volume criterion instead
//! (D-optimal) left the condition number of the Matsubara sampling matrices
//! unchanged (ratio 0.99 for 12% more points, over 63 bases of the
//! comparison in the review of #342), while the A-optimal choice halved it.
//!
//! Choices of this implementation, not taken from the references: the
//! selection in pairs of rows (the ±n symmetry); the candidates, the grid
//! of the sign-change search
//! ([`crate::polyfourier::pair_sampling_points`]); the rank tolerance
//! `RANK_TOL`; and, with `mitigate`, about 10% more frequencies chosen by
//! the A-optimal criterion.

use nalgebra::{DMatrix, DMatrixView, Matrix2};

/// Pairs whose residual rows are below this times the largest row norm are
/// linearly dependent on those already selected.
const RANK_TOL: f64 = 1e-12;

/// Select pairs of rows of `a` (pair `i` = rows `2i`, `2i + 1`).
///
/// `fixed` rows (with as many columns as `a`) are always part of the
/// selection. Pairs are added greedily (the ID) until there are at least
/// `a.ncols()` rows, then `n_extra` more pairs (oversampling), each the one
/// that most decreases `trace((F_SᵀF_S)⁻¹)`.
///
/// Returns pair indices in selection order. Fewer pairs are returned if the
/// rows of `a` do not have full column rank, or not enough pairs remain for
/// the oversampling.
pub(crate) fn select_pairs(a: &DMatrix<f64>, fixed: &DMatrix<f64>, n_extra: usize) -> Vec<usize> {
    assert_eq!(a.nrows() % 2, 0, "rows of `a` come in pairs");
    assert_eq!(a.ncols(), fixed.ncols());
    if a.ncols() == 0 {
        return Vec::new();
    }
    let (a, fixed) = (a.as_view(), fixed.as_view());
    let mut pairs = greedy_pairs(&a, &fixed, a.ncols());
    add_pairs_oversampling(&a, &fixed, &mut pairs, n_extra);
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

/// Greedy selection by volume (the ID): the pivoted QR of the rows in pairs,
/// [`sparse_ir_dlr::dlr_id::pivoted_pair_gram_schmidt`] (the pivoted
/// Gram–Schmidt that also chooses the Matsubara nodes of the DLR), after the
/// `fixed` rows, until at least `n_rows` rows are selected or the rank is
/// exhausted (residual rows below `RANK_TOL` times the largest row norm)
fn greedy_pairs(a: &DMatrixView<f64>, fixed: &DMatrixView<f64>, n_rows: usize) -> Vec<usize> {
    let (nf, n_pairs, m) = (fixed.nrows(), a.nrows() / 2, a.ncols());
    // The rows as contiguous vectors: fixed rows, then the pairs
    let mut rows = Vec::with_capacity((nf + 2 * n_pairs) * m);
    for i in 0..nf {
        rows.extend(fixed.row(i).iter());
    }
    for i in 0..2 * n_pairs {
        rows.extend(a.row(i).iter());
    }
    sparse_ir_dlr::dlr_id::pivoted_pair_gram_schmidt(&mut rows, m, nf, n_pairs, RANK_TOL, n_rows)
}

/// Add `n_extra` pairs, each the one that most decreases
/// `trace((F_SᵀF_S)⁻¹)` (A-optimal).
///
/// With `G = F_SᵀF_S`, `W = A G⁻¹` and, for the rows `I` of a pair,
/// `H = W_I A_Iᵀ`, the Sherman–Morrison–Woodbury formula gives
/// `trace((G + A_IᵀA_I)⁻¹) = trace(G⁻¹) − trace((I + H)⁻¹ W_I W_Iᵀ)`: the
/// decrease is a 2×2 computation per pair once `W` is formed.
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
        let w = a * gram_inv;
        let decrease = |p: usize| {
            let (i, j) = (2 * p, 2 * p + 1);
            let gram2 = |x: &DMatrix<f64>, y: &DMatrixView<f64>| {
                Matrix2::new(
                    x.row(i).dot(&y.row(i)),
                    x.row(i).dot(&y.row(j)),
                    x.row(j).dot(&y.row(i)),
                    x.row(j).dot(&y.row(j)),
                )
            };
            let h = gram2(&w, a);
            let m = gram2(&w, &w.as_view());
            (Matrix2::identity() + h)
                .try_inverse()
                .map_or(f64::NEG_INFINITY, |inv| (inv * m).trace())
        };
        let decreases: Vec<f64> = (0..n_pairs).map(decrease).collect();
        let Some(best) = (0..n_pairs)
            .filter(|&p| !selected[p])
            .max_by(|&p, &q| decreases[p].total_cmp(&decreases[q]))
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
