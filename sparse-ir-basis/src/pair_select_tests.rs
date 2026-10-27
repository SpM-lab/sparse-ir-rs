use super::*;

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

fn no_fixed(ncols: usize) -> DMatrix<f64> {
    DMatrix::zeros(0, ncols)
}

/// Smooth, generic test matrix: 2 * n_pairs rows, ncols columns
fn smooth(n_pairs: usize, ncols: usize) -> DMatrix<f64> {
    DMatrix::from_fn(2 * n_pairs, ncols, |i, j| {
        let x = (i / 2) as f64 / n_pairs as f64;
        let phase = if i % 2 == 0 { 0.0 } else { 0.7 };
        ((j as f64 + 1.0) * (3.0 * x + phase)).cos() / (1.0 + 0.3 * j as f64)
    })
}

/// `trace((F_SᵀF_S)⁻¹)` of a selection, computed directly
fn trace_inv_gram(a: &DMatrix<f64>, fixed: &DMatrix<f64>, pairs: &[usize]) -> f64 {
    let rows = selected_rows(&a.as_view(), &fixed.as_view(), pairs);
    (rows.transpose() * rows).try_inverse().unwrap().trace()
}

/// Pairs 1 and 3 are scaled unit vectors; the others are tiny.
#[test]
fn greedy_selects_dominant_pairs() {
    let mut a = DMatrix::from_element(10, 4, 1e-3);
    for (k, p) in [1usize, 3].iter().enumerate() {
        a[(2 * p, 2 * k)] = 10.0;
        a[(2 * p + 1, 2 * k + 1)] = 10.0;
    }
    assert_eq!(sorted(select_pairs(&a, &no_fixed(4), 0)), vec![1, 3]);
}

/// Each greedy pair has the largest projected area: the pivots of a
/// column-pivoted QR of the rows, taken in pairs
#[test]
fn greedy_takes_the_largest_projected_area() {
    let a = smooth(40, 8);
    let pairs = greedy_pairs(&a.as_view(), &no_fixed(8).as_view(), 8);
    assert_eq!(pairs.len(), 4);
    // Area of the rows of pair p projected onto the complement of the rows
    // of `taken`, computed afresh by a QR of those rows
    let area = |taken: &[usize], p: usize| {
        let mut res = a.rows(2 * p, 2).transpose();
        if !taken.is_empty() {
            let sel = selected_rows(&a.as_view(), &no_fixed(8).as_view(), taken);
            let q = sel.transpose().qr().q();
            res -= &q * (q.transpose() * &res);
        }
        (res.transpose() * res).determinant()
    };
    for k in 0..pairs.len() {
        let best = area(&pairs[..k], pairs[k]);
        for p in (0..40).filter(|p| !pairs[..k].contains(p)) {
            let other = area(&pairs[..k], p);
            assert!(
                other <= best * (1.0 + 1e-9) + 1e-14,
                "step {k}: pair {p} has a larger area {other:e} > {best:e}"
            );
        }
    }
}

/// Fixed rows are always selected, and count towards the rows.
#[test]
fn fixed_rows_count() {
    let a = smooth(40, 7);
    let fixed = DMatrix::from_fn(1, 7, |_, j| 1.0 / (1.0 + j as f64));
    let selected = select_pairs(&a, &fixed, 0);
    assert_eq!(selected.len(), 3); // 1 + 2 * 3 = 7 rows
    let rows = selected_rows(&a.as_view(), &fixed.as_view(), &selected);
    assert!(rows.try_inverse().is_some());
}

/// An odd number of columns without fixed rows cannot be square: one extra
/// row is selected.
#[test]
fn odd_columns_select_extra_row() {
    let a = smooth(40, 5);
    let selected = select_pairs(&a, &no_fixed(5), 0);
    assert_eq!(selected.len(), 3); // 6 rows for 5 columns
}

/// Rank-deficient rows: selection stops when the rank is exhausted.
#[test]
fn rank_deficient_returns_fewer_pairs() {
    // Every row is a multiple of (1, 1, 1, 1): rank 1
    let a = DMatrix::from_fn(20, 4, |i, _| (i + 1) as f64);
    let selected = select_pairs(&a, &no_fixed(4), 0);
    assert_eq!(selected.len(), 1);
}

/// Oversampling adds distinct pairs after the square selection, each the
/// one that minimizes `trace((F_SᵀF_S)⁻¹)` among all unselected pairs
/// (checked by brute force), which therefore decreases at every step.
#[test]
fn oversampling_takes_the_a_optimal_pair() {
    let a = smooth(60, 8);
    let fixed = DMatrix::from_fn(1, 8, |_, j| 0.5 / (1.0 + j as f64));
    let square = select_pairs(&a, &fixed, 0);
    let over = select_pairs(&a, &fixed, 4);
    assert_eq!(over.len(), square.len() + 4);
    assert_eq!(&over[..square.len()], &square[..]);
    assert!(
        sorted(over.clone()).windows(2).all(|w| w[0] != w[1]),
        "{over:?}"
    );
    for k in square.len()..over.len() {
        let chosen = trace_inv_gram(&a, &fixed, &over[..=k]);
        assert!(chosen < trace_inv_gram(&a, &fixed, &over[..k]));
        for p in (0..60).filter(|p| !over[..k].contains(p)) {
            let mut other = over[..k].to_vec();
            other.push(p);
            let t = trace_inv_gram(&a, &fixed, &other);
            assert!(
                chosen <= t * (1.0 + 1e-10),
                "step {k}: pair {p} gives {t:e} < {chosen:e}"
            );
        }
    }
}

#[test]
fn empty_selection() {
    assert!(select_pairs(&DMatrix::zeros(20, 0), &no_fixed(0), 3).is_empty());
    assert!(select_pairs(&DMatrix::zeros(0, 4), &no_fixed(4), 0).is_empty());
}
