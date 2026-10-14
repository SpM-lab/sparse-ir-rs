use super::*;

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

fn no_fixed(ncols: usize) -> DMatrix<f64> {
    DMatrix::zeros(0, ncols)
}

/// Objective `‖F_S⁻¹ T_S‖² + ‖F_S⁻¹‖²` of a square selection
fn objective(a: &DMatrix<f64>, fixed: &DMatrix<f64>, n_cols: usize, pairs: &[usize]) -> f64 {
    let rows = selected_rows(&a.as_view(), &fixed.as_view(), pairs);
    let finv = rows.columns(0, n_cols).into_owned().try_inverse().unwrap();
    let c = &finv * rows.columns(n_cols, a.ncols() - n_cols);
    c.norm_squared() + finv.norm_squared()
}

/// Smooth, generic test matrix: 2 * n_pairs rows, ncols columns
fn smooth(n_pairs: usize, ncols: usize) -> DMatrix<f64> {
    DMatrix::from_fn(2 * n_pairs, ncols, |i, j| {
        let x = (i / 2) as f64 / n_pairs as f64;
        let phase = if i % 2 == 0 { 0.0 } else { 0.7 };
        ((j as f64 + 1.0) * (3.0 * x + phase)).cos() / (1.0 + 0.3 * j as f64)
    })
}

/// Pairs 1 and 3 are scaled unit vectors; the others are tiny.
#[test]
fn greedy_selects_dominant_pairs() {
    let mut a = DMatrix::from_element(10, 4, 1e-3);
    for (k, p) in [1usize, 3].iter().enumerate() {
        a[(2 * p, 2 * k)] = 10.0;
        a[(2 * p + 1, 2 * k + 1)] = 10.0;
    }
    assert_eq!(sorted(select_pairs(&a, &no_fixed(4), 4, 0)), vec![1, 3]);
}

/// Pairs 0, 1 have a slightly larger volume than pairs 2, 3 (so the greedy
/// step picks one of them) but a nonzero tail; the exchanges move to a pair
/// on which the tail vanishes (the Gauss quadrature property).
#[test]
fn exchange_prefers_vanishing_tail() {
    // 4 pairs, 2 fitted columns + 1 tail column
    let a = DMatrix::from_row_slice(
        8,
        3,
        &[
            1.01, 0.0, 0.5, //
            0.0, 1.01, 0.5, //
            1.01, 0.1, 0.5, //
            0.1, 1.01, 0.5, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 0.1, 0.0, //
            0.1, 1.0, 0.0,
        ],
    );
    let greedy = greedy_pairs(&a.columns(0, 2), &no_fixed(3).columns(0, 2), 2);
    assert!(
        greedy[0] < 2,
        "greedy step should pick a larger pair: {greedy:?}"
    );
    let pairs = select_pairs(&a, &no_fixed(3), 2, 0);
    assert_eq!(pairs.len(), 1);
    assert!(pairs[0] >= 2, "selected pair {pairs:?} has a nonzero tail");
}

/// The exchanges never increase the objective of the greedy selection.
#[test]
fn exchange_lowers_objective() {
    let a = smooth(60, 12);
    let fixed = no_fixed(12);
    let n_cols = 10;
    let greedy = greedy_pairs(&a.columns(0, n_cols), &fixed.columns(0, n_cols), n_cols);
    let selected = select_pairs(&a, &fixed, n_cols, 0);
    assert_eq!(selected.len(), 5);
    assert!(
        objective(&a, &fixed, n_cols, &selected) < objective(&a, &fixed, n_cols, &greedy),
        "exchanges did not improve the greedy selection"
    );
}

/// Without tail columns, the selection still works (noise term only).
#[test]
fn no_tail_columns() {
    let a = smooth(40, 6);
    let selected = select_pairs(&a, &no_fixed(6), 6, 0);
    assert_eq!(selected.len(), 3);
    let rows = selected_rows(&a.as_view(), &no_fixed(6).as_view(), &selected);
    assert!(rows.try_inverse().is_some());
}

/// Fixed rows are always selected, and count towards the rows.
#[test]
fn fixed_rows_count() {
    let a = smooth(40, 8);
    let fixed = DMatrix::from_fn(1, 8, |_, j| 1.0 / (1.0 + j as f64));
    let selected = select_pairs(&a, &fixed, 7, 0);
    assert_eq!(selected.len(), 3); // 1 + 2 * 3 = 7 rows
    assert!(objective(&a, &fixed, 7, &selected).is_finite());
}

/// An odd number of columns without fixed rows cannot be square: one extra
/// row is selected and no exchange is made (greedy selection kept).
#[test]
fn odd_columns_select_extra_row() {
    let a = smooth(40, 7);
    let fixed = no_fixed(7);
    let selected = select_pairs(&a, &fixed, 5, 0);
    assert_eq!(selected.len(), 3); // 6 rows for 5 columns
    assert_eq!(
        selected,
        greedy_pairs(&a.columns(0, 5), &fixed.columns(0, 5), 5)
    );
}

/// Rank-deficient rows: selection stops when the rank is exhausted.
#[test]
fn rank_deficient_returns_fewer_pairs() {
    // Every row is a multiple of (1, 1, 1, 1): rank 1
    let a = DMatrix::from_fn(20, 4, |i, _| (i + 1) as f64);
    let selected = select_pairs(&a, &no_fixed(4), 4, 0);
    assert_eq!(selected.len(), 1);
}

/// Oversampling adds distinct pairs and grows det(F_Sᵀ F_S).
#[test]
fn oversampling_adds_pairs_and_grows_volume() {
    let a = smooth(60, 8);
    let fixed = no_fixed(8);
    let gram_det = |pairs: &[usize]| {
        let rows = selected_rows(&a.as_view(), &fixed.as_view(), pairs);
        (rows.transpose() * rows).determinant()
    };
    let square = select_pairs(&a, &fixed, 8, 0);
    let over = select_pairs(&a, &fixed, 8, 3);
    assert_eq!(over.len(), square.len() + 3);
    assert_eq!(&over[..square.len()], &square[..]);
    assert_eq!(
        sorted(over.clone())
            .windows(2)
            .filter(|w| w[0] == w[1])
            .count(),
        0
    );
    assert!(gram_det(&over) > gram_det(&square));
}

#[test]
fn empty_selection() {
    let a = smooth(10, 4);
    assert!(select_pairs(&a, &no_fixed(4), 0, 0).is_empty());
    assert!(select_pairs(&DMatrix::zeros(0, 4), &no_fixed(4), 4, 0).is_empty());
}

/// After a sequence of rank-2 swaps, the exchange state equals the one
/// computed afresh for the resulting selection, up to round-off amplified
/// by the condition number.
#[test]
fn exchange_state_swaps_match_fresh_state() {
    // Deterministic pseudo-random matrix: 60 pairs, 10 fitted + 2 tail columns
    let a = DMatrix::from_fn(120, 12, |i, j| {
        ((i as f64 * 12.9898 + j as f64 * 78.233).sin() * 43758.5453).fract()
    });
    let n_cols = 10;
    let f = a.columns(0, n_cols).into_owned();
    let t = a.columns(n_cols, 2);
    let none = no_fixed(12);
    let mut pairs = greedy_pairs(&f.columns(0, n_cols), &none.columns(0, n_cols), n_cols);
    let rows = selected_rows(&a.as_view(), &none.as_view(), &pairs);
    let mut state = ExchangeState::new(&f, &t, &rows).unwrap();
    // Swap in unselected pairs, at every position in turn
    let unselected: Vec<usize> = (0..60).filter(|p| !pairs.contains(p)).collect();
    for (&k, &p) in [0, 3, 1, 4, 2, 0].iter().zip(&unselected) {
        let minv = block(&state.b, 2 * p, 2 * k).try_inverse().unwrap();
        state.swap(2 * p, 2 * k, &minv);
        pairs[k] = p;
    }
    let rows = selected_rows(&a.as_view(), &none.as_view(), &pairs);
    let sv = rows.columns(0, n_cols).into_owned().singular_values();
    let tol = 1e3 * f64::EPSILON * sv.max() / sv.min();
    let fresh = ExchangeState::new(&f, &t, &rows).unwrap();
    let close = |x: &DMatrix<f64>, y: &DMatrix<f64>, name: &str| {
        let err = (x - y).amax() / y.amax();
        assert!(err < tol, "{name}: relative difference {err:e} > {tol:e}");
    };
    close(&state.finv, &fresh.finv, "F_S⁻¹");
    close(&state.b, &fresh.b, "B");
    close(&state.c, &fresh.c, "C");
    close(&state.r, &fresh.r, "R");
    close(&state.q, &fresh.q, "Q");
    close(&state.bq, &fresh.bq, "BQ");
}
