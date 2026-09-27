//! The τ convention, pinned down before any applied example relies on it.
//!
//! The applied notebooks of sparse-ir-tutorial are full of expressions like
//! `chi0 = G(τ) · G(β − τ)`. In Python that is written `grit * grit[::-1]`,
//! which works because the Python wrapper folds its sampling times into
//! `[0, β)` and the resulting grid is symmetric about `β/2`.
//!
//! The Rust sampling times live on `[−β/2, β/2]` instead, so `[::-1]` there
//! gives `G(−τ)`, not `G(β − τ)`. The two are related by
//!
//! ```text
//!     G(β − τ) = ζ G(−τ),     ζ = −1 (fermionic), +1 (bosonic),
//! ```
//!
//! and [`reverse_tau_rows`] is the only place in the tutorial code that knows
//! it. These tests check the two facts it stands on — that the grid is closed
//! under `τ → −τ`, and that the sign comes out right — against a direct
//! evaluation of the basis functions at `β − τ`, which needs no convention at
//! all as long as `β − τ` stays inside `[0, β]`.

use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, StatisticsType, TauSampling};
use sparse_ir_tutorial::mesh::reverse_tau_rows;

const BETA: f64 = 10.0;
const WMAX: f64 = 1.0;
const EPS: f64 = 1e-12;

fn basis<S: StatisticsType + 'static>() -> FiniteTempBasis<LogisticKernel, S> {
    let kernel = LogisticKernel::new(BETA * WMAX).expect("the kernel of a positive Λ exists");
    FiniteTempBasis::new(kernel, BETA, Some(EPS), None).expect("the basis of a valid kernel exists")
}

/// Coefficients with no symmetry of their own, so that a wrong permutation
/// cannot pass by accident.
fn coefficients(size: usize) -> Vec<f64> {
    (0..size)
        .map(|l| ((l + 1) as f64).sin() / ((l + 1) as f64))
        .collect()
}

fn symmetric_grid<S: StatisticsType + 'static>() {
    let basis = basis::<S>();
    let sampling = TauSampling::<S>::new(&basis).expect("the default sampling exists");
    let points = sampling.sampling_points();
    let n = points.len();

    assert!(
        points.windows(2).all(|pair| pair[0] < pair[1]),
        "the sampling times are expected to come back sorted"
    );

    for (i, &tau) in points.iter().enumerate() {
        let mirrored = points[n - 1 - i];
        assert!(
            (tau + mirrored).abs() < 1e-12 * BETA,
            "the grid is expected to be closed under τ → −τ, but {tau} has no partner \
             ({mirrored} is the best candidate)"
        );
    }
}

#[test]
fn the_default_tau_grid_is_symmetric_about_zero_fermionic() {
    symmetric_grid::<Fermionic>();
}

#[test]
fn the_default_tau_grid_is_symmetric_about_zero_bosonic() {
    symmetric_grid::<Bosonic>();
}

fn reversal_matches_direct_evaluation<S: StatisticsType + 'static>() {
    let basis = basis::<S>();
    let sampling = TauSampling::<S>::new(&basis).expect("the default sampling exists");
    let points = sampling.sampling_points().to_vec();
    let g_l = coefficients(basis.size());

    let g_tau = sampling
        .evaluate(&g_l)
        .expect("evaluating at the sampling times");
    let reversed = reverse_tau_rows::<S, f64>(&points, &g_tau, 1);

    let u = basis.u();
    let mut checked = 0;
    for (i, &tau) in points.iter().enumerate() {
        // `β − τ` only stays inside the domain of `u` for the non-negative half
        // of the grid. That half is enough: the grid is symmetric, so it pins
        // both the permutation and the sign.
        if tau < 0.0 {
            continue;
        }
        let direct: f64 = u
            .evaluate_at(BETA - tau)
            .iter()
            .zip(&g_l)
            .map(|(u_l, g)| u_l * g)
            .sum();
        let scale = direct.abs().max(1.0);
        assert!(
            (reversed[i] - direct).abs() < 1e-12 * scale,
            "G(β − τ) at τ = {tau} came out as {} but a direct evaluation says {direct}",
            reversed[i]
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no sampling time landed on the non-negative half"
    );
}

#[test]
fn reversal_matches_direct_evaluation_fermionic() {
    reversal_matches_direct_evaluation::<Fermionic>();
}

#[test]
fn reversal_matches_direct_evaluation_bosonic() {
    reversal_matches_direct_evaluation::<Bosonic>();
}

#[test]
fn reversal_is_its_own_inverse() {
    // Applying it twice returns G(τ) — for fermions only because the two sign
    // flips cancel, which is a cheap way of catching a stray ζ.
    let basis = basis::<Fermionic>();
    let sampling = TauSampling::<Fermionic>::new(&basis).expect("the default sampling exists");
    let points = sampling.sampling_points().to_vec();
    let g_l = coefficients(basis.size());
    let g_tau = sampling
        .evaluate(&g_l)
        .expect("evaluating at the sampling times");

    let once = reverse_tau_rows::<Fermionic, f64>(&points, &g_tau, 1);
    let twice = reverse_tau_rows::<Fermionic, f64>(&points, &once, 1);

    for (back, original) in twice.iter().zip(&g_tau) {
        assert!((back - original).abs() < 1e-14 * original.abs().max(1.0));
    }
}

#[test]
fn reversal_acts_on_rows_only() {
    // The applied examples keep G as a row-major (nτ, nk) array and reverse
    // along τ. Columns must come back untouched and in order.
    let points = [-2.0, -1.0, 1.0, 2.0];
    let values: Vec<f64> = (0..12).map(|v| v as f64).collect(); // 4 rows × 3 columns
    let reversed = reverse_tau_rows::<Bosonic, f64>(&points, &values, 3);
    assert_eq!(
        reversed,
        vec![9.0, 10.0, 11.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
    );

    let reversed = reverse_tau_rows::<Fermionic, f64>(&points, &values, 3);
    assert_eq!(
        reversed,
        vec![
            -9.0, -10.0, -11.0, -6.0, -7.0, -8.0, -3.0, -4.0, -5.0, -0.0, -1.0, -2.0
        ]
    );
}
