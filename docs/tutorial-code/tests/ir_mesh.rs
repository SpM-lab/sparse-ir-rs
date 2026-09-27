//! `IrMesh` transforms whole `(n, ncols)` arrays at once. These tests check
//! that batching changes nothing: every column must come out exactly as the
//! single-column API would have produced it, and the rows must stay in order.
//! A transposed batch is the kind of mistake that still produces a plausible
//! picture, so it is worth pinning before any example depends on it.

use num_complex::Complex64;
use sparse_ir::{Fermionic, FiniteTempBasis, LogisticKernel, MatsubaraSampling, TauSampling};
use sparse_ir_tutorial::mesh::IrMesh;

const BETA: f64 = 20.0;
const WMAX: f64 = 2.0;
const EPS: f64 = 1e-10;
const NCOLS: usize = 5;

fn basis() -> FiniteTempBasis<LogisticKernel, Fermionic> {
    let kernel = LogisticKernel::new(BETA * WMAX).expect("the kernel of a positive Λ exists");
    FiniteTempBasis::new(kernel, BETA, Some(EPS), None).expect("the basis of a valid kernel exists")
}

/// Column `c` of the coefficient array: unrelated to its neighbours, so that
/// a mix-up between columns cannot cancel out.
fn column(c: usize, size: usize) -> Vec<Complex64> {
    (0..size)
        .map(|l| {
            let x = (l + 1) as f64 * (c + 1) as f64;
            Complex64::new(x.sin() / x, x.cos() / (x + 3.0))
        })
        .collect()
}

#[test]
fn a_batched_round_trip_reproduces_the_single_column_one() {
    let basis = basis();
    let mesh = IrMesh::<Fermionic>::new(&basis).expect("the sampling objects exist");
    let wn = MatsubaraSampling::<Fermionic>::new(&basis).expect("the sampling exists");
    let tau = TauSampling::<Fermionic>::new(&basis).expect("the sampling exists");

    let columns: Vec<Vec<Complex64>> = (0..NCOLS).map(|c| column(c, basis.size())).collect();

    // The batched array, laid out row-major as (n_wn, NCOLS).
    let per_column: Vec<Vec<Complex64>> = columns
        .iter()
        .map(|g_l| wn.evaluate(g_l).expect("evaluating at the frequencies"))
        .collect();
    let mut batched = vec![Complex64::default(); mesh.n_wn() * NCOLS];
    for (c, values) in per_column.iter().enumerate() {
        for (i, value) in values.iter().enumerate() {
            batched[i * NCOLS + c] = *value;
        }
    }

    let batched_tau = mesh.wn_to_tau(&batched, NCOLS).expect("the transform");
    assert_eq!(batched_tau.len(), mesh.n_tau() * NCOLS);

    for (c, values) in per_column.iter().enumerate() {
        let g_l = wn.fit(values).expect("fitting the frequencies");
        let expected = tau.evaluate_zz(&g_l).expect("evaluating at the times");
        for (i, want) in expected.iter().enumerate() {
            let got = batched_tau[i * NCOLS + c];
            assert!(
                (got - want).norm() < 1e-12 * want.norm().max(1.0),
                "column {c}, row {i}: batched gave {got}, one column at a time gives {want}"
            );
        }
    }
}

#[test]
fn tau_and_frequency_transforms_are_inverses() {
    let basis = basis();
    let mesh = IrMesh::<Fermionic>::new(&basis).expect("the sampling objects exist");
    let wn = MatsubaraSampling::<Fermionic>::new(&basis).expect("the sampling exists");

    let mut values = vec![Complex64::default(); mesh.n_wn() * NCOLS];
    for c in 0..NCOLS {
        let evaluated = wn
            .evaluate(&column(c, basis.size()))
            .expect("evaluating at the frequencies");
        for (i, value) in evaluated.iter().enumerate() {
            values[i * NCOLS + c] = *value;
        }
    }

    let back = mesh
        .tau_to_wn(&mesh.wn_to_tau(&values, NCOLS).expect("iν → τ"), NCOLS)
        .expect("τ → iν");

    for (got, want) in back.iter().zip(&values) {
        // Two fits, so two condition numbers; the basis itself is good to 1e-10.
        assert!(
            (got - want).norm() < 1e-8 * want.norm().max(1.0),
            "the round trip moved {want} to {got}"
        );
    }
}
