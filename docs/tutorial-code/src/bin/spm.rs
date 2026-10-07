//! Sparse modeling: recovering a spectral function from noisy `G(τ)`.
//!
//! Ported from the Python notebook `spm_py.ipynb` of sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/>).
//!
//! Analytic continuation is the hard direction. Going from a spectral function
//! to `G(τ)` is a smoothing integral; going back amplifies whatever noise the
//! data carries, without bound. The sparse-modeling answer is to look for the
//! solution that explains the data with the fewest IR coefficients: minimise
//!
//! ```text
//! ½‖y − A x‖² + λ‖x‖₁,   y = −G(τ),  A_{il} = u_l(τ_i) s_l,  x = ρ_l,
//! ```
//!
//! over the IR coefficients `ρ_l` of the spectral function. The L1 penalty
//! drives most coefficients to exactly zero; `λ` decides how many survive.
//!
//! The notebook solves a constrained version of this — it adds `ρ(ω) ≥ 0` and
//! the sum rule `∫dω ρ = 1` as hard constraints, which needs ADMM and the
//! `admmsolver` package. Neither constraint has a cheap proximal operator, so
//! this port drops them and uses plain FISTA. The output shows what that
//! costs: the recovered spectrum dips slightly negative, and the sum rule is
//! satisfied to a few parts in a thousand rather than exactly.

use std::error::Error;

// ANCHOR: imports
use sparse_ir::Matrix;
use sparse_ir::{Basis, Fermionic, FiniteTempBasis, LogisticKernel};
// ANCHOR_END: imports
use sparse_ir_tutorial::{
    Table, fista, input_path, integrate_segments, output_path, provenance, read_table,
    soft_threshold, three_gaussians, write_table,
};

const EXAMPLE: &str = "spm";

// ANCHOR: parameters
// These must agree with `scripts/make_spm_input.py`, which wrote the input.
const BETA: f64 = 100.0;
const WMAX: f64 = 4.0;
const EPS: f64 = 1e-10;
// ANCHOR_END: parameters

/// The ω grid the recovered spectrum is reported on.
const N_OMEGA: usize = 501;

/// The regularisation strengths the example sweeps, and the one whose spectrum
/// it reports. `3e-5` is where the sweep bottoms out: it has the smallest
/// error against the exact spectrum, and its solution is the closest to
/// non-negative. The sweep is what picks it — see the `lambda_scan` table.
const LAMBDAS: [f64; 7] = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
const CHOSEN_LAMBDA: f64 = 3e-5;

/// Fixed-count power iteration for the Lipschitz constant of `∇f`, i.e. the
/// largest eigenvalue of `AᵀA`. A fixed count rather than a tolerance keeps
/// the number identical between this program and its Python reference, which
/// matters because every FISTA step is scaled by it.
const POWER_ITERATIONS: usize = 200;

/// A fixed iteration budget rather than a convergence tolerance.
///
/// FISTA's momentum makes the step-to-step change oscillate instead of
/// decreasing monotonically, so a small tolerance is either never reached or
/// reached at an iteration that a rounding difference can move. A fixed budget
/// keeps this program and its Python reference on the same trajectory for the
/// same number of steps, which is what makes comparing them meaningful. The
/// tolerance is left at zero so that a run which reaches an exact fixed point
/// — the larger `λ` do — still stops there.
const FISTA_MAX_ITER: usize = 20_000;
const FISTA_TOL: f64 = 0.0;

/// How settled the iterate must be by the end of the budget.
const SETTLED: f64 = 1e-6;

fn main() -> Result<(), Box<dyn Error>> {
    // ANCHOR: input
    let input = read_table(&input_path(EXAMPLE, "gtau"))?;
    let taus = input.expect_column("tau").to_vec();
    let g_tau = input.expect_column("g_tau").to_vec();
    let g_tau_clean = input.expect_column("g_tau_clean").to_vec();
    let n_tau = taus.len();
    // ANCHOR_END: input

    // ANCHOR: basis
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    let size = basis.size();
    // ANCHOR_END: basis

    // ANCHOR: design_matrix
    // `A_{il} = u_l(τ_i) s_l` maps IR coefficients of ρ to −G(τ).
    let u_at_taus: Matrix<f64> = basis.evaluate_tau(&taus)?;
    let mut a = vec![0.0; n_tau * size];
    for i in 0..n_tau {
        for l in 0..size {
            a[i * size + l] = *u_at_taus.get(&[i, l]).unwrap() * basis.s()[l];
        }
    }
    let y: Vec<f64> = g_tau.iter().map(|g| -g).collect();
    // ANCHOR_END: design_matrix

    let lipschitz = largest_eigenvalue_of_ata(&a, n_tau, size);
    println!("basis size {size}, {n_tau} times, Lipschitz bound {lipschitz:.6e}");

    // The exact answer, for the error columns. `ρ` is known here only because
    // the input was generated from it; a real continuation has no such column.
    let omegas = linspace(-WMAX, WMAX, N_OMEGA);
    let d_omega = omegas[1] - omegas[0];
    let rho_exact: Vec<f64> = omegas.iter().map(|&w| three_gaussians(w)).collect();
    let rho_l_exact = overlap_with_v(&basis, three_gaussians);
    let v_at_omegas: Matrix<f64> = basis.evaluate_omega(&omegas)?;

    let spectrum = |rho_l: &[f64]| -> Vec<f64> {
        (0..omegas.len())
            .map(|i| {
                (0..size)
                    .map(|l| *v_at_omegas.get(&[i, l]).unwrap() * rho_l[l])
                    .sum()
            })
            .collect()
    };

    let mut scan = Table::new(provenance(EXAMPLE));
    let (mut lambda_col, mut residual_col, mut l1_col) = (vec![], vec![], vec![]);
    let (mut error_col, mut sum_col, mut min_col, mut nonzero_col) =
        (vec![], vec![], vec![], vec![]);
    let mut chosen: Option<Vec<f64>> = None;

    for lambda in LAMBDAS {
        let (rho_l, report) = solve(&a, &y, n_tau, size, lipschitz, lambda);
        assert!(
            report.last_relative_change < SETTLED,
            "λ = {lambda:e} was still moving by {:e} of its size after \
             {} iterations, so the number below is not a solution of anything",
            report.last_relative_change,
            report.iterations
        );

        let rho = spectrum(&rho_l);
        let residual = residual_norm(&a, &y, n_tau, size, &rho_l);
        let l1: f64 = rho_l.iter().map(|x| x.abs()).sum();
        let error = (rho
            .iter()
            .zip(&rho_exact)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            * d_omega)
            .sqrt();
        let sum_rule: f64 = rho.iter().sum::<f64>() * d_omega;
        let minimum = rho.iter().cloned().fold(f64::INFINITY, f64::min);
        let nonzero = rho_l.iter().filter(|x| **x != 0.0).count();

        println!(
            "λ = {lambda:7.1e}  {:6} iterations  {nonzero:3}/{size} nonzero  \
             residual {residual:.4e}  ‖ρ−ρ_exact‖ {error:.4e}",
            report.iterations
        );

        lambda_col.push(lambda);
        residual_col.push(residual);
        l1_col.push(l1);
        error_col.push(error);
        sum_col.push(sum_rule);
        min_col.push(minimum);
        nonzero_col.push(nonzero as f64);

        if lambda == CHOSEN_LAMBDA {
            chosen = Some(rho_l);
        }
    }

    scan.push("lambda", lambda_col);
    scan.push("residual", residual_col);
    scan.push("l1_norm", l1_col);
    scan.push("l2_error", error_col);
    scan.push("sum_rule", sum_col);
    scan.push("min_rho", min_col);
    scan.push("nonzero", nonzero_col);
    write_table(&output_path(EXAMPLE, "lambda_scan")?, &scan)?;

    let rho_l = chosen.expect("CHOSEN_LAMBDA must be one of LAMBDAS");
    let rho = spectrum(&rho_l);

    // The point of the example: a noisy G(τ) still pins the spectrum down to
    // something recognisable. If this ever stops holding, the page is wrong.
    let peak = rho
        .iter()
        .zip(&omegas)
        .max_by(|a, b| a.0.partial_cmp(b.0).expect("no NaN in the spectrum"))
        .expect("the grid is not empty");
    assert!(
        peak.1.abs() < 0.1,
        "the recovered spectrum should peak at the central Gaussian, not at ω = {}",
        peak.1
    );

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omegas);
    table.push("rho_exact", rho_exact);
    table.push("rho_recovered", rho);
    write_table(&output_path(EXAMPLE, "spectrum")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..size).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push("rho_l_exact", rho_l_exact);
    table.push("rho_l_recovered", rho_l.clone());
    write_table(&output_path(EXAMPLE, "coefficients")?, &table)?;

    // What the fit says the data should have been, next to what it was.
    let g_tau_fit: Vec<f64> = (0..n_tau)
        .map(|i| -(0..size).map(|l| a[i * size + l] * rho_l[l]).sum::<f64>())
        .collect();
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", taus);
    table.push("g_tau_input", g_tau);
    table.push("g_tau_clean", g_tau_clean);
    table.push("g_tau_fit", g_tau_fit);
    write_table(&output_path(EXAMPLE, "gtau")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("basis_size", vec![size as f64]);
    table.push("n_tau", vec![n_tau as f64]);
    table.push("lipschitz", vec![lipschitz]);
    table.push("lambda", vec![CHOSEN_LAMBDA]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    Ok(())
}

// ANCHOR: solve
/// Minimises `½‖y − A x‖² + λ‖x‖₁` from `x = 0`.
fn solve(
    a: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    lipschitz: f64,
    lambda: f64,
) -> (Vec<f64>, sparse_ir_tutorial::FistaReport) {
    let mut x = vec![0.0; cols];
    let mut residual = vec![0.0; rows];
    let report = fista(
        &mut x,
        lipschitz,
        |x, grad| {
            multiply(a, cols, x, &mut residual);
            for (r, yi) in residual.iter_mut().zip(y) {
                *r -= yi;
            }
            multiply_transposed(a, rows, cols, &residual, grad);
        },
        |x, step| soft_threshold(x, step * lambda),
        FISTA_MAX_ITER,
        FISTA_TOL,
    );
    (x, report)
}
// ANCHOR_END: solve

/// `out = A x`, with `A` stored row by row.
fn multiply(a: &[f64], cols: usize, x: &[f64], out: &mut [f64]) {
    for (i, oi) in out.iter_mut().enumerate() {
        *oi = a[i * cols..(i + 1) * cols]
            .iter()
            .zip(x)
            .map(|(aij, xj)| aij * xj)
            .sum();
    }
}

/// `out = Aᵀ r`.
fn multiply_transposed(a: &[f64], rows: usize, cols: usize, r: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..rows {
        let row = &a[i * cols..(i + 1) * cols];
        let ri = r[i];
        for (oj, aij) in out.iter_mut().zip(row) {
            *oj += aij * ri;
        }
    }
}

fn residual_norm(a: &[f64], y: &[f64], rows: usize, cols: usize, x: &[f64]) -> f64 {
    let mut fit = vec![0.0; rows];
    multiply(a, cols, x, &mut fit);
    fit.iter()
        .zip(y)
        .map(|(f, yi)| (f - yi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The largest eigenvalue of `AᵀA` by power iteration from a fixed start.
fn largest_eigenvalue_of_ata(a: &[f64], rows: usize, cols: usize) -> f64 {
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    let mut w = vec![0.0; cols];
    let mut scratch = vec![0.0; rows];
    let mut eigenvalue = 0.0;
    for _ in 0..POWER_ITERATIONS {
        multiply(a, cols, &v, &mut scratch);
        multiply_transposed(a, rows, cols, &scratch, &mut w);
        eigenvalue = w.iter().map(|wi| wi * wi).sum::<f64>().sqrt();
        for (vi, wi) in v.iter_mut().zip(&w) {
            *vi = wi / eigenvalue;
        }
    }
    eigenvalue
}

/// `ρₗ = ∫ dω vₗ(ω) ρ(ω)` by composite Gauss-Legendre quadrature on the
/// basis' own knots, which is where `vₗ` is a polynomial.
fn overlap_with_v<F>(basis: &FiniteTempBasis<LogisticKernel, Fermionic>, f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let v = basis.v();
    let edges = v.get_knots(None);
    let order = v.get_polyorder() + 8;
    (0..basis.size())
        .map(|l| {
            let poly = &v[l];
            integrate_segments(|omega| poly.evaluate(omega) * f(omega), &edges, order)
        })
        .collect()
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    assert!(count > 1, "a grid needs at least two points");
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}
