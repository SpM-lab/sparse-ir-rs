//! Sparse sampling: recovering the IR coefficients from a handful of points.
//!
//! Ported from the Python notebook `sparse_sampling_demo_py.ipynb` of
//! sparse-ir-tutorial.
//!
//! The spectral function is the semicircle of full bandwidth 2,
//!
//! ```text
//!     ρ(ω) = (2/π) √(1 − ω²)   for |ω| < 1,   0 otherwise,
//! ```
//!
//! whose IR coefficients `Gₗ = −sₗ ρₗ` are known to machine precision. The
//! example then throws that knowledge away: it evaluates `G` on the default
//! sampling times and on the default sampling frequencies, fits the
//! coefficients back from those few values alone, and writes down how far the
//! result drifted from the exact one.

use std::error::Error;
use std::f64::consts::PI;

use num_complex::Complex64;
use sparse_ir::{Fermionic, FiniteTempBasis, LogisticKernel, MatsubaraSampling, TauSampling};
use sparse_ir_tutorial::{Table, output_path, provenance, semicircle_overlaps, write_table};

const EXAMPLE: &str = "sparse_sampling_demo";

/// Inverse temperature. Large enough that the basis is interesting (a few tens
/// of functions) while `ωmax = 1` keeps the spectral function simple.
const BETA: f64 = 10_000.0;
const WMAX: f64 = 1.0;
const EPS: f64 = 1e-15;

fn main() -> Result<(), Box<dyn Error>> {
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;

    // Gₗ = −sₗ ∫ dω vₗ(ω) ρ(ω), computed exactly enough to be a reference.
    let rho_l = semicircle_overlaps(&basis);
    let g_l: Vec<f64> = basis
        .s()
        .iter()
        .zip(&rho_l)
        .map(|(s, rho)| -s * rho)
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push("rho_l", rho_l.clone());
    table.push("g_l", g_l.clone());
    write_table(&output_path(EXAMPLE, "coefficients")?, &table)?;

    // --- from the sampling times -------------------------------------------
    let tau_sampling = TauSampling::<Fermionic>::new(&basis)?;
    let tau_points = tau_sampling.sampling_points().to_vec();
    // The default sampling times are folded around β/2, so they are reported
    // on the symmetric interval rather than on [0, β). See the conventions
    // page of the book.
    assert!(
        tau_points
            .iter()
            .all(|&tau| (-BETA / 2.0..=BETA / 2.0).contains(&tau)),
        "the default sampling times must lie in [-β/2, β/2]"
    );
    let g_tau = tau_sampling.evaluate(&g_l)?;
    let g_l_from_tau = tau_sampling.fit(&g_tau)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", tau_points.clone());
    table.push("g_tau", g_tau.clone());
    write_table(&output_path(EXAMPLE, "tau_sampling")?, &table)?;

    // --- from the sampling frequencies -------------------------------------
    let matsubara_sampling = MatsubaraSampling::<Fermionic>::new(&basis)?;
    let matsubara_points: Vec<i64> = matsubara_sampling
        .sampling_points()
        .iter()
        .map(|freq| freq.n())
        .collect();
    assert!(
        matsubara_points.iter().all(|n| n % 2 != 0),
        "fermionic Matsubara indices must be odd"
    );
    let g_l_complex: Vec<Complex64> = g_l.iter().map(|&g| Complex64::new(g, 0.0)).collect();
    let g_iv = matsubara_sampling.evaluate(&g_l_complex)?;
    let g_l_from_matsubara: Vec<f64> = matsubara_sampling
        .fit(&g_iv)?
        .iter()
        .map(|c| c.re)
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        matsubara_points
            .iter()
            .map(|&n| n as f64)
            .collect::<Vec<_>>(),
    );
    table.push(
        "nu",
        matsubara_points
            .iter()
            .map(|&n| n as f64 * PI / BETA)
            .collect::<Vec<_>>(),
    );
    table.push("g_iv_re", g_iv.iter().map(|c| c.re).collect::<Vec<_>>());
    table.push("g_iv_im", g_iv.iter().map(|c| c.im).collect::<Vec<_>>());
    write_table(&output_path(EXAMPLE, "matsubara_sampling")?, &table)?;

    // --- how far the round trip drifted ------------------------------------
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("g_l", g_l.clone());
    table.push("g_l_from_tau", g_l_from_tau.clone());
    table.push("g_l_from_matsubara", g_l_from_matsubara.clone());
    table.push(
        "error_tau",
        g_l_from_tau
            .iter()
            .zip(&g_l)
            .map(|(a, b)| (a - b).abs())
            .collect::<Vec<_>>(),
    );
    table.push(
        "error_matsubara",
        g_l_from_matsubara
            .iter()
            .zip(&g_l)
            .map(|(a, b)| (a - b).abs())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "reconstruction")?, &table)?;

    // --- the numbers the prose quotes --------------------------------------
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("accuracy", vec![basis.accuracy()]);
    table.push("n_tau_points", vec![tau_points.len() as f64]);
    table.push("n_matsubara_points", vec![matsubara_points.len() as f64]);
    table.push("cond_tau", vec![tau_sampling.condition_number()?]);
    table.push(
        "cond_matsubara",
        vec![matsubara_sampling.condition_number()?],
    );
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    Ok(())
}
