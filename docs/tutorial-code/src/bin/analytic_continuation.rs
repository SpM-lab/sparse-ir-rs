//! Numerical analytic continuation: why it is hard, and what regularisation
//! buys you.
//!
//! Ported from the Python notebook `analytic_continuation_py.ipynb` of
//! sparse-ir-tutorial-v2 (<https://spm-lab.github.io/sparse-ir-tutorial-v2/>).
//!
//! `G(τ) = −∫dω K(τ, ω) ρ(ω)` inverts, formally, to
//! `ρ(ω) = −Σ_l v_l(ω) G_l / s_l`. The singular values are all positive, so
//! the inverse exists; they fall off exponentially, so it is useless. Any
//! error in `G` — even the last bit of a double — is amplified by `1/s_l` and
//! swamps the answer. This example shows the failure, then two ways of
//! blunting it, and ends with the reason the IR basis is the wrong place to
//! represent the answer even when the inversion works.

use std::error::Error;

// ANCHOR: imports
use sparse_ir::Matrix;
use sparse_ir::{Basis, Fermionic, FiniteTempBasis, LogisticKernel};
// ANCHOR_END: imports
use sparse_ir_tutorial::{
    Table, input_path, integrate_segments, output_path, provenance, read_table, shifted_semicircle,
    shifted_semicircle_overlaps, write_table,
};

const EXAMPLE: &str = "analytic_continuation";

// ANCHOR: parameters
// These must agree with `scripts/make_analytic_continuation_input.py`.
const BETA: f64 = 40.0;
const WMAX: f64 = 2.0;
const EPS: f64 = 2e-8;
// ANCHOR_END: parameters

/// The noise put on `Gₗ`, as a fraction of the basis' own accuracy.
const NOISE_FRACTION: f64 = 0.3;

/// The ridge parameter, in units of the noise. Ideally it is tuned to the
/// noise level; a hundred times it is what the notebook uses.
const ALPHA_IN_NOISE: f64 = 100.0;

/// The ω grid the reconstructions are reported on.
const N_OMEGA: usize = 1001;

/// The "sharpness" of the Lorentzians of the real-axis basis, as a fraction of
/// `πT`. The `η → 0` limit is a set of delta peaks and is to be avoided.
const ETA_IN_PI_T: f64 = 0.1;

/// How many Lorentzians the real-axis basis has.
const N_LORENTZ: usize = 21;

/// The poles of the discrete model, in units of `ωmax`.
const DISCRETE_POLES: [f64; 4] = [-0.6, -0.1, 0.1, 0.6];

fn main() -> Result<(), Box<dyn Error>> {
    // ANCHOR: basis
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    let size = basis.size();
    let s = basis.s().to_vec();
    // ANCHOR_END: basis

    let noise = NOISE_FRACTION * s[size - 1] / s[0];
    let alpha = ALPHA_IN_NOISE * noise;
    let eta = ETA_IN_PI_T * std::f64::consts::PI / BETA;

    // --- the two models -----------------------------------------------------
    //
    // A semi-elliptic density of states filling the whole band, and an
    // insulating one: the same band split in two by a gap. Both are sums of
    // normalised semicircles, which is what lets `ρₗ` be computed exactly.
    // ANCHOR: models
    let rho_semi = shifted_semicircle_overlaps(&basis, 0.0, WMAX, 1.0);
    let rho_insul = {
        let right = shifted_semicircle_overlaps(&basis, WMAX / 2.0, WMAX / 4.0, 0.5);
        let left = shifted_semicircle_overlaps(&basis, -WMAX / 2.0, WMAX / 4.0, 0.5);
        right
            .iter()
            .zip(&left)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
    };
    let g_semi = coefficients(&s, &rho_semi);
    let g_insul = coefficients(&s, &rho_insul);
    // ANCHOR_END: models

    // --- and the noise on them ----------------------------------------------
    let draws = read_table(&input_path(EXAMPLE, "noise"))?;
    assert_eq!(
        draws.rows(),
        size,
        "the committed noise was drawn for a basis of {} functions, not {size}",
        draws.rows()
    );
    let g_semi_noisy = add_noise(&g_semi, draws.expect_column("z_semielliptic"), noise);
    let g_insul_noisy = add_noise(&g_insul, draws.expect_column("z_insulating"), noise);

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..size).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", s.clone());
    table.push("s_ratio", s.iter().map(|sl| sl / s[0]).collect::<Vec<_>>());
    table.push("rho_semielliptic", rho_semi.clone());
    table.push("g_semielliptic", g_semi.clone());
    table.push("g_semielliptic_noisy", g_semi_noisy.clone());
    table.push("rho_insulating", rho_insul);
    table.push("g_insulating", g_insul.clone());
    table.push("g_insulating_noisy", g_insul_noisy.clone());
    write_table(&output_path(EXAMPLE, "coefficients")?, &table)?;

    // --- truncated-SVD regularisation ---------------------------------------
    //
    // `ρₗ = −Gₗ/sₗ`, keeping only the first `L'`. Small `L'` throws away
    // features along with the noise; `L' = L` keeps everything, noise
    // included, and `1/s_{L−1}` is 4×10⁷ here.
    let omegas = linspace(-WMAX, WMAX, N_OMEGA);
    let v_at_omegas: Matrix<f64> = basis.evaluate_omega(&omegas)?;
    let half = size / 2;

    let rho_l_semi_noisy = divide_by_s(&s, &g_semi_noisy);
    let rho_l_insul_noisy = divide_by_s(&s, &g_insul_noisy);

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omegas.clone());
    table.push(
        "semielliptic_exact",
        omegas.iter().map(|&w| semielliptic(w)).collect::<Vec<_>>(),
    );
    table.push(
        "semielliptic_half",
        expand(&v_at_omegas, &rho_l_semi_noisy, half),
    );
    table.push(
        "semielliptic_full",
        expand(&v_at_omegas, &rho_l_semi_noisy, size),
    );
    table.push(
        "insulating_exact",
        omegas.iter().map(|&w| insulating(w)).collect::<Vec<_>>(),
    );
    table.push(
        "insulating_half",
        expand(&v_at_omegas, &rho_l_insul_noisy, half),
    );
    table.push(
        "insulating_full",
        expand(&v_at_omegas, &rho_l_insul_noisy, size),
    );
    write_table(&output_path(EXAMPLE, "tsvd")?, &table)?;

    // The failure this page is about: keeping every coefficient makes the
    // answer worse than keeping half of them. The noise was chosen to be
    // large enough to show that and small enough that the plot still fits on
    // a linear axis, so the gap is a factor of a few, not a factor of 10⁷.
    let worst_full = worst_deviation(
        &omegas,
        &expand(&v_at_omegas, &rho_l_semi_noisy, size),
        semielliptic,
    );
    let worst_half = worst_deviation(
        &omegas,
        &expand(&v_at_omegas, &rho_l_semi_noisy, half),
        semielliptic,
    );
    assert!(
        worst_full > 2.0 * worst_half,
        "the untruncated inversion is supposed to be far worse than the \
         truncated one, but they are {worst_full:.3e} and {worst_half:.3e}"
    );

    // --- ridge regression ---------------------------------------------------
    //
    // Instead of a hard cut-off, penalise `Σ|ρₗ|²`. The solution is
    // `ρₗ = −sₗ Gₗ / (sₗ² + α²)`, which passes the large singular values
    // through and rolls the small ones off smoothly.
    // ANCHOR: ridge
    let ridge: Vec<f64> = s.iter().map(|sl| -sl / (sl * sl + alpha * alpha)).collect();
    let rho_l_semi_ridge: Vec<f64> = ridge
        .iter()
        .zip(&g_semi_noisy)
        .map(|(r, g)| r * g)
        .collect();
    let rho_l_insul_ridge: Vec<f64> = ridge
        .iter()
        .zip(&g_insul_noisy)
        .map(|(r, g)| r * g)
        .collect();
    // ANCHOR_END: ridge

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omegas.clone());
    table.push(
        "semielliptic_ridge",
        expand(&v_at_omegas, &rho_l_semi_ridge, size),
    );
    table.push(
        "insulating_ridge",
        expand(&v_at_omegas, &rho_l_insul_ridge, size),
    );
    write_table(&output_path(EXAMPLE, "ridge")?, &table)?;

    // --- why ρₗ is the wrong thing to solve for -----------------------------
    //
    // `Gₗ` is compact because `sₗ` makes it so. `ρₗ` is not: for the
    // semi-elliptic model it decays like `1/l`, and for a set of delta peaks
    // it does not decay at all. A regulariser that acts on `ρₗ` is therefore
    // acting on coefficients that never become negligible.
    let poles: Vec<f64> = DISCRETE_POLES.iter().map(|x| x * WMAX).collect();
    let v_at_poles: Matrix<f64> = basis.evaluate_omega(&poles)?;
    let rho_discrete: Vec<f64> = (0..size)
        .map(|l| {
            (0..poles.len())
                .map(|p| *v_at_poles.get(&[p, l]).unwrap())
                .sum()
        })
        .collect();
    let g_discrete = coefficients(&s, &rho_discrete);

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..size).map(|l| l as f64).collect::<Vec<_>>());
    table.push("g_semielliptic", g_semi);
    table.push("rho_semielliptic", rho_semi);
    table.push("g_discrete", g_discrete);
    table.push("rho_discrete", rho_discrete);
    write_table(&output_path(EXAMPLE, "discrete")?, &table)?;

    // --- a real-axis basis instead ------------------------------------------
    //
    // Expand `ρ(ω) = Σₘ aₘ f(ω − ωₘ)` on a grid of Lorentzians. `f` is a
    // probability density, so non-negativity and the sum rule carry over from
    // `ρ` to `a`, which is what a constrained solver needs.
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omegas.clone());
    table.push(
        "f",
        omegas.iter().map(|&w| lorentz(w, eta)).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "lorentz")?, &table)?;

    let centres = linspace(-WMAX, WMAX, N_LORENTZ);
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..size).map(|l| l as f64).collect::<Vec<_>>());
    for (m, &centre) in centres.iter().enumerate() {
        let overlaps = lorentz_overlaps(&basis, centre, eta);
        let column: Vec<f64> = s
            .iter()
            .zip(&overlaps)
            .map(|(sl, overlap)| -sl * overlap)
            .collect();
        table.push(format!("k_{m}"), column);
    }
    write_table(&output_path(EXAMPLE, "lorentz_kernel")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("basis_size", vec![size as f64]);
    table.push("noise", vec![noise]);
    table.push("alpha", vec![alpha]);
    table.push("eta", vec![eta]);
    table.push("n_lorentz", vec![N_LORENTZ as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    println!(
        "basis size {size}, s_{}/s_0 = {:.3e}, noise {noise:.3e}, α {alpha:.3e}, η {eta:.5}",
        size - 1,
        s[size - 1] / s[0]
    );
    println!(
        "worst error of the reconstruction: {worst_half:.3e} at L' = {half}, \
         {worst_full:.3e} at L' = {size}"
    );

    Ok(())
}

// ANCHOR: coefficients
/// `Gₗ = −sₗ ρₗ`.
fn coefficients(s: &[f64], rho_l: &[f64]) -> Vec<f64> {
    s.iter().zip(rho_l).map(|(sl, rho)| -sl * rho).collect()
}
// ANCHOR_END: coefficients

/// `ρₗ = −Gₗ/sₗ`, the formal inverse.
fn divide_by_s(s: &[f64], g_l: &[f64]) -> Vec<f64> {
    g_l.iter().zip(s).map(|(g, sl)| g / -sl).collect()
}

fn add_noise(g_l: &[f64], draws: &[f64], noise: f64) -> Vec<f64> {
    // The notebook scales the noise by ‖Gₗ‖, so it is relative to the size of
    // the data rather than absolute.
    let norm = g_l.iter().map(|g| g * g).sum::<f64>().sqrt();
    g_l.iter()
        .zip(draws)
        .map(|(g, z)| g + noise * z * norm)
        .collect()
}

/// `Σ_{l<cutoff} vₗ(ω) ρₗ` on the grid `v` was evaluated on.
fn expand(v: &Matrix<f64>, rho_l: &[f64], cutoff: usize) -> Vec<f64> {
    let points = v.shape()[0];
    (0..points)
        .map(|i| {
            (0..cutoff)
                .map(|l| *v.get(&[i, l]).unwrap() * rho_l[l])
                .sum()
        })
        .collect()
}

fn worst_deviation<F>(omegas: &[f64], values: &[f64], exact: F) -> f64
where
    F: Fn(f64) -> f64,
{
    omegas
        .iter()
        .zip(values)
        .map(|(&w, &value)| (value - exact(w)).abs())
        .fold(0.0_f64, f64::max)
}

/// The semi-elliptic density of states filling the band.
fn semielliptic(omega: f64) -> f64 {
    shifted_semicircle(omega, 0.0, WMAX, 1.0)
}

/// The same band split in two by a gap around `ω = 0`.
fn insulating(omega: f64) -> f64 {
    shifted_semicircle(omega, WMAX / 2.0, WMAX / 4.0, 0.5)
        + shifted_semicircle(omega, -WMAX / 2.0, WMAX / 4.0, 0.5)
}

/// The Lorentz (Cauchy) density of half-width `eta`.
fn lorentz(omega: f64, eta: f64) -> f64 {
    eta / (std::f64::consts::PI * (omega * omega + eta * eta))
}

// ANCHOR: lorentz_overlaps
/// `∫ dω vₗ(ω) f(ω − centre)` over the basis' ω range, to machine precision.
///
/// `η` is a small fraction of the knot spacing, so quadrature on the knots
/// alone would step straight over the peak. The substitution
/// `ω = centre + η tan t` turns `f(ω − centre) dω` into `dt/π` and spreads the
/// peak over the whole integration range; the knots come along as the segment
/// edges in `t`, where `vₗ` is still a polynomial in `ω`.
fn lorentz_overlaps(
    basis: &FiniteTempBasis<LogisticKernel, Fermionic>,
    centre: f64,
    eta: f64,
) -> Vec<f64> {
    let v = basis.v();
    let edges: Vec<f64> = v
        .get_knots(None)
        .into_iter()
        .map(|omega| ((omega - centre) / eta).atan())
        .collect();
    // tan t varies fast at the ends of each segment, so take more points than
    // the polynomial degree alone would need.
    let order = v.get_polyorder() + 24;

    (0..basis.size())
        .map(|l| {
            let poly = &v[l];
            integrate_segments(
                |t| poly.evaluate(centre + eta * t.tan()) / std::f64::consts::PI,
                &edges,
                order,
            )
        })
        .collect()
}
// ANCHOR_END: lorentz_overlaps

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    assert!(count > 1, "a grid needs at least two points");
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}
