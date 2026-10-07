//! One GF2 & GW iteration, and then twenty of them.
//!
//! Ported from the Python notebook `GW_py.ipynb` of sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/>).
//!
//! The Hedin equations are each diagonal in one representation or the other,
//! so a GW iteration is a tour of all of them:
//!
//! ```text
//! G(iν) → Gₗ → G(τ) → P(τ) = G(τ)G(β − τ) → Pₗ → P(iω)
//!       → W(iω) = U/(1 − UP) − U → Wₗ → W(τ)
//!       → Σ(τ) = G(τ)W(τ) → Σₗ → Σ(iν) → G(iν)
//! ```
//!
//! `G` and `Σ` are fermionic, `P` and `W` bosonic, and the products in
//! imaginary time mix them: `P` is built from `G` at the *bosonic* sampling
//! times, and `Σ` from `W` at the *fermionic* ones. Which sign a function
//! picks up on the `[−β/2, β/2]` grid is decided by the function, never by the
//! grid it is sampled on — `G` stays anti-periodic at bosonic times, `W` stays
//! periodic at fermionic ones. `Basis::evaluate_tau` applies the right one,
//! because it knows the statistics of the basis it belongs to.

use std::error::Error as StdError;

use num_complex::Complex64;
use sparse_ir::{Basis, Bosonic, Fermionic, FiniteTempBasis, LogisticKernel};
use sparse_ir_tutorial::{
    IrMesh, Table, evaluate_rows, output_path, provenance, semicircle_coefficients, write_table,
};

const EXAMPLE: &str = "gw";

const T: f64 = 0.1;
const BETA: f64 = 1.0 / T;
const WMAX: f64 = 1.0;
/// The bare Coulomb interaction.
const U: f64 = 0.5;
/// How many iterations to run. The difference between successive self-energies
/// is down to 10⁻¹⁶ well before the last one, so the answer below is the fixed
/// point rather than a snapshot of a walk towards it.
const ITERATIONS: usize = 20;

fn main() -> Result<(), Box<dyn StdError>> {
    // ANCHOR: bases
    // Two bases with the same Λ = βω_max. `new` runs the SVE for each; for a
    // larger Λ, compute it once and use `FiniteTempBasis::from_sve_result`.
    let basis_f = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(BETA * WMAX)?,
        BETA,
        None,
        None,
    )?;
    let basis_b = FiniteTempBasis::<LogisticKernel, Bosonic>::new(
        LogisticKernel::new(BETA * WMAX)?,
        BETA,
        None,
        None,
    )?;
    let mesh_f = IrMesh::<Fermionic>::new(&basis_f)?;
    let mesh_b = IrMesh::<Bosonic>::new(&basis_b)?;
    // ANCHOR_END: bases

    // ANCHOR: cross
    // The two cross-statistics evaluation matrices, built once: the fermionic
    // basis functions at the bosonic sampling times, and the other way round.
    let uf_at_tau_b = basis_f.evaluate_tau(mesh_b.tau_points())?;
    let ub_at_tau_f = basis_b.evaluate_tau(mesh_f.tau_points())?;
    // u_l(β⁻), for the Hartree term.
    let uf_at_beta = basis_f.evaluate_tau(&[BETA])?;
    // ANCHOR_END: cross

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("u", vec![U]);
    table.push("iterations", vec![ITERATIONS as f64]);
    table.push("basis_size_f", vec![basis_f.size() as f64]);
    table.push("basis_size_b", vec![basis_b.size() as f64]);
    table.push("n_tau_f", vec![mesh_f.n_tau() as f64]);
    table.push("n_tau_b", vec![mesh_b.n_tau() as f64]);
    table.push("n_wn_f", vec![mesh_f.n_wn() as f64]);
    table.push("n_wn_b", vec![mesh_b.n_wn() as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // The non-interacting Green's function of a semicircular spectrum.
    let (_, g_l_0) = semicircle_coefficients(&basis_f);
    let g_l_0: Vec<Complex64> = g_l_0.iter().map(|&g| Complex64::new(g, 0.0)).collect();
    let g_iw_0 = mesh_f.l_to_wn(&g_l_0, 1)?;

    let n_f: Vec<f64> = mesh_f.wn().iter().map(|w| w.n() as f64).collect();
    let n_b: Vec<f64> = mesh_b.wn().iter().map(|w| w.n() as f64).collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n", n_f.clone());
    table.push("g_re", g_iw_0.iter().map(|z| z.re).collect::<Vec<_>>());
    table.push("g_im", g_iw_0.iter().map(|z| z.im).collect::<Vec<_>>());
    write_table(&output_path(EXAMPLE, "green_initial")?, &table)?;

    let mut g_iw_f = g_iw_0.clone();
    let mut differences = Vec::with_capacity(ITERATIONS - 1);
    let mut previous: Option<Vec<Complex64>> = None;
    let mut sigma = Vec::new();

    for iteration in 0..ITERATIONS {
        let g_l_f = mesh_f.wn_to_l(&g_iw_f, 1)?;
        let g_tau_f = mesh_f.l_to_tau(&g_l_f, 1)?;

        // --- into bosonic statistics ----------------------------------------
        // ANCHOR: polarization
        let g_tau_b = evaluate_rows(&uf_at_tau_b, &g_l_f, 1);
        // P(τ) = G(τ) G(β − τ), with G(β − τ) = −G(−τ): the reversed array
        // with the *fermionic* sign, even though the times are the bosonic
        // ones. This grid also carries a point at exactly τ = β/2, whose
        // mirror is itself one period away; `reverse_tau_as` handles that.
        let g_beta_minus_tau = mesh_b.reverse_tau_as::<Fermionic>(&g_tau_b, 1);
        let p_tau_b: Vec<Complex64> = g_tau_b
            .iter()
            .zip(&g_beta_minus_tau)
            .map(|(g, g_reversed)| g * g_reversed)
            .collect();
        let p_l_b = mesh_b.tau_to_l(&p_tau_b, 1)?;
        let p_iw_b = mesh_b.l_to_wn(&p_l_b, 1)?;
        // ANCHOR_END: polarization

        // --- the screened interaction ----------------------------------------
        // ANCHOR: screened
        // W = U/(1 − UP); only the frequency-dependent part U/(1 − UP) − U is
        // carried on, the constant being the bare interaction itself.
        let w_iw_b: Vec<Complex64> = p_iw_b.iter().map(|p| U / (1.0 - U * p) - U).collect();
        let w_l_b = mesh_b.wn_to_l(&w_iw_b, 1)?;
        // --- and back into fermionic statistics -------------------------------
        let w_tau_f = evaluate_rows(&ub_at_tau_f, &w_l_b, 1);
        // ANCHOR_END: screened

        // --- the self-energy ---------------------------------------------------
        // ANCHOR: self_energy
        let e_tau_f: Vec<Complex64> = g_tau_f.iter().zip(&w_tau_f).map(|(g, w)| g * w).collect();
        let e_l_f = mesh_f.tau_to_l(&e_tau_f, 1)?;
        let e_iw_f = mesh_f.l_to_wn(&e_l_f, 1)?;
        // The static term U G(β⁻) = −U⟨n⟩. The name follows the notebook; in
        // Hedin's equations this instantaneous piece is the exchange (Fock)
        // term. It is subtracted, so the reported Σ carries +U⟨n⟩.
        let hartree: Complex64 = U * evaluate_rows(&uf_at_beta, &g_l_f, 1)[0];
        let e_iw_f_hartree: Vec<Complex64> = e_iw_f.iter().map(|e| e - hartree).collect();
        // ANCHOR_END: self_energy

        if let Some(previous) = &previous {
            differences.push(
                e_iw_f_hartree
                    .iter()
                    .zip(previous)
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0, f64::max),
            );
        }
        previous = Some(e_iw_f_hartree.clone());

        if iteration == 0 {
            write_tau_table(
                EXAMPLE,
                "green_tau",
                "tau_f",
                mesh_f.tau_points(),
                "g",
                &g_tau_f,
            )?;
            write_tau_table(
                EXAMPLE,
                "green_tau_bosonic",
                "tau_b",
                mesh_b.tau_points(),
                "g",
                &g_tau_b,
            )?;
            write_tau_table(
                EXAMPLE,
                "green_tau_reversed",
                "tau_b",
                mesh_b.tau_points(),
                "g",
                &g_beta_minus_tau,
            )?;
            write_tau_table(
                EXAMPLE,
                "polarization_tau",
                "tau_b",
                mesh_b.tau_points(),
                "p",
                &p_tau_b,
            )?;
            write_abs_table(EXAMPLE, "polarization_coefficients", "p_l_abs", &p_l_b)?;
            write_freq_table(EXAMPLE, "polarization_matsubara", &n_b, "p", &p_iw_b)?;
            write_freq_table(EXAMPLE, "screened_matsubara", &n_b, "w", &w_iw_b)?;
            write_abs_table(EXAMPLE, "screened_coefficients", "w_l_abs", &w_l_b)?;
            write_tau_table(
                EXAMPLE,
                "screened_tau",
                "tau_f",
                mesh_f.tau_points(),
                "w",
                &w_tau_f,
            )?;
            write_tau_table(
                EXAMPLE,
                "self_energy_tau",
                "tau_f",
                mesh_f.tau_points(),
                "e",
                &e_tau_f,
            )?;
            write_abs_table(EXAMPLE, "self_energy_coefficients", "e_l_abs", &e_l_f)?;

            let mut table = Table::new(provenance(EXAMPLE));
            table.push("n", n_f.clone());
            table.push(
                "e_re",
                e_iw_f_hartree.iter().map(|z| z.re).collect::<Vec<_>>(),
            );
            table.push(
                "e_im",
                e_iw_f_hartree.iter().map(|z| z.im).collect::<Vec<_>>(),
            );
            table.push("hartree", vec![hartree.re; n_f.len()]);
            write_table(&output_path(EXAMPLE, "self_energy_matsubara")?, &table)?;
        }

        // ANCHOR: dyson
        // The Dyson equation. The static term is left out of it, as in the
        // notebook: it is a constant shift of the chemical potential.
        g_iw_f = g_iw_0
            .iter()
            .zip(&e_iw_f)
            .map(|(g0, e)| (g0.inv() - e).inv())
            .collect();
        // ANCHOR_END: dyson
        sigma = e_iw_f_hartree;
    }

    write_freq_table(EXAMPLE, "self_energy_final", &n_f, "e", &sigma)?;
    write_freq_table(EXAMPLE, "green_final", &n_f, "g", &g_iw_f)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "iteration",
        (1..ITERATIONS).map(|i| i as f64).collect::<Vec<_>>(),
    );
    table.push("difference", differences);
    write_table(&output_path(EXAMPLE, "convergence")?, &table)?;

    Ok(())
}

fn write_tau_table(
    example: &str,
    name: &str,
    tau_column: &str,
    tau: &[f64],
    quantity: &str,
    values: &[Complex64],
) -> Result<(), Box<dyn StdError>> {
    let mut table = Table::new(provenance(example));
    table.push(tau_column, tau.to_vec());
    table.push(
        format!("{quantity}_re"),
        values.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    table.push(
        format!("{quantity}_im"),
        values.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    write_table(&output_path(example, name)?, &table)?;
    Ok(())
}

fn write_freq_table(
    example: &str,
    name: &str,
    n: &[f64],
    quantity: &str,
    values: &[Complex64],
) -> Result<(), Box<dyn StdError>> {
    let mut table = Table::new(provenance(example));
    table.push("n", n.to_vec());
    table.push(
        format!("{quantity}_re"),
        values.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    table.push(
        format!("{quantity}_im"),
        values.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    write_table(&output_path(example, name)?, &table)?;
    Ok(())
}

fn write_abs_table(
    example: &str,
    name: &str,
    column: &str,
    coefficients: &[Complex64],
) -> Result<(), Box<dyn StdError>> {
    let mut table = Table::new(provenance(example));
    table.push(
        "l",
        (0..coefficients.len())
            .map(|l| l as f64)
            .collect::<Vec<_>>(),
    );
    table.push(
        column,
        coefficients.iter().map(|z| z.norm()).collect::<Vec<_>>(),
    );
    write_table(&output_path(example, name)?, &table)?;
    Ok(())
}
