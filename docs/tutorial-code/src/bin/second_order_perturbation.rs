//! Second-order perturbation theory for the Hubbard model on a square lattice.
//!
//! Ported from the Python notebook `second_order_perturbation_py.ipynb` of
//! sparse-ir-tutorial-v2 (<https://spm-lab.github.io/sparse-ir-tutorial-v2/>).
//!
//! The second-order self-energy is a product of three Green's functions in
//! imaginary time and real space,
//!
//! ```text
//! Σ(τ, r) = U² G(τ, r)² G(β − τ, r),
//! ```
//!
//! which is why the calculation goes back and forth between representations:
//! `G` starts life on the Matsubara axis, where it is a formula; the product
//! only takes this form in `(τ, r)`; and the answer is wanted back on the
//! Matsubara axis. Each hop is one sparse-sampling fit and one evaluation, and
//! the momentum ones are FFTs. No dense grid in `τ` or `ν` appears anywhere.
//!
//! One convention to keep an eye on: this library puts the sampling times on
//! `[−β/2, β/2]`, so `G(β − τ)` is `ζ G(−τ)` — the reversed rows with a sign
//! for fermions. `IrMesh::reverse_tau` is that operation, and
//! `tests/tau_convention.rs` is where it is pinned down.

use std::error::Error as StdError;
use std::f64::consts::PI;

use num_complex::Complex64;
// ANCHOR: imports
use sparse_ir::{Fermionic, FermionicFreq, FiniteTempBasis, LogisticKernel, MatsubaraSampling};
// ANCHOR_END: imports
use sparse_ir_tutorial::{IrMesh, MomentumGrid, Table, output_path, provenance, write_table};

const EXAMPLE: &str = "second_order_perturbation";

// ANCHOR: parameters
const LAMBDA: f64 = 1e5;
const BETA: f64 = 1e3;
const EPS: f64 = 1e-7;
const WMAX: f64 = LAMBDA / BETA;
// ANCHOR_END: parameters
/// Points along each reciprocal lattice vector. Below about 64 an unphysical
/// structure appears in the self-energy at low frequency.
const NK_LIN: usize = 256;
/// The onsite repulsion. At half filling `μ = U/2` cancels the Hartree shift
/// `U⟨n_σ̄⟩ = U/2`, so `G₀` is written with the shifted `μ̃ = μ − U/2 = 0` and
/// `ε(k)` is used as it stands.
const U: f64 = 2.0;

fn main() -> Result<(), Box<dyn StdError>> {
    // ANCHOR: basis
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    // ANCHOR_END: basis
    // ANCHOR: mesh
    let mesh = IrMesh::<Fermionic>::new(&basis)?;
    let grid = MomentumGrid::new(NK_LIN, NK_LIN);
    let nk = grid.len();
    // ANCHOR_END: mesh

    // Γ = (0, 0) is the first point of the grid, M = (π, π) the middle one.
    let gamma = 0;
    let m_point = (NK_LIN / 2) * NK_LIN + NK_LIN / 2;

    // ANCHOR: dispersion
    let ek = grid.square_lattice_dispersion(1.0);
    // ν_n = nπ/β, with the reduced (odd) index n of the sampling frequencies.
    let nu: Vec<f64> = mesh.wn().iter().map(|w| w.n() as f64 * PI / BETA).collect();
    // ANCHOR_END: dispersion

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("lambda", vec![LAMBDA]);
    table.push("u", vec![U]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("n_tau", vec![mesh.n_tau() as f64]);
    table.push("n_matsubara", vec![mesh.n_wn() as f64]);
    table.push("cond_tau", vec![mesh.tau_sampling().condition_number()?]);
    table.push(
        "cond_matsubara",
        vec![mesh.wn_sampling().condition_number()?],
    );
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // --- the non-interacting Green's function -------------------------------
    // ANCHOR: green
    // G₀(iν, k) = 1/(iν − ε(k)), stored row-major as (frequency, momentum).
    let mut gkf = Vec::with_capacity(mesh.n_wn() * nk);
    for &nu in &nu {
        for &e in &ek {
            gkf.push(Complex64::new(-e, nu).inv());
        }
    }
    // ANCHOR_END: green

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        mesh.wn().iter().map(|w| w.n() as f64).collect::<Vec<_>>(),
    );
    table.push("nu", nu.clone());
    table.push(
        "g_gamma_im",
        (0..mesh.n_wn())
            .map(|i| gkf[i * nk + gamma].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "g_gamma_re",
        (0..mesh.n_wn())
            .map(|i| gkf[i * nk + gamma].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "green_matsubara")?, &table)?;

    // --- into the basis, then onto the sampling times ------------------------
    // ANCHOR: to_tau
    let gkl = mesh.wn_to_l(&gkf, nk)?; // values → coefficients
    assert_eq!(gkl.len(), basis.size() * nk);
    let gkt = mesh.l_to_tau(&gkl, nk)?; // coefficients → values
    // ANCHOR_END: to_tau

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push(
        "g_gamma_abs",
        (0..basis.size())
            .map(|l| gkl[l * nk + gamma].norm())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "green_coefficients")?, &table)?;

    // --- and into real space -------------------------------------------------
    // ANCHOR: to_real_space
    let grt = grid.k_to_r(&gkt);
    // ANCHOR_END: to_real_space

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", mesh.tau_points().to_vec());
    table.push(
        "g_gamma",
        (0..mesh.n_tau())
            .map(|i| gkt[i * nk + gamma].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "g_m",
        (0..mesh.n_tau())
            .map(|i| gkt[i * nk + m_point].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "g_origin",
        (0..mesh.n_tau())
            .map(|i| grt[i * nk].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "green_tau")?, &table)?;

    // --- the self-energy -----------------------------------------------------
    // ANCHOR: self_energy
    // G(β − τ) on the symmetric grid: the reversed rows, times ζ = −1.
    let reversed = mesh.reverse_tau(&grt, nk);
    let srt: Vec<Complex64> = grt
        .iter()
        .zip(&reversed)
        .map(|(g, g_reversed)| U * U * g * g * g_reversed)
        .collect();
    // ANCHOR_END: self_energy

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", mesh.tau_points().to_vec());
    table.push(
        "sigma_origin",
        (0..mesh.n_tau())
            .map(|i| srt[i * nk].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy_tau")?, &table)?;

    // --- back to the basis and to momentum ------------------------------------
    // ANCHOR: back_to_l
    let srl = mesh.tau_to_l(&srt, nk)?; // values → coefficients
    let skl = grid.r_to_k(&srl); // and back to momentum
    // ANCHOR_END: back_to_l

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push(
        "sigma_origin_abs",
        (0..basis.size())
            .map(|l| srl[l * nk].norm())
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_gamma_abs",
        (0..basis.size())
            .map(|l| skl[l * nk + gamma].norm())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy_coefficients")?, &table)?;

    // --- and onto the Matsubara axis -------------------------------------------
    // ANCHOR: to_matsubara
    let sigma_iv = mesh.l_to_wn(&skl, nk)?;
    // ANCHOR_END: to_matsubara

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        mesh.wn().iter().map(|w| w.n() as f64).collect::<Vec<_>>(),
    );
    table.push("nu", nu);
    table.push(
        "sigma_gamma_im",
        (0..mesh.n_wn())
            .map(|i| sigma_iv[i * nk + gamma].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_gamma_re",
        (0..mesh.n_wn())
            .map(|i| sigma_iv[i * nk + gamma].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy_matsubara")?, &table)?;

    // ANCHOR: far
    // The coefficients are the whole answer: Σ can be evaluated on any
    // frequency, not only on the ones that were sampled. Here, every twentieth
    // fermionic frequency (m in steps of 20, so n = 2m + 1 in steps of 40) out
    // to |n| ≈ 20000 — far beyond the sampling set.
    let far: Vec<i64> = (-10000..10000).step_by(20).map(|n| 2 * n + 1).collect();
    let freqs: Vec<FermionicFreq> = far
        .iter()
        .map(|&n| FermionicFreq::new(n))
        .collect::<Result<_, _>>()?;
    let sampling = MatsubaraSampling::<Fermionic>::with_sampling_points(&basis, freqs)?;
    // Only Γ is wanted, so evaluate one column rather than all of them.
    let skl_gamma: Vec<Complex64> = (0..basis.size()).map(|l| skl[l * nk + gamma]).collect();
    let sigma_far = sampling.evaluate(&skl_gamma)?;
    // ANCHOR_END: far

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n", far.iter().map(|&n| n as f64).collect::<Vec<_>>());
    table.push(
        "nu",
        far.iter()
            .map(|&n| n as f64 * PI / BETA)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_gamma_im",
        sigma_far.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    table.push(
        "sigma_gamma_re",
        sigma_far.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy_far")?, &table)?;

    Ok(())
}
