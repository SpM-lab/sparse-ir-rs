//! Second-order perturbation theory for the Hubbard model on a square lattice.
//!
//! Ported from the Python notebook `second_order_perturbation_py.ipynb` of
//! sparse-ir-tutorial.
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
use sparse_ir::{Fermionic, FermionicFreq, FiniteTempBasis, LogisticKernel, MatsubaraSampling};
use sparse_ir_tutorial::{IrMesh, MomentumGrid, Table, output_path, provenance, write_table};

const EXAMPLE: &str = "second_order_perturbation";

const LAMBDA: f64 = 1e5;
const BETA: f64 = 1e3;
const EPS: f64 = 1e-7;
const WMAX: f64 = LAMBDA / BETA;
/// Points along each reciprocal lattice vector. Below about 64 an unphysical
/// structure appears in the self-energy at low frequency.
const NK_LIN: usize = 256;
/// The onsite repulsion. Half filling is arranged by measuring `ε(k)` from the
/// chemical potential `μ = U/2`, which for this dispersion means not shifting
/// it at all.
const U: f64 = 2.0;

fn main() -> Result<(), Box<dyn StdError>> {
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    let mesh = IrMesh::<Fermionic>::new(&basis)?;
    let grid = MomentumGrid::new(NK_LIN, NK_LIN);
    let nk = grid.len();

    // Γ = (0, 0) is the first point of the grid, M = (π, π) the middle one.
    let gamma = 0;
    let m_point = (NK_LIN / 2) * NK_LIN + NK_LIN / 2;

    let ek = grid.square_lattice_dispersion(1.0);
    let nu: Vec<f64> = mesh.wn().iter().map(|w| w.n() as f64 * PI / BETA).collect();

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
    // G₀(iν, k) = 1/(iν − ε(k)), stored row-major as (frequency, momentum).
    let mut gkf = Vec::with_capacity(mesh.n_wn() * nk);
    for &nu in &nu {
        for &e in &ek {
            gkf.push(Complex64::new(-e, nu).inv());
        }
    }

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
    let gkl = mesh.wn_to_l(&gkf, nk)?;
    assert_eq!(gkl.len(), basis.size() * nk);
    let gkt = mesh.l_to_tau(&gkl, nk)?;

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
    let grt = grid.k_to_r(&gkt);

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
    let reversed = mesh.reverse_tau(&grt, nk);
    let srt: Vec<Complex64> = grt
        .iter()
        .zip(&reversed)
        .map(|(g, g_reversed)| U * U * g * g * g_reversed)
        .collect();

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
    let srl = mesh.tau_to_l(&srt, nk)?;
    let skl = grid.r_to_k(&srl);

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
    let sigma_iv = mesh.l_to_wn(&skl, nk)?;

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

    // The coefficients are the whole answer: Σ can be evaluated on any
    // frequency, not only on the ones that were sampled. Here, every tenth
    // fermionic frequency out to |n| = 20000 — far beyond the sampling set.
    let far: Vec<i64> = (-10000..10000).step_by(20).map(|n| 2 * n + 1).collect();
    let freqs: Vec<FermionicFreq> = far
        .iter()
        .map(|&n| FermionicFreq::new(n))
        .collect::<Result<_, _>>()?;
    let sampling = MatsubaraSampling::<Fermionic>::with_sampling_points(&basis, freqs)?;
    // Only Γ is wanted, so evaluate one column rather than all of them.
    let skl_gamma: Vec<Complex64> = (0..basis.size()).map(|l| skl[l * nk + gamma]).collect();
    let sigma_far = sampling.evaluate(&skl_gamma)?;

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
