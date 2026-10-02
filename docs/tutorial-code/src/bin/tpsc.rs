//! The two-particle self-consistent approximation for the square-lattice
//! Hubbard model.
//!
//! Ported from the Python notebook `TPSC_py.ipynb` of sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/src/TPSC_py.html>), whose author is Niklas
//! Witt.
//!
//! TPSC is RPA with the interaction left unknown and then pinned down by the
//! local sum rules — one equation for the spin vertex, one for the charge
//! vertex. Both sum rules are Matsubara sums of a susceptibility over the
//! whole zone, which the basis turns into an evaluation at `τ = 0`, so the
//! root searches that solve them are cheap enough to run dozens of times.

use std::error::Error as StdError;

use sparse_ir_tutorial::lattice::{Lattice, high_symmetry_path};
use sparse_ir_tutorial::tpsc::solve;
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "tpsc";

/// Nearest-neighbour hopping; the bandwidth is `8t`.
const T_HOP: f64 = 1.0;
const TEMPERATURE: f64 = 0.1;
const BETA: f64 = 1.0 / TEMPERATURE;
/// The basis needs `ω_max` at least the bandwidth.
const WMAX: f64 = 10.0;
/// Electron filling per site, summed over both spins; `n = 1` is half filling.
const FILLING: f64 = 0.85;
const U: f64 = 4.0;
const NK_LIN: usize = 24;
const EPS: f64 = 1e-10;

fn main() -> Result<(), Box<dyn StdError>> {
    let lattice = Lattice::new(NK_LIN, NK_LIN, T_HOP, BETA, WMAX, EPS)?;
    let solution = solve(&lattice, U, FILLING)?;

    let nk = lattice.grid().len();
    let mesh_f = lattice.mesh_f();
    let mesh_b = lattice.mesh_b();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T_HOP]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("n", vec![FILLING]);
    table.push("u", vec![U]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("eps", vec![EPS]);
    table.push("basis_size_f", vec![lattice.basis_f().size() as f64]);
    table.push("basis_size_b", vec![lattice.basis_b().size() as f64]);
    table.push("n_tau", vec![mesh_f.n_tau() as f64]);
    table.push("n_wn_f", vec![mesh_f.n_wn() as f64]);
    table.push("n_wn_b", vec![mesh_b.n_wn() as f64]);
    table.push("mu_0", vec![solution.mu_0]);
    table.push("mu", vec![solution.mu]);
    table.push("u_crit", vec![solution.u_crit]);
    table.push("u_sp", vec![solution.u_sp]);
    table.push("u_ch", vec![solution.u_ch]);
    table.push("docc", vec![solution.docc]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // The three quantities the notebook plots over the zone, at the lowest
    // frequency of their statistics.
    let row_f = lattice.iw0_f() * nk;
    let row_b = lattice.iw0_b() * nk;
    let mut table = Table::new(provenance(EXAMPLE));
    let (mut kx, mut ky) = (Vec::with_capacity(nk), Vec::with_capacity(nk));
    for index in 0..nk {
        let (k1, k2) = lattice.grid().coordinates(index);
        kx.push(2.0 * k1);
        ky.push(2.0 * k2);
    }
    table.push("kx", kx);
    table.push("ky", ky);
    table.push("ek", lattice.dispersion().to_vec());
    table.push(
        "g_re",
        (0..nk)
            .map(|k| solution.green[row_f + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_im",
        (0..nk)
            .map(|k| solution.self_energy[row_f + k].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_0",
        (0..nk)
            .map(|k| solution.chi_0[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_spin",
        (0..nk)
            .map(|k| solution.chi_spin[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_charge",
        (0..nk)
            .map(|k| solution.chi_charge[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "momentum")?, &table)?;

    // A cut along Γ → X → M → Γ, for a one-dimensional view of the same data.
    let mut table = Table::new(provenance(EXAMPLE));
    let path = high_symmetry_path(NK_LIN);
    table.push(
        "distance",
        (0..path.len()).map(|i| i as f64).collect::<Vec<_>>(),
    );
    table.push(
        "chi_spin",
        path.iter()
            .map(|&k| solution.chi_spin[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_charge",
        path.iter()
            .map(|&k| solution.chi_charge[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_0",
        path.iter()
            .map(|&k| solution.chi_0[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "path")?, &table)?;

    // The self-energy at the antinode, where the spin fluctuations are
    // strongest, against frequency.
    let antinode = (NK_LIN / 2) * NK_LIN;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        mesh_f.wn().iter().map(|w| w.n() as f64).collect::<Vec<_>>(),
    );
    table.push("nu", lattice.nu().to_vec());
    table.push(
        "sigma_im",
        (0..mesh_f.n_wn())
            .map(|i| solution.self_energy[i * nk + antinode].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_re",
        (0..mesh_f.n_wn())
            .map(|i| solution.self_energy[i * nk + antinode].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy")?, &table)?;

    Ok(())
}
