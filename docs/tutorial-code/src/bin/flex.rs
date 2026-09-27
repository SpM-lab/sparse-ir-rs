//! The fluctuation-exchange approximation for the square-lattice Hubbard
//! model, and the linearised Eliashberg equation built on top of it.
//!
//! Ported from the Python notebook `FLEX_py.ipynb` of sparse-ir-tutorial,
//! whose author is Niklas Witt.
//!
//! Where `tpsc` fixed its vertices by sum rules and stopped, FLEX iterates
//! the Dyson equation with the bare `U` until the self-energy settles, and
//! then asks a second question of the converged state: does the
//! spin-fluctuation interaction bind a `d`-wave pair?

use std::error::Error as StdError;

use sparse_ir_tutorial::flex::{GapSolver, Settings, Solver};
use sparse_ir_tutorial::lattice::{Lattice, high_symmetry_path};
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "flex";

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
const MIX: f64 = 0.2;
const ITERATIONS: usize = 30;
const GAP_ITERATIONS: usize = 30;
/// The power method stops when `λ` moves by less than this.
const GAP_TOL: f64 = 1e-4;
const RENORMALISATION_ITERATIONS: usize = 50;

fn main() -> Result<(), Box<dyn StdError>> {
    let lattice = Lattice::new(NK_LIN, NK_LIN, T_HOP, BETA, WMAX, EPS)?;
    let settings = Settings {
        u: U,
        filling: FILLING,
        mix: MIX,
        iterations: ITERATIONS,
        renormalisation_iterations: RENORMALISATION_ITERATIONS,
    };
    let mut solver = Solver::from_scratch(&lattice, settings)?;
    solver.solve()?;

    let mut gap = GapSolver::new(&solver)?;
    gap.solve(&lattice, GAP_ITERATIONS, GAP_TOL)?;

    let nk = lattice.nk();
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
    table.push("mix", vec![MIX]);
    table.push("iterations", vec![ITERATIONS as f64]);
    table.push("gap_max_iterations", vec![GAP_ITERATIONS as f64]);
    table.push("gap_tol", vec![GAP_TOL]);
    table.push("gap_iterations", vec![gap.iterations() as f64]);
    table.push("basis_size_f", vec![lattice.basis_f().size() as f64]);
    table.push("basis_size_b", vec![lattice.basis_b().size() as f64]);
    table.push("n_tau", vec![mesh_f.n_tau() as f64]);
    table.push("n_wn_f", vec![mesh_f.n_wn() as f64]);
    table.push("n_wn_b", vec![mesh_b.n_wn() as f64]);
    table.push(
        "renormalisation_steps",
        vec![solver.renormalisation_steps() as f64],
    );
    table.push("mu", vec![solver.mu()]);
    table.push("residual", vec![solver.residual()]);
    table.push("chi_spin_max", vec![solver.max_chi_spin()]);
    table.push("lambda_d", vec![gap.lambda()]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // Everything the notebook plots over the zone, at the lowest frequency of
    // its statistics.
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
            .map(|k| solver.green_function()[row_f + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_im",
        (0..nk)
            .map(|k| solver.self_energy_values()[row_f + k].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_0",
        (0..nk)
            .map(|k| solver.chi_0()[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_spin",
        (0..nk)
            .map(|k| solver.chi_spin()[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "delta_re",
        (0..nk).map(|k| gap.gap()[row_f + k].re).collect::<Vec<_>>(),
    );
    table.push(
        "f_re",
        (0..nk)
            .map(|k| gap.anomalous_green()[row_f + k].re)
            .collect::<Vec<_>>(),
    );
    table.push("delta_seed", gap.seed().to_vec());
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
            .map(|&k| solver.chi_spin()[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_charge",
        path.iter()
            .map(|&k| solver.chi_charge()[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "chi_0",
        path.iter()
            .map(|&k| solver.chi_0()[row_b + k].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "path")?, &table)?;

    // The self-energy and the gap at the antinode, where the spin
    // fluctuations are strongest, against frequency.
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
            .map(|i| solver.self_energy_values()[i * nk + antinode].im)
            .collect::<Vec<_>>(),
    );
    table.push(
        "sigma_re",
        (0..mesh_f.n_wn())
            .map(|i| solver.self_energy_values()[i * nk + antinode].re)
            .collect::<Vec<_>>(),
    );
    table.push(
        "delta_re",
        (0..mesh_f.n_wn())
            .map(|i| gap.gap()[i * nk + antinode].re)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy")?, &table)?;

    Ok(())
}
