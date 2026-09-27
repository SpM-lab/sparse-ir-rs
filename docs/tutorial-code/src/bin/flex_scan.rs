//! The `d`-wave superconducting transition line of the square-lattice Hubbard
//! model in FLEX.
//!
//! Ported from the Python notebook `FLEX_py.ipynb` of sparse-ir-tutorial,
//! whose author is Niklas Witt; this is the calculation behind Figs. 3(b) and
//! 4 of Arita et al. (2000), and Fig. 2(a) of Witt et al. (2021).
//!
//! The point of interest for the basis is the sweep itself. `Λ = β ω_max` is
//! held fixed while `β` varies, so one singular value expansion serves every
//! temperature and the basis keeps the same size throughout — which means the
//! converged self-energy of one temperature is a legal starting point for the
//! next.

use std::error::Error as StdError;

use sparse_ir_tutorial::flex::{GapSolver, Settings, Solver};
use sparse_ir_tutorial::lattice::{Lattice, high_symmetry_path, sve_for};
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "flex_scan";

const T_HOP: f64 = 1.0;
const FILLING: f64 = 0.85;
const U: f64 = 4.0;
const NK_LIN: usize = 64;
/// `Λ` must cover the lowest temperature: `ω_max β_max = 10 × 40 = 400`.
const LAMBDA: f64 = 1e3;
const EPS: f64 = 1e-8;
const MIX: f64 = 0.2;
const ITERATIONS: usize = 30;
const GAP_ITERATIONS: usize = 30;
const GAP_TOL: f64 = 1e-4;
const RENORMALISATION_ITERATIONS: usize = 50;
const TEMPERATURES: [f64; 7] = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025];
/// The temperature whose momentum dependence is written out as well.
const PROBE: usize = 5;

fn main() -> Result<(), Box<dyn StdError>> {
    let beta_init = 1.0 / TEMPERATURES[0];
    let (kernel, sve) = sve_for(beta_init, LAMBDA / beta_init, EPS)?;

    let settings = Settings {
        u: U,
        filling: FILLING,
        mix: MIX,
        iterations: ITERATIONS,
        renormalisation_iterations: RENORMALISATION_ITERATIONS,
    };

    let mut sigma = None;
    let mut mu = Vec::new();
    let mut lambda_d = Vec::new();
    let mut chi_spin_max = Vec::new();
    let mut renormalisation_steps = Vec::new();
    let mut residual = Vec::new();
    let mut gap_iterations = Vec::new();
    let mut chi_path = Vec::new();
    let mut basis_report = None;

    for (index, &temperature) in TEMPERATURES.iter().enumerate() {
        let beta = 1.0 / temperature;
        let lattice = Lattice::from_sve(kernel, sve.clone(), NK_LIN, NK_LIN, T_HOP, beta, EPS)?;
        let mut solver = match sigma.take() {
            Some(previous) => Solver::new(&lattice, settings, previous)?,
            None => Solver::from_scratch(&lattice, settings)?,
        };
        solver.solve()?;
        sigma = Some(solver.self_energy_values().to_vec());

        let mut gap = GapSolver::new(&solver)?;
        gap.solve(&lattice, GAP_ITERATIONS, GAP_TOL)?;

        mu.push(solver.mu());
        lambda_d.push(gap.lambda());
        chi_spin_max.push(solver.max_chi_spin());
        renormalisation_steps.push(solver.renormalisation_steps() as f64);
        residual.push(solver.residual());
        gap_iterations.push(gap.iterations() as f64);

        if index == PROBE {
            let row_b = lattice.iw0_b() * lattice.nk();
            chi_path = high_symmetry_path(NK_LIN)
                .iter()
                .map(|&k| solver.chi_spin()[row_b + k].re)
                .collect();
        }
        basis_report = Some((
            lattice.basis_f().size() as f64,
            lattice.mesh_f().n_tau() as f64,
            lattice.mesh_f().n_wn() as f64,
            lattice.mesh_b().n_wn() as f64,
        ));
    }

    let (basis_size_f, n_tau, n_wn_f, n_wn_b) =
        basis_report.expect("the scan visits at least one temperature");

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T_HOP]);
    table.push("n", vec![FILLING]);
    table.push("u", vec![U]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("lambda_ir", vec![LAMBDA]);
    table.push("eps", vec![EPS]);
    table.push("mix", vec![MIX]);
    table.push("iterations", vec![ITERATIONS as f64]);
    table.push("gap_max_iterations", vec![GAP_ITERATIONS as f64]);
    table.push("gap_tol", vec![GAP_TOL]);
    table.push("t_num", vec![TEMPERATURES.len() as f64]);
    table.push("basis_size_f", vec![basis_size_f]);
    table.push("n_tau", vec![n_tau]);
    table.push("n_wn_f", vec![n_wn_f]);
    table.push("n_wn_b", vec![n_wn_b]);
    table.push("probe_temperature", vec![TEMPERATURES[PROBE]]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("temperature", TEMPERATURES.to_vec());
    table.push(
        "beta",
        TEMPERATURES.iter().map(|t| 1.0 / t).collect::<Vec<_>>(),
    );
    table.push("mu", mu);
    table.push("lambda_d", lambda_d);
    let inverse: Vec<f64> = chi_spin_max.iter().map(|c| 1.0 / c).collect();
    table.push("chi_spin_max", chi_spin_max);
    table.push("inverse_chi_spin_max", inverse);
    table.push("renormalisation_steps", renormalisation_steps);
    table.push("residual", residual);
    table.push("gap_iterations", gap_iterations);
    write_table(&output_path(EXAMPLE, "temperature")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "distance",
        (0..chi_path.len()).map(|i| i as f64).collect::<Vec<_>>(),
    );
    table.push("chi_spin", chi_path);
    write_table(&output_path(EXAMPLE, "chi_spin")?, &table)?;

    Ok(())
}
