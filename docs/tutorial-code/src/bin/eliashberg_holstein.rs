//! Eliashberg theory of the superconducting state of a Jahn-Teller-Hubbard
//! model, at one temperature.
//!
//! Ported from the Python notebook `eliashberg_holstein_py.ipynb` of
//! sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/src/eliashberg_holstein_py.html>), whose
//! authors are Shintaro Hoshino and Hiroshi Shinaoka.
//!
//! Every quantity here is local, so the loop is as small as a self-consistent
//! calculation gets: four transforms between `τ` and the Matsubara
//! frequencies per step, and the density of states handled by a quadrature
//! rule rather than a momentum grid.

use std::error::Error as StdError;

use num_complex::Complex64;

use sparse_ir_tutorial::eliashberg::{Settings, Solver, coupling};
use sparse_ir_tutorial::lattice::Bases;
use sparse_ir_tutorial::{Table, input_path, output_path, provenance, read_table, write_table};

const EXAMPLE: &str = "eliashberg_holstein";

/// Half bandwidth of the semicircular density of states.
const D: f64 = 0.5;
const U: f64 = 2.0;
/// Hund's coupling, `0.03 U` in the notebook.
const J: f64 = 0.03 * U;
const OMEGA0: f64 = 0.15;
const MU: f64 = 0.0;
const EPS: f64 = 1e-7;
/// The basis has to cover the whole band with room to spare.
const WMAX: f64 = 10.0 * D;
const DEG_LEGGAUSS: usize = 100;
const MIXING: f64 = 0.3;

const BETA: f64 = 500.0;
/// Dimensionless electron-phonon coupling.
const LAMBDA0: f64 = 0.125;
const MAX_ITERATIONS: usize = 10_000;
const ATOL: f64 = 1e-10;

fn main() -> Result<(), Box<dyn StdError>> {
    // ANCHOR: bases
    let bases = Bases::new(BETA, WMAX, EPS)?;
    // ANCHOR_END: bases
    let mesh_f = bases.mesh_f();
    let mesh_b = bases.mesh_b();
    let g = coupling(LAMBDA0, OMEGA0);
    let settings = Settings {
        d: D,
        u: U,
        j: J,
        omega0: OMEGA0,
        g,
        mu: MU,
        deg_leggauss: DEG_LEGGAUSS,
        mixing: MIXING,
        max_iterations: MAX_ITERATIONS,
        atol: ATOL,
    };

    let mut solver = Solver::new(&bases, settings, read_noise("noise", mesh_f.n_wn())?);
    let converged = solver.solve()?;
    assert!(
        converged,
        "the loop did not settle in {MAX_ITERATIONS} iterations"
    );
    let energy = solver.internal_energy()?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![BETA]);
    table.push("d", vec![D]);
    table.push("u", vec![U]);
    table.push("j", vec![J]);
    table.push("omega0", vec![OMEGA0]);
    table.push("lambda0", vec![LAMBDA0]);
    table.push("g", vec![g]);
    table.push("mu", vec![MU]);
    table.push("eps", vec![EPS]);
    table.push("wmax", vec![WMAX]);
    table.push("deg_leggauss", vec![DEG_LEGGAUSS as f64]);
    table.push("mixing", vec![MIXING]);
    table.push("max_iterations", vec![MAX_ITERATIONS as f64]);
    table.push("atol", vec![ATOL]);
    table.push("iterations", vec![solver.iterations() as f64]);
    table.push("basis_size_f", vec![bases.basis_f().size() as f64]);
    table.push("basis_size_b", vec![bases.basis_b().size() as f64]);
    table.push("n_tau_f", vec![mesh_f.n_tau() as f64]);
    table.push("n_wn_f", vec![mesh_f.n_wn() as f64]);
    table.push("n_wn_b", vec![mesh_b.n_wn() as f64]);
    // The gap at the frequency closest to zero, which is where it is largest.
    table.push("gap", vec![solver.gap()[mesh_f.n_wn() / 2].re]);
    table.push("energy", vec![energy]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        mesh_f.wn().iter().map(|w| w.n() as f64).collect::<Vec<_>>(),
    );
    table.push("nu", bases.nu().to_vec());
    table.push("sigma_im", imaginary(solver.self_energy()));
    table.push("delta_re", real(solver.gap()));
    table.push("g_im", imaginary(solver.green_function()));
    table.push("f_re", real(solver.anomalous_green_function()));
    write_table(&output_path(EXAMPLE, "matsubara_f")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        mesh_b.wn().iter().map(|w| w.n() as f64).collect::<Vec<_>>(),
    );
    table.push(
        "nu",
        mesh_b
            .wn()
            .iter()
            .map(|w| w.n() as f64 * std::f64::consts::PI / BETA)
            .collect::<Vec<_>>(),
    );
    table.push("d_re", real(solver.phonon_propagator()));
    table.push("phi_re", real(solver.polarisation()));
    write_table(&output_path(EXAMPLE, "matsubara_b")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", mesh_b.tau_points().to_vec());
    table.push("d_tau", real(solver.phonon_propagator_tau()));
    write_table(&output_path(EXAMPLE, "tau_b")?, &table)?;

    // The point of the whole exercise: the anomalous Green's function is as
    // compact in the basis as the singular values say it can be.
    let f_l = mesh_f.wn_to_l(solver.anomalous_green_function(), 1)?;
    let s = bases.basis_f().s();
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..f_l.len()).map(|l| l as f64).collect::<Vec<_>>());
    table.push(
        "f_l",
        f_l.iter().map(|value| value.norm()).collect::<Vec<_>>(),
    );
    table.push(
        "s_ratio",
        s.iter().map(|value| value / s[0]).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "basis")?, &table)?;

    Ok(())
}

/// The committed starting self-energy.
///
/// The normal state solves these equations too, so a loop started exactly on
/// it never leaves; the starting point has to be perturbed. The perturbation
/// is committed rather than drawn so that this run and its Python reference
/// start from bit-identical numbers.
fn read_noise(name: &str, expected: usize) -> Result<Vec<Complex64>, Box<dyn StdError>> {
    let table = read_table(&input_path(EXAMPLE, name))?;
    let re = table.expect_column("sigma_re");
    let im = table.expect_column("sigma_im");
    assert_eq!(
        re.len(),
        expected,
        "the committed starting point has {} values but the basis needs {expected}",
        re.len()
    );
    Ok(re
        .iter()
        .zip(im)
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect())
}

fn real(values: &[Complex64]) -> Vec<f64> {
    values.iter().map(|value| value.re).collect()
}

fn imaginary(values: &[Complex64]) -> Vec<f64> {
    values.iter().map(|value| value.im).collect()
}
