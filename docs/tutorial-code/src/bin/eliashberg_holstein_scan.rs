//! The specific heat of the Jahn-Teller-Hubbard model across its
//! superconducting transition.
//!
//! Ported from the Python notebook `eliashberg_holstein_py.ipynb` of
//! sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/src/eliashberg_holstein_py.html>), whose
//! authors are Shintaro Hoshino and Hiroshi Shinaoka.
//!
//! The specific heat is a numerical derivative of the internal energy, so
//! every temperature is solved twice, at `T` and at `T + dt`. The sweep holds
//! `Λ = β ω_max` fixed so that one singular value expansion serves the whole
//! range and the basis keeps the same size, which is what lets each solution
//! start from the one before it.

use std::error::Error as StdError;

use num_complex::Complex64;

use sparse_ir_tutorial::eliashberg::{Settings, Solver, coupling};
use sparse_ir_tutorial::lattice::{Bases, sve_for};
use sparse_ir_tutorial::{Table, input_path, output_path, provenance, read_table, write_table};

/// The example whose committed input this one shares.
const INPUT_EXAMPLE: &str = "eliashberg_holstein";
const EXAMPLE: &str = "eliashberg_holstein_scan";

const D: f64 = 0.5;
const U: f64 = 2.0;
const J: f64 = 0.03 * U;
const OMEGA0: f64 = 0.15;
const MU: f64 = 0.0;
const EPS: f64 = 1e-7;
const WMAX: f64 = 10.0 * D;
const DEG_LEGGAUSS: usize = 100;
const MIXING: f64 = 0.3;

/// Stronger coupling than the single-temperature example, so that the
/// transition falls inside the window below.
const LAMBDA0: f64 = 0.175;
const MAX_ITERATIONS: usize = 100_000;
const ATOL: f64 = 1e-6;
const T_MIN: f64 = 0.009;
const T_MAX: f64 = 0.013;
const T_NUM: usize = 10;
/// The step of the numerical derivative `C = dE/dT`.
const DT: f64 = 1e-5;

fn main() -> Result<(), Box<dyn StdError>> {
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

    let probes = linspace(T_MIN, T_MAX, T_NUM);
    // Both ends of every derivative, in one ascending sweep.
    let mut temperatures: Vec<f64> = probes.iter().flat_map(|&t| [t, t + DT]).collect();
    temperatures.sort_by(f64::total_cmp);

    // `Λ` is fixed by the coldest point, which is the one that needs the
    // largest basis.
    // ANCHOR: shared_sve
    let lambda_ir = WMAX / T_MIN;
    let (kernel, sve) = sve_for(1.0 / T_MIN, lambda_ir * T_MIN, EPS)?;
    // ANCHOR_END: shared_sve

    let mut sigma = read_noise("scan_noise")?;
    let mut delta: Option<Vec<Complex64>> = None;
    let mut sizes = Vec::new();
    let mut iterations = Vec::new();
    let mut energies = Vec::new();

    // ANCHOR: bases_per_beta
    for &temperature in &temperatures {
        let beta = 1.0 / temperature;
        let bases = Bases::from_sve(kernel, sve.clone(), beta, EPS)?;
        // ANCHOR_END: bases_per_beta
        let mut solver = Solver::new(&bases, settings, sigma.clone());
        if let Some(delta) = delta.clone() {
            solver.set_gap(delta);
        }
        let converged = solver.solve()?;
        assert!(
            converged,
            "the loop at T = {temperature} did not settle in {MAX_ITERATIONS} iterations"
        );
        sigma = solver.self_energy().to_vec();
        delta = Some(solver.gap().to_vec());
        sizes.push(bases.mesh_f().n_wn());
        iterations.push(solver.iterations());
        energies.push(solver.internal_energy()?);
    }

    assert!(
        sizes.windows(2).all(|pair| pair[0] == pair[1]),
        "the basis changed size along the sweep: {sizes:?}"
    );

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("d", vec![D]);
    table.push("u", vec![U]);
    table.push("j", vec![J]);
    table.push("omega0", vec![OMEGA0]);
    table.push("lambda0", vec![LAMBDA0]);
    table.push("g", vec![g]);
    table.push("mu", vec![MU]);
    table.push("eps", vec![EPS]);
    table.push("lambda_ir", vec![lambda_ir]);
    table.push("deg_leggauss", vec![DEG_LEGGAUSS as f64]);
    table.push("mixing", vec![MIXING]);
    table.push("max_iterations", vec![MAX_ITERATIONS as f64]);
    table.push("atol", vec![ATOL]);
    table.push("t_num", vec![T_NUM as f64]);
    table.push("dt", vec![DT]);
    table.push("basis_size_f", vec![sizes[0] as f64]);
    table.push("n_wn_f", vec![sizes[0] as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("temperature", temperatures.clone());
    table.push(
        "beta",
        temperatures.iter().map(|t| 1.0 / t).collect::<Vec<_>>(),
    );
    table.push(
        "iterations",
        iterations.iter().map(|&n| n as f64).collect::<Vec<_>>(),
    );
    table.push("energy", energies.clone());
    write_table(&output_path(EXAMPLE, "temperature")?, &table)?;

    // `C = (E(T + dt) − E(T)) / dt`; the sweep put the two ends of each
    // derivative next to one another.
    let specific_heat: Vec<f64> = (0..probes.len())
        .map(|i| (energies[2 * i + 1] - energies[2 * i]) / DT)
        .collect();
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("temperature", probes.clone());
    table.push("specific_heat", specific_heat.clone());
    table.push(
        "c_over_t",
        specific_heat
            .iter()
            .zip(&probes)
            .map(|(c, t)| c / t)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "specific_heat")?, &table)?;

    Ok(())
}

/// `T_NUM` points from `T_MIN` to `T_MAX` inclusive, as `numpy.linspace`
/// computes them.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    assert!(count > 1, "a derivative needs at least two temperatures");
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}

/// The committed starting self-energy of the coldest point; see the
/// single-temperature example for why it cannot be zero.
fn read_noise(name: &str) -> Result<Vec<Complex64>, Box<dyn StdError>> {
    let table = read_table(&input_path(INPUT_EXAMPLE, name))?;
    let re = table.expect_column("sigma_re");
    let im = table.expect_column("sigma_im");
    Ok(re
        .iter()
        .zip(im)
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect())
}
