//! The Mott transition of the `dmft_ipt` model, and its hysteresis.
//!
//! Ported from the Python notebook `DMFT_IPT_py.ipynb` of sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/src/DMFT_IPT_py.html>), whose author is
//! Niklas Witt.
//!
//! The same loop as `dmft_ipt`, run for 66 interaction strengths from three
//! different starting points: always from the non-interacting Green's
//! function, walking up in `U` from the metal, and walking down from the
//! insulator. The transition is first order, so the three do not agree —
//! between `U_c1` and `U_c2` both solutions exist and the loop keeps whichever
//! one it was started in.
//!
//! Each run is a fixed number of iterations rather than a threshold on the
//! change of `Σ`, so that every point is a fixed point to machine precision
//! and not a snapshot on the way to one. The loop enforces particle-hole
//! symmetry (see `dmft::Symmetry`); without it, rounding drives every run off
//! the symmetric solution and the scan maps that instability instead of the
//! Mott transition.
//!
//! This is the expensive half of the example, so it lives in its own binary
//! and runs only in the applied-examples CI job.

use std::error::Error as StdError;

use num_complex::Complex64;
use sparse_ir_tutorial::dmft::Dmft;
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "dmft_ipt_scan";

const D: f64 = 2.0;
const BETA: f64 = D / 0.1;
const EPS: f64 = 1e-15;
const U_MAX: f64 = 6.5;
const U_NUM: usize = 66;
const ITERATIONS: usize = 5000;
/// The interaction strengths whose self-energy is written out, as indices into
/// the scan: `U = 5.0, 5.4` below the coexistence window, `5.7` the last metal
/// reached from `G⁰`, and `5.8, 6.0` above `U_c2`.
const PROBES: [usize; 5] = [50, 54, 57, 58, 60];

fn main() -> Result<(), Box<dyn StdError>> {
    let dmft = Dmft::new(BETA, D, EPS)?;
    let g0 = dmft.noninteracting()?;
    let u_arr: Vec<f64> = (0..U_NUM)
        .map(|i| i as f64 * (U_MAX / (U_NUM - 1) as f64))
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("d", vec![D]);
    table.push("beta", vec![BETA]);
    table.push("eps", vec![EPS]);
    table.push("u_max", vec![U_MAX]);
    table.push("u_num", vec![U_NUM as f64]);
    table.push("iterations", vec![ITERATIONS as f64]);
    table.push("basis_size", vec![dmft.basis().size() as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // Every calculation from the non-interacting Green's function.
    let mut from_g0 = Vec::with_capacity(U_NUM);
    let mut probes: Vec<(usize, Vec<Complex64>)> = Vec::with_capacity(PROBES.len());
    for (index, &u) in u_arr.iter().enumerate() {
        // A tolerance of zero is never met, so every run is `ITERATIONS` long.
        let solution = dmft.solve(&g0, u, ITERATIONS, 0.0)?;
        from_g0.push(dmft.renormalisation(&solution.self_energy));
        if PROBES.contains(&index) {
            probes.push((index, solution.self_energy));
        }
    }

    // And the hysteresis: walking up in `U` from the metal and down from the
    // insulator, each calculation starting where the previous one stopped.
    let mut metal = vec![0.0; U_NUM];
    let mut insulator = vec![0.0; U_NUM];
    let (mut g_metal, mut g_insulator) = (g0.clone(), g0.clone());
    for index in 0..U_NUM {
        let solution = dmft.solve(&g_metal, u_arr[index], ITERATIONS, 0.0)?;
        metal[index] = dmft.renormalisation(&solution.self_energy);
        g_metal = solution.green;

        let other = U_NUM - 1 - index;
        let solution = dmft.solve(&g_insulator, u_arr[other], ITERATIONS, 0.0)?;
        insulator[other] = dmft.renormalisation(&solution.self_energy);
        g_insulator = solution.green;
    }

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("u", u_arr);
    table.push("from_g0", from_g0);
    table.push("metal", metal);
    table.push("insulator", insulator);
    write_table(&output_path(EXAMPLE, "renormalisation")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n", dmft.frequency_column());
    for (index, sigma) in &probes {
        table.push(
            format!("sigma_im_u{index}"),
            sigma.iter().map(|z| z.im).collect::<Vec<_>>(),
        );
    }
    write_table(&output_path(EXAMPLE, "self_energy")?, &table)?;

    Ok(())
}
