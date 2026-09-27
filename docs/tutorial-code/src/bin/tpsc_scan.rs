//! The interaction dependence of the TPSC vertices, at half filling.
//!
//! Ported from the last section of `TPSC_py.ipynb` of sparse-ir-tutorial
//! (author: Niklas Witt), which reproduces Fig. 2 of Vilk and Tremblay (1997).
//!
//! The same solver as `tpsc`, run at fifty-one interaction strengths. The
//! temperature is high enough that the spin sum rule has a solution below
//! `U_crit` for every one of them, so the scan never runs into the ordered
//! phase.

use std::error::Error as StdError;

use sparse_ir_tutorial::tpsc::{Lattice, high_symmetry_path, solve};
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "tpsc_scan";

const T_HOP: f64 = 1.0;
const TEMPERATURE: f64 = 0.4;
const BETA: f64 = 1.0 / TEMPERATURE;
const WMAX: f64 = 10.0;
/// Half filling, where the spin fluctuations are strongest.
const FILLING: f64 = 1.0;
const NK_LIN: usize = 24;
const EPS: f64 = 1e-8;

const U_MIN: f64 = 1e-2;
const U_MAX: f64 = 5.0;
const U_NUM: usize = 51;
/// The three interaction strengths whose spin susceptibility is written out
/// along the high-symmetry path as well: the weakest, the middle one and the
/// strongest of the grid.
const PROBES: [usize; 3] = [0, 25, 50];

fn main() -> Result<(), Box<dyn StdError>> {
    let lattice = Lattice::new(NK_LIN, NK_LIN, T_HOP, BETA, WMAX, EPS)?;
    let nk = lattice.grid().len();
    let row_b = lattice.iw0_b() * nk;
    let path = high_symmetry_path(NK_LIN);

    // `np.linspace(U_MIN, U_MAX, U_NUM)`, written the way NumPy computes it so
    // that the two grids are the same doubles.
    let step = (U_MAX - U_MIN) / (U_NUM - 1) as f64;
    let grid: Vec<f64> = (0..U_NUM).map(|i| i as f64 * step + U_MIN).collect();

    let mut u_sp = Vec::with_capacity(U_NUM);
    let mut u_ch = Vec::with_capacity(U_NUM);
    let mut u_crit = Vec::with_capacity(U_NUM);
    let mut docc = Vec::with_capacity(U_NUM);
    let mut probes: Vec<(usize, Vec<f64>)> = Vec::new();
    for (index, &u) in grid.iter().enumerate() {
        let solution = solve(&lattice, u, FILLING)?;
        u_sp.push(solution.u_sp);
        u_ch.push(solution.u_ch);
        u_crit.push(solution.u_crit);
        docc.push(solution.docc);
        if PROBES.contains(&index) {
            probes.push((
                index,
                path.iter()
                    .map(|&k| solution.chi_spin[row_b + k].re)
                    .collect(),
            ));
        }
    }

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T_HOP]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("n", vec![FILLING]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("eps", vec![EPS]);
    table.push("u_num", vec![U_NUM as f64]);
    table.push("u_min", vec![grid[0]]);
    table.push("u_max", vec![grid[U_NUM - 1]]);
    table.push("basis_size_f", vec![lattice.basis_f().size() as f64]);
    for index in PROBES {
        table.push(format!("probe_u_{index}"), vec![grid[index]]);
    }
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("u", grid);
    table.push("u_sp", u_sp);
    table.push("u_ch", u_ch);
    table.push("u_crit", u_crit);
    table.push("docc", docc);
    write_table(&output_path(EXAMPLE, "vertices")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "distance",
        (0..path.len()).map(|i| i as f64).collect::<Vec<_>>(),
    );
    for (index, values) in probes {
        // Named by grid index rather than by `U`, so that the column names
        // cannot depend on how a half-way value is rounded.
        table.push(format!("u_{index}"), values);
    }
    write_table(&output_path(EXAMPLE, "chi_spin")?, &table)?;

    Ok(())
}
