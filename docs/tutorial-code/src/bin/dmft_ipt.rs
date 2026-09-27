//! DMFT on the Bethe lattice, solved by iterated perturbation theory.
//!
//! Ported from the Python notebook `DMFT_IPT_py.ipynb` of sparse-ir-tutorial.
//!
//! The impurity solver is one line — `Σ(τ) = U² 𝒢(τ)³` — because the
//! self-energy is a *product* in imaginary time and the Dyson equation is a
//! product in Matsubara frequency. Neither is a product in the other
//! representation, so every iteration passes through both, and the basis is
//! what makes the trip cheap: 37 coefficients carry a function that would
//! need thousands of frequencies.
//!
//! The convergence criterion is the notebook's: stop when the relative change
//! of `Σ` from one iteration to the next falls below `1e-5`. That happens
//! after 65 iterations and leaves `Z = 0.26`, a metal. The example then runs
//! the same loop 5000 times without a criterion, and the relative change
//! climbs back to order one some forty iterations later before settling on an
//! insulator with `Z = 0`. Exactly when it climbs is set by rounding — the
//! trajectory is passing an unstable fixed point, and how it leaves depends on
//! the last bits of where it is. A threshold on the *change* says nothing
//! about the distance
//! still to go, which is the lesson worth taking out of any self-consistent
//! loop.

use std::error::Error as StdError;

use num_complex::Complex64;
use sparse_ir_tutorial::dmft::{Dmft, MIX, Solution};
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "dmft_ipt";

/// Half-bandwidth of the Bethe lattice; the hopping is `t = D/2 = 1`.
const D: f64 = 2.0;
const TEMPERATURE: f64 = 0.1 / D;
const BETA: f64 = 1.0 / TEMPERATURE;
const U: f64 = 5.0;
const EPS: f64 = 1e-15;
const MAXITER: usize = 300;
const SFC_TOL: f64 = 1e-5;
/// The same calculation carried on for a fixed number of iterations instead of
/// stopped at a threshold, which is what shows the threshold up.
const LONG_ITERATIONS: usize = 5000;

fn main() -> Result<(), Box<dyn StdError>> {
    let dmft = Dmft::new(BETA, D, EPS)?;
    let g0 = dmft.noninteracting()?;
    let solution = dmft.solve(&g0, U, MAXITER, SFC_TOL)?;
    // A tolerance of zero is never met, so this one runs the full count.
    let long = dmft.solve(&g0, U, LONG_ITERATIONS, 0.0)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("d", vec![D]);
    table.push("wmax", vec![2.0 * D]);
    table.push("beta", vec![BETA]);
    table.push("u", vec![U]);
    table.push("eps", vec![EPS]);
    table.push("mix", vec![MIX]);
    table.push("sfc_tol", vec![SFC_TOL]);
    table.push("maxiter", vec![MAXITER as f64]);
    table.push("basis_size", vec![dmft.basis().size() as f64]);
    table.push("n_tau", vec![dmft.mesh().n_tau() as f64]);
    table.push("n_wn", vec![dmft.mesh().n_wn() as f64]);
    table.push("iterations", vec![solution.residuals.len() as f64]);
    table.push("z", vec![dmft.renormalisation(&solution.self_energy)]);
    table.push("long_iterations", vec![long.residuals.len() as f64]);
    table.push("long_z", vec![dmft.renormalisation(&long.self_energy)]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let n = dmft.frequency_column();
    write_frequencies("green", &n, "g", &solution.green)?;
    write_frequencies("self_energy", &n, "sigma", &solution.self_energy)?;
    write_residuals("convergence", &solution)?;
    write_residuals("long_convergence", &long)?;

    // The self-energy back in imaginary time, on the `[−β/2, β/2]` grid this
    // implementation reports. `Σ` is fermionic, so the Python reference has to
    // fold its own `[0, β)` grid with a sign change to land on these numbers.
    let sigma_l = dmft.mesh().wn_to_l(&solution.self_energy, 1)?;
    let sigma_tau = dmft.mesh().l_to_tau(&sigma_l, 1)?;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", dmft.mesh().tau_points().to_vec());
    table.push(
        "sigma_re",
        sigma_tau.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    table.push(
        "sigma_im",
        sigma_tau.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "self_energy_tau")?, &table)?;

    Ok(())
}

fn write_frequencies(
    name: &str,
    n: &[f64],
    quantity: &str,
    values: &[Complex64],
) -> Result<(), Box<dyn StdError>> {
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n", n.to_vec());
    table.push(
        format!("{quantity}_re"),
        values.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    table.push(
        format!("{quantity}_im"),
        values.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, name)?, &table)?;
    Ok(())
}

fn write_residuals(name: &str, solution: &Solution) -> Result<(), Box<dyn StdError>> {
    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "iteration",
        (1..=solution.residuals.len())
            .map(|i| i as f64)
            .collect::<Vec<_>>(),
    );
    table.push("residual", solution.residuals.clone());
    write_table(&output_path(EXAMPLE, name)?, &table)?;
    Ok(())
}
