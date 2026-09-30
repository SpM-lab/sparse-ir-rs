//! The discrete Lehmann representation: the same Green's function as a sum of
//! poles.
//!
//! Ported from the Python notebook `DLR_py.ipynb` of sparse-ir-tutorial.
//!
//! The DLR models the spectral function as `ρ(ω) = Σₚ cₚ δ(ω − ω̄ₚ)`, with the
//! poles taken to be the roots of the first basis function the truncation
//! discarded. That choice is heuristic, but it makes the matrix
//! `Vₗₚ = vₗ(ω̄ₚ)` well conditioned, so `ρₗ ↔ cₚ` is a stable transform in
//! both directions. What you get for it is an analytic form: once the `cₚ`
//! are known, `G(iν) = Σₚ cₚ/(iν − ω̄ₚ)` on any frequency, with no basis
//! functions to evaluate.

use std::error::Error;
use std::f64::consts::PI;

use num_complex::Complex64;
use sparse_ir::TypedTensor;
use sparse_ir::{
    DiscreteLehmannRepresentation, Fermionic, FermionicFreq, FiniteTempBasis, LogisticKernel,
    MatsubaraSampling,
};
use sparse_ir_tutorial::{Table, output_path, provenance, semicircle_coefficients, write_table};

const EXAMPLE: &str = "dlr";

const WMAX: f64 = 1.0;
const LAMBDA: f64 = 1e4;
const BETA: f64 = LAMBDA / WMAX;
const EPS: f64 = 1e-15;

fn main() -> Result<(), Box<dyn Error>> {
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;

    let (rho_l, g_l) = semicircle_coefficients(&basis);

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push("rho_l", rho_l);
    table.push("g_l", g_l.clone());
    write_table(&output_path(EXAMPLE, "coefficients")?, &table)?;

    // --- into the DLR -------------------------------------------------------
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis)?;
    let poles = dlr.poles().to_vec();
    assert_eq!(
        poles.len(),
        basis.size(),
        "the default DLR has one pole per basis function"
    );

    let g_l_tensor = TypedTensor::from_vec_col_major(vec![basis.size()], g_l.clone())?;
    let c_p = dlr.from_ir_nd::<f64>(None, &g_l_tensor, 0)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("p", (0..poles.len()).map(|p| p as f64).collect::<Vec<_>>());
    table.push("pole", poles.clone());
    table.push(
        "c_p",
        (0..poles.len())
            .map(|p| *c_p.get(&[p]).unwrap())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "dlr_coefficients")?, &table)?;

    // --- and back out of it -------------------------------------------------
    let g_l_reconstructed = dlr.to_ir_nd::<f64>(None, &c_p, 0)?;
    let g_l_from_dlr: Vec<f64> = (0..basis.size())
        .map(|l| *g_l_reconstructed.get(&[l]).unwrap())
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("g_l", g_l.clone());
    table.push("g_l_from_dlr", g_l_from_dlr.clone());
    table.push(
        "error",
        g_l_from_dlr
            .iter()
            .zip(&g_l)
            .map(|(a, b)| (a - b).abs())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "reconstruction")?, &table)?;

    // --- on the Matsubara axis ----------------------------------------------
    // Every tenth fermionic frequency out to |n| = 2000, far beyond the
    // sampling frequencies the basis picked for itself.
    let frequencies: Vec<i64> = (-1000..1000).step_by(10).map(|k| 2 * k + 1).collect();
    let nu: Vec<f64> = frequencies.iter().map(|&n| n as f64 * PI / BETA).collect();

    // The DLR needs no basis functions here: it is a sum over poles.
    let g_iv_dlr: Vec<Complex64> = nu
        .iter()
        .map(|&nu| {
            (0..poles.len())
                .map(|p| *c_p.get(&[p]).unwrap() / (Complex64::new(0.0, nu) - poles[p]))
                .sum()
        })
        .collect();

    let freqs: Vec<FermionicFreq> = frequencies
        .iter()
        .map(|&n| FermionicFreq::new(n))
        .collect::<Result<_, _>>()?;
    let sampling = MatsubaraSampling::<Fermionic>::with_sampling_points(&basis, freqs)?;
    let g_l_complex: Vec<Complex64> = g_l.iter().map(|&g| Complex64::new(g, 0.0)).collect();
    let g_iv_exact = sampling.evaluate(&g_l_complex)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        frequencies.iter().map(|&n| n as f64).collect::<Vec<_>>(),
    );
    table.push("nu", nu);
    table.push(
        "g_iv_exact_im",
        g_iv_exact.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    table.push(
        "g_iv_dlr_im",
        g_iv_dlr.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    table.push(
        "g_iv_exact_re",
        g_iv_exact.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    table.push(
        "g_iv_dlr_re",
        g_iv_dlr.iter().map(|z| z.re).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "matsubara")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("lambda", vec![LAMBDA]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("n_poles", vec![poles.len() as f64]);
    table.push("accuracy", vec![basis.accuracy()]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    Ok(())
}
