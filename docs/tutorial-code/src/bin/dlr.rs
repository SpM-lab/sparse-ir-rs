//! The discrete Lehmann representation: the same Green's function as a sum of
//! poles.
//!
//! Adapted from the Python notebook `DLR_py.ipynb` of the sparse-ir tutorials
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/>).
//!
//! The DLR models the spectral function as `ρ(ω) = Σₚ cₚ δ(ω − ωₚ)`. Two
//! constructions are shown:
//!
//! * the default one, `DiscreteLehmannRepresentation::new(β, ωmax, ε)`, picks
//!   the poles by an interpolative decomposition of the logistic kernel (Kaye,
//!   Chen and Parcollet, PRB 105, 235115 (2022)) without any IR basis. The DLR
//!   then supplies its own imaginary-time and Matsubara nodes, and the
//!   coefficients are fitted from values there;
//! * `from_ir` takes the poles from an existing IR basis — the roots of the
//!   first basis function the truncation discarded — and carries the IR ↔ DLR
//!   transform, so `gₗ ↔ cₚ` is one call in each direction.
//!
//! Either way the result is an analytic form: once the `cₚ` are known,
//! `G(iν) = Σₚ cₚ wₚ/(iν − ωₚ)` on any frequency, where `wₚ` is
//! `pole_weights()` (1 for fermions, `tanh(βωₚ/2)` for bosons).

use std::error::Error;
use std::f64::consts::PI;

use num_complex::Complex64;
use sparse_ir::DlrFromIr;
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
    independent()?;

    // ANCHOR: basis
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    // ANCHOR_END: basis

    let (rho_l, g_l) = semicircle_coefficients(&basis);

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push("rho_l", rho_l);
    table.push("g_l", g_l.clone());
    write_table(&output_path(EXAMPLE, "coefficients")?, &table)?;

    // --- into the DLR -------------------------------------------------------
    // ANCHOR: from_ir
    // One pole per basis function, at the roots of the first discarded v_l.
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::from_ir(&basis)?;
    let poles = dlr.poles().to_vec();
    assert_eq!(
        poles.len(),
        basis.size(),
        "an IR-derived DLR has one pole per basis function"
    );
    // ANCHOR_END: from_ir

    // ANCHOR: from_ir_nd
    // g_l -> c_p: one axis of a tensor, here a rank-1 tensor.
    let g_l_tensor = TypedTensor::from_vec_col_major(vec![basis.size()], g_l.clone())?;
    let c_p = dlr.from_ir_nd::<f64>(None, &g_l_tensor, 0)?;
    // ANCHOR_END: from_ir_nd

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
    // ANCHOR: to_ir_nd
    // c_p -> g_l
    let g_l_reconstructed = dlr.to_ir_nd::<f64>(None, &c_p, 0)?;
    // ANCHOR_END: to_ir_nd
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

    // ANCHOR: pole_sum
    // The DLR needs no basis functions here: it is a sum over poles,
    // G(iν) = Σ_p c_p w_p / (iν − ω_p). The weight w_p is 1 for fermions and
    // tanh(βω_p/2) for bosons.
    let weights = dlr.pole_weights();
    let g_iv_dlr: Vec<Complex64> = nu
        .iter()
        .map(|&nu| {
            (0..poles.len())
                .map(|p| {
                    *c_p.get(&[p]).unwrap() * weights[p] / (Complex64::new(0.0, nu) - poles[p])
                })
                .sum()
        })
        .collect();
    // ANCHOR_END: pole_sum

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

/// The default construction: a DLR built without any IR basis, sampled at its
/// own nodes and fitted there.
///
/// The model is the semicircle `ρ(ω) = (2/π)√(1 − ω²)` again, whose Green's
/// function is known in closed form, `G(iν) = 2i(ν − sgn(ν)√(ν² + 1))`
/// `= −2i sgn(ν)/(|ν| + √(ν² + 1))`.
fn independent() -> Result<(), Box<dyn Error>> {
    // ANCHOR: independent_build
    use num_complex::Complex64;
    use sparse_ir::{
        Basis, DiscreteLehmannRepresentation, Fermionic, FermionicFreq, MatsubaraSampling,
        TauSampling,
    };

    let (beta, wmax, eps) = (1e4, 1.0, 1e-14);
    // The poles come from an interpolative decomposition of the logistic
    // kernel; eps sets how many are kept.
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, eps)?;
    // ANCHOR_END: independent_build
    assert_eq!((beta, wmax), (BETA, WMAX));

    // ANCHOR: independent_fit
    // The exact G(iν) of the semicircle at ν = nπ/β, written so that it does
    // not cancel at large |ν|.
    let g_exact = |nu: f64| Complex64::new(0.0, -2.0 * nu.signum() / (nu.abs() + nu.hypot(1.0)));

    // MatsubaraSampling::new samples at the DLR's own nodes,
    // dlr.matsubara_nodes(false): one frequency per pole.
    let sampling = MatsubaraSampling::<Fermionic>::new(&dlr)?;
    let g_iv: Vec<Complex64> = sampling
        .sampling_points()
        .iter()
        .map(|freq| g_exact(freq.value(beta)))
        .collect();
    // ρ is real, so ask for real coefficients.
    let c_p = sampling.fit_real(&g_iv)?;
    // ANCHOR_END: independent_fit

    // ANCHOR: independent_eval
    // Evaluate the fitted DLR anywhere: here odd n from 1 to about 10⁷.
    let mut frequencies: Vec<i64> = (0..=140)
        .map(|k| (10f64.powf(k as f64 / 20.0).round() as i64) | 1)
        .collect();
    frequencies.dedup();
    let freqs: Vec<FermionicFreq> = frequencies
        .iter()
        .map(|&n| FermionicFreq::new(n))
        .collect::<Result<_, _>>()?;
    let dense = MatsubaraSampling::<Fermionic>::with_sampling_points(&dlr, freqs.clone())?;
    let g_iv_dlr = dense.evaluate_real(&c_p)?;
    let worst = freqs
        .iter()
        .zip(&g_iv_dlr)
        .map(|(freq, g)| (g - g_exact(freq.value(beta))).norm())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "max |ΔG(iν)| = {worst:e}");
    // ANCHOR_END: independent_eval

    // ANCHOR: independent_tau
    // TauSampling::new samples at dlr.tau_nodes(). Fitting from G(τ) there
    // gives a DLR that is just as good on the Matsubara axis.
    let tau_sampling = TauSampling::<Fermionic>::new(&dlr)?;
    let g_tau = tau_sampling.evaluate(&c_p)?;
    let c_p_from_tau = tau_sampling.fit(&g_tau)?;
    let g_iv_from_tau = dense.evaluate_real(&c_p_from_tau)?;
    let worst_from_tau = freqs
        .iter()
        .zip(&g_iv_from_tau)
        .map(|(freq, g)| (g - g_exact(freq.value(beta))).norm())
        .fold(0.0, f64::max);
    assert!(worst_from_tau < 1e-11, "max |ΔG(iν)| = {worst_from_tau:e}");

    // G(0⁺) + G(β⁻) = −∫ρ(ω)dω = −1, a check that does not involve the fit
    // frequencies at all.
    let ends = dlr.evaluate_tau(&[0.0, beta])?;
    let sum_rule: f64 = (0..dlr.size())
        .map(|p| (ends.get(&[0, p]).unwrap() + ends.get(&[1, p]).unwrap()) * c_p[p])
        .sum();
    assert!((sum_rule + 1.0).abs() < 1e-11, "G(0) + G(β) = {sum_rule}");
    // ANCHOR_END: independent_tau

    // For comparison: at eps = 1e-15 the ID keeps far more poles than at
    // 1e-14 (192 against 98 when this was written), with no gain in accuracy.
    let n_poles_eps15 = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-15)?.size();

    let nu: Vec<f64> = freqs.iter().map(|f| f.value(beta)).collect();
    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "n",
        frequencies.iter().map(|&n| n as f64).collect::<Vec<_>>(),
    );
    table.push("nu", nu.clone());
    table.push(
        "g_iv_exact_im",
        nu.iter().map(|&nu| g_exact(nu).im).collect::<Vec<_>>(),
    );
    table.push(
        "g_iv_dlr_im",
        g_iv_dlr.iter().map(|z| z.im).collect::<Vec<_>>(),
    );
    table.push(
        "error",
        g_iv_dlr
            .iter()
            .zip(&nu)
            .map(|(g, &nu)| (g - g_exact(nu)).norm())
            .collect::<Vec<_>>(),
    );
    table.push(
        "error_from_tau",
        g_iv_from_tau
            .iter()
            .zip(&nu)
            .map(|(g, &nu)| (g - g_exact(nu)).norm())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "independent_matsubara")?, &table)?;

    let nodes: Vec<f64> = sampling
        .sampling_points()
        .iter()
        .map(|f| f.n() as f64)
        .collect();
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n", nodes.clone());
    table.push(
        "nu",
        nodes.iter().map(|&n| n * PI / beta).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "independent_nodes")?, &table)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![beta]);
    table.push("wmax", vec![wmax]);
    table.push("eps", vec![eps]);
    table.push("n_poles", vec![dlr.size() as f64]);
    table.push(
        "n_matsubara_nodes",
        vec![sampling.sampling_points().len() as f64],
    );
    table.push(
        "n_tau_nodes",
        vec![tau_sampling.sampling_points().len() as f64],
    );
    table.push("max_error_matsubara", vec![worst]);
    table.push("max_error_from_tau", vec![worst_from_tau]);
    table.push("sum_rule", vec![sum_rule]);
    table.push("n_poles_eps_1e-15", vec![n_poles_eps15 as f64]);
    write_table(&output_path(EXAMPLE, "independent_summary")?, &table)?;

    Ok(())
}
