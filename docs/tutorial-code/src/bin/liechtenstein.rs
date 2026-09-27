//! Exchange interactions by the Liechtenstein method.
//!
//! Ported from the Python notebook `liechtenstein_py.ipynb` of
//! sparse-ir-tutorial, whose author is Takuya Nomoto.
//!
//! The Liechtenstein formula reads the parameters of a classical Heisenberg
//! model off an itinerant one by comparing second derivatives of the total
//! energy with respect to spin angles. For a single orbital in a field `B ẑ`
//! that leaves
//!
//! ```text
//! J_ij = −B² T Σ_ν G_ij,+(iν) G_ji,−(iν)
//! J_0  = (B/2)(n_+ − n_−) + B² T Σ_ν G_00,+(iν) G_00,−(iν)
//! ```
//!
//! — in both cases a Matsubara sum of a *product* of two Green's functions.
//! That is the whole point of the example. Summed naively the series
//! converges like `1/N_M`, because the product falls off as `1/ν²`; fitted to
//! the basis it is one evaluation at `τ = 0`, since
//! `T Σ_ν F(iν) = F(τ = 0)`, and `F` needs 37 coefficients no matter how cold
//! the system is.

use std::error::Error as StdError;
use std::f64::consts::PI;

use num_complex::Complex64;
use sparse_ir::{Basis, Fermionic, FiniteTempBasis, LogisticKernel};
use sparse_ir_tutorial::{
    IrMesh, MomentumGrid, Table, evaluate_rows, output_path, provenance, write_table,
};

const EXAMPLE: &str = "liechtenstein";

const T_HOPPING: f64 = 1.0;
const BETA: f64 = 50.0;
const NK_LIN: usize = 36;
/// The effective field that polarizes the reference state.
const B_EFF: f64 = 3.0;
const EPS: f64 = 1e-7;
/// `2 max(W, B) β` with the bandwidth `W = 8t`: the basis has to hold the
/// whole of both spin-split bands.
const LAMBDA: f64 = 2.0 * 8.0 * T_HOPPING * BETA;
/// How many chemical potentials `J₀` is scanned over, from −10 to 10.
const N_MU: usize = 41;
/// The truncated Matsubara grids `J₀` is also evaluated on, for comparison.
const NAIVE: [usize; 5] = [100, 200, 400, 800, 1600];

fn chemical_potentials() -> Vec<f64> {
    (0..N_MU)
        .map(|i| -10.0 + 20.0 * i as f64 / (N_MU - 1) as f64)
        .collect()
}

/// The first term of `J₀`: the difference in occupation between the two spin
/// species, which is a Fermi function and has nothing to do with the
/// frequency grid.
fn occupation_term(ek: &[f64], mu: &[f64]) -> Vec<f64> {
    mu.iter()
        .map(|&mu| {
            let occupation = |shift: f64| {
                ek.iter()
                    .map(|&e| 0.5 * (1.0 - (0.5 * BETA * (e - mu - shift)).tanh()))
                    .sum::<f64>()
                    / ek.len() as f64
            };
            0.5 * B_EFF * (occupation(B_EFF) - occupation(-B_EFF))
        })
        .collect()
}

/// `B² ⟨G₊⟩_k ⟨G₋⟩_k` at one frequency, for every chemical potential.
fn product_at(nu: f64, ek: &[f64], mu: &[f64]) -> Vec<Complex64> {
    mu.iter()
        .map(|&mu| {
            let average = |shift: f64| {
                ek.iter()
                    .map(|&e| Complex64::new(-(e - shift - mu), nu).inv())
                    .sum::<Complex64>()
                    / ek.len() as f64
            };
            B_EFF * B_EFF * average(B_EFF) * average(-B_EFF)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn StdError>> {
    let wmax = LAMBDA / BETA;
    let kernel = LogisticKernel::new(LAMBDA)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    let mesh = IrMesh::<Fermionic>::new(&basis)?;
    let grid = MomentumGrid::new(NK_LIN, NK_LIN);
    let ek = grid.square_lattice_dispersion(T_HOPPING);
    let mu = chemical_potentials();

    // `T Σ_ν F(iν) = F(τ = 0)`: the Matsubara sum is the basis expansion read
    // at one point. This is the row of basis functions that reads it.
    let u_at_zero = basis.evaluate_tau(&[0.0])?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T_HOPPING]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![wmax]);
    table.push("eps", vec![EPS]);
    table.push("lambda", vec![LAMBDA]);
    table.push("b_eff", vec![B_EFF]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("n_matsubara", vec![mesh.n_wn() as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    // --- J₀ through the basis ------------------------------------------------
    let occupation = occupation_term(&ek, &mu);
    let mut sampled = Vec::with_capacity(mesh.n_wn() * N_MU);
    for freq in mesh.wn() {
        sampled.extend(product_at(freq.value(BETA), &ek, &mu));
    }
    let coefficients = mesh.wn_to_l(&sampled, N_MU)?;
    let summed = evaluate_rows(&u_at_zero, &coefficients, N_MU);
    let j0: Vec<f64> = occupation
        .iter()
        .zip(&summed)
        .map(|(n, s)| n + s.re)
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("mu", mu.clone());
    table.push("j0", j0.clone());

    // --- and J₀ by summing the series ---------------------------------------
    // Every grid is a prefix of the largest, so one pass accumulates all five.
    let largest = *NAIVE.iter().max().expect("NAIVE is not empty");
    let mut running = vec![Complex64::default(); N_MU];
    let mut naive: Vec<(usize, Vec<f64>)> = Vec::with_capacity(NAIVE.len());
    for step in 0..largest {
        // The grid is symmetric: `n = −step − 1` and `n = step` at once.
        for n in [-(step as i64) - 1, step as i64] {
            let nu = (2 * n + 1) as f64 * PI / BETA;
            for (slot, value) in running.iter_mut().zip(product_at(nu, &ek, &mu)) {
                *slot += value;
            }
        }
        if let Some(&nm) = NAIVE.iter().find(|&&nm| nm == step + 1) {
            naive.push((
                nm,
                occupation
                    .iter()
                    .zip(&running)
                    .map(|(n, s)| n + s.re / BETA)
                    .collect(),
            ));
        }
    }
    for (nm, values) in &naive {
        table.push(format!("j0_nm{nm}"), values.clone());
    }
    write_table(&output_path(EXAMPLE, "j0")?, &table)?;

    // --- J_ij at half filling ------------------------------------------------
    // `G_ij` carries `e^{−ik·r}` and `G_ji` carries `e^{+ik·r}`, both averaged
    // over the zone. The dispersion is even in `k`, so the second transform is
    // the first one applied to the same array — there is no separate direction
    // to get wrong here.
    assert!(
        (0..grid.len()).all(|index| {
            let (k1, k2) = (index / NK_LIN, index % NK_LIN);
            let mirrored = ((NK_LIN - k1) % NK_LIN) * NK_LIN + (NK_LIN - k2) % NK_LIN;
            (ek[index] - ek[mirrored]).abs() < 1e-12
        }),
        "the dispersion must be even in k for G_ji to be the transform of G_ij"
    );
    let nk = grid.len();
    let mut jij_iw = Vec::with_capacity(mesh.n_wn() * nk);
    for freq in mesh.wn() {
        let nu = freq.value(BETA);
        let green = |shift: f64| -> Vec<Complex64> {
            grid.k_to_r(
                &ek.iter()
                    .map(|&e| Complex64::new(-(e - shift), nu).inv())
                    .collect::<Vec<_>>(),
            )
        };
        let (up, down) = (green(B_EFF), green(-B_EFF));
        jij_iw.extend(
            up.iter()
                .zip(&down)
                .map(|(u, d)| -B_EFF * B_EFF * u * d)
                .collect::<Vec<_>>(),
        );
    }
    let coefficients = mesh.wn_to_l(&jij_iw, nk)?;
    let mut jij: Vec<f64> = evaluate_rows(&u_at_zero, &coefficients, nk)
        .iter()
        .map(|z| z.re)
        .collect();
    // The on-site term is not an exchange interaction; the notebook drops it.
    jij[0] = 0.0;

    let distance: Vec<f64> = (0..nk)
        .map(|index| {
            let (r1, r2) = (index / NK_LIN, index % NK_LIN);
            ((r1 * r1 + r2 * r2) as f64).sqrt()
        })
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("distance", distance);
    table.push("j_ij", jij.clone());
    write_table(&output_path(EXAMPLE, "jij")?, &table)?;

    // --- and the sum rule the two obey --------------------------------------
    // `J₀ = Σ_j J_0j` is a statement about the calculation, not about the
    // method: the two sides take different routes to the same number.
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("mu", vec![0.0]);
    table.push("j0_direct", vec![j0[N_MU / 2]]);
    table.push("j0_from_jij", vec![jij.iter().sum::<f64>()]);
    write_table(&output_path(EXAMPLE, "sum_rule")?, &table)?;

    Ok(())
}
