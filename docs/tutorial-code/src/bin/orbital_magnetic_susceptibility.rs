//! Orbital magnetic susceptibility of a tight-binding model.
//!
//! Ported from the Python notebook `orbital_magnetic_susceptibility_py.ipynb`
//! of sparse-ir-tutorial, whose authors are Soshun Ozaki and Takashi
//! Koretsune.
//!
//! For a tight-binding model the orbital susceptibility is the Matsubara sum
//!
//! ```text
//! χ = T Σ_ν χ(iν)
//! χ(iν) = Σ_k Tr[γx G γy G γx G γy G
//!                + ½ (γx G γy G + γy G γx G) γxy G]
//! ```
//!
//! with `γi = ∂H/∂kᵢ` and `γxy = ∂²H/∂kx∂ky`. The summand falls off as
//! `1/ν³`, so summing it directly converges slowly; through the basis the sum
//! is one evaluation at `τ = 0`, because `T Σ_ν F(iν) = F(τ = 0)`.
//!
//! Two lattices are computed. The square lattice has one band, so every
//! matrix above is a number and the trace is a product; graphene has two
//! sites per cell, so `H` has to be diagonalised at every momentum and the
//! velocity matrices rotated into its eigenbasis, where `G` is diagonal. The
//! square lattice has a closed form at `T = 0` in complete elliptic
//! integrals, which the finite-temperature curve is compared against.

use std::error::Error as StdError;
use std::f64::consts::{PI, TAU};

use num_complex::Complex64;
use sparse_ir::{Basis, Fermionic, FiniteTempBasis, LogisticKernel};
use sparse_ir_tutorial::elliptic::{ellipe, ellipk};
use sparse_ir_tutorial::linalg::Hermitian2;
use sparse_ir_tutorial::{
    IrMesh, MomentumGrid, Table, evaluate_rows, output_path, provenance, write_table,
};

const EXAMPLE: &str = "orbital_magnetic_susceptibility";

const T_HOPPING: f64 = 1.0;
const LATTICE: f64 = 1.0;
const TEMPERATURE: f64 = 0.1;
const BETA: f64 = 1.0 / TEMPERATURE;
const WMAX: f64 = 10.0;
const EPS: f64 = 1e-10;
const NK_LIN: usize = 200;
/// The chemical potentials `χ` is scanned over, from −4.5 to 4.5.
const N_MU: usize = 91;
const MU_MIN: f64 = -4.5;
const MU_MAX: f64 = 4.5;
/// The chemical potential the sampled `χ(iν)` is written out at. Not μ = 0:
/// both lattices are particle-hole symmetric there and `χ(iν)` comes out
/// real, which would hide a mistake in its imaginary part.
const PROBE_MU: f64 = -1.0;

fn chemical_potentials() -> Vec<f64> {
    let step = (MU_MAX - MU_MIN) / (N_MU - 1) as f64;
    (0..N_MU).map(|i| i as f64 * step + MU_MIN).collect()
}

/// A 2×2 matrix with `upper` above the diagonal and nothing on it, which is
/// the shape of every operator of the two-site cell below.
fn offdiagonal(upper: Complex64) -> [[Complex64; 2]; 2] {
    [
        [Complex64::default(), upper],
        [upper.conj(), Complex64::default()],
    ]
}

/// `ε(k)`, `γx`, `γy`, `γxy` of the nearest-neighbour square lattice.
fn square(k1: f64, k2: f64) -> (f64, f64, f64, f64) {
    let (kx, ky) = (TAU * k1, TAU * k2);
    (
        -2.0 * T_HOPPING * (kx.cos() + ky.cos()),
        2.0 * T_HOPPING * LATTICE * kx.sin(),
        2.0 * T_HOPPING * LATTICE * ky.sin(),
        // ∂²ε/∂kx∂ky vanishes for a dispersion that separates.
        0.0,
    )
}

/// `H(k)`, `γx`, `γy`, `γxy` of graphene, each a 2×2 matrix over the two
/// sites of the cell.
#[allow(clippy::type_complexity)]
fn graphene(k1: f64, k2: f64) -> (Hermitian2, [[[Complex64; 2]; 2]; 3]) {
    let sqrt3 = 3.0_f64.sqrt();
    let kx = TAU * k1 / LATTICE;
    let ky = TAU * (k1 + 2.0 * k2) / (LATTICE * sqrt3);

    let phase = Complex64::new(0.0, -ky * LATTICE / (2.0 * sqrt3)).exp();
    let h = -T_HOPPING
        * (Complex64::new(0.0, ky * LATTICE / sqrt3).exp()
            + 2.0 * (kx * LATTICE / 2.0).cos() * phase);
    let hx = T_HOPPING * LATTICE * (kx / 2.0).sin() * phase;
    let hy = -T_HOPPING
        * LATTICE
        * (Complex64::new(0.0, 1.0 / sqrt3)
            * (Complex64::new(0.0, ky / sqrt3).exp() - (kx / 2.0).cos() * phase));
    let hxy = -T_HOPPING
        * LATTICE
        * LATTICE
        * Complex64::new(0.0, 1.0 / (2.0 * sqrt3))
        * (kx / 2.0).sin()
        * phase;

    (
        Hermitian2::new(0.0, 0.0, h),
        [offdiagonal(hx), offdiagonal(hy), offdiagonal(hxy)],
    )
}

/// `M · G` for a diagonal `G`, which is `M` with its columns scaled.
fn times_green(m: &[[Complex64; 2]; 2], g: &[Complex64; 2]) -> [[Complex64; 2]; 2] {
    let mut out = [[Complex64::default(); 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = m[i][j] * g[j];
        }
    }
    out
}

fn product(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    let mut out = [[Complex64::default(); 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = (0..2).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn trace(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> Complex64 {
    (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| a[i][j] * b[j][i])
        .sum()
}

/// `χ(iν)` of the square lattice, summed over the zone and averaged.
///
/// One band leaves the trace a product of numbers, so this is the formula in
/// the module comment with every matrix replaced by a scalar.
fn chi_square(grid: &MomentumGrid, nu: &[f64], mu: &[f64]) -> Vec<Complex64> {
    let n_mu = mu.len();
    let mut chi = vec![Complex64::default(); nu.len() * n_mu];
    for index in 0..grid.len() {
        let (k1, k2) = grid.coordinates(index);
        let (ek, gx, gy, gxy) = square(k1, k2);
        for (i, &nu) in nu.iter().enumerate() {
            for (m, &mu) in mu.iter().enumerate() {
                let g = Complex64::new(-(ek - mu), nu).inv();
                let g2 = g * g;
                chi[i * n_mu + m] += gx * gx * gy * gy * g2 * g2 + gx * gy * gxy * g2 * g;
            }
        }
    }
    let nk = grid.len() as f64;
    chi.iter().map(|z| z / nk).collect()
}

/// `χ(iν)` of graphene, summed over the zone and averaged.
fn chi_graphene(grid: &MomentumGrid, nu: &[f64], mu: &[f64]) -> Vec<Complex64> {
    let n_mu = mu.len();
    let mut chi = vec![Complex64::default(); nu.len() * n_mu];
    for index in 0..grid.len() {
        let (k1, k2) = grid.coordinates(index);
        let (hamiltonian, velocities) = graphene(k1, k2);
        let eigen = hamiltonian.eigen();
        // In the eigenbasis `G` is diagonal, so the traces below are products
        // of 2×2 matrices and two numbers rather than of four matrices.
        let [gx, gy, gxy] = velocities.map(|m| Hermitian2::rotate(&eigen, &m));
        for (i, &nu) in nu.iter().enumerate() {
            for (m, &mu) in mu.iter().enumerate() {
                let g = [
                    Complex64::new(-(eigen.values[0] - mu), nu).inv(),
                    Complex64::new(-(eigen.values[1] - mu), nu).inv(),
                ];
                let (xg, yg, xyg) = (
                    times_green(&gx, &g),
                    times_green(&gy, &g),
                    times_green(&gxy, &g),
                );
                let xy = product(&xg, &yg);
                let yx = product(&yg, &xg);
                chi[i * n_mu + m] += trace(&xy, &xy) + 0.5 * (trace(&xy, &xyg) + trace(&yx, &xyg));
            }
        }
    }
    let nk = grid.len() as f64;
    chi.iter().map(|z| z / nk).collect()
}

/// The square lattice at `T = 0`:
/// `χ = −(2/3π²) [E(m) − K(m)/2]` with `m = 1 − μ²/16`, zero outside the band.
///
/// `K(1)` is infinite, so the closed form diverges at the van Hove filling
/// μ = 0; that one chemical potential is left out rather than written as an
/// infinity nothing can be compared against.
fn analytic(mu: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let kept: Vec<f64> = mu.iter().copied().filter(|&mu| mu != 0.0).collect();
    let values = kept
        .iter()
        .map(|&mu| {
            let m = 1.0 - mu * mu / 16.0;
            if m >= 0.0 {
                -(ellipe(m) - ellipk(m) / 2.0) * (2.0 / 3.0) / (PI * PI)
            } else {
                0.0
            }
        })
        .collect();
    (kept, values)
}

fn main() -> Result<(), Box<dyn StdError>> {
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;
    let mesh = IrMesh::<Fermionic>::new(&basis)?;
    let grid = MomentumGrid::new(NK_LIN, NK_LIN);
    let mu = chemical_potentials();
    let nu: Vec<f64> = mesh.wn().iter().map(|w| w.value(BETA)).collect();

    // `T Σ_ν F(iν) = F(τ = 0)`: the Matsubara sum is the basis expansion read
    // at one point. This is the row of basis functions that reads it.
    let u_at_zero = basis.evaluate_tau(&[0.0])?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("t", vec![T_HOPPING]);
    table.push("a", vec![LATTICE]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("nk_lin", vec![NK_LIN as f64]);
    table.push("n_mu", vec![N_MU as f64]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("n_matsubara", vec![mesh.n_wn() as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    let probe = mu
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - PROBE_MU).abs().total_cmp(&(*b - PROBE_MU).abs()))
        .map(|(index, _)| index)
        .expect("there is at least one chemical potential");
    let n: Vec<f64> = mesh.wn().iter().map(|w| w.n() as f64).collect();

    for (name, chi_iw) in [
        ("square", chi_square(&grid, &nu, &mu)),
        ("graphene", chi_graphene(&grid, &nu, &mu)),
    ] {
        let mut table = Table::new(provenance(EXAMPLE));
        table.push("mu", vec![mu[probe]; mesh.n_wn()]);
        table.push("wn", n.clone());
        table.push(
            "chi_re",
            (0..mesh.n_wn())
                .map(|i| chi_iw[i * N_MU + probe].re)
                .collect::<Vec<_>>(),
        );
        table.push(
            "chi_im",
            (0..mesh.n_wn())
                .map(|i| chi_iw[i * N_MU + probe].im)
                .collect::<Vec<_>>(),
        );
        write_table(&output_path(EXAMPLE, &format!("{name}_matsubara"))?, &table)?;

        let coefficients = mesh.wn_to_l(&chi_iw, N_MU)?;
        let chi: Vec<f64> = evaluate_rows(&u_at_zero, &coefficients, N_MU)
            .iter()
            .map(|z| z.re)
            .collect();
        let mut table = Table::new(provenance(EXAMPLE));
        table.push("mu", mu.clone());
        table.push("chi", chi);
        write_table(&output_path(EXAMPLE, name)?, &table)?;
    }

    let (kept, values) = analytic(&mu);
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("mu", kept);
    table.push("chi", values);
    write_table(&output_path(EXAMPLE, "square_analytic")?, &table)?;

    Ok(())
}
