//! Getting numerical data into the IR basis, and back out again.
//!
//! Ported from the Python notebook `transformation_py.ipynb` of
//! sparse-ir-tutorial.
//!
//! Four routes in, one route out:
//!
//! * from poles, where `ρₗ = Σₚ cₚ vₗ(ω̄ₚ)` is a sum rather than an integral —
//!   and the same thing through the DLR;
//! * from a smooth spectral function, by composite Gauss-Legendre quadrature
//!   over the segments the basis functions themselves are built on;
//! * from `Gₗ` to `G(τ)` on any grid you like, directly or through
//!   `TauSampling`;
//! * back from `G(τ)` by `Gₗ = ∫₀^β dτ G(τ) uₗ(τ)`.
//!
//! The last section shows what a too-small `ωmax` looks like: the coefficients
//! stop following the singular values, which is the signal to widen the basis.

use std::error::Error;

use mdarray::{DTensor, Tensor};
use sparse_ir::{
    Basis, Bosonic, DiscreteLehmannRepresentation, Fermionic, FiniteTempBasis, LogisticKernel,
    MatsubaraSampling, TauSampling,
};
use sparse_ir_tutorial::{Table, integrate_segments, output_path, provenance, write_table};

const EXAMPLE: &str = "transformation";

/// The pole section: one bosonic pole, placed just off zero so that the
/// `1/tanh(βω/2)` regularizer is large but finite.
const POLE_BETA: f64 = 15.0;
const POLE_WMAX: f64 = 10.0;
const POLE_EPS: f64 = 1e-10;
const POLE_POSITION: f64 = 0.1;
const POLE_WEIGHT: f64 = 1.0;

/// The smooth section: three Gaussian peaks.
const BETA: f64 = 10.0;
const WMAX: f64 = 10.0;
const EPS: f64 = 1e-10;

/// The remark at the end: a basis whose `ωmax` is far too small.
const NARROW_WMAX: f64 = 0.5;

const N_OMEGA: usize = 1000;
const N_TAU: usize = 1000;

fn main() -> Result<(), Box<dyn Error>> {
    poles()?;
    let basis = smooth_spectrum()?;
    let g_l = imaginary_time(&basis)?;
    let narrow_size = narrow_basis(&basis, &g_l)?;
    matrix_valued(&basis, &g_l)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("pole_beta", vec![POLE_BETA]);
    table.push("pole_wmax", vec![POLE_WMAX]);
    table.push("pole_basis_size", vec![pole_basis()?.size() as f64]);
    table.push("beta", vec![BETA]);
    table.push("wmax", vec![WMAX]);
    table.push("eps", vec![EPS]);
    table.push("basis_size", vec![basis.size() as f64]);
    table.push("accuracy", vec![basis.accuracy()]);
    table.push("narrow_wmax", vec![NARROW_WMAX]);
    table.push("narrow_basis_size", vec![narrow_size as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;

    Ok(())
}

fn pole_basis() -> Result<FiniteTempBasis<LogisticKernel, Bosonic>, Box<dyn Error>> {
    let kernel = LogisticKernel::new(POLE_BETA * POLE_WMAX)?;
    Ok(FiniteTempBasis::<LogisticKernel, Bosonic>::new(
        kernel,
        POLE_BETA,
        Some(POLE_EPS),
        None,
    )?)
}

/// `ρ(ω) = Σₚ cₚ δ(ω − ω̄ₚ)` needs no quadrature: the overlap integral is the
/// value of `vₗ` at the pole.
fn poles() -> Result<(), Box<dyn Error>> {
    let basis = pole_basis()?;

    // For the logistic kernel the bosonic spectral function carries the
    // `1/tanh(βω̄/2)` factor, so that is what the basis expands.
    let regularized = POLE_WEIGHT / (0.5 * POLE_BETA * POLE_POSITION).tanh();

    let v_at_pole: DTensor<f64, 2> = basis.evaluate_omega(&[POLE_POSITION])?;
    let rho_l: Vec<f64> = (0..basis.size())
        .map(|l| v_at_pole[[0, l]] * regularized)
        .collect();
    let g_l: Vec<f64> = basis
        .s()
        .iter()
        .zip(&rho_l)
        .map(|(s, rho)| -s * rho)
        .collect();

    // The DLR says the same thing in one call: it knows how a pole maps onto
    // the basis, so it takes the pole weights and returns `Gₗ` directly.
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, vec![POLE_POSITION])?;
    let weights = Tensor::<f64, _>::from_fn([1], |_| regularized).into_dyn();
    let g_l_dlr = dlr.to_ir_nd::<f64>(None, &weights, 0)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("rho_l", rho_l);
    table.push("g_l", g_l);
    table.push(
        "g_l_dlr",
        (0..basis.size()).map(|l| g_l_dlr[[l]]).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "pole_coefficients")?, &table)?;

    Ok(())
}

/// Three Gaussian peaks, normalized to one.
fn rho(omega: f64) -> f64 {
    let gaussian = |mu: f64, sigma: f64| {
        (-((omega - mu) / sigma).powi(2)).exp() / (std::f64::consts::PI.sqrt() * sigma)
    };
    0.2 * gaussian(0.0, 0.15) + 0.4 * gaussian(1.0, 0.8) + 0.4 * gaussian(-1.0, 0.8)
}

/// `ρₗ = ∫ dω vₗ(ω) ρ(ω)` by composite Gauss-Legendre quadrature.
///
/// A single Gauss-Legendre rule over `[−ωmax, ωmax]` would not do: the roots
/// of `vₗ` crowd together near `ω = 0`, so the integrand varies on a scale the
/// rule cannot see. Splitting the interval at the knots the basis functions
/// are built on — where `vₗ` is a polynomial on each piece — and using a rule
/// on every piece converges exponentially in the order, as long as `ρ` is
/// smooth within a piece.
fn overlap_with_v<F>(basis: &FiniteTempBasis<LogisticKernel, Fermionic>, f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let v = basis.v();
    let edges = v.get_knots(None);
    let order = v.get_polyorder() + 8;
    (0..basis.size())
        .map(|l| {
            let poly = &v[l];
            integrate_segments(|omega| poly.evaluate(omega) * f(omega), &edges, order)
        })
        .collect()
}

/// The same quadrature on the imaginary-time side: `Gₗ = ∫₀^β dτ G(τ) uₗ(τ)`.
fn overlap_with_u<F>(basis: &FiniteTempBasis<LogisticKernel, Fermionic>, f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let u = basis.u();
    let edges = u.get_knots(None);
    let order = u.get_polyorder() + 8;
    (0..basis.size())
        .map(|l| {
            let poly = &u[l];
            integrate_segments(|tau| poly.evaluate(tau) * f(tau), &edges, order)
        })
        .collect()
}

fn smooth_spectrum() -> Result<FiniteTempBasis<LogisticKernel, Fermionic>, Box<dyn Error>> {
    let kernel = LogisticKernel::new(BETA * WMAX)?;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;

    let rho_l = overlap_with_v(&basis, rho);
    let g_l: Vec<f64> = basis
        .s()
        .iter()
        .zip(&rho_l)
        .map(|(s, rho)| -s * rho)
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("s_l", basis.s().to_vec());
    table.push("rho_l", rho_l.clone());
    table.push("g_l", g_l);
    write_table(&output_path(EXAMPLE, "smooth_coefficients")?, &table)?;

    // The expansion is good on any ω you ask for, not only on the knots.
    let omegas = linspace(-5.0, 5.0, N_OMEGA);
    let v_at_omegas: DTensor<f64, 2> = basis.evaluate_omega(&omegas)?;
    let reconstructed: Vec<f64> = (0..omegas.len())
        .map(|i| {
            (0..basis.size())
                .map(|l| v_at_omegas[[i, l]] * rho_l[l])
                .sum()
        })
        .collect();

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omegas.clone());
    table.push(
        "rho_exact",
        omegas.iter().map(|&omega| rho(omega)).collect::<Vec<_>>(),
    );
    table.push("rho_reconstructed", reconstructed);
    write_table(&output_path(EXAMPLE, "spectrum")?, &table)?;

    Ok(basis)
}

/// Writes `G(τ)` on a dense grid two ways, and the coefficients recovered from
/// it. Returns `Gₗ`, which the rest of the example reuses.
fn imaginary_time(
    basis: &FiniteTempBasis<LogisticKernel, Fermionic>,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let rho_l = overlap_with_v(basis, rho);
    let g_l: Vec<f64> = basis
        .s()
        .iter()
        .zip(&rho_l)
        .map(|(s, rho)| -s * rho)
        .collect();

    let taus = linspace(0.0, BETA, N_TAU);

    // Directly: `G(τ) = Σₗ uₗ(τ) Gₗ`.
    let u_at_taus: DTensor<f64, 2> = basis.evaluate_tau(&taus)?;
    let g_tau_direct: Vec<f64> = (0..taus.len())
        .map(|i| (0..basis.size()).map(|l| u_at_taus[[i, l]] * g_l[l]).sum())
        .collect();

    // Or through `TauSampling`, which builds the same matrix once and can also
    // go the other way. Any set of τ points will do — these are not the
    // default sampling times.
    let sampling = TauSampling::<Fermionic>::with_sampling_points(basis, taus.clone())?;
    let g_tau_sampling = sampling.evaluate(&g_l)?;

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("tau", taus);
    table.push("g_tau_direct", g_tau_direct);
    table.push("g_tau_sampling", g_tau_sampling);
    write_table(&output_path(EXAMPLE, "gtau")?, &table)?;

    // The stable way back from imaginary-time data is the overlap integral,
    // not a fit: it uses `G(τ)` everywhere rather than at a few points.
    let g_l_reconstructed = overlap_with_u(basis, |tau| {
        let u_at_tau = basis
            .evaluate_tau(&[tau])
            .expect("τ is inside [0, β] by construction");
        (0..basis.size()).map(|l| u_at_tau[[0, l]] * g_l[l]).sum()
    });

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("l", (0..basis.size()).map(|l| l as f64).collect::<Vec<_>>());
    table.push("g_l", g_l.clone());
    table.push("g_l_reconstructed", g_l_reconstructed.clone());
    table.push(
        "error",
        g_l_reconstructed
            .iter()
            .zip(&g_l)
            .map(|(a, b)| (a - b).abs())
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "roundtrip")?, &table)?;

    Ok(g_l)
}

/// Expands the very same `G(τ)` in a basis built for `ωmax = 0.5`, which is
/// far too narrow for a spectral function that reaches out to `ω ≈ 3`.
fn narrow_basis(
    basis: &FiniteTempBasis<LogisticKernel, Fermionic>,
    g_l: &[f64],
) -> Result<usize, Box<dyn Error>> {
    let kernel = LogisticKernel::new(BETA * NARROW_WMAX)?;
    let narrow = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, BETA, Some(EPS), None)?;

    let g_l_narrow = overlap_with_u(&narrow, |tau| {
        let u_at_tau = basis
            .evaluate_tau(&[tau])
            .expect("τ is inside [0, β] by construction");
        (0..basis.size()).map(|l| u_at_tau[[0, l]] * g_l[l]).sum()
    });

    // The point of the section: the coefficients do not follow the singular
    // values down, so the last ones are not small and the expansion has not
    // converged. `ρ` is even in `ω`, so odd `l` vanishes by symmetry; the last
    // even index is where the tail is visible.
    let last_even = narrow.size() - 2;
    assert!(
        g_l_narrow[last_even].abs() > 1e4 * narrow.s()[last_even],
        "a too-narrow basis must fail to converge; if this ever stops holding, \
         the section no longer shows what it claims"
    );

    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "l",
        (0..narrow.size()).map(|l| l as f64).collect::<Vec<_>>(),
    );
    table.push("s_l", narrow.s().to_vec());
    table.push("g_l", g_l_narrow);
    write_table(&output_path(EXAMPLE, "narrow_basis")?, &table)?;

    Ok(narrow.size())
}

/// `evaluate` and `fit` also come in `_nd` flavours that transform one axis of
/// a whole array, which is what you want when many Green's functions share a
/// basis.
fn matrix_valued(
    basis: &FiniteTempBasis<LogisticKernel, Fermionic>,
    g_l: &[f64],
) -> Result<(), Box<dyn Error>> {
    let size = basis.size();
    // Two orbital indices times the basis, each Green's function a fixed
    // multiple of `Gₗ` so that the round trip has something to check.
    let coeffs = Tensor::<f64, _>::from_fn([2, 3, size], |index| {
        let (i, j, l) = (index[0], index[1], index[2]);
        (1.0 + i as f64 + 2.0 * j as f64) * g_l[l]
    })
    .into_dyn();

    let sampling = MatsubaraSampling::<Fermionic>::new(basis)?;
    let values = sampling.evaluate_nd_real(None, &coeffs, 2)?;
    let recovered = sampling.fit_nd_real(None, &values, 2)?;

    let mut worst: f64 = 0.0;
    for i in 0..2 {
        for j in 0..3 {
            for l in 0..size {
                worst = worst.max((recovered[[i, j, l]] - coeffs[[i, j, l]]).abs());
            }
        }
    }
    let scale = g_l.iter().fold(0.0f64, |acc, g| acc.max(g.abs()));
    assert!(
        worst < 1e-12 * scale,
        "the N-D round trip must recover the coefficients: worst deviation {worst:e}"
    );

    Ok(())
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    assert!(count > 1, "a grid needs at least two points");
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}
