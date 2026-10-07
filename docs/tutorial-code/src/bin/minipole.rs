//! MiniPole: a few complex poles from Matsubara data.
//!
//! Part 1 compresses the semicircular density of states into a handful of
//! complex poles and writes the complex-plane picture that opens the book page,
//! together with scans over the ESPRIT tolerance `err` and the contour start
//! `n0`. Part 2 is a harder case, a low-energy bosonic pole pair, where the
//! upper end `nmax` of the contour matters; its analytic model follows
//! sparse-ir-minipole's `from_dlr_bosonic_odd_spectrum_with_low_energy_pair`
//! test.

// ANCHOR: imports
use num_complex::Complex64;
use sparse_ir::minipole::{MiniPoleDlrParams, MiniPoleResult, mini_pole_dlr_from};
use sparse_ir::{
    Bosonic, DiscreteLehmannRepresentation, Fermionic, MatsubaraSampling, TypedTensor,
};
use std::error::Error;
use std::f64::consts::PI;
// ANCHOR_END: imports
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};

const EXAMPLE: &str = "minipole";

// ANCHOR: semicircle_model
/// Semicircular DOS rho(w) = (2/pi) sqrt(1 - w^2) on [-1, 1]. Its Green's
/// function on the physical sheet is G(z) = 2 (z - sqrt(z-1) sqrt(z+1)).
fn semicircle(z: Complex64) -> Complex64 {
    2.0 * (z - (z - 1.0).sqrt() * (z + 1.0).sqrt())
}

/// Largest |G_MP(iv_n) - G(iv_n)| over the fermionic frequencies
/// iv_n = i n pi / beta with odd n, 0 < n < 2000. The poles lie in the lower
/// half plane, so G_MP represents G in the upper half plane only.
fn max_matsubara_error(rep: &MiniPoleResult, beta: f64) -> Result<f64, Box<dyn Error>> {
    let z: Vec<Complex64> = (0..1000)
        .map(|m| Complex64::new(0.0, (2 * m + 1) as f64 * PI / beta))
        .collect();
    let got = rep.evaluate(&z)?;
    Ok(got
        .host_data()?
        .iter()
        .zip(&z)
        .map(|(g, &zi)| (*g - semicircle(zi)).norm())
        .fold(0.0, f64::max))
}
// ANCHOR_END: semicircle_model

fn pole_table(rep: &MiniPoleResult) -> Result<Table, Box<dyn Error>> {
    let weights = rep.pole_weight.host_data()?;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push(
        "pole_re",
        rep.pole_location.iter().map(|p| p.re).collect::<Vec<_>>(),
    );
    table.push(
        "pole_im",
        rep.pole_location.iter().map(|p| p.im).collect::<Vec<_>>(),
    );
    table.push(
        "weight_re",
        weights.iter().map(|a| a.re).collect::<Vec<_>>(),
    );
    table.push(
        "weight_im",
        weights.iter().map(|a| a.im).collect::<Vec<_>>(),
    );
    Ok(table)
}

fn main() -> Result<(), Box<dyn Error>> {
    semicircle_example()?;
    bosonic_pair_example()?;
    Ok(())
}

fn semicircle_example() -> Result<(), Box<dyn Error>> {
    // ANCHOR: semicircle
    let (beta, wmax) = (100.0, 1.5);
    // An IR-independent DLR and its sparse Matsubara nodes.
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-12)?;
    let sampling = MatsubaraSampling::new(&dlr)?;
    let values: Vec<Complex64> = sampling
        .sampling_points()
        .iter()
        .map(|n| semicircle(n.value_imaginary(beta)))
        .collect();
    let values = TypedTensor::from_vec_col_major(vec![values.len()], values)?;
    let coeffs = sampling.fit_nd(None, &values, 0)?;

    // Contour from iw_5 = 11 pi / beta up to the default nmax = beta;
    // ESPRIT tolerance err = 1e-8.
    let rep = mini_pole_dlr_from(&dlr, &coeffs, &MiniPoleDlrParams::new(5, 1e-8))?;
    for pole in &rep.pole_location {
        println!("pole {:+.4} {:+.4}i", pole.re, pole.im);
        assert!(pole.im < 0.0, "pole {pole} is not in the lower half plane");
    }
    let error = max_matsubara_error(&rep, beta)?;
    println!(
        "{} poles (DLR size {}), max |G_MP - G| on iv_n = {error:.1e}",
        rep.pole_location.len(),
        dlr.poles().len()
    );
    assert!(error < 1e-3, "Matsubara error {error}");
    // ANCHOR_END: semicircle

    write_table(
        &output_path(EXAMPLE, "semicircle_poles")?,
        &pole_table(&rep)?,
    )?;

    // The complex plane picture: exact G and G_MP on a grid of z.
    let (nx, ny) = (201, 151);
    let mut z = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            z.push(Complex64::new(
                -2.0 + 4.0 * i as f64 / (nx - 1) as f64,
                -1.5 + 3.0 * j as f64 / (ny - 1) as f64,
            ));
        }
    }
    let g = rep.evaluate(&z)?;
    let g = g.host_data()?;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("x", z.iter().map(|zi| zi.re).collect::<Vec<_>>());
    table.push("y", z.iter().map(|zi| zi.im).collect::<Vec<_>>());
    table.push(
        "log10_abs_exact",
        z.iter()
            .map(|&zi| semicircle(zi).norm().log10())
            .collect::<Vec<_>>(),
    );
    table.push(
        "log10_abs_mp",
        g.iter().map(|gi| gi.norm().log10()).collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "semicircle_grid")?, &table)?;

    // Spectral function on the real axis. All poles lie below it, so
    // G_MP(w + i0) is G_MP(w).
    let omega: Vec<f64> = (0..801).map(|i| -1.6 + 3.2 * i as f64 / 800.0).collect();
    let real_axis: Vec<Complex64> = omega.iter().map(|&w| Complex64::new(w, 0.0)).collect();
    let g = rep.evaluate(&real_axis)?;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("omega", omega.clone());
    table.push(
        "rho_exact",
        omega
            .iter()
            .map(|&w| 2.0 / PI * (1.0 - w * w).max(0.0).sqrt())
            .collect::<Vec<_>>(),
    );
    table.push(
        "rho_mp",
        g.host_data()?
            .iter()
            .map(|gi| -gi.im / PI)
            .collect::<Vec<_>>(),
    );
    write_table(&output_path(EXAMPLE, "semicircle_spectral")?, &table)?;

    // ANCHOR: err_scan
    // Fewer poles for a looser tolerance; err is not an error bound.
    let mut errs = Vec::new();
    let mut counts = Vec::new();
    let mut errors = Vec::new();
    for err in [1e-4, 1e-6, 1e-8, 1e-10] {
        let rep = mini_pole_dlr_from(&dlr, &coeffs, &MiniPoleDlrParams::new(5, err))?;
        assert!(rep.pole_location.iter().all(|p| p.im < 0.0));
        let error = max_matsubara_error(&rep, beta)?;
        println!(
            "err = {err:.0e}: {} poles, max error {error:.1e}",
            rep.pole_location.len()
        );
        errs.push(err);
        counts.push(rep.pole_location.len() as f64);
        errors.push(error);
    }
    // ANCHOR_END: err_scan
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("err", errs);
    table.push("n_poles", counts);
    table.push("max_error", errors);
    write_table(&output_path(EXAMPLE, "semicircle_err_scan")?, &table)?;

    // ANCHOR: n0_scan
    // Same coefficients and err, three lower ends of the contour.
    let mut scan = Vec::new();
    for n0 in [0, 2, 5] {
        let rep = mini_pole_dlr_from(&dlr, &coeffs, &MiniPoleDlrParams::new(n0, 1e-8))?;
        let upper = rep.pole_location.iter().filter(|p| p.im > 0.0).count();
        let error = max_matsubara_error(&rep, beta)?;
        println!(
            "n0 = {n0}: {} poles, {upper} in the upper half plane, max error {error:.1e}",
            rep.pole_location.len()
        );
        scan.push((n0, upper, error, rep));
    }
    // n0 = 0 and 2 give spurious upper-half-plane poles; n0 = 5 does not.
    assert!(
        scan.iter()
            .all(|(n0, upper, ..)| (*n0 == 5) == (*upper == 0))
    );
    // ANCHOR_END: n0_scan
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("n0", scan.iter().map(|s| s.0 as f64).collect::<Vec<_>>());
    table.push(
        "n_poles",
        scan.iter()
            .map(|s| s.3.pole_location.len() as f64)
            .collect::<Vec<_>>(),
    );
    table.push(
        "n_upper",
        scan.iter().map(|s| s.1 as f64).collect::<Vec<_>>(),
    );
    table.push("max_error", scan.iter().map(|s| s.2).collect::<Vec<_>>());
    for (n0, _, _, rep) in &scan {
        write_table(
            &output_path(EXAMPLE, &format!("semicircle_poles_n0_{n0}"))?,
            &pole_table(rep)?,
        )?;
    }
    write_table(&output_path(EXAMPLE, "semicircle_n0_scan")?, &table)?;
    // ANCHOR_END: n0_scan

    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![beta]);
    table.push("wmax", vec![wmax]);
    table.push("dlr_size", vec![dlr.poles().len() as f64]);
    table.push("max_error", vec![error]);
    write_table(&output_path(EXAMPLE, "semicircle_summary")?, &table)?;
    Ok(())
}

fn bosonic_pair_example() -> Result<(), Box<dyn Error>> {
    // ANCHOR: dlr
    let (beta, wmax) = (20.0, 3.0);
    // Odd spectral function: residues have opposite signs at +/- xi.
    let spectrum = [(-1.2, -0.2), (-0.1, -0.3), (0.1, 0.3), (1.2, 0.2)];
    let exact = |z: Complex64| -> Complex64 { spectrum.iter().map(|&(xi, a)| a / (z - xi)).sum() };
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(beta, wmax, 1e-12)?;
    let sampling = MatsubaraSampling::new(&dlr)?;
    let values: Vec<Complex64> = sampling
        .sampling_points()
        .iter()
        .map(|n| exact(n.value_imaginary(beta)))
        .collect();
    let values = TypedTensor::from_vec_col_major(vec![values.len()], values)?;
    let coeffs = sampling.fit_nd(None, &values, 0)?;

    // These contour parameters resolve this particular low-energy pair.
    let params = MiniPoleDlrParams {
        nmax: Some(50.0),
        ..MiniPoleDlrParams::new(5, 1e-8)
    };
    let poles = mini_pole_dlr_from(&dlr, &coeffs, &params)?;
    assert_eq!(poles.pole_location.len(), 4);

    // Independent analytic checks: pole and residue errors below 1e-3, and
    // the static value chi(0) within 1 %.
    for ((location, weight), &(xi, a)) in poles
        .pole_location
        .iter()
        .zip(poles.pole_weight.host_data()?)
        .zip(&spectrum)
    {
        assert!((*location - xi).norm() < 1e-3, "pole {xi}: {location}");
        assert!((*weight - a).norm() < 1e-3, "residue {a}: {weight}");
    }
    let zero = Complex64::new(0.0, 0.0);
    let chi0 = poles.evaluate(&[zero])?.host_data()?[0];
    assert!((chi0 - exact(zero)).norm() < 1e-2 * exact(zero).norm());
    // ANCHOR_END: dlr

    // Non-negative physical bosonic frequencies, including the static value.
    let z: Vec<Complex64> = (0..201)
        .map(|m| Complex64::new(0.0, 2.0 * m as f64 * PI / beta))
        .collect();
    let reference: Vec<Complex64> = z.iter().map(|&zi| exact(zi)).collect();
    let scale = reference.iter().map(|v| v.norm()).fold(0.0, f64::max);

    let mut summary = Table::new(provenance(EXAMPLE));
    let mut n0s = Vec::new();
    let mut nmaxs = Vec::new();
    let mut counts = Vec::new();
    let mut errors = Vec::new();
    for n0 in [0, 2, 5, 8] {
        for nmax in [10.0, 20.0, 50.0, 100.0] {
            let rep = mini_pole_dlr_from(
                &dlr,
                &coeffs,
                &MiniPoleDlrParams {
                    nmax: Some(nmax),
                    ..MiniPoleDlrParams::new(n0, 1e-8)
                },
            )?;
            let evaluated = rep.evaluate(&z)?;
            let max_error = evaluated
                .host_data()?
                .iter()
                .zip(&reference)
                .map(|(got, expected)| (*got - expected).norm())
                .fold(0.0, f64::max)
                / scale;
            assert!(max_error.is_finite());
            n0s.push(n0 as f64);
            nmaxs.push(nmax);
            counts.push(rep.pole_location.len() as f64);
            errors.push(max_error);
        }
    }
    summary.push("n0", n0s);
    summary.push("nmax", nmaxs);
    summary.push("n_poles", counts);
    summary.push("max_relative_error", errors);
    write_table(&output_path(EXAMPLE, "contours")?, &summary)?;

    let default = mini_pole_dlr_from(&dlr, &coeffs, &MiniPoleDlrParams::new(5, 1e-8))?;
    assert_eq!(default.pole_location.len(), 3, "default contour changed");
    for (name, rep) in [("default", default), ("extended", poles)] {
        write_table(
            &output_path(EXAMPLE, &format!("poles_{name}"))?,
            &pole_table(&rep)?,
        )?;

        let evaluated = rep.evaluate(&z)?;
        let evaluated = evaluated.host_data()?;
        let mut table = Table::new(provenance(EXAMPLE));
        table.push("nu", z.iter().map(|zi| zi.im).collect::<Vec<_>>());
        table.push(
            "relative_error",
            evaluated
                .iter()
                .zip(&reference)
                .map(|(got, expected)| (*got - expected).norm() / scale)
                .collect::<Vec<_>>(),
        );
        write_table(&output_path(EXAMPLE, &format!("values_{name}"))?, &table)?;
    }
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("pole", spectrum.iter().map(|p| p.0).collect::<Vec<_>>());
    table.push("weight", spectrum.iter().map(|p| p.1).collect::<Vec<_>>());
    write_table(&output_path(EXAMPLE, "exact_poles")?, &table)?;
    let mut table = Table::new(provenance(EXAMPLE));
    table.push("beta", vec![beta]);
    table.push("wmax", vec![wmax]);
    table.push("dlr_size", vec![dlr.poles().len() as f64]);
    write_table(&output_path(EXAMPLE, "summary")?, &table)?;
    Ok(())
}
