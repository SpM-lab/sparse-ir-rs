//! DLR -> MiniPole: contour sensitivity of a low-energy bosonic pole pair.
//! The analytic model and tolerances follow sparse-ir-minipole's
//! `from_dlr_bosonic_odd_spectrum_with_low_energy_pair` regression test.

// ANCHOR: imports
use num_complex::Complex64;
use sparse_ir::minipole::{MiniPoleDlrParams, mini_pole_dlr_from};
use sparse_ir::{Bosonic, DiscreteLehmannRepresentation, MatsubaraSampling, TypedTensor};
// ANCHOR_END: imports
use sparse_ir_tutorial::{Table, output_path, provenance, write_table};
use std::error::Error;
use std::f64::consts::PI;

const EXAMPLE: &str = "minipole";

fn main() -> Result<(), Box<dyn Error>> {
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

    // Independent analytic checks, with the bounds of the regression test:
    // pole/residue errors < 1e-3 and static relative error < 1e-2.
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
        write_table(&output_path(EXAMPLE, &format!("poles_{name}"))?, &table)?;

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
