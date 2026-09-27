//! The momentum-space Fourier convention, checked against a plain DFT.
//!
//! The applied notebooks rely on the two transforms being each other's
//! inverse, with the `1/nk` sitting on the momentum sum, which is what makes
//! the product of two real-space arrays come back as the momentum convolution
//! `(1/nk) Σ_q A(q) B(k − q)`. A sign or a factor in the wrong place changes
//! χ₀ and Σ by a constant and is very hard to spot in a picture, so it is
//! pinned here.

use std::f64::consts::TAU;

use num_complex::Complex64;
use sparse_ir_tutorial::mesh::MomentumGrid;

const NK1: usize = 4;
const NK2: usize = 6;

fn sample_rows(nrows: usize, nk: usize) -> Vec<Complex64> {
    (0..nrows * nk)
        .map(|n| {
            let x = n as f64;
            Complex64::new((0.7 * x).sin(), (0.3 * x + 1.0).cos())
        })
        .collect()
}

/// `Σ_k e^{−ik·r} A(k)`, written out.
fn dft_forward(values: &[Complex64], nk1: usize, nk2: usize) -> Vec<Complex64> {
    let nk = nk1 * nk2;
    let mut out = vec![Complex64::default(); values.len()];
    for (row_in, row_out) in values.chunks(nk).zip(out.chunks_mut(nk)) {
        for r1 in 0..nk1 {
            for r2 in 0..nk2 {
                let mut sum = Complex64::default();
                for k1 in 0..nk1 {
                    for k2 in 0..nk2 {
                        let phase =
                            -TAU * ((k1 * r1) as f64 / nk1 as f64 + (k2 * r2) as f64 / nk2 as f64);
                        sum += row_in[k1 * nk2 + k2] * Complex64::from_polar(1.0, phase);
                    }
                }
                row_out[r1 * nk2 + r2] = sum;
            }
        }
    }
    out
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let scale = e.norm().max(1.0);
        assert!(
            (a - e).norm() <= tol * scale,
            "entry {i}: {a} differs from {e} by {}",
            (a - e).norm()
        );
    }
}

#[test]
fn k_to_r_is_the_forward_dft_averaged_over_the_zone() {
    let grid = MomentumGrid::new(NK1, NK2);
    let nk = grid.len() as f64;
    let values = sample_rows(3, grid.len());
    let expected: Vec<Complex64> = dft_forward(&values, NK1, NK2)
        .iter()
        .map(|v| v / nk)
        .collect();
    assert_close(&grid.k_to_r(&values), &expected, 1e-13);
}

#[test]
fn r_to_k_undoes_k_to_r() {
    let grid = MomentumGrid::new(NK1, NK2);
    let values = sample_rows(3, grid.len());
    assert_close(&grid.r_to_k(&grid.k_to_r(&values)), &values, 1e-13);
}

#[test]
fn a_product_in_real_space_is_a_convolution_in_momentum_space() {
    let grid = MomentumGrid::new(NK1, NK2);
    let nk = grid.len();
    let a = sample_rows(1, nk);
    let b: Vec<Complex64> = sample_rows(1, nk).iter().map(|v| v.conj() + 0.5).collect();

    let product: Vec<Complex64> = grid
        .k_to_r(&a)
        .iter()
        .zip(grid.k_to_r(&b))
        .map(|(x, y)| x * y)
        .collect();
    let convolved = grid.r_to_k(&product);

    // (1/nk) Σ_q A(q) B(k − q), with k − q taken on the grid.
    let mut expected = vec![Complex64::default(); nk];
    for k1 in 0..NK1 {
        for k2 in 0..NK2 {
            let mut sum = Complex64::default();
            for q1 in 0..NK1 {
                for q2 in 0..NK2 {
                    let d1 = (k1 + NK1 - q1) % NK1;
                    let d2 = (k2 + NK2 - q2) % NK2;
                    sum += a[q1 * NK2 + q2] * b[d1 * NK2 + d2];
                }
            }
            expected[k1 * NK2 + k2] = sum / nk as f64;
        }
    }
    assert_close(&convolved, &expected, 1e-13);
}

#[test]
fn the_square_lattice_dispersion_has_its_extrema_where_it_should() {
    let grid = MomentumGrid::new(8, 8);
    let ek = grid.square_lattice_dispersion(1.0);
    assert!((ek[0] + 4.0).abs() < 1e-14, "Γ is the band bottom, −4t");
    let corner = grid.len() / 2 + 4; // (k₁, k₂) = (1/2, 1/2), the M point
    assert!((ek[corner] - 4.0).abs() < 1e-14, "M is the band top, +4t");
}
