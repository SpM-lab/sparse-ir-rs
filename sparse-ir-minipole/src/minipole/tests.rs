use super::*;
use crate::dlr::DiscreteLehmannRepresentation;
use sparse_ir_core::{Bosonic, Fermionic, MatsubaraSampling, StatisticsType};
use std::f64::consts::PI;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Discrete test spectrum `(ξ_j, A_j)` with poles inside `[-wmax, wmax]`.
fn spectrum(wmax: f64) -> Vec<(f64, f64)> {
    vec![(-0.71 * wmax, 0.3), (0.13 * wmax, 0.5), (0.58 * wmax, 0.2)]
}

fn g_exact(spec: &[(f64, f64)], z: C64) -> C64 {
    spec.iter().map(|&(x, a)| a / (z - x)).sum()
}

/// Every pole of `spec` is matched by a pole of `rep` within `pole_tol`, with
/// its residue within `res_tol`; the other poles have negligible weights.
fn assert_poles_close(rep: &MiniPoleResult, spec: &[(f64, f64)], pole_tol: f64, res_tol: f64) {
    let a = rep.pole_weight.host_data().unwrap();
    let r = rep.pole_location.len();
    let mut used = vec![false; r];
    for &(x, w) in spec {
        let j = (0..r)
            .min_by(|&i, &k| {
                (rep.pole_location[i] - x)
                    .norm()
                    .total_cmp(&(rep.pole_location[k] - x).norm())
            })
            .unwrap();
        assert!(
            (rep.pole_location[j] - x).norm() < pole_tol,
            "pole {x}: {:?}",
            rep.pole_location
        );
        assert!((a[j] - w).norm() < res_tol, "residue {w}: {}", a[j]);
        used[j] = true;
    }
    for j in (0..r).filter(|&j| !used[j]) {
        assert!(
            a[j].norm() < res_tol,
            "extra pole {} with weight {}",
            rep.pole_location[j],
            a[j]
        );
    }
}

/// DLR coefficients of `spec` from exact values at the DLR Matsubara nodes.
fn dlr_coeffs<S: StatisticsType + 'static>(
    dlr: &DiscreteLehmannRepresentation<S>,
    spec: &[(f64, f64)],
) -> TypedTensor<C64> {
    let s = MatsubaraSampling::new(dlr).unwrap();
    let beta = dlr_beta(dlr);
    let v: Vec<C64> = s
        .sampling_points()
        .iter()
        .map(|f| g_exact(spec, f.value_imaginary(beta)))
        .collect();
    let n = v.len();
    let v = TypedTensor::from_vec_col_major(vec![n], v).unwrap();
    s.fit_nd(None, &v, 0).unwrap()
}

fn dlr_beta<S: StatisticsType + 'static>(dlr: &DiscreteLehmannRepresentation<S>) -> f64 {
    use crate::basis_trait::Basis;
    dlr.beta()
}

/// Non-negative Matsubara frequencies `ω_n`, `n < nw`, and `G(iω_n)` with
/// deterministic noise of size `eta`.
fn matsubara_data(
    beta: f64,
    zeta: f64,
    nw: usize,
    spec: &[(f64, f64)],
    eta: f64,
    seed: u64,
) -> (Vec<f64>, TypedTensor<C64>) {
    let w: Vec<f64> = (0..nw)
        .map(|n| (2.0 * n as f64 + zeta) * PI / beta)
        .collect();
    let mut s = seed;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    let g: Vec<C64> = w
        .iter()
        .map(|&x| g_exact(spec, c(0.0, x)) + c(eta * rnd(), eta * rnd()))
        .collect();
    (w, TypedTensor::from_vec_col_major(vec![nw], g).unwrap())
}

#[test]
fn from_dlr_recovers_discrete_poles() {
    let (beta, wmax) = (50.0, 2.0);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-12).unwrap();
    let spec = spectrum(wmax);
    let g = dlr_coeffs(&dlr, &spec);
    let rep = mini_pole_dlr_from(&dlr, &g, &MiniPoleDlrParams::new(5, 1e-8)).unwrap();
    assert_eq!(rep.pole_location.len(), 3, "{:?}", rep.pole_location);
    assert_poles_close(&rep, &spec, 1e-6, 1e-6);

    // The pole representation reproduces G off the Matsubara axis too.
    for z in [c(0.3, 0.05), c(-1.0, 0.5), c(5.0, 0.0)] {
        let v = rep.evaluate(&[z]).unwrap();
        assert!((v.host_data().unwrap()[0] - g_exact(&spec, z)).norm() < 1e-5);
    }
}

#[test]
fn from_dlr_bosonic() {
    let (beta, wmax) = (20.0, 3.0);
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(beta, wmax, 1e-12).unwrap();
    let spec = spectrum(wmax);
    let g = dlr_coeffs(&dlr, &spec);
    let rep = mini_pole_dlr_from(&dlr, &g, &MiniPoleDlrParams::new(5, 1e-8)).unwrap();
    assert_eq!(rep.pole_location.len(), 3, "{:?}", rep.pole_location);
    assert_poles_close(&rep, &spec, 1e-6, 1e-6);
}

/// Odd bosonic spectrum `ρ(-ω) = -ρ(ω)` (a susceptibility-like `χ(iν)`,
/// finite at `ν = 0`) with a low-energy pair at `β|ξ| = 2`. The pair is
/// resolved only with a contour longer than the default `nmax = β`.
#[test]
fn from_dlr_bosonic_odd_spectrum_with_low_energy_pair() {
    let (beta, wmax) = (20.0, 3.0);
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::new(beta, wmax, 1e-12).unwrap();
    let x = 2.0 / beta;
    let spec = vec![(-0.4 * wmax, -0.2), (-x, -0.3), (x, 0.3), (0.4 * wmax, 0.2)];
    let g = dlr_coeffs(&dlr, &spec);
    let params = MiniPoleDlrParams {
        nmax: Some(50.0),
        ..MiniPoleDlrParams::new(5, 1e-8)
    };
    let rep = mini_pole_dlr_from(&dlr, &g, &params).unwrap();
    assert_eq!(rep.pole_location.len(), 4, "{:?}", rep.pole_location);
    assert_poles_close(&rep, &spec, 1e-3, 1e-3);
    // χ(0) is real and finite; its error is amplified by 1/ξ² of the pair.
    let v = rep.evaluate(&[c(0.0, 0.0)]).unwrap().host_data().unwrap()[0];
    let exact = g_exact(&spec, c(0.0, 0.0));
    assert!((v - exact).norm() < 1e-2 * exact.norm(), "{v} vs {exact}");
}

#[test]
fn from_dlr_matrix_valued_shares_poles() {
    let (beta, wmax) = (40.0, 1.0);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-12).unwrap();
    let xs = [-0.6, 0.2, 0.75];
    // Hermitian residue matrices A_j (2x2), channel c = a + 2 b.
    let res: [[C64; 4]; 3] = [
        [c(0.5, 0.0), c(0.1, -0.2), c(0.1, 0.2), c(0.2, 0.0)],
        [c(0.3, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.6, 0.0)],
        [c(0.2, 0.0), c(-0.1, 0.1), c(-0.1, -0.1), c(0.2, 0.0)],
    ];
    let s = MatsubaraSampling::new(&dlr).unwrap();
    let nf = s.sampling_points().len();
    let mut v = vec![c(0.0, 0.0); nf * 4];
    for (i, f) in s.sampling_points().iter().enumerate() {
        let z = f.value_imaginary(beta);
        for ch in 0..4 {
            v[i + nf * ch] = (0..3).map(|j| res[j][ch] / (z - xs[j])).sum();
        }
    }
    let v = TypedTensor::from_vec_col_major(vec![nf, 2, 2], v).unwrap();
    let g = s.fit_nd(None, &v, 0).unwrap();
    let rep = mini_pole_dlr_from(&dlr, &g, &MiniPoleDlrParams::new(8, 1e-8)).unwrap();
    assert_eq!(rep.pole_weight.shape(), &[3, 2, 2]);
    let a = rep.pole_weight.host_data().unwrap();
    for j in 0..3 {
        assert!(
            (rep.pole_location[j] - xs[j]).norm() < 1e-6,
            "{:?}",
            rep.pole_location
        );
        for ch in 0..4 {
            assert!((a[j + 3 * ch] - res[j][ch]).norm() < 1e-6);
        }
    }
}

#[test]
fn from_matsubara_with_noise() {
    let (beta, wmax, eta) = (100.0, 1.0, 1e-7);
    let spec = spectrum(wmax);
    let (w, g) = matsubara_data(beta, 1.0, 300, &spec, eta, 12345);
    let rep = mini_pole(&g, &w, &MiniPoleParams::new(1e-6)).unwrap();
    assert_eq!(rep.pole_location.len(), 3, "{:?}", rep.pole_location);
    assert_poles_close(&rep, &spec, 2e-3, 2e-3);

    // Frequencies not in the input, and the negative axis.
    for n in [7_i64, 301, 450, -45] {
        let z = c(0.0, (2 * n + 1) as f64 * PI / beta);
        let got = rep.evaluate(&[z]).unwrap().host_data().unwrap()[0];
        assert!((got - g_exact(&spec, z)).norm() < 1e-3, "n = {n}");
    }
}

#[test]
fn from_matsubara_bosonic_with_noise() {
    let (beta, wmax, eta) = (100.0, 1.0, 1e-7);
    let spec = spectrum(wmax);
    let (w, g) = matsubara_data(beta, 0.0, 300, &spec, eta, 54321);
    let rep = mini_pole(&g, &w, &MiniPoleParams::new(1e-6)).unwrap();
    assert_eq!(rep.pole_location.len(), 3, "{:?}", rep.pole_location);
    assert_poles_close(&rep, &spec, 2e-3, 2e-3);
    for n in [7_i64, 301, -45] {
        let z = c(0.0, (2 * n) as f64 * PI / beta);
        let got = rep.evaluate(&[z]).unwrap().host_data().unwrap()[0];
        assert!((got - g_exact(&spec, z)).norm() < 1e-3, "n = {n}");
    }
    // At ν = 0 the pole and residue errors are amplified by 1/ξ² and 1/ξ of
    // the pole closest to the origin.
    let got = rep.evaluate(&[c(0.0, 0.0)]).unwrap().host_data().unwrap()[0];
    assert!((got - g_exact(&spec, c(0.0, 0.0))).norm() < 1e-2);
}

#[test]
fn invalid_parameters_are_rejected() {
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(10.0, 1.0, 1e-8).unwrap();
    let r = dlr.poles().len();
    let g = TypedTensor::from_vec_col_major(vec![r], vec![0.0; r]).unwrap();
    let g1 = TypedTensor::from_vec_col_major(vec![r], vec![1.0; r]).unwrap();
    // Neither err nor M; nmax not above n0; wrong number of coefficients.
    let none = MiniPoleDlrParams {
        err: None,
        ..MiniPoleDlrParams::new(2, 1e-8)
    };
    assert!(mini_pole_dlr_from(&dlr, &g1, &none).is_err());
    let nmax = MiniPoleDlrParams {
        nmax: Some(1.0),
        ..MiniPoleDlrParams::new(2, 1e-8)
    };
    assert!(mini_pole_dlr_from(&dlr, &g, &nmax).is_err());
    let bad = TypedTensor::from_vec_col_major(vec![3], vec![0.0; 3]).unwrap();
    assert!(mini_pole_dlr_from(&dlr, &bad, &MiniPoleDlrParams::new(2, 1e-8)).is_err());
    // Zero coefficients give an empty representation.
    let rep = mini_pole_dlr_from(&dlr, &g, &MiniPoleDlrParams::new(2, 1e-8)).unwrap();
    assert!(rep.pole_location.is_empty());

    // Matsubara input: non-uniform or negative grid, n0 too large, a
    // non-square matrix, symmetry with the constant term.
    let spec = spectrum(1.0);
    let (w, g) = matsubara_data(10.0, 1.0, 40, &spec, 0.0, 1);
    let p = MiniPoleParams::new(1e-8);
    let mut w_bad = w.clone();
    w_bad[5] += 0.01;
    assert!(mini_pole(&g, &w_bad, &p).is_err());
    let w_neg: Vec<f64> = w.iter().map(|x| x - w[1]).collect();
    assert!(mini_pole(&g, &w_neg, &p).is_err());
    assert!(
        mini_pole(
            &g,
            &w,
            &MiniPoleParams {
                n0: N0::Fixed(39),
                ..p.clone()
            }
        )
        .is_err()
    );
    let rect = TypedTensor::from_vec_col_major(vec![40, 2, 3], vec![c(0.0, 0.0); 240]).unwrap();
    assert!(mini_pole(&rect, &w, &p).is_err());
    let sym = MiniPoleParams {
        symmetry: true,
        compute_const: true,
        ..p
    };
    assert!(mini_pole(&g, &w, &sym).is_err());
}
