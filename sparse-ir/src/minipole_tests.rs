use super::*;
use crate::{Bosonic, Fermionic, MatsubaraSampling, Statistics};

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

fn assert_poles_close(rep: &PoleRepresentation, spec: &[(f64, f64)], pole_tol: f64, res_tol: f64) {
    assert_eq!(rep.poles.len(), spec.len(), "poles {:?}", rep.poles);
    let a = rep.residues.host_data().unwrap();
    for (j, &(x, w)) in spec.iter().enumerate() {
        assert!(
            (rep.poles[j] - x).norm() < pole_tol,
            "pole {j}: {} vs {x}",
            rep.poles[j]
        );
        assert!((a[j] - w).norm() < res_tol, "residue {j}: {} vs {w}", a[j]);
    }
}

fn zeta<S: StatisticsType>() -> i64 {
    match S::STATISTICS {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    }
}

/// DLR coefficients of `spec` from exact values at the DLR Matsubara nodes.
fn dlr_coeffs<S: StatisticsType + 'static>(
    dlr: &DiscreteLehmannRepresentation<S>,
    spec: &[(f64, f64)],
) -> TypedTensor<C64> {
    let s = MatsubaraSampling::new(dlr).unwrap();
    let v: Vec<C64> = s
        .sampling_points()
        .iter()
        .map(|f| g_exact(spec, f.value_imaginary(dlr.beta())))
        .collect();
    let n = v.len();
    let v = TypedTensor::from_vec_col_major(vec![n], v).unwrap();
    s.fit_nd(None, &v, 0).unwrap()
}

#[test]
fn from_dlr_recovers_discrete_poles() {
    let (beta, wmax) = (50.0, 2.0);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-12).unwrap();
    let spec = spectrum(wmax);
    let g = dlr_coeffs(&dlr, &spec);
    let rep = minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8)).unwrap();
    assert_poles_close(&rep, &spec, 1e-6, 1e-6);
    assert_eq!(rep.diagnostics.esprit.order, 3);

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
    let rep = minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8)).unwrap();
    assert_poles_close(&rep, &spec, 1e-6, 1e-6);
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
    let rep = minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8)).unwrap();
    assert_eq!(rep.residues.shape(), &[3, 2, 2]);
    let a = rep.residues.host_data().unwrap();
    for j in 0..3 {
        assert!((rep.poles[j] - xs[j]).norm() < 1e-6);
        for ch in 0..4 {
            assert!((a[j + 3 * ch] - res[j][ch]).norm() < 1e-6);
        }
    }
}

#[test]
fn from_matsubara_with_noise() {
    let (beta, wmax, eta) = (100.0, 1.0, 1e-7);
    let spec = spectrum(wmax);
    let zt = zeta::<Fermionic>();
    // Irregular, unsorted frequency set.
    let ns: Vec<i64> = (0..120)
        .map(|k| {
            let m = if k % 2 == 0 { k } else { -k - 3 };
            2 * m + zt
        })
        .collect();
    let freqs: Vec<MatsubaraFreq<Fermionic>> =
        ns.iter().map(|&n| MatsubaraFreq::new(n).unwrap()).collect();
    let mut s = 12345u64;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    let v: Vec<C64> = freqs
        .iter()
        .map(|f| g_exact(&spec, f.value_imaginary(beta)) + c(eta * rnd(), eta * rnd()))
        .collect();
    let v = TypedTensor::from_vec_col_major(vec![freqs.len()], v).unwrap();
    let rep = minipole_from_matsubara(beta, wmax, 1e-10, &freqs, &v, &MiniPoleOptions::new(1e-5))
        .unwrap();
    assert!(rep.diagnostics.dlr_fit_residual.unwrap() < 1e-5);
    assert_poles_close(&rep, &spec, 2e-3, 2e-3);

    // Held-out frequencies.
    let held: Vec<MatsubaraFreq<Fermionic>> = [7, 301, -45]
        .iter()
        .map(|&k| MatsubaraFreq::new(2 * k + zt).unwrap())
        .collect();
    let ev = rep.evaluate_matsubara(beta, &held).unwrap();
    for (f, got) in held.iter().zip(ev.host_data().unwrap()) {
        assert!((got - g_exact(&spec, f.value_imaginary(beta))).norm() < 1e-3);
    }
}

#[test]
fn invalid_options_are_rejected() {
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(10.0, 1.0, 1e-8).unwrap();
    let g = TypedTensor::from_vec_col_major(vec![dlr.poles().len()], vec![0.0; dlr.poles().len()])
        .unwrap();
    assert!(minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(-1.0)).is_err());
    assert!(
        minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8).with_segment(1.0, 0.5)).is_err()
    );
    assert!(minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8).with_moments(2)).is_err());
    let bad = TypedTensor::from_vec_col_major(vec![3], vec![0.0; 3]).unwrap();
    assert!(minipole_from_dlr(&dlr, &bad, &MiniPoleOptions::new(1e-8)).is_err());
    // Zero coefficients give an empty representation.
    let rep = minipole_from_dlr(&dlr, &g, &MiniPoleOptions::new(1e-8)).unwrap();
    assert!(rep.poles.is_empty());
}
