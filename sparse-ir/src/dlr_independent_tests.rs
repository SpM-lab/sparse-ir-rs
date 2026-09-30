//! Tests for the independent (interpolative-decomposition) DLR.

use super::*;
use crate::basis_trait::Basis;
use crate::{
    Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, MatsubaraSampling,
    MatsubaraSamplingPositiveOnly, TauSampling,
};
use crate::{DlrFromIr, IrBasis};

/// Off-grid test poles and weights inside `[-wmax, wmax]`.
fn test_spectrum(wmax: f64) -> Vec<(f64, f64)> {
    vec![
        (-0.913 * wmax, 0.2),
        (-0.237 * wmax, 0.3),
        (0.0071 * wmax, 0.1),
        (0.4417 * wmax, 0.25),
        (0.977 * wmax, 0.15),
    ]
}

/// `G(τ) = -Σ_k c_k e^{-τω_k} / (1 + e^{-βω_k})` for both statistics
/// (for bosons this is the logistic-weighted spectral representation).
fn g_tau(spec: &[(f64, f64)], tau: f64, beta: f64) -> f64 {
    spec.iter()
        .map(|&(w, c)| c * fermionic_single_pole(tau, w, beta).unwrap())
        .sum()
}

fn g_iw<S: StatisticsType>(spec: &[(f64, f64)], n: i64, beta: f64) -> Complex<f64> {
    let freq = MatsubaraFreq::<S>::new(n).unwrap();
    let weight = |w: f64| match S::STATISTICS {
        Statistics::Fermionic => 1.0,
        Statistics::Bosonic => (0.5 * beta * w).tanh(),
    };
    spec.iter()
        .map(|&(w, c)| c * weight(w) * giwn_single_pole(&freq, w, beta).unwrap())
        .sum()
}

fn eval_tau_dlr<S: StatisticsType + 'static>(
    dlr: &DiscreteLehmannRepresentation<S>,
    coeffs: &[f64],
    taus: &[f64],
) -> Vec<f64> {
    let m = crate::sampling::mat_from_matrix(&dlr.evaluate_tau(taus).unwrap()).unwrap();
    (0..taus.len())
        .map(|i| (0..coeffs.len()).map(|j| m[[i, j]] * coeffs[j]).sum())
        .collect()
}

fn test_taus(beta: f64) -> Vec<f64> {
    (0..97).map(|i| beta * (i as f64 + 0.37) / 97.0).collect()
}

#[test]
fn independent_rank_matches_ir_size() {
    let (beta, wmax, eps) = (100.0, 10.0, 1e-10);
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, eps).unwrap();
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(beta * wmax).unwrap(),
        beta,
        Some(eps),
        None,
    )
    .unwrap();
    let (r, l) = (dlr.poles().len() as i64, basis.size() as i64);
    assert!((r - l).abs() <= 4, "DLR rank {r} vs IR size {l}");
    assert!(dlr.poles().iter().all(|p| p.abs() <= wmax));
    assert!(dlr.ir_transform().is_none());
    assert_eq!(dlr.ir_basis_size(), None);
}

fn check_tau_roundtrip<S: StatisticsType + 'static>(beta: f64, wmax: f64, eps: f64) {
    let dlr = DlrBuilder::<S>::new(beta, wmax)
        .accuracy(eps)
        .build()
        .unwrap();
    let spec = test_spectrum(wmax);
    let sampling = TauSampling::new(&dlr).unwrap();
    assert_eq!(sampling.sampling_points().len(), dlr.poles().len());
    let values: Vec<f64> = sampling
        .sampling_points()
        .iter()
        .map(|&t| g_tau(&spec, t, beta))
        .collect();
    let coeffs = sampling.fit(&values).unwrap();

    let taus = test_taus(beta);
    let approx = eval_tau_dlr(&dlr, &coeffs, &taus);
    let err = taus
        .iter()
        .zip(&approx)
        .map(|(&t, &a)| (a - g_tau(&spec, t, beta)).abs())
        .fold(0.0, f64::max);
    assert!(err < 100.0 * eps, "tau round trip error {err:e}");
}

#[test]
fn tau_nodes_interpolate_fermionic() {
    check_tau_roundtrip::<Fermionic>(100.0, 10.0, 1e-10);
    check_tau_roundtrip::<Fermionic>(1e3, 10.0, 1e-12);
}

#[test]
fn tau_nodes_interpolate_bosonic() {
    check_tau_roundtrip::<Bosonic>(100.0, 10.0, 1e-10);
}

fn check_matsubara_roundtrip<S: StatisticsType + 'static>(beta: f64, wmax: f64, eps: f64) {
    let dlr = DiscreteLehmannRepresentation::<S>::new(beta, wmax, eps).unwrap();
    let spec = test_spectrum(wmax);
    let zeta = match S::STATISTICS {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    };
    let check_ns: Vec<i64> = (-300..300)
        .map(|k| 2 * k + zeta)
        .chain([2 * 12345 + zeta, -2 * 98765 - zeta])
        .collect();
    let check_freqs: Vec<MatsubaraFreq<S>> = check_ns
        .iter()
        .map(|&n| MatsubaraFreq::new(n).unwrap())
        .collect();
    let eval =
        crate::sampling::mat_from_matrix(&dlr.evaluate_matsubara(&check_freqs).unwrap()).unwrap();
    let max_err = |coeffs: &[Complex<f64>]| {
        check_ns
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let a: Complex<f64> = (0..coeffs.len()).map(|j| eval[[i, j]] * coeffs[j]).sum();
                (a - g_iw::<S>(&spec, n, beta)).norm()
            })
            .fold(0.0, f64::max)
    };
    let scale = beta;

    let full = MatsubaraSampling::new(&dlr).unwrap();
    assert_eq!(full.sampling_points().len(), dlr.poles().len());
    let values: Vec<Complex<f64>> = full
        .sampling_points()
        .iter()
        .map(|f| g_iw::<S>(&spec, f.n(), beta))
        .collect();
    let coeffs = full.fit(&values).unwrap();
    let err = max_err(&coeffs);
    assert!(err < 100.0 * eps * scale, "full Matsubara error {err:e}");

    let pos = MatsubaraSamplingPositiveOnly::new(&dlr).unwrap();
    assert!(pos.sampling_points().iter().all(|f| f.n() >= 0));
    let values: Vec<Complex<f64>> = pos
        .sampling_points()
        .iter()
        .map(|f| g_iw::<S>(&spec, f.n(), beta))
        .collect();
    let coeffs: Vec<Complex<f64>> = pos
        .fit(&values)
        .unwrap()
        .into_iter()
        .map(|x| Complex::new(x, 0.0))
        .collect();
    let err = max_err(&coeffs);
    assert!(
        err < 100.0 * eps * scale,
        "positive-only Matsubara error {err:e}"
    );
}

#[test]
fn matsubara_nodes_interpolate_fermionic() {
    check_matsubara_roundtrip::<Fermionic>(100.0, 10.0, 1e-10);
}

#[test]
fn matsubara_nodes_interpolate_bosonic() {
    check_matsubara_roundtrip::<Bosonic>(100.0, 10.0, 1e-10);
}

#[test]
fn ir_transform_connects_independent_dlr_and_ir() {
    let (beta, wmax, eps) = (50.0, 4.0, 1e-10);
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(beta * wmax).unwrap(),
        beta,
        Some(eps),
        None,
    )
    .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, eps).unwrap();
    assert!(
        dlr.to_ir_nd::<f64>(
            None,
            &crate::TypedTensor::from_vec_col_major(
                vec![dlr.poles().len()],
                vec![0.0; dlr.poles().len()]
            )
            .unwrap(),
            0
        )
        .is_err()
    );

    let t = basis.dlr_transform(&dlr).unwrap();
    assert_eq!(
        (t.ir_size(), t.dlr_size()),
        (basis.size(), dlr.poles().len())
    );

    // DLR coefficients of the test spectrum from τ interpolation.
    let spec = test_spectrum(wmax);
    let tau_s = TauSampling::new(&dlr).unwrap();
    let values: Vec<f64> = tau_s
        .sampling_points()
        .iter()
        .map(|&x| g_tau(&spec, x, beta))
        .collect();
    let g_dlr = tau_s.fit(&values).unwrap();
    let r = g_dlr.len();
    let g_dlr_t = crate::TypedTensor::from_vec_col_major(vec![r], g_dlr.clone()).unwrap();
    let gl = t.dlr_to_ir_nd(None, &g_dlr_t, 0).unwrap();
    let gl = gl.host_data().unwrap().to_vec();

    // The IR expansion reproduces G(τ).
    let taus = test_taus(beta);
    let u = crate::sampling::mat_from_matrix(&basis.evaluate_tau(&taus).unwrap()).unwrap();
    for (i, &x) in taus.iter().enumerate() {
        let v: f64 = (0..gl.len()).map(|l| u[[i, l]] * gl[l]).sum();
        assert!((v - g_tau(&spec, x, beta)).abs() < 1e-8, "tau = {x}");
    }

    // IR -> DLR inverts DLR -> IR.
    let gl_t = crate::TypedTensor::from_vec_col_major(vec![basis.size()], gl).unwrap();
    let back = t.ir_to_dlr_nd(None, &gl_t, 0).unwrap();
    let back = eval_tau_dlr(&dlr, back.host_data().unwrap(), &taus);
    let orig = eval_tau_dlr(&dlr, &g_dlr, &taus);
    for (a, b) in back.iter().zip(&orig) {
        assert!((a - b).abs() < 1e-8);
    }

    // Mismatched β is rejected.
    let other =
        DiscreteLehmannRepresentation::<Fermionic>::new(2.0 * beta, wmax / 2.0, eps).unwrap();
    assert!(matches!(
        basis.dlr_transform(&other),
        Err(Error::InvalidParameter { name: "basis", .. })
    ));
}

#[test]
fn ir_derived_dlr_keeps_transform() {
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(
        LogisticKernel::new(100.0).unwrap(),
        10.0,
        Some(1e-8),
        None,
    )
    .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();
    assert_eq!(dlr.ir_basis_size(), Some(basis.size()));
    // Default nodes are available for IR-derived DLRs as well.
    assert_eq!(
        dlr.default_tau_sampling_points().unwrap().len(),
        dlr.poles().len()
    );
}

#[test]
fn invalid_parameters_are_rejected() {
    for (beta, wmax, eps) in [
        (0.0, 1.0, 1e-8),
        (1.0, f64::NAN, 1e-8),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 2.0),
    ] {
        assert!(matches!(
            DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, eps),
            Err(Error::InvalidParameter { .. })
        ));
    }
    let capped = DlrBuilder::<Fermionic>::new(10.0, 10.0)
        .max_size(7)
        .build()
        .unwrap();
    assert_eq!(capped.poles().len(), 7);
}
