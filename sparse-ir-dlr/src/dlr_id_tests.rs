use super::*;
use crate::fitters::RealMatrixFitter;
use crate::matrix::Mat;

/// Deterministic uniform samples in `[0, 1)`.
fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

fn naive_kernel(tau: f64, omega: f64) -> f64 {
    (-tau * omega).exp() / (1.0 + (-omega).exp())
}

#[test]
fn candidate_grids_are_symmetric_and_in_range() {
    for &lambda in &[1.0, 10.0, 1e3, 1e5] {
        let taus = tau_candidates(lambda);
        assert_eq!(taus.len() % 2, 0);
        for (a, b) in taus.iter().zip(taus.iter().rev()) {
            assert!((a.value() + b.value() - 1.0).abs() < 1e-15);
        }
        assert!(taus.windows(2).all(|w| w[0].value() < w[1].value()));
        assert!(taus.iter().all(|t| t.t > 0.0 && t.t <= 0.5));

        let omegas = omega_candidates(lambda);
        for (a, b) in omegas.iter().zip(omegas.iter().rev()) {
            assert_eq!(*a, -*b);
        }
        assert!(omegas.windows(2).all(|w| w[0] < w[1]));
        assert!(omegas.iter().all(|w| w.abs() < lambda && *w != 0.0));
    }
}

#[test]
fn logistic_kernel_matches_naive_formula() {
    for &w in &[-30.0, -1.5, -1e-3, 0.0, 2e-3, 1.0, 40.0] {
        for &t in &[1e-4, 0.1, 0.3, 0.5] {
            for from_end in [false, true] {
                let c = TauCandidate { t, from_end };
                let naive = naive_kernel(c.value(), w);
                let got = logistic_kernel(c, w);
                assert!(
                    (got - naive).abs() <= 1e-13 * naive.abs(),
                    "t={t} from_end={from_end} w={w}: {got} vs {naive}"
                );
            }
        }
    }
    // Large |ω| near τ = 1 stays finite and accurate.
    let c = TauCandidate {
        t: 1e-6,
        from_end: true,
    };
    let got = logistic_kernel(c, -1e5);
    assert!((got - (-0.1f64).exp()).abs() < 1e-14);
}

#[test]
fn matsubara_candidates_have_parity_and_cover_range() {
    for zeta in [0, 1] {
        let pos = matsubara_candidates(1e4, zeta, true);
        assert!(pos.iter().all(|n| n.rem_euclid(2) == zeta));
        assert!(pos.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(pos[0], zeta);
        assert!(*pos.last().unwrap() as f64 >= 8e4 - 2.0);

        let all = matsubara_candidates(1e4, zeta, false);
        assert!(all.windows(2).all(|w| w[0] < w[1]));
        let n_neg = all.iter().filter(|&&n| n < 0).count();
        let n_pos = all.iter().filter(|&&n| n > 0).count();
        assert_eq!(n_neg, n_pos);
    }
}

#[test]
fn gram_schmidt_detects_rank() {
    // 6 x 5 matrix of rank 3: columns 3 and 4 are combinations of 0..3.
    let m = 6;
    let base: Vec<Vec<f64>> = (0..3)
        .map(|j| {
            (0..m)
                .map(|i| ((i * (j + 2)) as f64).sin() + j as f64)
                .collect()
        })
        .collect();
    let mut a = Vec::new();
    for col in &base {
        a.extend_from_slice(col);
    }
    a.extend((0..m).map(|i| base[0][i] - 2.0 * base[1][i]));
    a.extend((0..m).map(|i| 0.5 * base[2][i] + base[1][i]));
    let piv = pivoted_gram_schmidt(&mut a, m, 5, 1e-12, usize::MAX);
    assert_eq!(piv.len(), 3);

    let mut z: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 12];
    z[0] = Complex::new(1.0, 1.0);
    z[5] = Complex::new(0.0, 2.0);
    let piv = pivoted_gram_schmidt(&mut z, 4, 3, 1e-12, usize::MAX);
    assert_eq!(piv, vec![1, 0]);
}

/// `K(τ, ω) ≈ Σ_i K(τ, ω_i) c_i(ω)` uniformly on random points not in the
/// candidate grids.
fn check_pole_reconstruction(lambda: f64, eps: f64) -> usize {
    let poles = select_poles(lambda, eps, None);
    let r = poles.len();
    let taus = tau_candidates(lambda);
    let kfit = Mat::from_fn([taus.len(), r], |idx| {
        logistic_kernel(taus[idx[0]], poles[idx[1]])
    });
    let fitter = RealMatrixFitter::new(kfit);

    let test_tau: Vec<f64> = uniform(300, 1).into_iter().collect();
    let test_omega: Vec<f64> = uniform(200, 2)
        .into_iter()
        .map(|u| lambda * (2.0 * u - 1.0))
        .collect();
    let mut max_err: f64 = 0.0;
    for &w in &test_omega {
        let rhs: Vec<f64> = taus.iter().map(|&t| logistic_kernel(t, w)).collect();
        let c = fitter.fit(None, &rhs).unwrap();
        for &t in &test_tau {
            let approx: f64 = poles
                .iter()
                .zip(&c)
                .map(|(&p, &ci)| naive_kernel(t, p) * ci)
                .sum();
            max_err = max_err.max((approx - naive_kernel(t, w)).abs());
        }
    }
    assert!(
        max_err < 100.0 * eps,
        "Λ={lambda} ε={eps}: rank {r}, max error {max_err:e}"
    );
    r
}

#[test]
fn poles_reconstruct_kernel() {
    let r_lo = check_pole_reconstruction(100.0, 1e-6);
    let r_hi = check_pole_reconstruction(100.0, 1e-12);
    assert!(r_lo < r_hi);
    check_pole_reconstruction(1e4, 1e-10);
}

#[test]
fn poles_are_nearly_antisymmetric_and_rank_capped() {
    let poles = select_poles(1e3, 1e-10, None);
    assert!(poles.windows(2).all(|w| w[0] < w[1]));
    assert!(poles.iter().all(|w| w.abs() <= 1e3));
    let capped = select_poles(1e3, 1e-10, Some(10));
    assert_eq!(capped.len(), 10);
}
