use super::model::{Problem, Workspace, gram};
use super::*;
use std::f64::consts::PI;

/// Inverse temperature of the Matsubara grids of the end-to-end tests.
const BETA: f64 = 20.0;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn with_session<R: Send>(f: impl FnOnce(&mut dyn BackendSession) -> R + Send) -> R {
    CpuBackend::new().with_backend_session(f)
}

/// Fermionic Matsubara points `i (2n + 1) π / β`, `n = 0..nf`.
fn matsubara(beta: f64, nf: usize) -> Vec<C64> {
    (0..nf)
        .map(|n| c(0.0, (2 * n + 1) as f64 * PI / beta))
        .collect()
}

/// `Σ_j w_j / (z - p_j)` per channel, `[len z, X]` column-major.
fn pole_sum(p: &[C64], w: &[C64], z: &[C64], x: usize) -> Vec<C64> {
    let m = p.len();
    let mut out = vec![c(0.0, 0.0); z.len() * x];
    for ch in 0..x {
        for (i, &zi) in z.iter().enumerate() {
            for (j, &pj) in p.iter().enumerate() {
                out[i + z.len() * ch] += w[j + m * ch] / (zi - pj);
            }
        }
    }
    out
}

fn rel_l2(a: &[C64], b: &[C64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
    let den: f64 = b.iter().map(|y| y.norm_sqr()).sum();
    (num / den).sqrt()
}

fn real_grid(n: usize, half_width: f64) -> Vec<f64> {
    (0..n)
        .map(|i| -half_width + 2.0 * half_width * i as f64 / (n - 1) as f64)
        .collect()
}

/// Smallest eigenvalue over `ω` of the 2 x 2 (or scalar) spectral matrix
/// `-(Σ(ω) - Σ(ω)†) / (2i)` of a result.
fn min_spectral(res: &CausalPoles, n: usize, omega: &[f64]) -> f64 {
    let z: Vec<C64> = omega.iter().map(|&w| c(w, 0.0)).collect();
    let s = res.evaluate(&z).unwrap();
    let s = s.host_data().unwrap();
    let l = z.len();
    let mut min = f64::INFINITY;
    for i in 0..l {
        let e = |p: usize, q: usize| s[i + l * (p + n * q)];
        // A = -(S - S†) / (2i), Hermitian
        let a = |p: usize, q: usize| (e(p, q) - e(q, p).conj()) / c(0.0, -2.0);
        let ev = if n == 1 {
            a(0, 0).re
        } else {
            let (a00, a11, a01) = (a(0, 0).re, a(1, 1).re, a(0, 1));
            let tr = 0.5 * (a00 + a11);
            let disc = (0.25 * (a00 - a11).powi(2) + a01.norm_sqr()).sqrt();
            tr - disc
        };
        min = min.min(ev);
    }
    min
}

#[test]
fn gram_matches_quadrature() {
    // ω = tan t maps the real axis onto (-π/2, π/2); midpoint rule.
    let nq = 400_000;
    let quad = |p: C64, a: i32, q: C64, b: i32| {
        let mut acc = c(0.0, 0.0);
        let dt = PI / nq as f64;
        for i in 0..nq {
            let t = -0.5 * PI + (i as f64 + 0.5) * dt;
            let w = t.tan();
            let jac = 1.0 + w * w;
            acc += (c(w, 0.0) - p).powi(-a).conj() * (c(w, 0.0) - q).powi(-b) * jac * dt;
        }
        acc
    };
    let pairs = [
        (c(-0.3, -0.5), c(0.4, -0.7)),
        (c(0.2, 0.6), c(-0.5, 0.9)),
        (c(0.2, -0.6), c(-0.5, 0.9)),
    ];
    for (p, q) in pairs {
        for (a, b) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
            let exact = gram(p, a as u8, q, b as u8);
            let num = quad(p, a, q, b);
            assert!(
                (exact - num).norm() < 1e-6 * (1.0 + num.norm()),
                "p={p} q={q} ({a},{b}): {exact} vs {num}"
            );
        }
    }
}

/// A random-ish parameter vector for `m` poles, `n x k` factors.
fn sample_theta(m: usize, n: usize, k: usize) -> Vec<f64> {
    let np = m * (2 + 2 * n * k);
    (0..np)
        .map(|i| {
            let v = ((i as f64 + 1.0) * 0.7548776662466927).fract() - 0.5;
            if i < m {
                2.0 * v
            } else if i < 2 * m {
                -1.0 + v
            } else {
                0.8 * v
            }
        })
        .collect()
}

fn sample_problem<'a>(
    m: usize,
    n: usize,
    k: usize,
    z: &'a [C64],
    d: &'a [C64],
    ref_poles: Vec<C64>,
    ref_weights: Vec<C64>,
) -> Problem<'a> {
    // Exercise the shifted line: the reference poles stay in the lower half
    // plane after the shift by -iη.
    let eta = 0.2;
    let ref_poles: Vec<C64> = ref_poles.iter().map(|p| p - c(0.0, eta)).collect();
    let x = n * n;
    let mr = ref_poles.len();
    let mut ref_norm = 0.0;
    for ch in 0..x {
        for i in 0..mr {
            for j in 0..mr {
                ref_norm += (ref_weights[i + mr * ch].conj()
                    * gram(ref_poles[i], 1, ref_poles[j], 1)
                    * ref_weights[j + mr * ch])
                    .re;
            }
        }
    }
    Problem {
        m,
        n,
        k,
        z,
        d,
        eta,
        ref_poles,
        ref_weights,
        ref_norm,
        h2_weight: 1.0 / ref_norm,
        eps: 1e-3,
    }
}

#[test]
fn gradient_matches_finite_differences() {
    for (m, n, k) in [(3, 1, 1), (2, 2, 2), (3, 2, 1)] {
        let x = n * n;
        let z = matsubara(10.0, 8);
        let ref_poles = vec![c(-0.4, -0.3), c(0.6, -0.5)];
        let ref_weights: Vec<C64> = (0..2 * x)
            .map(|i| c(0.3 + 0.1 * i as f64, 0.05 * i as f64 - 0.1))
            .collect();
        let d = pole_sum(&ref_poles, &ref_weights, &z, x)
            .iter()
            .map(|v| v * 1.01 + c(1e-3, 0.0))
            .collect::<Vec<_>>();
        let pb = sample_problem(m, n, k, &z, &d, ref_poles, ref_weights);
        let theta = sample_theta(m, n, k);
        with_session(|s| {
            let mut ws = Workspace::new(&pb).unwrap();
            for alpha in [0.0, 0.7] {
                let ev = ws.eval(&pb, &theta, alpha, true, s).unwrap();
                for i in 0..theta.len() {
                    let h = 1e-6;
                    let mut tp = theta.clone();
                    tp[i] += h;
                    let fp = ws.eval(&pb, &tp, alpha, false, s).unwrap().phi;
                    tp[i] -= 2.0 * h;
                    let fm = ws.eval(&pb, &tp, alpha, false, s).unwrap().phi;
                    let fd = (fp - fm) / (2.0 * h);
                    assert!(
                        (fd - ev.grad[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                        "(m,n,k)=({m},{n},{k}) alpha={alpha} param {i}: fd {fd} vs {}",
                        ev.grad[i]
                    );
                }
            }
        });
    }
}

#[test]
fn gauss_newton_is_hessian_at_zero_residual() {
    // Reference and data generated by the parametrization itself: at the
    // true parameters both residuals vanish and Gauss–Newton is exact.
    for (m, n, k) in [(2, 1, 1), (2, 2, 2)] {
        let x = n * n;
        let z = matsubara(10.0, 6);
        let theta = sample_theta(m, n, k);
        let dummy = sample_problem(m, n, k, &z, &[], vec![c(0.0, -1.0)], vec![c(1.0, 0.0); x]);
        let (poles, weights) = with_session(|s| {
            let mut ws = Workspace::new(&dummy).unwrap();
            let pb = Problem {
                d: &vec![c(0.0, 0.0); z.len() * x],
                ..dummy
            };
            ws.eval(&pb, &theta, 0.0, false, s).unwrap();
            (ws.poles().to_vec(), ws.w.clone())
        });
        let d = pole_sum(&poles, &weights, &z, x);
        let pb = sample_problem(m, n, k, &z, &d, poles, weights);
        with_session(|s| {
            let mut ws = Workspace::new(&pb).unwrap();
            let alpha = 0.5;
            let ev = ws.eval(&pb, &theta, alpha, true, s).unwrap();
            assert!(ev.phi < 1e-14, "phi at the truth: {}", ev.phi);
            let p = theta.len();
            for j in 0..p {
                let h = 1e-5;
                let mut tp = theta.clone();
                tp[j] += h;
                let gp = ws.eval(&pb, &tp, alpha, true, s).unwrap().grad;
                tp[j] -= 2.0 * h;
                let gm = ws.eval(&pb, &tp, alpha, true, s).unwrap().grad;
                for i in 0..p {
                    let fd = (gp[i] - gm[i]) / (2.0 * h);
                    let gn = ev.gn[i + p * j];
                    assert!(
                        (fd - gn).abs() < 1e-5 * (1.0 + gn.abs()),
                        "(m,n,k)=({m},{n},{k}) [{i},{j}]: fd {fd} vs {gn}"
                    );
                }
            }
        });
    }
}

/// Causal truth with `m` Lorentzians; a perturbed, non-causal reference
/// (complex weights, a negative weight, a pole in the upper half plane).
fn scalar_case() -> (Vec<C64>, Vec<C64>, Vec<C64>, Vec<C64>) {
    let truth_p = vec![c(-0.8, -0.3), c(0.5, -0.2), c(1.4, -0.4)];
    let truth_w = vec![c(0.5, 0.0), c(0.3, 0.0), c(0.2, 0.0)];
    let ref_p = vec![
        c(-0.82, -0.28),
        c(0.51, -0.21),
        c(1.38, -0.42),
        c(0.13, 0.11),
        c(2.0, -0.1),
    ];
    let ref_w = vec![
        c(0.49, 0.02),
        c(0.31, -0.015),
        c(0.2, 0.01),
        c(0.004, 0.002),
        c(-0.01, 0.0),
    ];
    (truth_p, truth_w, ref_p, ref_w)
}

fn no_prune() -> Params {
    Params {
        prune: false,
        ..Default::default()
    }
}

#[test]
fn scalar_refinement_is_causal_and_fits_the_data() {
    let (tp, tw, rp, rw) = scalar_case();
    let z = matsubara(20.0, 40);
    let d = pole_sum(&tp, &tw, &z, 1);
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d.clone()).unwrap();
    let weights = TypedTensor::from_vec_col_major(vec![rp.len()], rw.clone()).unwrap();
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &no_prune()).unwrap();
    println!("{info:?}");
    assert!(res.poles.iter().all(|p| p.im < 0.0));
    assert!(res.poles.windows(2).all(|p| p[0].re <= p[1].re));
    let omega = real_grid(2001, 5.0);
    let min_a = min_spectral(&res, 1, &omega);
    assert!(min_a >= -1e-12, "negative spectral weight {min_a}");
    assert!(info.residual_rms <= info.target_rms * (1.0 + 1e-12));
    assert_eq!(info.target_rms, info.input_residual_rms);
    // The data residual agrees with the returned function.
    let fit = res.evaluate(&z).unwrap();
    let r = (fit
        .host_data()
        .unwrap()
        .iter()
        .zip(&d)
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        / z.len() as f64)
        .sqrt();
    assert!((r - info.residual_rms).abs() < 1e-10 * (1.0 + r));
    // Close to the truth near the real axis, no worse than the input.
    let near: Vec<C64> = omega.iter().map(|&w| c(w, 0.05)).collect();
    let truth = pole_sum(&tp, &tw, &near, 1);
    let e_in = rel_l2(&pole_sum(&rp, &rw, &near, 1), &truth);
    let e_out = rel_l2(res.evaluate(&near).unwrap().host_data().unwrap(), &truth);
    println!("relL2 input {e_in:.4} refined {e_out:.4}");
    assert!(e_out < 1.5 * e_in, "input {e_in} refined {e_out}");
    assert_eq!(res.poles.len(), rp.len());
    // Pruning is off by default; with it the data at this ε are explained by
    // fewer poles.
    let (_, info) = refine(&rp, &weights, &z, &values, BETA, &Params::default()).unwrap();
    assert_eq!(info.pruned, 0);
    let params = Params {
        prune: true,
        ..Default::default()
    };
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &params).unwrap();
    assert!(
        info.pruned > 0 && res.poles.len() == rp.len() - info.pruned,
        "{info:?}"
    );
    assert!(info.residual_rms <= info.target_rms * (1.0 + 1e-12));
    assert!(min_spectral(&res, 1, &omega) >= -1e-12);
}

#[test]
fn causal_input_is_reproduced() {
    let (tp, tw, _, _) = scalar_case();
    let z = matsubara(20.0, 40);
    let d = pole_sum(&tp, &tw, &z, 1);
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d).unwrap();
    let weights = TypedTensor::from_vec_col_major(vec![tp.len()], tw.clone()).unwrap();
    let (res, info) = refine(&tp, &weights, &z, &values, BETA, &Params::default()).unwrap();
    println!("{info:?}");
    assert!(info.h2_distance < 1e-6, "{info:?}");
    assert!(info.residual_rms < 1e-8, "{info:?}");
    let w = res.weights.host_data().unwrap();
    for (j, (&p, &a)) in tp.iter().zip(&tw).enumerate() {
        assert!((res.poles[j] - p).norm() < 1e-6, "{:?}", res.poles);
        assert!((w[j] - a).norm() < 1e-6, "{w:?}");
    }
}

#[test]
fn matrix_refinement_is_positive_semidefinite() {
    let n = 2;
    let x = n * n;
    // Truth: PSD weights v v† + diagonal
    let tp = vec![c(-0.6, -0.25), c(0.7, -0.35)];
    let vecs = [[c(0.6, 0.0), c(0.3, 0.2)], [c(0.2, -0.1), c(0.5, 0.0)]];
    let mut tw = vec![c(0.0, 0.0); 2 * x];
    for j in 0..2 {
        for q in 0..n {
            for p in 0..n {
                let mut v = vecs[j][p] * vecs[j][q].conj();
                if p == q {
                    v += 0.05;
                }
                tw[j + 2 * (p + n * q)] = v;
            }
        }
    }
    // Reference: non-Hermitian perturbation and a pole in the upper half plane
    let rp = vec![c(-0.62, -0.24), c(0.69, -0.36), c(0.0, 0.08)];
    let mut rw = vec![c(0.0, 0.0); 3 * x];
    for ch in 0..x {
        for j in 0..2 {
            rw[j + 3 * ch] = tw[j + 2 * ch] + c(0.01 * (ch as f64 - 1.5), 0.008 * j as f64);
        }
        rw[2 + 3 * ch] = c(0.003, -0.002 * ch as f64);
    }
    let z = matsubara(20.0, 40);
    let d = pole_sum(&tp, &tw, &z, x);
    let values = TypedTensor::from_vec_col_major(vec![z.len(), n, n], d).unwrap();
    let weights = TypedTensor::from_vec_col_major(vec![3, n, n], rw.clone()).unwrap();
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &Params::default()).unwrap();
    println!("{info:?}");
    let mo = res.poles.len();
    assert!((1..=3).contains(&mo));
    assert_eq!(res.weights.shape(), &[mo, n, n]);
    assert_eq!(res.factors.shape(), &[mo, n, n]);
    assert!(res.poles.iter().all(|p| p.im < 0.0));
    let omega = real_grid(2001, 5.0);
    let min_a = min_spectral(&res, n, &omega);
    assert!(min_a >= -1e-12, "negative spectral eigenvalue {min_a}");
    assert!(
        info.residual_rms <= info.target_rms * (1.0 + 1e-12),
        "{info:?}"
    );
    let near: Vec<C64> = omega.iter().map(|&w| c(w, 0.05)).collect();
    let truth = pole_sum(&tp, &tw, &near, x);
    let e_in = rel_l2(&pole_sum(&rp, &rw, &near, x), &truth);
    let e_out = rel_l2(res.evaluate(&near).unwrap().host_data().unwrap(), &truth);
    println!("relL2 input {e_in:.4} refined {e_out:.4}");
    assert!(e_out < 1.5 * e_in, "input {e_in} refined {e_out}");
}

#[test]
fn rank_one_factors() {
    let (tp, tw, rp, rw) = scalar_case();
    let _ = (tp, tw);
    let z = matsubara(20.0, 20);
    let d = pole_sum(&rp, &rw, &z, 1);
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d).unwrap();
    let weights = TypedTensor::from_vec_col_major(vec![rp.len()], rw).unwrap();
    let params = Params {
        eps: Some(1e-3),
        ..Default::default()
    };
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &params).unwrap();
    assert_eq!(res.factors.shape(), &[res.poles.len(), 1, 1]);
    assert_eq!(info.target_rms, 1e-3);
    assert!(info.residual_rms <= 1e-3 * (1.0 + 1e-12), "{info:?}");
}

#[test]
fn invalid_inputs_are_rejected() {
    let z = matsubara(10.0, 4);
    let values = TypedTensor::from_vec_col_major(vec![4], vec![c(0.0, -0.1); 4]).unwrap();
    let w1 = TypedTensor::from_vec_col_major(vec![1], vec![c(1.0, 0.0)]).unwrap();
    let ok = [c(0.0, -0.5)];
    let def = Params::default();
    let err = |r: Result<(CausalPoles, Info)>| r.err().unwrap();

    assert!(matches!(
        err(refine(&[], &w1, &z, &values, BETA, &def)),
        Error::EmptyInput { name: "poles" }
    ));
    assert!(matches!(
        err(refine(&ok, &w1, &[], &values, BETA, &def)),
        Error::EmptyInput { name: "z" }
    ));
    // On the line Im z = π / β
    assert!(matches!(
        err(refine(&[c(0.3, PI / 10.0)], &w1, &z, &values, 10.0, &def)),
        Error::InvalidParameter { name: "poles", .. }
    ));
    for beta in [0.0, -1.0, f64::INFINITY] {
        assert!(matches!(
            err(refine(&ok, &w1, &z, &values, beta, &def)),
            Error::InvalidParameter { name: "beta", .. }
        ));
    }
    let zbad = [c(0.0, 1.0), c(0.0, -1.0), c(0.0, 2.0), c(0.0, 3.0)];
    assert!(matches!(
        err(refine(&ok, &w1, &zbad, &values, BETA, &def)),
        Error::InvalidParameter { name: "z", .. }
    ));
    let w2 = TypedTensor::from_vec_col_major(vec![2], vec![c(1.0, 0.0); 2]).unwrap();
    assert!(matches!(
        err(refine(&ok, &w2, &z, &values, BETA, &def)),
        Error::ShapeMismatch { .. }
    ));
    let wm = TypedTensor::from_vec_col_major(vec![1, 2, 2], vec![c(1.0, 0.0); 4]).unwrap();
    assert!(matches!(
        err(refine(&ok, &wm, &z, &values, BETA, &def)),
        Error::ShapeMismatch { .. }
    ));
    let wnan = TypedTensor::from_vec_col_major(vec![1], vec![c(f64::NAN, 0.0)]).unwrap();
    assert!(matches!(
        err(refine(&ok, &wnan, &z, &values, BETA, &def)),
        Error::NonFiniteInput {
            name: "weights",
            ..
        }
    ));
    let w0 = TypedTensor::from_vec_col_major(vec![1], vec![c(0.0, 0.0)]).unwrap();
    assert!(matches!(
        err(refine(&ok, &w0, &z, &values, BETA, &def)),
        Error::InvalidParameter {
            name: "weights",
            ..
        }
    ));
    for (params, name) in [
        (
            Params {
                rank: Some(2),
                ..Default::default()
            },
            "rank",
        ),
        (
            Params {
                eps: Some(-1.0),
                ..Default::default()
            },
            "eps",
        ),
        (
            Params {
                discrepancy_rtol: 1.0,
                ..Default::default()
            },
            "discrepancy_rtol",
        ),
        (
            Params {
                max_solves: 0,
                ..Default::default()
            },
            "max_solves",
        ),
    ] {
        match err(refine(&ok, &w1, &z, &values, BETA, &params)) {
            Error::InvalidParameter { name: got, .. } => assert_eq!(got, name),
            e => panic!("{name}: {e:?}"),
        }
    }
}

#[test]
fn spurious_poles_are_pruned() {
    // Causal truth plus two tiny spurious poles, one non-causal.
    let (tp, tw, _, _) = scalar_case();
    let z = matsubara(BETA, 40);
    let d = pole_sum(&tp, &tw, &z, 1);
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d.clone()).unwrap();
    let mut rp = tp.clone();
    let mut rw = tw.clone();
    rp.extend([c(0.1, 0.05), c(-2.0, -0.3)]);
    rw.extend([c(1e-4, 2e-5), c(2e-4, 0.0)]);
    let weights = TypedTensor::from_vec_col_major(vec![rp.len()], rw).unwrap();
    let params = Params {
        prune: true,
        ..Default::default()
    };
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &params).unwrap();
    println!("{info:?} {:?}", res.poles);
    assert_eq!(info.pruned, 2, "{info:?}");
    assert_eq!(res.poles.len(), 3);
    assert!(info.residual_rms <= info.target_rms * (1.0 + 1e-12));
    for (p, q) in res.poles.iter().zip(&tp) {
        assert!((p - q).norm() < 1e-2, "{:?}", res.poles);
    }
    // Without pruning (the default) all poles are kept.
    let (res, info) = refine(&rp, &weights, &z, &values, BETA, &Params::default()).unwrap();
    assert_eq!(info.pruned, 0);
    assert_eq!(res.poles.len(), 5);
}

#[test]
fn bath_representation_on_the_real_axis() {
    // A discrete bath Δ(z) = Σ_b v_b v_b† / (z - ε_b) with real levels is a
    // valid input; the result is causal and fits the bath data.
    let n = 2;
    let x = n * n;
    let levels = [-1.0, -0.3, 0.4, 1.1];
    let v = [[0.4, 0.1], [0.2, -0.3], [0.1, 0.35], [-0.25, 0.2]];
    let mut bw = vec![c(0.0, 0.0); levels.len() * x];
    for (b, vb) in v.iter().enumerate() {
        for q in 0..n {
            for p in 0..n {
                bw[b + levels.len() * (p + n * q)] = c(vb[p] * vb[q], 0.0);
            }
        }
    }
    let bp: Vec<C64> = levels.iter().map(|&e| c(e, 0.0)).collect();
    let z = matsubara(BETA, 40);
    let d = pole_sum(&bp, &bw, &z, x);
    let values = TypedTensor::from_vec_col_major(vec![z.len(), n, n], d.clone()).unwrap();
    let weights = TypedTensor::from_vec_col_major(vec![levels.len(), n, n], bw).unwrap();
    let params = Params {
        eps: Some(1e-6),
        ..Default::default()
    };
    let (res, info) = refine(&bp, &weights, &z, &values, BETA, &params).unwrap();
    println!("{info:?} {:?}", res.poles);
    assert_ne!(info.status, Status::DiscrepancyNotReached, "{info:?}");
    assert!(info.residual_rms <= 1e-6 * (1.0 + 1e-12), "{info:?}");
    assert!(res.poles.iter().all(|p| p.im < 0.0));
    let min_a = min_spectral(&res, n, &real_grid(2001, 3.0));
    assert!(min_a >= -1e-12, "negative spectral eigenvalue {min_a}");
    // Each level is reproduced by a sharp pole with weight v_b v_b†.
    let w = res.weights.host_data().unwrap();
    let m = res.poles.len();
    for (b, &e) in levels.iter().enumerate() {
        let j = (0..m)
            .min_by(|&i, &k| {
                (res.poles[i].re - e)
                    .abs()
                    .total_cmp(&(res.poles[k].re - e).abs())
            })
            .unwrap();
        assert!((res.poles[j].re - e).abs() < 1e-3, "{:?}", res.poles);
        assert!(res.poles[j].im > -1e-3, "{:?}", res.poles);
        for q in 0..n {
            for p in 0..n {
                let want = v[b][p] * v[b][q];
                assert!((w[j + m * (p + n * q)] - want).norm() < 1e-3, "{w:?}");
            }
        }
    }
}

/// Matrix bath data `Σ_b v_b v_b† / (z - ε_b)` on the Matsubara grid.
fn bath_data(levels: &[f64], v: &[[f64; 2]], z: &[C64]) -> Vec<C64> {
    let (n, mb) = (2, levels.len());
    let mut bw = vec![c(0.0, 0.0); mb * n * n];
    for (b, vb) in v.iter().enumerate() {
        for q in 0..n {
            for p in 0..n {
                bw[b + mb * (p + n * q)] = c(vb[p] * vb[q], 0.0);
            }
        }
    }
    let bp: Vec<C64> = levels.iter().map(|&e| c(e, 0.0)).collect();
    pole_sum(&bp, &bw, z, n * n)
}

#[test]
fn scalar_fit_from_levels() {
    // Lorentzian-broadened truth fitted from an equidistant level grid.
    let (tp, tw, _, _) = scalar_case();
    let z = matsubara(BETA, 40);
    let d = pole_sum(&tp, &tw, &z, 1);
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d.clone()).unwrap();
    let levels = real_grid(9, 2.0);
    let (res, info) = fit(&levels, &z, &values, BETA, &Params::default()).unwrap();
    println!("{info:?} {:?}", res.poles);
    assert_ne!(info.status, Status::DiscrepancyNotReached, "{info:?}");
    assert_eq!(res.poles.len(), levels.len());
    assert!(res.poles.iter().all(|p| p.im < 0.0));
    let omega = real_grid(2001, 5.0);
    assert!(min_spectral(&res, 1, &omega) >= -1e-12);
    // Default ε: (1 + τ) times the least-squares residual.
    assert_eq!(info.target_rms, 1.1 * info.input_residual_rms);
    assert!(
        info.residual_rms <= info.target_rms * (1.0 + 1e-12),
        "{info:?}"
    );
    assert!(info.residual_rms >= 0.9 * info.target_rms, "{info:?}");
    // The data residual agrees with the returned function and is small.
    let r = rel_l2(res.evaluate(&z).unwrap().host_data().unwrap(), &d);
    assert!(r < 1e-3, "relative residual {r}");
    // Fewer levels with pruning, at the same bound.
    let params = Params {
        prune: true,
        ..Default::default()
    };
    let (res, info) = fit(&levels, &z, &values, BETA, &params).unwrap();
    println!("{info:?} {:?}", res.poles);
    assert!(info.pruned > 0, "{info:?}");
    assert_eq!(res.poles.len(), levels.len() - info.pruned);
    assert!(info.residual_rms <= info.target_rms * (1.0 + 1e-12));
    assert!(min_spectral(&res, 1, &omega) >= -1e-12);
}

#[test]
fn matrix_bath_fit_from_levels() {
    // A 2 x 2 discrete bath fitted from a grid containing its levels.
    let n = 2;
    let truth = [-1.0, 0.0, 1.0];
    let v = [[0.4, 0.1], [0.2, -0.3], [-0.25, 0.2]];
    let z = matsubara(BETA, 40);
    let d = bath_data(&truth, &v, &z);
    let values = TypedTensor::from_vec_col_major(vec![z.len(), n, n], d.clone()).unwrap();
    let levels = real_grid(5, 1.0);
    let params = Params {
        eps: Some(1e-6),
        ..Default::default()
    };
    let (res, info) = fit(&levels, &z, &values, BETA, &params).unwrap();
    println!("{info:?} {:?}", res.poles);
    assert_ne!(info.status, Status::DiscrepancyNotReached, "{info:?}");
    assert!(info.residual_rms <= 1e-6 * (1.0 + 1e-12), "{info:?}");
    assert!(res.poles.iter().all(|p| p.im < 0.0));
    let min_a = min_spectral(&res, n, &real_grid(2001, 3.0));
    assert!(min_a >= -1e-12, "negative spectral eigenvalue {min_a}");
    let r = rel_l2(res.evaluate(&z).unwrap().host_data().unwrap(), &d);
    assert!(r < 1e-4, "relative residual {r}");
}

#[test]
fn invalid_fit_inputs_are_rejected() {
    let z = matsubara(BETA, 10);
    let d: Vec<C64> = z.iter().map(|&z| 1.0 / (z - 0.2)).collect();
    let values = TypedTensor::from_vec_col_major(vec![z.len()], d.clone()).unwrap();
    let p = Params::default();
    let levels = [-0.5, 0.5];
    assert!(matches!(
        fit(&[], &z, &values, BETA, &p),
        Err(Error::EmptyInput { name: "levels" })
    ));
    assert!(matches!(
        fit(&[0.0, f64::NAN], &z, &values, BETA, &p),
        Err(Error::NonFiniteInput { name: "levels", .. })
    ));
    let mut bad = d.clone();
    bad[3] = c(f64::INFINITY, 0.0);
    let bad = TypedTensor::from_vec_col_major(vec![z.len()], bad).unwrap();
    assert!(matches!(
        fit(&levels, &z, &bad, BETA, &p),
        Err(Error::NonFiniteInput { name: "values", .. })
    ));
    assert!(matches!(
        fit(&levels, &z, &values, -1.0, &p),
        Err(Error::InvalidParameter { name: "beta", .. })
    ));
    let zero = TypedTensor::from_vec_col_major(vec![z.len()], vec![c(0.0, 0.0); z.len()]).unwrap();
    assert!(matches!(
        fit(&levels, &z, &zero, BETA, &p),
        Err(Error::InvalidParameter { name: "values", .. })
    ));
    let q = Params {
        rank: Some(2),
        ..Default::default()
    };
    assert!(matches!(
        fit(&levels, &z, &values, BETA, &q),
        Err(Error::InvalidParameter { name: "rank", .. })
    ));
}
