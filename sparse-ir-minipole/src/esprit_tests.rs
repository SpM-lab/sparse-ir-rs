use super::*;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn test_nodes() -> Vec<C64> {
    vec![
        C64::from_polar(0.95, 0.4),
        C64::from_polar(0.8, -1.3),
        c(0.6, 0.0),
    ]
}

/// `y_k = Σ_j a_j z_j^k`, `k < n`.
fn signal(nodes: &[C64], amps: &[C64], n: usize) -> Vec<C64> {
    (0..n)
        .map(|k| {
            nodes
                .iter()
                .zip(amps)
                .map(|(z, a)| a * z.powi(k as i32))
                .sum()
        })
        .collect()
}

/// Every wanted node has a found node within `tol`.
fn assert_nodes_close(got: &[C64], want: &[C64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{got:?}");
    for w in want {
        let d = got
            .iter()
            .map(|g| (g - w).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(d < tol, "node {w} not found in {got:?}");
    }
}

/// Deterministic uniform noise in [-1, 1].
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
        })
        .collect()
}

#[test]
fn scalar_fixed_order_recovers_nodes_and_amplitudes() {
    let nodes = test_nodes();
    let amps = vec![c(1.0, 0.0), c(-0.5, 0.2), c(0.0, 0.3)];
    let h = signal(&nodes, &amps, 60);
    let e = Esprit::new(
        &h,
        60,
        1,
        &EspritParams {
            m: Some(3),
            ..EspritParams::default()
        },
    )
    .unwrap();
    assert_eq!(e.m, 3);
    assert_nodes_close(&e.gamma, &nodes, 1e-10);
    for (z, a) in nodes.iter().zip(&amps) {
        let j = (0..3)
            .min_by(|&i, &k| (e.gamma[i] - z).norm().total_cmp(&(e.gamma[k] - z).norm()))
            .unwrap();
        assert!((e.omega[j] - a).norm() < 1e-10);
    }
    assert!(e.err_max < 1e-12);
    // The approximation interpolates between the samples.
    let x = 0.5 / 59.0; // halfway between samples 0 and 1
    let want: C64 = nodes.iter().zip(&amps).map(|(z, a)| a * z.powf(0.5)).sum();
    assert!((e.get_value_indiv(x, 0) - want).norm() < 1e-10);
}

#[test]
fn tolerance_selects_model_order() {
    let nodes = test_nodes();
    let amps = vec![c(1.0, 0.0), c(1e-3, 0.0), c(1e-6, 0.0)];
    let h = signal(&nodes, &amps, 60);
    for (err, err_type, m) in [
        (1e-9, ErrType::Abs, 3),
        (1e-5, ErrType::Abs, 2),
        (1e-2, ErrType::Abs, 1),
        (1e-9, ErrType::Rel, 3),
    ] {
        let p = EspritParams {
            err: Some(err),
            err_type,
            ..EspritParams::default()
        };
        let e = Esprit::new(&h, 60, 1, &p).unwrap();
        assert_eq!(e.m, m, "err = {err}, {err_type:?}");
        // First discarded singular value below the cutoff, the kept ones above.
        let cutoff = if err_type == ErrType::Abs {
            err
        } else {
            err * e.s[0]
        };
        assert!(e.sigma < cutoff && e.s[m - 1] >= cutoff);
    }
}

#[test]
fn noisy_samples_with_tolerance() {
    let nodes = test_nodes();
    let amps = vec![c(1.0, 0.0), c(-0.5, 0.2), c(0.0, 0.3)];
    let eta = 1e-8;
    let (nr, ni) = (noise(60, 1), noise(60, 2));
    let h: Vec<C64> = signal(&nodes, &amps, 60)
        .iter()
        .enumerate()
        .map(|(k, v)| v + c(eta * nr[k], eta * ni[k]))
        .collect();
    let p = EspritParams {
        err: Some(1e-6),
        ..EspritParams::default()
    };
    let e = Esprit::new(&h, 60, 1, &p).unwrap();
    assert_nodes_close(&e.gamma, &nodes, 1e-6);
    assert!(e.err_max < 1e-7);
}

#[test]
fn matrix_valued_samples_share_nodes() {
    let nodes = test_nodes();
    let n = 60;
    let a0 = [c(1.0, 0.0), c(0.0, 0.0), c(0.3, 0.0)];
    let a1 = [c(0.0, 0.0), c(0.5, -0.1), c(0.2, 0.0)];
    let mut h = signal(&nodes, &a0, n);
    h.extend(signal(&nodes, &a1, n));
    let p = EspritParams {
        err: Some(1e-10),
        ..EspritParams::default()
    };
    let e = Esprit::new(&h, n, 2, &p).unwrap();
    // Neither column alone has all three nodes.
    assert_eq!(e.m, 3);
    assert_nodes_close(&e.gamma, &nodes, 1e-9);
    let approx = e.get_value(&linspace(0.0, 1.0, n));
    let dev = approx
        .iter()
        .zip(&h)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(dev < 1e-10);
}

#[test]
fn invalid_input_is_rejected() {
    let h = signal(&test_nodes(), &[c(1.0, 0.0); 3], 20);
    // Neither err nor M.
    assert!(Esprit::new(&h, 20, 1, &EspritParams::default()).is_err());
    // Wrong length, L too large, empty interval.
    let p = EspritParams {
        m: Some(2),
        ..EspritParams::default()
    };
    assert!(Esprit::new(&h, 21, 1, &p).is_err());
    assert!(
        Esprit::new(
            &h,
            20,
            1,
            &EspritParams {
                lfactor: 0.7,
                ..p.clone()
            }
        )
        .is_err()
    );
    assert!(
        Esprit::new(
            &h,
            20,
            1,
            &EspritParams {
                x_min: 1.0,
                x_max: 1.0,
                ..p.clone()
            }
        )
        .is_err()
    );
    // Zero input gives an empty approximation.
    let e = Esprit::new(&[c(0.0, 0.0); 20], 20, 1, &p).unwrap();
    assert!(e.gamma.is_empty() && e.err_max == 0.0);
}
