use super::*;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn test_nodes() -> Vec<C64> {
    vec![
        C64::from_polar(0.95, 0.3),
        C64::from_polar(0.8, -1.1),
        C64::from_polar(0.6, 2.0),
        c(0.4, 0.0),
    ]
}

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

fn assert_nodes_close(got: &[C64], want: &[C64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for w in want {
        let best = got
            .iter()
            .map(|g| (g - w).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(best < tol, "node {w} not recovered: best distance {best:e}");
    }
}

/// Deterministic pseudo-random noise in `[-1, 1]`.
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
    let amps = vec![c(1.0, 0.5), c(-0.3, 0.2), c(0.7, 0.0), c(0.1, -0.4)];
    let y = signal(&nodes, &amps, 40);
    let res = esprit_scalar(&y, &EspritOptions::fixed(4)).unwrap();
    assert_nodes_close(&res.nodes, &nodes, 1e-10);
    assert!(res.diagnostics.max_residual < 1e-12);
    assert!(res.diagnostics.truncation < 1e-12);

    // Amplitudes follow the node ordering.
    let a = res.amplitudes.host_data().unwrap();
    for (j, z) in res.nodes.iter().enumerate() {
        let i = nodes.iter().position(|w| (w - z).norm() < 1e-8).unwrap();
        assert!((a[j] - amps[i]).norm() < 1e-9);
    }

    // Evaluation extrapolates beyond the sampled range.
    let ev = res.evaluate(&[45.0, 2.5]).unwrap();
    let ev = ev.host_data().unwrap();
    let exact = signal(&nodes, &amps, 46)[45];
    assert!((ev[0] - exact).norm() < 1e-10);
    let half: C64 = nodes.iter().zip(&amps).map(|(z, a)| a * z.powf(2.5)).sum();
    assert!((ev[1] - half).norm() < 1e-10);
}

#[test]
fn tolerance_selects_model_order() {
    let nodes = test_nodes();
    let amps = vec![c(1.0, 0.0); 4];
    let y = signal(&nodes, &amps, 50);
    let res = esprit_scalar(&y, &EspritOptions::tolerance(1e-10)).unwrap();
    assert_eq!(res.diagnostics.order, 4);
    assert_nodes_close(&res.nodes, &nodes, 1e-8);

    let capped = esprit_scalar(&y, &EspritOptions::tolerance(1e-10).with_max_order(2)).unwrap();
    assert_eq!(capped.nodes.len(), 2);
    assert!(capped.diagnostics.relative_residual > 1e-6);

    // All-zero data gives an empty model.
    let zero = esprit_scalar(&[c(0.0, 0.0); 10], &EspritOptions::tolerance(1e-8)).unwrap();
    assert!(zero.nodes.is_empty());
    assert_eq!(zero.amplitudes.shape(), &[0]);
}

#[test]
fn noisy_samples_with_tolerance() {
    let nodes = test_nodes();
    let amps = vec![c(1.0, 0.0), c(0.5, 0.5), c(-0.8, 0.0), c(0.3, 0.0)];
    let n = 80;
    let eta = 1e-9;
    let (nr, ni) = (noise(n, 3), noise(n, 4));
    let y: Vec<C64> = signal(&nodes, &amps, n)
        .into_iter()
        .enumerate()
        .map(|(k, v)| v + c(eta * nr[k], eta * ni[k]))
        .collect();
    let res = esprit_scalar(&y, &EspritOptions::tolerance(1e-6)).unwrap();
    assert_eq!(res.diagnostics.order, 4);
    assert_nodes_close(&res.nodes, &nodes, 1e-6);
    assert!(res.diagnostics.max_residual < 10.0 * eta);
}

#[test]
fn matrix_valued_samples_share_nodes() {
    let nodes = test_nodes();
    let n = 30;
    // 2x2 channels, each a different combination of the same nodes; one
    // channel alone does not contain every node.
    let amp_sets = [
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.5, 0.0), c(0.0, 0.0)],
        vec![c(0.0, 0.0), c(1.0, 1.0), c(0.0, 0.0), c(0.2, 0.0)],
        vec![c(0.3, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        vec![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0), c(0.4, 0.0)],
    ];
    let mut data = Vec::new();
    for a in &amp_sets {
        data.extend(signal(&nodes, a, n));
    }
    let y = TypedTensor::from_vec_col_major(vec![n, 2, 2], data).unwrap();
    let res = esprit(&y, &EspritOptions::tolerance(1e-10)).unwrap();
    assert_eq!(res.diagnostics.order, 4);
    assert_eq!(res.amplitudes.shape(), &[4, 2, 2]);
    assert_nodes_close(&res.nodes, &nodes, 1e-9);
    assert!(res.diagnostics.max_residual < 1e-11);
    let ev = res.evaluate(&[3.0]).unwrap();
    assert_eq!(ev.shape(), &[1, 2, 2]);
    let ev = ev.host_data().unwrap();
    for (ch, a) in amp_sets.iter().enumerate() {
        assert!((ev[ch] - signal(&nodes, a, 4)[3]).norm() < 1e-11);
    }
}

#[test]
fn invalid_input_is_rejected() {
    let y = vec![c(1.0, 0.0); 10];
    assert!(esprit_scalar(&y[..1], &EspritOptions::fixed(1)).is_err());
    assert!(esprit_scalar(&y, &EspritOptions::fixed(10)).is_err());
    assert!(esprit_scalar(&y, &EspritOptions::fixed(1).with_pencil(11)).is_err());
    assert!(esprit_scalar(&y, &EspritOptions::tolerance(f64::NAN)).is_err());
}
