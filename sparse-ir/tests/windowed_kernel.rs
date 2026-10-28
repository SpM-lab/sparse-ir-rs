//! Tests for the asymmetric-window logistic kernel (a non-centrosymmetric
//! kernel) through the kernel, SVE, basis and DLR layers.

use sparse_ir::kernel::{
    AbstractKernel, KernelProperties, LogisticKernel, LogisticKernelWindowed, SVEKernel,
};
use sparse_ir::sve::compute_sve;
use sparse_ir::traits::{Bosonic, Fermionic};
use sparse_ir::{
    Basis, DiscreteLehmannRepresentation, DlrFromIr, Error, FiniteTempBasis, TworkType,
};

fn windowed(beta: f64, omega_min: f64, omega_max: f64) -> LogisticKernelWindowed {
    LogisticKernelWindowed::from_physical(beta, omega_min, omega_max).unwrap()
}

#[test]
fn symmetric_kernel_has_symmetric_default_domain() {
    let k = LogisticKernel::new(10.0).unwrap();
    assert_eq!(k.xmin(), -k.xmax());
    assert_eq!(k.ymin(), -k.ymax());
}

#[test]
fn windowed_kernel_dimensionless_params() {
    // beta = 10, window [-0.2, 1.0] => wmax = 1.0, lambda = 10
    let k = windowed(10.0, -0.2, 1.0);
    assert_eq!(k.lambda(), 10.0);
    assert_eq!(k.ymax(), 1.0);
    assert_eq!(k.ymin(), -0.2);
    assert_eq!((k.xmin(), k.xmax()), (-1.0, 1.0));

    // omega_min-dominant window: wmax = |omega_min|
    let k = windowed(20.0, -1.0, 0.3);
    assert_eq!(k.lambda(), 20.0);
    assert_eq!((k.ymin(), k.ymax()), (-1.0, 0.3));
}

#[test]
fn windowed_kernel_formula_matches_logistic() {
    let base = LogisticKernel::new(10.0).unwrap();
    let win = windowed(10.0, -0.2, 1.0);
    for &(x, y) in &[(0.3_f64, 0.5_f64), (-0.7, 0.9), (0.1, -0.15)] {
        let a: f64 = base.compute(x, y);
        let b: f64 = win.compute(x, y);
        assert_eq!(a, b, "mismatch at ({x},{y})");
    }
}

#[test]
fn windowed_kernel_regularizer_matches_logistic() {
    let base = LogisticKernel::new(10.0).unwrap();
    let win = windowed(10.0, -0.2, 1.0);
    for omega in [-0.2, 0.0, 0.3, 1.0] {
        assert_eq!(
            win.regularizer::<Fermionic>(10.0, omega),
            base.regularizer::<Fermionic>(10.0, omega)
        );
        assert_eq!(
            win.regularizer::<Bosonic>(10.0, omega),
            base.regularizer::<Bosonic>(10.0, omega)
        );
    }
}

#[test]
fn windowed_kernel_rejects_invalid_parameters() {
    for (beta, omega_min, omega_max) in [
        (10.0, 1.0, 1.0),      // degenerate window
        (10.0, 1.0, 0.5),      // inverted window
        (10.0, 0.0, 0.0),      // degenerate at 0
        (0.0, -1.0, 1.0),      // beta not positive
        (f64::NAN, -1.0, 1.0), // beta not finite
        (10.0, f64::NEG_INFINITY, 1.0),
        (10.0, -1.0, f64::NAN),
    ] {
        let err = LogisticKernelWindowed::from_physical(beta, omega_min, omega_max).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { .. }),
            "({beta}, {omega_min}, {omega_max}): {err:?}"
        );
    }
}

/// The window need not contain 0: Λ = β·max(|ωmin|, |ωmax|) still holds.
#[test]
fn one_sided_windows_are_accepted() {
    for (omega_min, omega_max, lambda, ymin, ymax) in [
        (0.0, 2.0, 20.0, 0.0, 1.0),
        (0.5, 2.0, 20.0, 0.25, 1.0),
        (-2.0, 0.0, 20.0, -1.0, 0.0),
        (-2.0, -0.5, 20.0, -1.0, -0.25),
    ] {
        let k = windowed(10.0, omega_min, omega_max);
        assert_eq!(k.lambda(), lambda, "[{omega_min}, {omega_max}]");
        assert_eq!(
            (k.ymin(), k.ymax()),
            (ymin, ymax),
            "[{omega_min}, {omega_max}]"
        );
    }
}

/// The truncated SVE of a one-sided window reproduces the kernel on it
/// to the accuracy, and the windows [0, 1] and [-1, 0], which the kernel's
/// symmetry K(-x, -y) = K(x, y) maps onto each other, have the same spectrum.
#[test]
fn one_sided_window_sve_reproduces_kernel() {
    let eps = 1e-10;
    let sve = |omega_min, omega_max| {
        compute_sve(
            windowed(100.0, omega_min, omega_max),
            Some(eps),
            None,
            None,
            TworkType::Auto,
        )
        .unwrap()
    };
    for (omega_min, omega_max) in [(0.0, 1.0), (0.2, 1.0), (-1.0, -0.3)] {
        let kernel = windowed(100.0, omega_min, omega_max);
        let result = sve(omega_min, omega_max);
        let s = result.s();
        let xs: Vec<f64> = (0..=40).map(|i| -1.0 + i as f64 / 20.0).collect();
        let (ymin, ymax) = (kernel.ymin(), kernel.ymax());
        let ys: Vec<f64> = (0..=40)
            .map(|i| ymin + (ymax - ymin) * i as f64 / 40.0)
            .collect();
        let u = result.u().evaluate_at_many(&xs);
        let v = result.v().evaluate_at_many(&ys);
        let mut max_err = 0.0_f64;
        for (i, &x) in xs.iter().enumerate() {
            for (j, &y) in ys.iter().enumerate() {
                let approx: f64 = (0..s.len()).map(|l| u[[l, i]] * s[l] * v[[l, j]]).sum();
                let exact: f64 = kernel.compute(x, y);
                max_err = max_err.max((approx - exact).abs());
            }
        }
        assert!(
            max_err < 10.0 * eps * s[0],
            "[{omega_min}, {omega_max}]: {} singular values, error {max_err:e}",
            s.len()
        );
    }

    let (pos, neg) = (sve(0.0, 1.0), sve(-1.0, 0.0));
    assert_eq!(pos.s().len(), neg.s().len());
    for (l, (a, b)) in pos.s().iter().zip(neg.s()).enumerate() {
        assert!((a - b).abs() < 1e-12 * pos.s()[0], "s[{l}]: {a} vs {b}");
    }
}

/// A basis on [0, ωmax] samples and fits in τ and Matsubara frequencies, and
/// its bosonic DLR reproduces it, with the regularizer vanishing at the edge
/// ω = 0.
#[test]
fn one_sided_window_basis_round_trips() {
    use num_complex::Complex;
    use sparse_ir::{MatsubaraSampling, TauSampling};

    let beta = 10.0;
    let kernel = windowed(beta, 0.0, 1.0);
    let fermionic = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(1e-10), None).unwrap();
    let bosonic = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, Some(1e-10), None).unwrap();
    assert_eq!((bosonic.omega_min(), bosonic.omega_max()), (0.0, 1.0));

    let g: Vec<f64> = (0..fermionic.size())
        .map(|i| 1.0 / (i as f64 + 1.0))
        .collect();
    let tau = TauSampling::new(&fermionic).unwrap();
    let g_tau = tau.fit(&tau.evaluate(&g).unwrap()).unwrap();
    let matsu = MatsubaraSampling::new(&fermionic).unwrap();
    let gc: Vec<Complex<f64>> = g.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let g_matsu = matsu.fit(&matsu.evaluate(&gc).unwrap()).unwrap();
    for i in 0..g.len() {
        assert!((g[i] - g_tau[i]).abs() < 1e-8, "tau: {i}");
        assert!((gc[i] - g_matsu[i]).norm() < 1e-8, "matsubara: {i}");
    }

    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&bosonic).unwrap();
    assert!(
        dlr.poles().iter().all(|&w| 0.0 < w && w < 1.0),
        "{:?}",
        dlr.poles()
    );
    let taus = [0.1, 2.5, 7.3, 9.9];
    let u = bosonic.evaluate_tau(&taus).unwrap();
    let dlr_tau = dlr.evaluate_tau(&taus).unwrap();
    let s = bosonic.svals();
    for (p, &pole) in dlr.poles().iter().enumerate() {
        let v = bosonic.evaluate_omega(&[pole]).unwrap();
        for (t, tau) in taus.iter().enumerate() {
            let ir: f64 = (0..bosonic.size())
                .map(|l| -s[l] * *v.get(&[0, l]).unwrap() * *u.get(&[t, l]).unwrap())
                .sum();
            let err = (ir - *dlr_tau.get(&[t, p]).unwrap()).abs();
            assert!(err < 1e-8, "pole {pole}, tau {tau}: error {err:e}");
        }
    }
}

/// A symmetric window goes through the full-domain SVE strategy but must give
/// the same spectrum as the centrosymmetric strategy for the logistic kernel.
#[test]
fn symmetric_window_matches_centrosymmetric_sve() {
    let eps = Some(1e-10);
    let sym = compute_sve(
        LogisticKernel::new(10.0).unwrap(),
        eps,
        None,
        None,
        TworkType::Auto,
    )
    .unwrap();
    let win = compute_sve(windowed(10.0, -1.0, 1.0), eps, None, None, TworkType::Auto).unwrap();
    assert_eq!(sym.s().len(), win.s().len());
    for (i, (a, b)) in sym.s().iter().zip(win.s()).enumerate() {
        assert!((a - b).abs() < 1e-12 * sym.s()[0], "s[{i}]: {a} vs {b}");
    }
}

#[test]
fn asymmetric_window_singular_values_strictly_decrease() {
    let win = compute_sve(
        windowed(10.0, -0.2, 1.0),
        Some(1e-10),
        None,
        None,
        TworkType::Auto,
    )
    .unwrap();
    assert!(win.s()[0] > 0.0);
    for i in 1..win.s().len() {
        assert!(
            win.s()[i] > 0.0 && win.s()[i] < win.s()[i - 1],
            "not decreasing at {i}"
        );
    }
}

#[test]
fn windowed_basis_maps_to_physical_window() {
    // omega_max-dominant and omega_min-dominant windows
    for &(beta, wmin, wmax) in &[(10.0, -0.2, 1.0), (20.0, -1.0, 0.3)] {
        let b = FiniteTempBasis::<_, Fermionic>::new(
            windowed(beta, wmin, wmax),
            beta,
            Some(1e-10),
            None,
        )
        .unwrap();

        assert!(b.size() > 5);
        assert!(
            (b.omega_min() - wmin).abs() < 1e-12,
            "omega_min = {}",
            b.omega_min()
        );
        assert!(
            (b.omega_max() - wmax).abs() < 1e-12,
            "omega_max = {}",
            b.omega_max()
        );

        // v lives on the asymmetric window, not on [-wmax, wmax].
        let v0 = &b.v().get_polys()[0];
        assert_eq!(
            (v0.get_xmin(), v0.get_xmax()),
            (b.omega_min(), b.omega_max())
        );

        // The default omega points (the DLR poles) lie inside the window.
        let omegas = b.default_omega_sampling_points().unwrap();
        assert!(omegas.iter().all(|&w| wmin < w && w < wmax), "{omegas:?}");
    }
}

/// The bosonic DLR of a windowed basis reproduces its IR basis functions:
/// `G_p(τ) = -Σ_l s_l V_l(ω_p) U_l(τ)` for each pole `ω_p`.
#[test]
fn windowed_bosonic_dlr_matches_ir() {
    let beta = 10.0;
    let basis =
        FiniteTempBasis::<_, Bosonic>::new(windowed(beta, -0.5, 1.0), beta, Some(1e-10), None)
            .unwrap();
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::from_ir(&basis).unwrap();
    let taus = [0.1, 2.5, 7.3, 9.9];
    let u = basis.evaluate_tau(&taus).unwrap();
    let dlr_tau = dlr.evaluate_tau(&taus).unwrap();
    let s = basis.svals();
    for (p, &pole) in dlr.poles().iter().enumerate() {
        let v = basis.evaluate_omega(&[pole]).unwrap();
        for (t, tau) in taus.iter().enumerate() {
            let ir: f64 = (0..basis.size())
                .map(|l| -s[l] * *v.get(&[0, l]).unwrap() * *u.get(&[t, l]).unwrap())
                .sum();
            let err = (ir - *dlr_tau.get(&[t, p]).unwrap()).abs();
            assert!(err < 1e-8, "pole {pole}, tau {tau}: error {err:e}");
        }
    }
}

/// A DLR pole inside [-wmax, wmax] but outside the window [ωmin, ωmax] is
/// reported as out of the basis's domain, under the name `poles`.
#[test]
fn windowed_dlr_rejects_pole_outside_window() {
    let beta = 10.0;
    let basis =
        FiniteTempBasis::<_, Fermionic>::new(windowed(beta, -0.5, 1.0), beta, Some(1e-6), None)
            .unwrap();
    let err =
        DiscreteLehmannRepresentation::<Fermionic>::from_ir_with_poles(&basis, vec![-0.8, 0.2])
            .err()
            .unwrap();
    assert!(
        matches!(err, Error::OutOfDomain { name: "poles", value, domain }
            if value == -0.8 && domain == (-0.5, 1.0)),
        "{err:?}"
    );
}

fn windowed_basis() -> FiniteTempBasis<LogisticKernelWindowed, Fermionic> {
    FiniteTempBasis::new(windowed(20.0, -0.2, 1.0), 20.0, Some(1e-10), None).unwrap()
}

/// The default tau sampling points do not assume centrosymmetry.
#[test]
fn windowed_tau_round_trip() {
    use sparse_ir::TauSampling;

    let basis = windowed_basis();
    let sampling = TauSampling::new(&basis).unwrap();
    let g: Vec<f64> = (0..basis.size()).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let g_rec = sampling.fit(&sampling.evaluate(&g).unwrap()).unwrap();
    for (i, (a, b)) in g.iter().zip(&g_rec).enumerate() {
        assert!((a - b).abs() < 1e-8, "tau mismatch at {i}: {a} vs {b}");
    }
}

/// Neither do the default Matsubara sampling points.
#[test]
fn windowed_matsubara_round_trip() {
    use num_complex::Complex;
    use sparse_ir::MatsubaraSampling;

    let basis = windowed_basis();
    let sampling = MatsubaraSampling::new(&basis).unwrap();
    let g: Vec<Complex<f64>> = (0..basis.size())
        .map(|i| Complex::new(1.0 / (i as f64 + 1.0), 0.2 * i as f64))
        .collect();
    let g_rec = sampling.fit(&sampling.evaluate(&g).unwrap()).unwrap();
    for (i, (a, b)) in g.iter().zip(&g_rec).enumerate() {
        assert!(
            (a - b).norm() < 1e-8,
            "matsubara mismatch at {i}: {a} vs {b}"
        );
    }
}
