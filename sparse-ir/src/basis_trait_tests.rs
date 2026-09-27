//! Tests for Basis trait

use crate::{Basis, Bosonic, Fermionic, FiniteTempBasis, LogisticKernel};

#[test]
fn test_basis_trait_fermionic() {
    let beta = 1000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Test Basis trait methods
    assert_eq!(basis.beta(), beta);
    assert_eq!(basis.wmax(), wmax);
    assert_eq!(basis.lambda(), beta * wmax);
    assert!(basis.size() > 0);
    assert!(basis.accuracy() > 0.0);
    assert_eq!(basis.significance().len(), basis.size());

    // First significance should be 1.0
    assert!((basis.significance()[0] - 1.0).abs() < 1e-10);

    // Significance should be monotonically decreasing
    let sig = basis.significance();
    for i in 1..sig.len() {
        assert!(sig[i] <= sig[i - 1], "Significance should decrease");
    }

    // Test sampling points
    let tau_points = basis.default_tau_sampling_points().unwrap();
    assert_eq!(tau_points.len(), basis.size());

    let matsubara_points = basis.default_matsubara_sampling_points(false).unwrap();
    assert!(!matsubara_points.is_empty());
}

#[test]
fn test_basis_trait_omega_sampling() {
    let beta = 1000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Test omega sampling via Basis trait
    let omega_points = basis.default_omega_sampling_points().unwrap();
    assert_eq!(omega_points.len(), basis.size());

    // All points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(omega.abs() <= wmax + 1e-10);
    }
}

#[test]
fn test_basis_trait_generic() {
    // Test that Basis trait works with generic code
    fn check_basis<S: crate::StatisticsType + 'static>(basis: &impl Basis<S>) {
        assert!(basis.beta() > 0.0);
        assert!(basis.wmax() > 0.0);
        assert!(basis.size() > 0);
        assert_eq!(basis.significance().len(), basis.size());
    }

    let beta = 500.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    let kernel_f = LogisticKernel::new(beta * wmax).unwrap();
    let basis_f =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel_f, beta, Some(epsilon), None)
            .unwrap();
    check_basis(&basis_f);

    let kernel_b = LogisticKernel::new(beta * wmax).unwrap();
    let basis_b =
        FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel_b, beta, Some(epsilon), None)
            .unwrap();
    check_basis(&basis_b);
}

#[test]
fn test_basis_trait_evaluate_tau() {
    let beta = 100.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Test evaluate_tau via Basis trait
    let tau_points = vec![0.0, beta / 4.0, beta / 2.0, 3.0 * beta / 4.0, beta];
    let matrix = basis.evaluate_tau(&tau_points).unwrap();

    // Check shape
    assert_eq!(*matrix.shape(), (tau_points.len(), basis.size()));

    // Values should be finite
    for i in 0..tau_points.len() {
        for l in 0..basis.size() {
            assert!(
                matrix[[i, l]].is_finite(),
                "matrix[{}, {}] = {}",
                i,
                l,
                matrix[[i, l]]
            );
        }
    }
}

#[test]
fn test_basis_trait_evaluate_matsubara() {
    use crate::MatsubaraFreq;

    let beta = 100.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None)
            .unwrap();

    // Test evaluate_matsubara via Basis trait
    // For fermions, n must be odd
    let freqs: Vec<MatsubaraFreq<Fermionic>> = vec![
        MatsubaraFreq::new(1).unwrap(),
        MatsubaraFreq::new(3).unwrap(),
        MatsubaraFreq::new(5).unwrap(),
        MatsubaraFreq::new(-1).unwrap(),
    ];

    let matrix = basis.evaluate_matsubara(&freqs).unwrap();

    // Check shape
    assert_eq!(*matrix.shape(), (freqs.len(), basis.size()));

    // Values should be finite
    for i in 0..freqs.len() {
        for l in 0..basis.size() {
            let val = matrix[[i, l]];
            assert!(
                val.re.is_finite() && val.im.is_finite(),
                "matrix[{}, {}] = {:?}",
                i,
                l,
                val
            );
        }
    }
}

/// Basis::evaluate_* reject points outside their domain and NaN (they
/// panicked), and return an empty matrix of the right width for no points
/// (from_fn ran its closure for a zero extent, mdarray#21, and panicked).
#[test]
fn test_basis_evaluate_checks_the_points() {
    use crate::error::Error;
    use crate::freq::MatsubaraFreq;

    let beta = 10.0;
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        LogisticKernel::new(beta).unwrap(),
        beta,
        Some(1e-6),
        None,
    )
    .unwrap();
    let size = basis.size();
    let wmax = Basis::wmax(&basis);

    for tau in [2.0 * beta, f64::NAN] {
        let err = Basis::evaluate_tau(&basis, &[0.0, tau]).unwrap_err();
        assert!(
            matches!(err, Error::OutOfDomain { name: "tau", .. }),
            "{err:?}"
        );
    }
    for omega in [2.0 * wmax, -2.0 * wmax, f64::NAN] {
        let err = Basis::evaluate_omega(&basis, &[0.0, omega]).unwrap_err();
        assert!(
            matches!(err, Error::OutOfDomain { name: "omega", domain, .. } if domain == (-wmax, wmax)),
            "{err:?}"
        );
    }
    // The ends of the domains are inside.
    Basis::evaluate_tau(&basis, &[-beta, beta]).unwrap();
    Basis::evaluate_omega(&basis, &[-wmax, wmax]).unwrap();

    assert_eq!(
        *Basis::evaluate_tau(&basis, &[]).unwrap().shape(),
        (0, size)
    );
    assert_eq!(
        *Basis::evaluate_omega(&basis, &[]).unwrap().shape(),
        (0, size)
    );
    let no_freqs: [MatsubaraFreq<Fermionic>; 0] = [];
    assert_eq!(
        *Basis::evaluate_matsubara(&basis, &no_freqs)
            .unwrap()
            .shape(),
        (0, size)
    );
}
