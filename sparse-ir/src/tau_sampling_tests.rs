// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]
use crate::basis::FiniteTempBasis;
use crate::kernel::{CentrosymmKernel, KernelProperties};
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::sampling::TauSampling;
use crate::traits::{Bosonic, Fermionic, StatisticsType};
use num_complex::Complex;

use crate::test_utils::{ErrorNorm, movedim};

/// Test for evaluate_nd/fit_nd roundtrip (real)
fn test_evaluate_nd_roundtrip_real<S>()
where
    S: StatisticsType + 'static,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let n_k = 5;
    let n_omega = 7;

    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate random test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = crate::test_utils::generate_nd_test_data::<f64, _, _>(
            &basis,
            sampling.sampling_points(),
            &[],
            42 + dim as u64,
            &[n_k, n_omega],
        );

        // Move to target dimension
        let coeffs_dim = movedim(&coeffs_0, 0, dim);

        // Evaluate along target dimension
        let evaluated_values = sampling.evaluate_nd(None, &coeffs_dim, dim).unwrap();

        // Fit back along target dimension
        let fitted_coeffs_dim = sampling.fit_nd(None, &evaluated_values, dim).unwrap();

        // Move back to dim=0 for comparison
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();

        // Check roundtrip
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    let abs_error = (orig - fitted).abs();

                    assert!(
                        abs_error < 1e-10,
                        "ND roundtrip (dim={}) error at ({},{},{}): error={}",
                        dim,
                        l,
                        k,
                        omega,
                        abs_error
                    );
                }
            }
        }
    }
}

#[test]
fn test_evaluate_nd_fermionic_real() {
    test_evaluate_nd_roundtrip_real::<Fermionic>();
}

#[test]
fn test_evaluate_nd_bosonic_real() {
    test_evaluate_nd_roundtrip_real::<Bosonic>();
}

/// Test for evaluate_nd/fit_nd roundtrip (complex)
fn test_evaluate_nd_roundtrip_complex<S>()
where
    S: StatisticsType + 'static,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let n_k = 5;
    let n_omega = 7;

    for dim in 0..3 {
        let (coeffs_0, _gtau_0, _giwn_0) =
            crate::test_utils::generate_nd_test_data::<Complex<f64>, _, _>(
                &basis,
                sampling.sampling_points(),
                &[],
                42 + dim as u64,
                &[n_k, n_omega],
            );

        let coeffs_dim = movedim(&coeffs_0, 0, dim);
        let evaluated_values = sampling.evaluate_nd_zz(None, &coeffs_dim, dim).unwrap();
        let fitted_coeffs_dim = sampling.fit_nd_zz(None, &evaluated_values, dim).unwrap();
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();

        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    let abs_error = (orig - fitted).error_norm();

                    assert!(
                        abs_error < 1e-10,
                        "ND roundtrip (dim={}) error at ({},{},{}): error={}",
                        dim,
                        l,
                        k,
                        omega,
                        abs_error
                    );
                }
            }
        }
    }
}

#[test]
fn test_evaluate_nd_fermionic_complex() {
    test_evaluate_nd_roundtrip_complex::<Fermionic>();
}

#[test]
fn test_evaluate_nd_bosonic_complex() {
    test_evaluate_nd_roundtrip_complex::<Bosonic>();
}

// ====================
// RegularizedBoseKernel TauSampling Tests
// ====================

/// Test for RegularizedBoseKernel evaluate_nd/fit_nd roundtrip (real)
fn test_regularized_bose_evaluate_nd_roundtrip_real() {
    let beta = 10.0;
    let wmax = 1.0;
    let epsilon = Some(1e-4);

    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let n_k = 5;
    let n_omega = 7;

    for dim in 0..3 {
        let (coeffs_0, _gtau_0, _giwn_0) = crate::test_utils::generate_nd_test_data::<f64, _, _>(
            &basis,
            sampling.sampling_points(),
            &[],
            42 + dim as u64,
            &[n_k, n_omega],
        );

        let coeffs_dim = movedim(&coeffs_0, 0, dim);
        let evaluated_values = sampling.evaluate_nd(None, &coeffs_dim, dim).unwrap();
        let fitted_coeffs_dim = sampling.fit_nd(None, &evaluated_values, dim).unwrap();
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();
        let mut max_error = 0.0;
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    let abs_error = (orig - fitted).abs();
                    if abs_error > max_error {
                        max_error = abs_error;
                    }
                }
            }
        }

        assert!(
            max_error < 1e-7,
            "RegularizedBose ND roundtrip (dim={}) error too large: {}",
            dim,
            max_error
        );
    }
}

/// Test for RegularizedBoseKernel evaluate_nd/fit_nd roundtrip (complex)
fn test_regularized_bose_evaluate_nd_roundtrip_complex() {
    let beta = 10.0;
    let wmax = 1.0;
    let epsilon = Some(1e-4);

    let kernel = RegularizedBoseKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Bosonic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let n_k = 5;
    let n_omega = 7;

    for dim in 0..3 {
        let (coeffs_0, _gtau_0, _giwn_0) =
            crate::test_utils::generate_nd_test_data::<Complex<f64>, _, _>(
                &basis,
                sampling.sampling_points(),
                &[],
                42 + dim as u64,
                &[n_k, n_omega],
            );

        let coeffs_dim = movedim(&coeffs_0, 0, dim);
        let evaluated_values = sampling.evaluate_nd_zz(None, &coeffs_dim, dim).unwrap();
        let fitted_coeffs_dim = sampling.fit_nd_zz(None, &evaluated_values, dim).unwrap();
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);

        let basis_size = basis.size();
        let mut max_error = 0.0;
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    let abs_error = (orig - fitted).error_norm();
                    if abs_error > max_error {
                        max_error = abs_error;
                    }
                }
            }
        }

        assert!(
            max_error < 1e-7,
            "RegularizedBose ND roundtrip (dim={}) error too large: {}",
            dim,
            max_error
        );
    }
}

#[test]
fn test_regularized_bose_evaluate_nd_real() {
    test_regularized_bose_evaluate_nd_roundtrip_real();
}

#[test]
fn test_regularized_bose_evaluate_nd_complex() {
    test_regularized_bose_evaluate_nd_roundtrip_complex();
}

/// Test that evaluate_nd_to produces identical results to evaluate_nd
#[test]
fn test_evaluate_nd_to_matches_fermionic_real() {
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    let expected = sampling.evaluate_nd(None, &coeffs, 0).unwrap();

    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_points, n_k, n_omega][..], 0.0);
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .evaluate_nd_to(None, &coeffs, 0, &mut actual_view)
            .unwrap();
    }

    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

#[test]
fn test_evaluate_nd_to_matches_fermionic_complex() {
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    let coeffs =
        Tensor::<Complex<f64>, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                idx[2] as f64 * 0.3,
            )
        });

    let expected = sampling.evaluate_nd_zz(None, &coeffs, 0).unwrap();

    let mut actual = Tensor::<Complex<f64>, crate::DynRank>::from_elem(
        &[n_points, n_k, n_omega][..],
        Complex::new(0.0, 0.0),
    );
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .evaluate_nd_zz_to(None, &coeffs, 0, &mut actual_view)
            .unwrap();
    }

    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).error_norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

/// Test that fit_nd_to produces identical results to fit_nd
#[test]
fn test_fit_nd_to_matches_fermionic_real() {
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    let values = Tensor::<f64, crate::DynRank>::from_fn(&[n_points, n_k, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    let expected = sampling.fit_nd(None, &values, 0).unwrap();

    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[basis_size, n_k, n_omega][..], 0.0);
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .fit_nd_to(None, &values, 0, &mut actual_view)
            .unwrap();
    }

    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..basis_size {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

#[test]
fn test_fit_nd_to_matches_fermionic_complex() {
    use mdarray::{Shape, Tensor};

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    let values =
        Tensor::<Complex<f64>, crate::DynRank>::from_fn(&[n_points, n_k, n_omega][..], |idx| {
            Complex::new(
                (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5),
                idx[2] as f64 * 0.3,
            )
        });

    let expected = sampling.fit_nd_zz(None, &values, 0).unwrap();

    let mut actual = Tensor::<Complex<f64>, crate::DynRank>::from_elem(
        &[basis_size, n_k, n_omega][..],
        Complex::new(0.0, 0.0),
    );
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .fit_nd_zz_to(None, &values, 0, &mut actual_view)
            .unwrap();
    }

    let expected_shape = expected.shape().with_dims(|d| d.to_vec());
    let actual_shape = actual.shape().with_dims(|d| d.to_vec());
    assert_eq!(expected_shape, actual_shape);

    for i in 0..basis_size {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).error_norm();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

// ============================================================================
// Tests for evaluate_nd_to at different dims
// ============================================================================

#[test]
fn test_evaluate_nd_to_dim0() {
    use mdarray::Tensor;

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[basis_size, n_k, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    let expected = sampling.evaluate_nd(None, &coeffs, 0).unwrap();

    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_points, n_k, n_omega][..], 0.0);
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .evaluate_nd_to(None, &coeffs, 0, &mut actual_view)
            .unwrap();
    }

    for i in 0..n_points {
        for j in 0..n_k {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

#[test]
fn test_evaluate_nd_to_dim1() {
    use mdarray::Tensor;

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients with basis_size in middle dimension
    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[n_k, basis_size, n_omega][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    // Expected result
    let expected = sampling.evaluate_nd(None, &coeffs, 1).unwrap();

    // Actual result using to_viewmut
    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_k, n_points, n_omega][..], 0.0);
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .evaluate_nd_to(None, &coeffs, 1, &mut actual_view)
            .unwrap();
    }

    // Compare
    for i in 0..n_k {
        for j in 0..n_points {
            for k in 0..n_omega {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-14,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}",
                    i,
                    j,
                    k,
                    e,
                    a
                );
            }
        }
    }
}

#[test]
fn test_evaluate_nd_to_dim_last() {
    // Test dim == N-1 (last dimension) fast path
    use mdarray::Tensor;

    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);

    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let basis_size = basis.size();
    let n_points = sampling.n_sampling_points();
    let n_k = 3;
    let n_omega = 4;

    // Create test coefficients with basis_size in LAST dimension (dim=2)
    let coeffs = Tensor::<f64, crate::DynRank>::from_fn(&[n_k, n_omega, basis_size][..], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 0.5) * (idx[2] as f64 + 0.3)
    });

    // Expected result (dim=2, which is rank-1)
    let expected = sampling.evaluate_nd(None, &coeffs, 2).unwrap();

    // Actual result using to_viewmut (should use fast path for dim == N-1)
    let mut actual = Tensor::<f64, crate::DynRank>::from_elem(&[n_k, n_omega, n_points][..], 0.0);
    {
        let mut actual_view = actual.expr_mut();
        sampling
            .evaluate_nd_to(None, &coeffs, 2, &mut actual_view)
            .unwrap();
    }

    // Compare
    for i in 0..n_k {
        for j in 0..n_omega {
            for k in 0..n_points {
                let e = expected[&[i, j, k][..]];
                let a = actual[&[i, j, k][..]];
                let diff = (e - a).abs();
                assert!(
                    diff < 1e-12,
                    "Mismatch at [{}, {}, {}]: expected={:?}, actual={:?}, diff={}",
                    i,
                    j,
                    k,
                    e,
                    a,
                    diff
                );
            }
        }
    }
}

/// TauSampling reports the condition number of its real sampling matrix
fn check_tau_condition_number<S: StatisticsType + 'static>() {
    use crate::test_utils::{assert_condition_number_close, oracle_condition_number};

    let (beta, wmax, epsilon) = (10.0, 1.0, 1e-6);
    let kernel = LogisticKernel::new(beta * wmax).unwrap();
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None).unwrap();
    let sampling = TauSampling::new(&basis).unwrap();

    let oracle = oracle_condition_number(sampling.matrix());
    let label = format!(
        "tau {:?}, beta={beta}, wmax={wmax}, eps={epsilon:e}, L={}, n={}",
        S::STATISTICS,
        basis.size(),
        sampling.n_sampling_points()
    );
    assert_condition_number_close(&label, sampling.condition_number(), oracle);
}

#[test]
fn test_tau_condition_number_fermionic() {
    check_tau_condition_number::<Fermionic>();
}

#[test]
fn test_tau_condition_number_bosonic() {
    check_tau_condition_number::<Bosonic>();
}

/// `out` of `TauSampling::*_nd_to` must match the input on every axis, not
/// only in rank and target extent. Before PR-0, `evaluate_nd_to` with coeffs
/// of shape [L, 50] and an out view of shape [n_points, 1] wrote the
/// 49 × n_points values that do not fit past the end of the view; PR-0 made
/// it a panic, and it is ShapeMismatch of the output now.
#[test]
fn test_nd_to_rejects_out_with_wrong_batch_extent() {
    use crate::error::{ArrayRole, Error};
    use mdarray::{DenseMapping, DynRank, Shape, Tensor, ViewMut};

    let basis = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let sampling = TauSampling::new(&basis).unwrap();
    let (l, np, extra) = (sampling.basis_size(), sampling.n_sampling_points(), 50);
    const CANARY: f64 = -12345.0;

    // Each call gets a buffer large enough for the correct output, and a view
    // that claims only its first `extra` = 1 column.
    let run = |fit: bool, complex: bool| -> (Result<(), Error>, usize, bool) {
        let (n_in, n_out) = if fit { (np, l) } else { (l, np) };
        let mut buffer = vec![CANARY; 2 * n_out * extra];
        let shape = DynRank::from_dims(&[n_out, 1]);
        let result = if complex {
            let input = Tensor::<Complex<f64>, DynRank>::from_elem(
                &[n_in, extra][..],
                Complex::new(1.0, 0.5),
            );
            // SAFETY: the view covers the first `n_out` of `2 * n_out * extra` complex-sized slots.
            let mut out = unsafe {
                ViewMut::<'_, Complex<f64>, DynRank>::new_unchecked(
                    buffer.as_mut_ptr() as *mut Complex<f64>,
                    DenseMapping::new(shape.clone()),
                )
            };
            if fit {
                sampling.fit_nd_zz_to(None, &input, 0, &mut out)
            } else {
                sampling.evaluate_nd_zz_to(None, &input, 0, &mut out)
            }
        } else {
            let input = Tensor::<f64, DynRank>::from_elem(&[n_in, extra][..], 1.0);
            // SAFETY: the view covers the first `n_out` elements of `buffer`.
            let mut out = unsafe {
                ViewMut::<'_, f64, DynRank>::new_unchecked(
                    buffer.as_mut_ptr(),
                    DenseMapping::new(shape.clone()),
                )
            };
            if fit {
                sampling.fit_nd_to(None, &input, 0, &mut out)
            } else {
                sampling.evaluate_nd_to(None, &input, 0, &mut out)
            }
        };
        (result, n_out, buffer.iter().all(|&x| x == CANARY))
    };

    for fit in [false, true] {
        for complex in [false, true] {
            let (result, n_out, untouched) = run(fit, complex);
            assert_eq!(
                result,
                Err(Error::ShapeMismatch {
                    which: ArrayRole::Output,
                    expected: vec![n_out, extra],
                    actual: vec![n_out, 1],
                }),
                "fit={fit}, complex={complex}"
            );
            assert!(
                untouched,
                "fit={fit}, complex={complex}: out buffer written"
            );
        }
    }
}

/// Extents of `dims` permuted as `movedim(_, src, dst)` does
fn moved_dims(dims: &[usize], src: usize, dst: usize) -> Vec<usize> {
    let mut moved = dims.to_vec();
    let d = moved.remove(src);
    moved.insert(dst, d);
    moved
}

/// mdarray 0.7.2 copies a permuted (strided) view out of bounds when an
/// extent other than the last is zero (https://github.com/fre-hu/mdarray/issues/21).
/// Before the fix `movedim` of e.g. a [2, 0] array from axis 1 to 0
/// segfaulted. Every zero-extent array must move to an empty array of the
/// permuted shape.
#[test]
fn test_movedim_with_zero_extent() {
    use crate::sampling::movedim;
    use mdarray::{DynRank, Tensor};

    for dims in [
        vec![0usize, 3],
        vec![2, 0],
        vec![0, 2, 3],
        vec![2, 0, 3],
        vec![2, 3, 0],
        vec![0, 0, 4],
    ] {
        let arr = Tensor::<f64, DynRank>::zeros(&dims[..]);
        for src in 0..dims.len() {
            for dst in 0..dims.len() {
                let moved = movedim(&arr, src, dst);
                assert_eq!(moved.shape().dims(), &moved_dims(&dims, src, dst)[..]);
                assert_eq!(moved.len(), 0);
            }
        }
    }

    // A non-empty array still moves its elements
    let arr = Tensor::<usize, DynRank>::from_fn(&[2, 3, 4][..], |idx| {
        100 * idx[0] + 10 * idx[1] + idx[2]
    });
    let moved = movedim(&arr, 2, 0);
    assert_eq!(moved.shape().dims(), &[4, 2, 3]);
    assert_eq!(moved[&[3, 1, 2][..]], arr[&[1, 2, 3][..]]);
}

/// movedim checks both axes also when they are equal: an axis past the rank
/// was returned as a copy of the array.
#[test]
#[should_panic(expected = "src axis 2 out of bounds for rank 2")]
fn test_movedim_rejects_equal_axes_past_the_rank() {
    use crate::sampling::movedim;
    use mdarray::{DynRank, Tensor};

    let arr = Tensor::<f64, DynRank>::zeros(&[2, 3][..]);
    let _ = movedim(&arr, 2, 2);
}

/// Evaluating or fitting an empty batch gives an empty result of the right
/// shape, for every target axis
#[test]
fn test_tau_nd_with_empty_batch() {
    use mdarray::{DynRank, Tensor};

    let basis = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let sampling = TauSampling::new(&basis).unwrap();
    let (l, np) = (sampling.basis_size(), sampling.n_sampling_points());

    for (batch, dim) in [
        (vec![0usize, 3], 0),
        (vec![3, 0], 1),
        (vec![0, 3], 2),
        (vec![2, 0], 1),
    ] {
        let mut dims = batch.clone();
        dims.insert(dim, l);
        let mut expected = batch.clone();
        expected.insert(dim, np);

        let coeffs = Tensor::<f64, DynRank>::zeros(&dims[..]);
        let values = sampling.evaluate_nd(None, &coeffs, dim).unwrap();
        assert_eq!(values.shape().dims(), &expected[..]);
        let fitted = sampling.fit_nd(None, &values, dim).unwrap();
        assert_eq!(fitted.shape().dims(), &dims[..]);

        let coeffs_z = Tensor::<Complex<f64>, DynRank>::zeros(&dims[..]);
        let values_z = sampling.evaluate_nd_zz(None, &coeffs_z, dim).unwrap();
        assert_eq!(values_z.shape().dims(), &expected[..]);
        let fitted_z = sampling.fit_nd_zz(None, &values_z, dim).unwrap();
        assert_eq!(fitted_z.shape().dims(), &dims[..]);
    }
}

/// A sampling matrix without columns describes no basis function. Before the
/// fix `from_matrix` accepted it and the first fit or condition number
/// segfaulted in the SVD (mdarray#21).
#[test]
#[should_panic(expected = "Matrix must have at least one column")]
fn test_from_matrix_rejects_zero_columns() {
    use mdarray::DTensor;
    let matrix = DTensor::<f64, 2>::zeros([3, 0]);
    let sampling = TauSampling::<Fermionic>::from_matrix(vec![0.1, 0.2, 0.3], matrix);
    // Not reached after the fix; crashed before it
    sampling.condition_number();
}

/// A tau sampling point outside [-β, β] or NaN is OutOfDomain (from
/// Basis::evaluate_tau), and no points is EmptyInput; both were panics.
#[test]
fn test_tau_sampling_rejects_invalid_points() {
    use crate::error::Error;

    let beta = 10.0;
    let basis = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(beta).unwrap(),
        beta,
        Some(1e-6),
        None,
    )
    .unwrap();
    for tau in [beta * 1.5, -beta * 1.5, f64::NAN, f64::INFINITY] {
        let err = TauSampling::with_sampling_points(&basis, vec![0.5, tau])
            .err()
            .unwrap();
        assert!(
            matches!(err, Error::OutOfDomain { name: "tau", value, domain }
                if (value == tau || (value.is_nan() && tau.is_nan())) && domain == (-beta, beta)),
            "{err:?}"
        );
    }
    assert_eq!(
        TauSampling::with_sampling_points(&basis, vec![]).err(),
        Some(Error::EmptyInput {
            name: "sampling_points"
        })
    );
    // ±β are inside the domain.
    TauSampling::with_sampling_points(&basis, vec![-beta, 0.0, beta]).unwrap();
}

/// The N-D methods of TauSampling report an axis that is not an axis of the
/// input and an input of the wrong extent along it (they panicked), before
/// allocating the output.
#[test]
fn test_tau_nd_methods_check_the_axis_and_the_input() {
    use crate::error::{ArrayRole, Error};
    use mdarray::{DynRank, Tensor};

    let basis = FiniteTempBasis::<_, Fermionic>::new(
        LogisticKernel::new(10.0).unwrap(),
        1.0,
        Some(1e-6),
        None,
    )
    .unwrap();
    let sampling = TauSampling::new(&basis).unwrap();
    let (l, np) = (sampling.basis_size(), sampling.n_sampling_points());

    let coeffs = Tensor::<f64, DynRank>::zeros(&[l, 3][..]);
    let values = Tensor::<f64, DynRank>::zeros(&[np, 3][..]);
    let coeffs_z = Tensor::<Complex<f64>, DynRank>::zeros(&[l, 3][..]);
    let values_z = Tensor::<Complex<f64>, DynRank>::zeros(&[np, 3][..]);
    let axis = Some(Error::AxisOutOfRange { axis: 2, rank: 2 });
    assert_eq!(sampling.evaluate_nd(None, &coeffs, 2).err(), axis);
    assert_eq!(sampling.fit_nd(None, &values, 2).err(), axis);
    assert_eq!(sampling.evaluate_nd_zz(None, &coeffs_z, 2).err(), axis);
    assert_eq!(sampling.fit_nd_zz(None, &values_z, 2).err(), axis);

    // Inputs of the wrong extent along `dim` (the default τ points are as
    // many as the basis functions here, so l + 1 and np + 1 are both wrong)
    let values_bad = Tensor::<f64, DynRank>::zeros(&[l + 1, 3][..]);
    let coeffs_bad = Tensor::<f64, DynRank>::zeros(&[np + 1, 3][..]);
    assert_eq!(
        sampling.fit_nd(None, &values_bad, 0).err(),
        Some(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected: vec![np, 3],
            actual: vec![l + 1, 3],
        })
    );
    assert_eq!(
        sampling.evaluate_nd(None, &coeffs_bad, 0).err(),
        Some(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected: vec![l, 3],
            actual: vec![np + 1, 3],
        })
    );
}
