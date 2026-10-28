//! The SVE of the kernels of the IR basis: accuracy, parity of the singular
//! functions, and the full-domain path. The mechanics of the SVE are tested
//! in `sparse-ir-sve` on kernels of its own.
// RegularizedBoseKernel is deprecated (#273) but tested until it is removed.
#![allow(deprecated)]

use crate::kernel::{
    AbstractKernel, BasisKernel, LogisticKernel, LogisticKernelWindowed, RegularizedBoseKernel,
    SVEKernel,
};
use crate::sve::{
    SVEOptions, SVEResult, TworkType, compute_sve, compute_sve_dense, compute_sve_with,
};

/// Assert that an SVE satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// This verifies the fundamental SVE relation:
/// K(x, y) = sum_{l=0}^{L-1} s_l * u_l(x) * v_l(y)
///
/// where:
/// - x, y ∈ [-1, 1] (scaled variables used in SVE computation)
/// - u_l(x) are left singular functions
/// - v_l(y) are right singular functions
/// - s_l are singular values
///
/// The singular functions must be defined on the full kernel domain
/// `[-xmax, xmax] x [-ymax, ymax]`, so the test points below cover all four
/// quadrants.
fn assert_sve_reconstructs_kernel<K>(kernel: &K, sve_result: &SVEResult, tolerance: f64)
where
    K: AbstractKernel + SVEKernel,
{
    // Test points: midpoints of knots of u and v in scaled domain [-1, 1]
    //
    // Evaluating at midpoints between knots is more representative of how the
    // piecewise Legendre polynomials approximate the kernel between collocation
    // points, and avoids testing exactly at the knots used during SVE.
    let u_polys = sve_result.u().get_polys();
    let v_polys = sve_result.v().get_polys();
    let x_knots = u_polys[0].get_knots();
    let y_knots = v_polys[0].get_knots();

    for (name, knots, min, max) in [
        ("x", x_knots, kernel.xmin(), kernel.xmax()),
        ("y", y_knots, kernel.ymin(), kernel.ymax()),
    ] {
        let (first, last) = (knots[0], knots[knots.len() - 1]);
        let tol = f64::EPSILON * min.abs().max(max);
        assert!(
            (first - min).abs() <= tol && (last - max).abs() <= tol,
            "singular functions must cover the full {name} domain [{min}, {max}], got [{first}, {last}]"
        );
    }

    let test_x: Vec<f64> = x_knots.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let test_y: Vec<f64> = y_knots.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();

    // Track worst-case error for diagnostics
    let mut max_error = 0.0f64;
    let mut worst_x = 0.0f64;
    let mut worst_y = 0.0f64;
    let mut worst_direct = 0.0f64;
    let mut worst_sve = 0.0f64;
    let mut max_abs_direct = 0.0f64;

    for &x in &test_x {
        for &y in &test_y {
            // Compute kernel value directly
            let k_direct = kernel.compute(x, y);

            // Compute kernel value from SVE decomposition
            let mut k_sve = 0.0;
            for l in 0..sve_result.s().len() {
                let u_l_x = sve_result.u()[l].evaluate(x);
                let v_l_y = sve_result.v()[l].evaluate(y);
                k_sve += sve_result.s()[l] * u_l_x * v_l_y;
            }

            // Compute absolute error only
            // Note: We don't use relative error because kernel values can be very small
            let error = (k_direct - k_sve).abs();

            if error > max_error {
                max_error = error;
                worst_x = x;
                worst_y = y;
                worst_direct = k_direct;
                worst_sve = k_sve;
            }

            let abs_direct = k_direct.abs();
            if abs_direct > max_abs_direct {
                max_abs_direct = abs_direct;
            }
        }
    }

    eprintln!(
        "Max SVE abs error: error={:.15e}, x={:.15e}, y={:.15e}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}, max|K|={:.15e}",
        max_error, worst_x, worst_y, worst_direct, worst_sve, tolerance, max_abs_direct
    );

    assert!(
        max_error < tolerance,
        "SVE decomposition failed: max_error={:.15e} at x={}, y={}, direct={:.15e}, sve={:.15e}, abs_tol={:.15e}",
        max_error,
        worst_x,
        worst_y,
        worst_direct,
        worst_sve,
        tolerance
    );
}

/// Test that the SVE computed by `compute_sve` reconstructs the kernel
fn test_sve_decomposition_kernel_impl<K>(kernel: K, lambda: f64, epsilon: f64)
where
    K: BasisKernel,
{
    // Tolerance for comparison (absolute error)
    // Note: We use absolute error only because kernel values can be very small.
    // Tolerance is set based on epsilon to account for numerical errors in SVE decomposition.
    // Use a tolerance that is modestly larger than epsilon to account for accumulated errors.
    let tolerance = epsilon * 200.0;
    test_sve_decomposition_kernel_impl_with_tolerance(kernel, lambda, epsilon, tolerance);
}

/// Same as test_sve_decomposition_kernel_impl but with explicit tolerance parameter
fn test_sve_decomposition_kernel_impl_with_tolerance<K>(
    kernel: K,
    _lambda: f64,
    epsilon: f64,
    tolerance: f64,
) where
    K: BasisKernel,
{
    let sve_result =
        compute_sve(kernel.clone(), Some(epsilon), None, None, TworkType::Auto).unwrap();
    assert_sve_reconstructs_kernel(&kernel, &sve_result, tolerance);
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for LogisticKernel
#[test]
fn test_sve_decomposition_logistic_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl(LogisticKernel::new(100.0).unwrap(), 100.0, 1e-6);

    // Test with lambda = 10^5, epsilon = 1e-12
    test_sve_decomposition_kernel_impl(LogisticKernel::new(1e5).unwrap(), 1e5, 1e-12);
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for RegularizedBoseKernel
/// Note: Uses relaxed tolerance due to numerical challenges with RegularizedBoseKernel
#[test]
fn test_sve_decomposition_regularized_bose_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl_with_tolerance(
        RegularizedBoseKernel::new(100.0).unwrap(),
        100.0,
        1e-6,
        1e-6 * 1.0,
    );

    // Test with lambda = 10^5, epsilon = 1e-12
    // Use relaxed tolerance: epsilon * 100000 (much more lenient than epsilon * 200)
    // For very small kernel values, use absolute tolerance of 1e-5
    //test_sve_decomposition_kernel_impl_with_tolerance(
    //RegularizedBoseKernel::new(1e5),
    //1e5,
    //1e-12,
    //1e-5  // Use fixed absolute tolerance for very small values
    //);
}

/// In a computed SVE, `l` is the index of the singular function and, since the
/// even and odd singular values interlace, `(-1)^l` is its parity -- the
/// relation the asymptotic expansion of uhat relies on (#265).
#[test]
fn test_compute_sve_global_index_matches_parity() {
    let results = [
        compute_sve(
            LogisticKernel::new(10.0).unwrap(),
            Some(1e-10),
            None,
            None,
            TworkType::Auto,
        )
        .unwrap(),
        compute_sve(
            RegularizedBoseKernel::new(10.0).unwrap(),
            Some(1e-10),
            None,
            None,
            TworkType::Auto,
        )
        .unwrap(),
    ];
    for sve in &results {
        assert!(sve.s().len() >= 8, "SVE too small: {}", sve.s().len());
        for (i, (u, v)) in sve
            .u()
            .get_polys()
            .iter()
            .zip(sve.v().get_polys())
            .enumerate()
        {
            let parity = if i % 2 == 0 { 1 } else { -1 };
            assert_eq!(u.get_l(), i as i32, "u[{i}].l");
            assert_eq!(v.get_l(), i as i32, "v[{i}].l");
            assert_eq!(u.get_symm(), parity, "u[{i}].symm");
            assert_eq!(v.get_symm(), parity, "v[{i}].symm");
        }
    }
}

/// the full-domain SVE reconstructs centrosymmetric kernels on the full
/// domain `[-1, 1] x [-1, 1]` to the same tolerances as the `compute_sve`
/// decomposition tests above (issue #246)
#[test]
fn test_full_domain_sve_reconstructs_centrosymmetric_kernels() {
    let kernel = LogisticKernel::new(100.0).unwrap();
    let sve = compute_sve_with(
        kernel,
        &SVEOptions::new()
            .epsilon(1e-6)
            .twork(TworkType::Auto)
            .full_domain(true),
    )
    .unwrap();
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-6 * 200.0);

    // TworkType::Auto selects Float64X2 for this epsilon.
    let kernel = LogisticKernel::new(10.0).unwrap();
    let sve = compute_sve_with(
        kernel,
        &SVEOptions::new()
            .epsilon(1e-10)
            .twork(TworkType::Auto)
            .full_domain(true),
    )
    .unwrap();
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-10 * 200.0);

    let kernel = RegularizedBoseKernel::new(100.0).unwrap();
    let sve = compute_sve_with(
        kernel,
        &SVEOptions::new()
            .epsilon(1e-6)
            .twork(TworkType::Auto)
            .full_domain(true),
    )
    .unwrap();
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-6);
}

/// Deviations of an SVE from a reference: `(max relative error of s_l,
/// max error of u_l and v_l relative to their maximum)`, over the singular
/// values and functions `ls`
fn sve_deviation(sve: &SVEResult, reference: &SVEResult, ls: std::ops::Range<usize>) -> (f64, f64) {
    let grid = |lo: f64, hi: f64| -> Vec<f64> {
        (0..=2000)
            .map(|i| lo + (hi - lo) * i as f64 / 2000.0)
            .collect()
    };
    let (u_knots, v_knots) = (
        reference.u().get_polys()[0].get_knots(),
        reference.v().get_polys()[0].get_knots(),
    );
    let xs = grid(u_knots[0], *u_knots.last().unwrap());
    let ys = grid(v_knots[0], *v_knots.last().unwrap());
    let mut ds: f64 = 0.0;
    let mut df: f64 = 0.0;
    for l in ls {
        ds = ds.max((sve.s()[l] - reference.s()[l]).abs() / reference.s()[l]);
        for (a, b, pts) in [(sve.u(), reference.u(), &xs), (sve.v(), reference.v(), &ys)] {
            let (p, q) = (&a.get_polys()[l], &b.get_polys()[l]);
            let scale = pts.iter().fold(0.0f64, |m, &x| m.max(q.evaluate(x).abs()));
            let err = pts
                .iter()
                .fold(0.0f64, |m, &x| m.max((p.evaluate(x) - q.evaluate(x)).abs()));
            df = df.max(err / scale);
        }
    }
    (ds, df)
}

/// Machine epsilon of the working precision `compute_sve` uses for `twork`
/// (`Float64` or `Float64X2`)
fn working_machine_epsilon(twork: TworkType) -> f64 {
    match twork {
        TworkType::Float64 => f64::EPSILON,
        TworkType::Float64X2 => {
            crate::CustomNumeric::to_f64(<crate::Df64 as crate::CustomNumeric>::epsilon())
        }
        TworkType::Auto => unreachable!("the cases give the working precision"),
    }
}

/// Checks the SVE of `compute_sve` (or, with `full_domain`, that over the
/// full domain) against the SVE by the dense CPQR-based SVD of the
/// fully formed blocks (`compute_sve_dense`), the method of both before the
/// ACA
///
/// In f64, and over the full domain of a centrosymmetric kernel, they use
/// that method: the SVEs are identical. Otherwise, in Df64, they take the
/// block SVDs by ACA. The CPQR SVE is accurate only to the working precision
/// ε_w relative to s₀, so the singular values are compared to 64·ε_w·s₀
/// (measured: under ε_w·s₀). Neither method resolves the values near the
/// cutoff 2·ε_w·s₀, so their number can differ there (by one, in values at
/// 2.0–2.5·ε_w·s₀, for Λ = 3e5 and 1e6): the values only one SVE has must be
/// below 8·ε_w·s₀. Relative to each singular value, and for the functions,
/// the comparison is over the values a basis for ε uses (s_l ≥ ε·s₀, plus
/// one more), where they agree to 1e-15 and 1e-14 (measured: 1.4e-16 and
/// 1.2e-15).
fn check_sve_against_dense_cpqr<K>(kernel: K, eps: f64, twork: TworkType, full_domain: bool)
where
    K: BasisKernel,
{
    let sve = if full_domain {
        let options = SVEOptions::new()
            .epsilon(eps)
            .twork(twork)
            .full_domain(true);
        compute_sve_with(kernel.clone(), &options).unwrap()
    } else {
        compute_sve(kernel.clone(), Some(eps), None, None, twork).unwrap()
    };
    let identical =
        twork == TworkType::Float64 || (full_domain && kernel.implements_reduced_kernel());
    let dense = compute_sve_dense(kernel, Some(eps), twork, full_domain).unwrap();
    let case = format!("{twork:?}, full_domain = {full_domain}");
    if identical {
        assert_eq!(sve.s(), dense.s(), "{case}");
        for (a, b) in [(sve.u(), dense.u()), (sve.v(), dense.v())] {
            for (p, q) in a.get_polys().iter().zip(b.get_polys()) {
                assert_eq!(p.get_data().as_slice(), q.get_data().as_slice(), "{case}");
            }
        }
        return;
    }

    let eps_w = working_machine_epsilon(twork);
    let s0 = dense.s()[0];
    let n = sve.s().len().min(dense.s().len());
    for extra in sve.s()[n..].iter().chain(&dense.s()[n..]) {
        assert!(
            *extra < 8.0 * eps_w * s0,
            "{case}: {} vs {} singular values, the last {extra:e}",
            sve.s().len(),
            dense.s().len()
        );
    }
    for (l, (a, d)) in sve.s().iter().zip(dense.s()).enumerate() {
        assert!(
            (a - d).abs() <= 64.0 * eps_w * s0,
            "{case}: s[{l}] = {a:e} vs {d:e}"
        );
    }
    let l_basis = dense.s().iter().take_while(|&&s| s >= eps * s0).count() + 1;
    let (ds, df) = sve_deviation(&sve, &dense, 0..l_basis);
    assert!(ds <= 1e-15, "{case}: s_l relative error {ds:e}");
    assert!(df <= 1e-14, "{case}: u_l, v_l error {df:e}");
}

/// The cases of [`check_sve_against_dense_cpqr`] at one Λ: f64 and Df64;
/// logistic (even/odd blocks and full domain), regularized Bose and
/// windowed kernels
fn check_sve_against_dense_cpqr_at(lambda: f64) {
    for (eps, twork) in [(1e-6, TworkType::Float64), (1e-15, TworkType::Float64X2)] {
        for full_domain in [false, true] {
            check_sve_against_dense_cpqr(
                LogisticKernel::new(lambda).unwrap(),
                eps,
                twork,
                full_domain,
            );
        }
        check_sve_against_dense_cpqr(
            RegularizedBoseKernel::new(lambda).unwrap(),
            eps,
            twork,
            false,
        );
        check_sve_against_dense_cpqr(
            LogisticKernelWindowed::from_physical(1.0, -0.3 * lambda, lambda).unwrap(),
            eps,
            twork,
            false,
        );
    }
}

#[test]
fn test_sve_matches_dense_cpqr_lambda_1e2() {
    check_sve_against_dense_cpqr_at(1e2);
}

#[test]
fn test_sve_matches_dense_cpqr_lambda_1e4() {
    check_sve_against_dense_cpqr_at(1e4);
}

/// Λ = 1e6: the dense Df64 SVEs take a few minutes (release), so this runs
/// on request: `cargo test --release -- --ignored`
#[test]
#[ignore = "slow: dense Df64 SVE at Λ = 1e6 (a few minutes in release)"]
fn test_sve_matches_dense_cpqr_lambda_1e6() {
    check_sve_against_dense_cpqr_at(1e6);
}
