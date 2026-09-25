//! Tests for SVE module functions

use super::utils::{extend_to_full_domain, merge_results, mirror_segments_to_full_domain};
use super::{SVDStrategy, SVEResult, TworkType, compute_sve, compute_sve_general, safe_epsilon};
use crate::kernel::{
    AbstractKernel, CentrosymmKernel, KernelProperties, LogisticKernel, LogisticSVEHints,
    RegularizedBoseKernel, SVEHints, SymmetryType,
};
use crate::numeric::CustomNumeric;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::traits::StatisticsType;
use mdarray::DTensor;
use std::fmt::Debug;

/// Create a simple polynomial on positive domain [0, 1]
fn create_simple_poly_on_positive_domain() -> PiecewiseLegendrePoly {
    // Create a simple polynomial: f(x) = 1 + 2x on [0, 1]
    // Legendre basis: P_0(x) = 1, P_1(x) = x
    // On [0, 1], we need to map to [-1, 1] internally
    let data = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 2.0 });
    let knots = vec![0.0, 1.0];
    let delta_x = vec![1.0];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0)
}

/// Create polynomial with multiple segments [0, 0.5, 1.0]
fn create_poly_with_segments() -> PiecewiseLegendrePoly {
    // Two segments: [0, 0.5] and [0.5, 1.0]
    let data_vec = [1.0, 1.5, 0.5, 1.0];
    let data = DTensor::<f64, 2>::from_fn([2, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    let knots = vec![0.0, 0.5, 1.0];
    let delta_x = vec![0.5, 0.5];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0)
}

#[test]
fn test_extend_even_symmetry() {
    let poly_positive = create_simple_poly_on_positive_domain();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even, 1.0);

    // Test: f(-x) = f(x) for Even symmetry
    let poly = &polys_full[0];
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let val_pos = poly.evaluate(x);
        let val_neg = poly.evaluate(-x);
        assert!(
            (val_pos - val_neg).abs() < 1e-14,
            "Even symmetry violated: f({}) = {}, f({}) = {}",
            x,
            val_pos,
            -x,
            val_neg
        );
    }
}

#[test]
fn test_extend_odd_symmetry() {
    let poly_positive = create_simple_poly_on_positive_domain();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Odd, 1.0);

    // Test: f(-x) = -f(x) for Odd symmetry
    let poly = &polys_full[0];
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let val_pos = poly.evaluate(x);
        let val_neg = poly.evaluate(-x);
        assert!(
            (val_pos + val_neg).abs() < 1e-14,
            "Odd symmetry violated: f({}) = {}, f({}) = {}",
            x,
            val_pos,
            -x,
            val_neg
        );
    }
}

#[test]
fn test_positive_domain_preserved() {
    let poly_positive = create_simple_poly_on_positive_domain();

    // Save original values
    let original_values: Vec<f64> = (0..10)
        .map(|i| poly_positive.evaluate(i as f64 * 0.1))
        .collect();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even, 1.0);

    // Check that positive domain values are preserved (with 1/sqrt(2) normalization)
    // The extended polynomial applies 1/sqrt(2) normalization to both parts
    let poly = &polys_full[0];
    let norm_factor = 1.0 / 2.0_f64.sqrt();

    for (i, &expected) in original_values.iter().enumerate() {
        let x = i as f64 * 0.1;
        let actual = poly.evaluate(x);
        let expected_normalized = expected * norm_factor;
        assert!(
            (actual - expected_normalized).abs() < 1e-14,
            "Positive domain not preserved: f({}) = {} (expected {})",
            x,
            actual,
            expected_normalized
        );
    }
}

#[test]
fn test_segment_structure() {
    let poly = create_poly_with_segments();

    let polys_full = extend_to_full_domain(vec![poly], SymmetryType::Even, 1.0);

    // Extended from [0, 0.5, 1.0] to [-1.0, -0.5, 0.0, 0.5, 1.0]
    let expected_knots = [-1.0, -0.5, 0.0, 0.5, 1.0];

    // Check segment structure
    for (i, &expected) in expected_knots.iter().enumerate() {
        assert!(
            (polys_full[0].knots[i] - expected).abs() < 1e-14,
            "Segment {} mismatch: got {}, expected {}",
            i,
            polys_full[0].knots[i],
            expected
        );
    }
}

#[test]
fn test_multiple_polynomials() {
    let poly1 = create_simple_poly_on_positive_domain();
    let poly2 = create_poly_with_segments();

    let polys_full = extend_to_full_domain(vec![poly1, poly2], SymmetryType::Even, 1.0);

    // Should have extended both polynomials
    assert_eq!(polys_full.len(), 2);

    // Both should satisfy even symmetry
    for poly in &polys_full {
        let val_pos = poly.evaluate(0.3);
        let val_neg = poly.evaluate(-0.3);
        assert!(
            (val_pos - val_neg).abs() < 1e-14,
            "Even symmetry violated for one of the polynomials"
        );
    }
}

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
    K: AbstractKernel + KernelProperties,
{
    // Test points: midpoints of knots of u and v in scaled domain [-1, 1]
    //
    // Evaluating at midpoints between knots is more representative of how the
    // piecewise Legendre polynomials approximate the kernel between collocation
    // points, and avoids testing exactly at the knots used during SVE.
    let u_polys = sve_result.u.get_polys();
    let v_polys = sve_result.v.get_polys();
    let x_knots = &u_polys[0].knots;
    let y_knots = &v_polys[0].knots;

    for (name, knots, max) in [("x", x_knots, kernel.xmax()), ("y", y_knots, kernel.ymax())] {
        let (first, last) = (knots[0], knots[knots.len() - 1]);
        assert!(
            (first + max).abs() <= f64::EPSILON * max && (last - max).abs() <= f64::EPSILON * max,
            "singular functions must cover the full {name} domain [-{max}, {max}], got [{first}, {last}]"
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
            for l in 0..sve_result.s.len() {
                let u_l_x = sve_result.u[l].evaluate(x);
                let v_l_y = sve_result.v[l].evaluate(y);
                k_sve += sve_result.s[l] * u_l_x * v_l_y;
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
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
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
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    let sve_result = compute_sve(kernel.clone(), epsilon, None, None, TworkType::Auto);
    assert_sve_reconstructs_kernel(&kernel, &sve_result, tolerance);
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for LogisticKernel
#[test]
fn test_sve_decomposition_logistic_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl(LogisticKernel::new(100.0), 100.0, 1e-6);

    // Test with lambda = 10^5, epsilon = 1e-12
    test_sve_decomposition_kernel_impl(LogisticKernel::new(1e5), 1e5, 1e-12);
}

/// Test that SVE decomposition satisfies k(x, y) = sum_l s_l u_l(x) v_l(y)
///
/// Tests multiple parameter combinations for RegularizedBoseKernel
/// Note: Uses relaxed tolerance due to numerical challenges with RegularizedBoseKernel
#[test]
fn test_sve_decomposition_regularized_bose_kernel() {
    // Test with lambda = 100, epsilon = 1e-6
    test_sve_decomposition_kernel_impl_with_tolerance(
        RegularizedBoseKernel::new(100.0),
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

/// Polynomial on [0, 1] with Legendre coefficients `(c, 1)`, i.e.
/// `sqrt(2) * (c + 2x - 1)`, carrying `l = k` as `svd_to_polynomials` does for
/// column `k` of one even/odd SVD block.
fn half_domain_poly(c: f64, k: i32) -> PiecewiseLegendrePoly {
    let data = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { c } else { 1.0 });
    PiecewiseLegendrePoly::new(data, vec![0.0, 1.0], k, Some(vec![1.0]), 0)
}

/// `merge_results` must renumber `l` from the index within the even or odd
/// block to the position in the merged result (#265).
#[test]
fn test_merge_results_assigns_global_index() {
    let even = extend_to_full_domain(
        vec![half_domain_poly(1.0, 0), half_domain_poly(2.0, 1)],
        SymmetryType::Even,
        1.0,
    );
    let odd = extend_to_full_domain(
        vec![half_domain_poly(3.0, 0), half_domain_poly(4.0, 1)],
        SymmetryType::Odd,
        1.0,
    );
    // Interlacing singular values, as for a totally positive kernel.
    let merged = merge_results(
        (
            PiecewiseLegendrePolyVector::new(even.clone()),
            vec![1.0, 0.1],
            PiecewiseLegendrePolyVector::new(even),
        ),
        (
            PiecewiseLegendrePolyVector::new(odd.clone()),
            vec![0.5, 0.05],
            PiecewiseLegendrePolyVector::new(odd),
        ),
        1e-10,
    );

    assert_eq!(merged.s, vec![1.0, 0.5, 0.1, 0.05]);
    // The extension divides by sqrt(2), so f(1) = c + 1 identifies the source
    // polynomial: even 0, odd 0, even 1, odd 1.
    let expected_value_at_1 = [2.0, 4.0, 3.0, 5.0];
    for (i, (u, v)) in merged
        .u
        .get_polys()
        .iter()
        .zip(merged.v.get_polys())
        .enumerate()
    {
        let parity = if i % 2 == 0 { 1 } else { -1 };
        assert_eq!(u.l, i as i32, "u[{i}].l");
        assert_eq!(v.l, i as i32, "v[{i}].l");
        assert_eq!(u.symm, parity, "u[{i}].symm");
        assert_eq!(v.symm, parity, "v[{i}].symm");
        assert!((u.evaluate(1.0) - expected_value_at_1[i]).abs() < 1e-14);
        assert!((v.evaluate(1.0) - expected_value_at_1[i]).abs() < 1e-14);
    }
}

/// In a computed SVE, `l` is the index of the singular function and, since the
/// even and odd singular values interlace, `(-1)^l` is its parity -- the
/// relation the asymptotic expansion of uhat relies on (#265).
#[test]
fn test_compute_sve_global_index_matches_parity() {
    let results = [
        compute_sve(
            LogisticKernel::new(10.0),
            1e-10,
            None,
            None,
            TworkType::Auto,
        ),
        compute_sve(
            RegularizedBoseKernel::new(10.0),
            1e-10,
            None,
            None,
            TworkType::Auto,
        ),
    ];
    for sve in &results {
        assert!(sve.s.len() >= 8, "SVE too small: {}", sve.s.len());
        for (i, (u, v)) in sve.u.get_polys().iter().zip(sve.v.get_polys()).enumerate() {
            let parity = if i % 2 == 0 { 1 } else { -1 };
            assert_eq!(u.l, i as i32, "u[{i}].l");
            assert_eq!(v.l, i as i32, "v[{i}].l");
            assert_eq!(u.symm, parity, "u[{i}].symm");
            assert_eq!(v.symm, parity, "v[{i}].symm");
        }
    }
}

/// Machine epsilon of the working precision that `compute_sve*` selects for
/// `(epsilon, twork)`
fn working_machine_epsilon(epsilon: f64, twork: TworkType) -> f64 {
    match safe_epsilon(epsilon, twork, SVDStrategy::Auto).1 {
        TworkType::Float64 => f64::EPSILON,
        TworkType::Float64X2 => CustomNumeric::to_f64(<crate::Df64 as CustomNumeric>::epsilon()),
        TworkType::Auto => unreachable!("safe_epsilon resolves TworkType::Auto"),
    }
}

/// Compare `compute_sve_general` with `compute_sve` for a centrosymmetric kernel
///
/// Regression test for issue #246: `compute_sve_general` handed the
/// half-domain `SVEHints` segments of centrosymmetric kernels to the
/// full-domain strategy, so it expanded only the restriction of the kernel to
/// `[0, 1] x [0, 1]` (LogisticKernel, lambda = 10, eps = 1e-6: 12 singular
/// values with s_0 = 0.1733 instead of 19 with s_0 = 0.5645).
///
/// With mirrored segments the full-domain matrix uses the same Gauss points and
/// weights as the even/odd blocks of `compute_sve` and is orthogonally
/// equivalent to their direct sum, so the singular values agree in exact
/// arithmetic. Both paths use backward-stable SVDs, so each singular value
/// moves by O(u_work * s_0), u_work being the machine epsilon of the working
/// precision; `1000 * u_work * s_0` allows a generous dimension-dependent
/// constant. Float64X2 singular values are also rounded to f64 on output,
/// which adds at most one ulp, bounded by `2 * f64::EPSILON * s_l`.
///
/// `expect_same_count` should be set only when the smallest kept singular
/// value is far above the common cutoff, so that rounding cannot move a value
/// across it. Otherwise, a singular value kept by only one path must lie at
/// the cutoff within the same error bound.
fn assert_general_path_matches_compute_sve<K>(
    kernel: K,
    epsilon: f64,
    twork: TworkType,
    expect_same_count: bool,
) where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    let u_work = working_machine_epsilon(epsilon, twork);
    // Explicit cutoff, so that this comparison does not depend on the
    // default-cutoff policy of either path.
    let cutoff = 2.0 * u_work;
    let reference = compute_sve(kernel.clone(), epsilon, Some(cutoff), None, twork);
    let general = compute_sve_general(kernel, epsilon, Some(cutoff), None, twork);

    let s0 = reference.s[0];
    let tol = |s: f64| 1000.0 * u_work * s0 + 2.0 * f64::EPSILON * s;

    let n = reference.s.len().min(general.s.len());
    let mut max_rel_diff = 0.0_f64;
    for l in 0..n {
        let diff = (general.s[l] - reference.s[l]).abs();
        max_rel_diff = max_rel_diff.max(diff / s0);
        assert!(
            diff <= tol(reference.s[l]),
            "s[{l}] mismatch (eps={epsilon:e}, twork={twork:?}): compute_sve_general={:.16e}, \
             compute_sve={:.16e}, |diff|/s0={:.3e}, tol/s0={:.3e}",
            general.s[l],
            reference.s[l],
            diff / s0,
            tol(reference.s[l]) / s0
        );
    }
    for &s in reference.s[n..].iter().chain(general.s[n..].iter()) {
        assert!(
            s <= cutoff * s0 + tol(s),
            "singular value {s:.3e} = {:.3e} s0 is kept by only one path but lies above \
             the cutoff {cutoff:.3e} s0 by more than the error bound",
            s / s0
        );
    }
    if expect_same_count {
        assert_eq!(
            general.s.len(),
            reference.s.len(),
            "number of singular values (eps={epsilon:e}, twork={twork:?})"
        );
    }
    eprintln!(
        "compute_sve_general vs compute_sve (eps={epsilon:e}, twork={twork:?}): n={}/{}, max|ds|/s0={max_rel_diff:.3e}",
        general.s.len(),
        reference.s.len()
    );
}

/// `compute_sve_general` reproduces `compute_sve` for LogisticKernel (issue #246)
#[test]
fn test_compute_sve_general_matches_compute_sve_logistic() {
    // Equal counts are asserted at lambda = 10, where the smallest kept
    // singular value is well above the cutoff 2 u_work s_0 (about 8.6 u s_0 in
    // Float64 and 27 u s_0 in Float64X2); at lambda = 1e4 it is about
    // 2.7 u s_0, i.e. within rounding of the cutoff.
    assert_general_path_matches_compute_sve(
        LogisticKernel::new(10.0),
        1e-6,
        TworkType::Float64,
        true,
    );
    assert_general_path_matches_compute_sve(
        LogisticKernel::new(10.0),
        1e-10,
        TworkType::Float64X2,
        true,
    );
    assert_general_path_matches_compute_sve(
        LogisticKernel::new(1e4),
        1e-6,
        TworkType::Float64,
        false,
    );
}

/// `compute_sve_general` reproduces `compute_sve` for RegularizedBoseKernel
/// (issue #246)
#[test]
fn test_compute_sve_general_matches_compute_sve_regularized_bose() {
    // Same margins as for LogisticKernel (smallest kept singular value about
    // 12 u s_0 in Float64 and 38 u s_0 in Float64X2 at lambda = 10).
    assert_general_path_matches_compute_sve(
        RegularizedBoseKernel::new(10.0),
        1e-6,
        TworkType::Float64,
        true,
    );
    assert_general_path_matches_compute_sve(
        RegularizedBoseKernel::new(10.0),
        1e-10,
        TworkType::Float64X2,
        true,
    );
    assert_general_path_matches_compute_sve(
        RegularizedBoseKernel::new(1e4),
        1e-6,
        TworkType::Float64,
        false,
    );
}

/// `compute_sve_general` reconstructs centrosymmetric kernels on the full
/// domain `[-1, 1] x [-1, 1]` to the same tolerances as the `compute_sve`
/// decomposition tests above (issue #246)
#[test]
fn test_compute_sve_general_reconstructs_centrosymmetric_kernels() {
    let kernel = LogisticKernel::new(100.0);
    let sve = compute_sve_general(kernel, 1e-6, None, None, TworkType::Auto);
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-6 * 200.0);

    // TworkType::Auto selects Float64X2 for this epsilon.
    let kernel = LogisticKernel::new(10.0);
    let sve = compute_sve_general(kernel, 1e-10, None, None, TworkType::Auto);
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-10 * 200.0);

    let kernel = RegularizedBoseKernel::new(100.0);
    let sve = compute_sve_general(kernel, 1e-6, None, None, TworkType::Auto);
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-6);
}

/// Logistic kernel tilted by `1 + x/2`, so that `K(-x, -y) != K(x, y)`
///
/// Exercises the non-centrosymmetric branch of `compute_sve_general`.
#[derive(Debug, Clone, Copy)]
struct TiltedLogisticKernel(LogisticKernel);

/// Full-domain SVE hints for `TiltedLogisticKernel`: the logistic segments
/// mirrored onto `[-1, 1]`
#[derive(Debug, Clone)]
struct TiltedLogisticSVEHints<T>(LogisticSVEHints<T>);

impl<T> SVEHints<T> for TiltedLogisticSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        mirror_segments_to_full_domain(&self.0.segments_x())
    }

    fn segments_y(&self) -> Vec<T> {
        mirror_segments_to_full_domain(&self.0.segments_y())
    }

    fn nsvals(&self) -> usize {
        self.0.nsvals()
    }

    fn ngauss(&self) -> usize {
        self.0.ngauss()
    }
}

impl AbstractKernel for TiltedLogisticKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        let tilt = T::from_f64_unchecked(1.0) + T::from_f64_unchecked(0.5) * x;
        self.0.compute(x, y) * tilt
    }
}

impl KernelProperties for TiltedLogisticKernel {
    type SVEHintsType<T>
        = TiltedLogisticSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn ypower(&self) -> i32 {
        self.0.ypower()
    }

    fn conv_radius(&self) -> f64 {
        self.0.conv_radius()
    }

    fn xmax(&self) -> f64 {
        self.0.xmax()
    }

    fn ymax(&self) -> f64 {
        self.0.ymax()
    }

    fn regularizer<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        self.0.regularizer::<S>(beta, omega)
    }

    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        TiltedLogisticSVEHints(self.0.sve_hints::<T>(epsilon))
    }
}

/// `compute_sve_general` on a non-centrosymmetric kernel uses its full-domain
/// hint segments as given, reconstructs the kernel, and fixes the SVD sign
/// gauge by `u_l(xmax) >= 0` like `compute_sve`
#[test]
fn test_compute_sve_general_non_centrosymmetric_kernel() {
    let kernel = TiltedLogisticKernel(LogisticKernel::new(10.0));
    assert!(!kernel.is_centrosymmetric());
    let epsilon = 1e-6;
    let sve = compute_sve_general(kernel, epsilon, None, None, TworkType::Float64);

    assert!(
        sve.s.windows(2).all(|w| w[0] >= w[1]) && sve.s[sve.s.len() - 1] > 0.0,
        "singular values must be positive and non-increasing: {:?}",
        sve.s
    );
    // Same tolerance as for the untilted LogisticKernel above.
    assert_sve_reconstructs_kernel(&kernel, &sve, epsilon * 200.0);
    for (l, u) in sve.u.get_polys().iter().enumerate() {
        let u_at_xmax = u.evaluate(kernel.xmax());
        assert!(
            u_at_xmax >= 0.0,
            "sign gauge not fixed: u[{l}](xmax) = {u_at_xmax:e} < 0"
        );
    }
}

/// Pin the default truncation cutoff of both SVE paths to `2 * u_work`
/// (issue #249), `u_work` being the machine epsilon of the working precision,
/// as in libsparseir
///
/// The cases are chosen so that the cutoff is observable: an odd-sector
/// singular value lies at about 1.4 u_work s_0, between u_work s_0 and
/// 2 u_work s_0 (found by scanning lambda; the geometric centre of that window
/// is sqrt(2) u_work s_0), so `compute_sve` keeps one more singular value with
/// cutoff `u_work` than with `2 * u_work`. That value is also about 14 u_work
/// times the largest odd-sector singular value, well above the accuracy floor
/// of its SVD block, so its position is stable across platforms.
///
/// `compute_sve_general` computes a single SVD block whose truncated SVD
/// already drops singular values below `2 * u_work * s_0`, so a smaller default
/// cutoff would not be observable there; the test still pins that its default
/// equals an explicit `2 * u_work` and keeps as many values as `compute_sve`.
fn assert_default_cutoff_is_two_machine_epsilon<K>(kernel: K, epsilon: f64, twork: TworkType)
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    let u_work = working_machine_epsilon(epsilon, twork);

    let explicit_2u = compute_sve(kernel.clone(), epsilon, Some(2.0 * u_work), None, twork);
    let explicit_u = compute_sve(kernel.clone(), epsilon, Some(u_work), None, twork);
    assert_eq!(
        explicit_u.s.len(),
        explicit_2u.s.len() + 1,
        "precondition: exactly one singular value must lie in [u, 2u) s_0 so that the \
         cutoff is observable (eps={epsilon:e}, twork={twork:?}); choose another lambda"
    );

    let centro_default = compute_sve(kernel.clone(), epsilon, None, None, twork);
    assert_eq!(
        centro_default.s, explicit_2u.s,
        "compute_sve default cutoff must be 2 * machine epsilon (eps={epsilon:e}, twork={twork:?})"
    );

    let general_2u = compute_sve_general(kernel.clone(), epsilon, Some(2.0 * u_work), None, twork);
    let general_default = compute_sve_general(kernel, epsilon, None, None, twork);
    assert_eq!(
        general_default.s, general_2u.s,
        "compute_sve_general default cutoff must be 2 * machine epsilon (eps={epsilon:e}, twork={twork:?})"
    );
    assert_eq!(
        general_default.s.len(),
        centro_default.s.len(),
        "both paths must keep the same number of singular values by default \
         (eps={epsilon:e}, twork={twork:?})"
    );
}

#[test]
fn test_default_cutoff_is_two_machine_epsilon_logistic() {
    assert_default_cutoff_is_two_machine_epsilon(
        LogisticKernel::new(1.48),
        1e-6,
        TworkType::Float64,
    );
    assert_default_cutoff_is_two_machine_epsilon(
        LogisticKernel::new(1.37),
        1e-10,
        TworkType::Float64X2,
    );
}

#[test]
fn test_default_cutoff_is_two_machine_epsilon_regularized_bose() {
    assert_default_cutoff_is_two_machine_epsilon(
        RegularizedBoseKernel::new(1.48),
        1e-6,
        TworkType::Float64,
    );
    assert_default_cutoff_is_two_machine_epsilon(
        RegularizedBoseKernel::new(1.37),
        1e-10,
        TworkType::Float64X2,
    );
}
