//! Tests for SVE module functions

use super::types::safe_epsilon;
use super::utils::{extend_to_full_domain, merge_results};
use super::{SVEResult, TworkType, compute_sve, compute_sve_general};
use crate::error::Error;
use crate::kernel::{AbstractKernel, SVEHints, SVEKernel, SVEStrategyKernel, SymmetryType};
use crate::matrix::Mat;
use crate::numeric::CustomNumeric;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::test_kernels::{AsymmetricDomainExpKernel, ExpKernel, HalfExpKernel, TiltedExpKernel};
use std::fmt::Debug;

/// Create a simple polynomial on positive domain [0, 1]
fn create_simple_poly_on_positive_domain() -> PiecewiseLegendrePoly {
    // Create a simple polynomial: f(x) = 1 + 2x on [0, 1]
    // Legendre basis: P_0(x) = 1, P_1(x) = x
    // On [0, 1], we need to map to [-1, 1] internally
    let data = Mat::<f64>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 2.0 });
    let knots = vec![0.0, 1.0];
    let delta_x = vec![1.0];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0).unwrap()
}

/// Create polynomial with multiple segments [0, 0.5, 1.0]
fn create_poly_with_segments() -> PiecewiseLegendrePoly {
    // Two segments: [0, 0.5] and [0.5, 1.0]
    let data_vec = [1.0, 1.5, 0.5, 1.0];
    let data = Mat::<f64>::from_fn([2, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    let knots = vec![0.0, 0.5, 1.0];
    let delta_x = vec![0.5, 0.5];
    PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0).unwrap()
}

#[test]
fn test_extend_even_symmetry() {
    let poly_positive = create_simple_poly_on_positive_domain();

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even).unwrap();

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

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Odd).unwrap();

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

    let polys_full = extend_to_full_domain(vec![poly_positive], SymmetryType::Even).unwrap();

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

    let polys_full = extend_to_full_domain(vec![poly], SymmetryType::Even).unwrap();

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

    let polys_full = extend_to_full_domain(vec![poly1, poly2], SymmetryType::Even).unwrap();

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
    K: AbstractKernel + SVEKernel,
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

/// Polynomial on [0, 1] with Legendre coefficients `(c, 1)`, i.e.
/// `sqrt(2) * (c + 2x - 1)`, carrying `l = k` as `svd_to_polynomials` does for
/// column `k` of one even/odd SVD block.
fn half_domain_poly(c: f64, k: i32) -> PiecewiseLegendrePoly {
    let data = Mat::<f64>::from_fn([2, 1], |idx| if idx[0] == 0 { c } else { 1.0 });
    PiecewiseLegendrePoly::new(data, vec![0.0, 1.0], k, Some(vec![1.0]), 0).unwrap()
}

/// `merge_results` must renumber `l` from the index within the even or odd
/// block to the position in the merged result (#265).
#[test]
fn test_merge_results_assigns_global_index() {
    let even = extend_to_full_domain(
        vec![half_domain_poly(1.0, 0), half_domain_poly(2.0, 1)],
        SymmetryType::Even,
    )
    .unwrap();
    let odd = extend_to_full_domain(
        vec![half_domain_poly(3.0, 0), half_domain_poly(4.0, 1)],
        SymmetryType::Odd,
    )
    .unwrap();
    // Interlacing singular values, as for a totally positive kernel.
    let merged = merge_results(
        (
            PiecewiseLegendrePolyVector::new(even.clone()).unwrap(),
            vec![1.0, 0.1],
            PiecewiseLegendrePolyVector::new(even).unwrap(),
        ),
        (
            PiecewiseLegendrePolyVector::new(odd.clone()).unwrap(),
            vec![0.5, 0.05],
            PiecewiseLegendrePolyVector::new(odd).unwrap(),
        ),
        1e-10,
    )
    .unwrap();

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

/// Machine epsilon of the working precision that `compute_sve*` selects for
/// `(epsilon, twork)`
fn working_machine_epsilon(epsilon: f64, twork: TworkType) -> f64 {
    match safe_epsilon(Some(epsilon), twork).1 {
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
/// `[0, 1] x [0, 1]`, with fewer and wrong singular values.
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
    K: SVEStrategyKernel,
{
    let u_work = working_machine_epsilon(epsilon, twork);
    // Explicit cutoff, so that this comparison does not depend on the
    // default-cutoff policy of either path.
    let cutoff = 2.0 * u_work;
    let reference = compute_sve(kernel.clone(), Some(epsilon), Some(cutoff), None, twork).unwrap();
    let general = compute_sve_general(kernel, Some(epsilon), Some(cutoff), None, twork).unwrap();

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

/// `compute_sve_general` reproduces `compute_sve` for a centrosymmetric
/// kernel, singular values and functions (issue #246)
#[test]
fn test_compute_sve_general_matches_compute_sve() {
    // Equal counts are asserted at a = 10, where the smallest kept singular
    // value is well above the cutoff 2 u_work s_0 (about 6.2 u s_0 in Float64
    // and 4.3 u s_0 in Float64X2); at a = 20 it is about 2.04 u s_0, i.e.
    // within rounding of the cutoff.
    assert_general_path_matches_compute_sve(ExpKernel::new(10.0), 1e-6, TworkType::Float64, true);
    assert_general_path_matches_compute_sve(
        ExpKernel::new(10.0),
        1e-10,
        TworkType::Float64X2,
        true,
    );
    assert_general_path_matches_compute_sve(
        ExpKernel { a: 20.0, nseg: 8 },
        1e-6,
        TworkType::Float64,
        false,
    );

    // TworkType::Auto selects Float64X2 for 1e-10.
    for epsilon in [1e-6, 1e-10] {
        let kernel = ExpKernel::new(10.0);
        let sve = compute_sve_general(kernel, Some(epsilon), None, None, TworkType::Auto).unwrap();
        assert_sve_reconstructs_kernel(&kernel, &sve, epsilon * 200.0);
    }
}

/// The SVE of a kernel with zero rows (ACA moves on to other pivot rows)
/// reproduces the kernel
#[test]
fn test_sve_of_kernel_with_zero_rows() {
    let kernel = HalfExpKernel(ExpKernel::new(10.0));
    for (eps, twork) in [(1e-6, TworkType::Float64), (1e-10, TworkType::Float64X2)] {
        let sve = compute_sve(kernel, Some(eps), None, None, twork).unwrap();
        assert!(
            sve.s.len() > 5,
            "{twork:?}: {} singular values",
            sve.s.len()
        );
        let mut err: f64 = 0.0;
        for i in 0..=40 {
            let x = -1.0 + 0.05 * i as f64;
            for j in 0..=40 {
                let y = -1.0 + 0.05 * j as f64;
                let approx: f64 = (0..sve.s.len())
                    .map(|l| sve.u[l].evaluate(x) * sve.s[l] * sve.v[l].evaluate(y))
                    .sum();
                let exact: f64 = kernel.compute(x, y);
                err = err.max((approx - exact).abs());
            }
        }
        assert!(err < 100.0 * eps, "{twork:?}: reconstruction error {err:e}");
    }
}

/// `compute_sve_general` on a non-centrosymmetric kernel uses its full-domain
/// hint segments as given, reconstructs the kernel, and fixes the SVD sign
/// gauge by `u_l(xmax) >= 0` like `compute_sve`
#[test]
fn test_compute_sve_general_non_centrosymmetric_kernel() {
    let kernel = TiltedExpKernel(ExpKernel::new(10.0));
    assert!(!kernel.is_centrosymmetric());
    let epsilon = 1e-6;
    let sve = compute_sve_general(kernel, Some(epsilon), None, None, TworkType::Float64).unwrap();

    assert!(
        sve.s.windows(2).all(|w| w[0] >= w[1]) && sve.s[sve.s.len() - 1] > 0.0,
        "singular values must be positive and non-increasing: {:?}",
        sve.s
    );
    assert_sve_reconstructs_kernel(&kernel, &sve, epsilon * 200.0);
    for (l, u) in sve.u.get_polys().iter().enumerate() {
        let u_at_xmax = u.evaluate(kernel.xmax());
        assert!(
            u_at_xmax >= 0.0,
            "sign gauge not fixed: u[{l}](xmax) = {u_at_xmax:e} < 0"
        );
    }
}

/// `compute_sve` dispatches on the kernel: one that is not a
/// `ReducedKernel` goes through the full-domain strategy, exactly as with
/// `compute_sve_general`
#[test]
fn test_compute_sve_of_non_centrosymmetric_kernel_uses_full_domain() {
    let kernel = TiltedExpKernel(ExpKernel::new(10.0));
    let general = compute_sve_general(kernel, Some(1e-6), None, None, TworkType::Float64).unwrap();
    let sve = compute_sve(kernel, Some(1e-6), None, None, TworkType::Float64).unwrap();
    assert_eq!(sve.s, general.s);
    assert!(sve.u.get_polys().iter().all(|u| u.symm == 0));
    assert_sve_reconstructs_kernel(&kernel, &sve, 1e-6 * 200.0);
}

/// The even/odd strategy, and so `compute_sve`, rejects a centrosymmetric
/// kernel whose domain is not symmetric instead of returning the SVE of its
/// symmetric part
#[test]
fn test_centrosymm_sve_rejects_asymmetric_domain() {
    use super::CentrosymmSVE;

    let kernel = AsymmetricDomainExpKernel(ExpKernel::new(10.0));
    let expected = Error::InvalidParameter {
        name: "kernel domain [ymin, ymax]",
        value: "[-0.5, 1.0]".to_string(),
        reason: "must be symmetric about 0 for a centrosymmetric kernel".to_string(),
    };
    let err = CentrosymmSVE::<f64, _>::new(kernel, 1e-6).err().unwrap();
    assert_eq!(err, expected);
    let err = compute_sve(kernel, Some(1e-6), None, None, TworkType::Float64).unwrap_err();
    assert_eq!(err, expected);
}

// A non-centrosymmetric kernel with an analytically known SVE: a finite sum of
// products of L2[-1,1]-orthonormal normalized Legendre polynomials
// P̃_n(x) = sqrt((2n+1)/2) P_n(x),
//
//   K(x, y) = s0 P̃_0(x) P̃_1(y) + s1 P̃_1(x) P̃_0(y) + s2 P̃_2(x) P̃_2(y).
//
// The left family {P̃_0, P̃_1, P̃_2} and the right family {P̃_1, P̃_0, P̃_2} are
// each orthonormal, so this is the SVE: the singular values are exactly
// {s0, s1, s2}. Under (x, y) -> (-x, -y) the first two terms flip sign and the
// third does not, so K is not centrosymmetric.

fn norm_legendre_p0<T: CustomNumeric>() -> T {
    T::from_f64_unchecked(0.5_f64.sqrt())
}

fn norm_legendre_p1<T: CustomNumeric + Copy>(x: T) -> T {
    T::from_f64_unchecked(1.5_f64.sqrt()) * x
}

fn norm_legendre_p2<T: CustomNumeric + Copy>(x: T) -> T {
    let c = T::from_f64_unchecked(2.5_f64.sqrt());
    c * T::from_f64_unchecked(0.5)
        * (T::from_f64_unchecked(3.0) * x * x - T::from_f64_unchecked(1.0))
}

#[derive(Debug, Clone, Copy)]
struct SeparableNonsymmKernel {
    s: [f64; 3],
}

impl AbstractKernel for SeparableNonsymmKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        let [s0, s1, s2] = self.s.map(T::from_f64_unchecked);
        s0 * norm_legendre_p0::<T>() * norm_legendre_p1(y)
            + s1 * norm_legendre_p1(x) * norm_legendre_p0::<T>()
            + s2 * norm_legendre_p2(x) * norm_legendre_p2(y)
    }
}

/// Full-domain hints: one segment per half of [-1, 1], with enough Gauss
/// points to integrate the degree-4 products of the degree-2 functions
#[derive(Debug, Clone)]
struct SeparableNonsymmSVEHints<T>(std::marker::PhantomData<T>);

impl<T> SVEHints<T> for SeparableNonsymmSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        [-1.0, 0.0, 1.0].map(T::from_f64_unchecked).to_vec()
    }

    fn segments_y(&self) -> Vec<T> {
        [-1.0, 0.0, 1.0].map(T::from_f64_unchecked).to_vec()
    }

    fn nsvals(&self) -> usize {
        6
    }

    fn ngauss(&self) -> usize {
        16
    }
}

impl SVEKernel for SeparableNonsymmKernel {
    type SVEHintsType<T>
        = SeparableNonsymmSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn sve_hints<T>(&self, _epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        SeparableNonsymmSVEHints(std::marker::PhantomData)
    }
}

impl SVEStrategyKernel for SeparableNonsymmKernel {}

/// `compute_sve` of the separable kernel recovers its exact singular values,
/// reconstructs the kernel and returns functions without parity (symm == 0),
/// which keeps the Matsubara transforms from assuming one
#[test]
fn test_compute_sve_recovers_analytic_sve_of_non_centrosymmetric_kernel() {
    let kernel = SeparableNonsymmKernel { s: [1.0, 0.6, 0.3] };
    let k_pos: f64 = kernel.compute(0.4, 0.7);
    let k_neg: f64 = kernel.compute(-0.4, -0.7);
    assert!((k_pos - k_neg).abs() > 1e-6);

    for twork in [TworkType::Float64, TworkType::Float64X2] {
        let sve = compute_sve(kernel, Some(1e-10), None, None, twork).unwrap();
        // Rank 3 by construction: further singular values are SVD noise.
        assert!(sve.s.len() >= 3, "{twork:?}: {:?}", sve.s);
        for (l, (&got, want)) in sve.s.iter().zip([1.0, 0.6, 0.3]).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "{twork:?}: s[{l}] = {got:.15e}, expected {want}"
            );
        }
        if let Some(&s3) = sve.s.get(3) {
            assert!(s3 < 1e-12, "{twork:?}: s[3] = {s3:e}");
        }
        let polys = sve.u.get_polys().iter().chain(sve.v.get_polys());
        assert!(polys.into_iter().all(|p| p.symm == 0), "{twork:?}");
        assert_sve_reconstructs_kernel(&kernel, &sve, 1e-12);
    }
}

/// Pin the default truncation cutoff of both SVE paths to `2 * u_work`
/// (issue #249), `u_work` being the machine epsilon of the working precision,
/// as in libsparseir
///
/// The cases are chosen so that the cutoff is observable: an odd-sector
/// singular value lies at about 1.4 u_work s_0, between u_work s_0 and
/// 2 u_work s_0 (found by scanning a; the geometric centre of that window
/// is sqrt(2) u_work s_0), so `compute_sve` keeps one more singular value with
/// cutoff `u_work` than with `2 * u_work`. That value is also 11 to 14 u_work
/// times the largest odd-sector singular value, well above the accuracy floor
/// of its SVD block, so its position is stable across platforms.
///
/// `compute_sve_general` computes a single SVD block whose truncated SVD
/// already drops singular values below `2 * u_work * s_0`, so a smaller default
/// cutoff would not be observable there; the test still pins that its default
/// equals an explicit `2 * u_work` and keeps as many values as `compute_sve`.
fn assert_default_cutoff_is_two_machine_epsilon<K>(kernel: K, epsilon: f64, twork: TworkType)
where
    K: SVEStrategyKernel,
{
    let u_work = working_machine_epsilon(epsilon, twork);

    let explicit_2u = compute_sve(
        kernel.clone(),
        Some(epsilon),
        Some(2.0 * u_work),
        None,
        twork,
    )
    .unwrap();
    let explicit_u = compute_sve(kernel.clone(), Some(epsilon), Some(u_work), None, twork).unwrap();
    assert_eq!(
        explicit_u.s.len(),
        explicit_2u.s.len() + 1,
        "precondition: exactly one singular value must lie in [u, 2u) s_0 so that the \
         cutoff is observable (eps={epsilon:e}, twork={twork:?}); choose another a"
    );

    let centro_default = compute_sve(kernel.clone(), Some(epsilon), None, None, twork).unwrap();
    assert_eq!(
        centro_default.s, explicit_2u.s,
        "compute_sve default cutoff must be 2 * machine epsilon (eps={epsilon:e}, twork={twork:?})"
    );

    let general_2u = compute_sve_general(
        kernel.clone(),
        Some(epsilon),
        Some(2.0 * u_work),
        None,
        twork,
    )
    .unwrap();
    let general_default = compute_sve_general(kernel, Some(epsilon), None, None, twork).unwrap();
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
fn test_default_cutoff_is_two_machine_epsilon() {
    assert_default_cutoff_is_two_machine_epsilon(ExpKernel::new(0.3), 1e-6, TworkType::Float64);
    assert_default_cutoff_is_two_machine_epsilon(
        ExpKernel::new(0.395),
        1e-10,
        TworkType::Float64X2,
    );
}

/// A kernel matrix with infinite entries (exp(a x y) with a = ∞ or a so large
/// that it overflows) is reported. Its SVD first iterated forever, then the
/// TSVD panicked ("TSVD computation failed"); compute_sve now reports it.
#[test]
fn test_compute_sve_reports_a_non_finite_kernel_matrix() {
    for a in [f64::INFINITY, f64::MAX] {
        let result = compute_sve(ExpKernel::new(a), Some(1e-6), None, None, TworkType::Auto);
        assert!(
            matches!(result, Err(Error::NonFiniteInput { name: "matrix", .. })),
            "a = {a}: {:?}",
            result.err()
        );
    }
}

/// Assert that `truncated` holds exactly the first `n` singular values and
/// functions of `full`. Both come from the same even/odd SVDs, so they agree
/// exactly.
fn assert_leading_part_of(truncated: &SVEResult, full: &SVEResult, n: usize) {
    assert_eq!(truncated.s, full.s[..n].to_vec());
    assert_eq!(truncated.u.get_polys().len(), n);
    assert_eq!(truncated.v.get_polys().len(), n);
    for l in 0..n {
        let (u, u_full) = (&truncated.u.get_polys()[l], &full.u.get_polys()[l]);
        let (v, v_full) = (&truncated.v.get_polys()[l], &full.v.get_polys()[l]);
        assert_eq!((u.l, u.symm), (u_full.l, u_full.symm));
        for x in [-1.0, -0.3, 0.0, 0.7, 1.0] {
            assert_eq!(u.evaluate(x), u_full.evaluate(x), "u[{l}]({x})");
            assert_eq!(v.evaluate(x), v_full.evaluate(x), "v[{l}]({x})");
        }
    }
}

/// Truncating the SVE to its largest singular value empties the odd block
/// (the largest singular value is even). Before the fix the odd block was
/// wrapped in a `PiecewiseLegendrePolyVector`, which cannot be empty:
/// `compute_sve` panicked with "Cannot create empty
/// PiecewiseLegendrePolyVector" for `max_num_svals = Some(1)` and for
/// `cutoff = Some(1.0)`.
fn check_sve_truncated_to_one<K>(kernel: K, epsilon: f64, twork: TworkType)
where
    K: SVEStrategyKernel,
{
    let full = compute_sve(kernel.clone(), Some(epsilon), None, None, twork).unwrap();
    assert!(full.s.len() >= 4, "SVE too small: {}", full.s.len());
    for (cutoff, max_num_svals) in [(None, Some(1)), (Some(1.0), None)] {
        let sve = compute_sve(kernel.clone(), Some(epsilon), cutoff, max_num_svals, twork).unwrap();
        assert_leading_part_of(&sve, &full, 1);
        assert_eq!(
            sve.u.get_polys()[0].symm,
            1,
            "the largest singular value is even"
        );
    }
    // Two singular values keep one of each parity, as before the fix
    let sve = compute_sve(kernel, Some(epsilon), None, Some(2), twork).unwrap();
    assert_leading_part_of(&sve, &full, 2);
}

#[test]
fn test_compute_sve_truncated_to_one_singular_value() {
    check_sve_truncated_to_one(ExpKernel::new(10.0), 1e-6, TworkType::Float64);
    check_sve_truncated_to_one(ExpKernel::new(10.0), 1e-10, TworkType::Float64X2);
}

/// `merge_results` accepts an empty block and renumbers `l` in the merged
/// result. A `PiecewiseLegendrePolyVector` built with `new` cannot be
/// empty; the public field allows it, and truncation can produce one.
#[test]
fn test_merge_results_with_an_empty_block() {
    let even = extend_to_full_domain(
        vec![half_domain_poly(1.0, 0), half_domain_poly(2.0, 1)],
        SymmetryType::Even,
    )
    .unwrap();
    let odd = extend_to_full_domain(vec![half_domain_poly(3.0, 0)], SymmetryType::Odd).unwrap();
    let empty = || PiecewiseLegendrePolyVector::from_polys_unchecked(vec![]);
    let block = |polys: &Vec<PiecewiseLegendrePoly>, s: Vec<f64>| {
        (
            PiecewiseLegendrePolyVector::new(polys.clone()).unwrap(),
            s,
            PiecewiseLegendrePolyVector::new(polys.clone()).unwrap(),
        )
    };

    let merged = merge_results(
        block(&even, vec![1.0, 0.1]),
        (empty(), vec![], empty()),
        1e-10,
    )
    .unwrap();
    assert_eq!(merged.s, vec![1.0, 0.1]);
    let merged = merge_results((empty(), vec![], empty()), block(&odd, vec![0.5]), 1e-10).unwrap();
    assert_eq!(merged.s, vec![0.5]);
    assert_eq!(merged.u.get_polys()[0].symm, -1);
    for (i, u) in merged.u.get_polys().iter().enumerate() {
        assert_eq!(u.l, i as i32);
    }
}

#[test]
fn test_merge_results_rejects_two_empty_blocks() {
    let empty = || PiecewiseLegendrePolyVector::from_polys_unchecked(vec![]);
    let err = merge_results(
        (empty(), vec![], empty()),
        (empty(), vec![], empty()),
        1e-10,
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::EmptyInput {
            name: "singular values"
        }
    );
}

fn invalid(name: &'static str, value: &str, reason: &str) -> Error {
    Error::InvalidParameter {
        name,
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

/// Both SVE functions check their parameters before any work. A NaN epsilon
/// used to mean "automatic" here and to panic in `part`; `None` is the only
/// automatic value now.
#[test]
fn test_compute_sve_rejects_invalid_parameters() {
    let kernel = ExpKernel::new(10.0);
    let cases = [
        (
            Some(0.0),
            None,
            None,
            invalid("epsilon", "0.0", "must be in (0, 1)"),
        ),
        (
            Some(1.0),
            None,
            None,
            invalid("epsilon", "1.0", "must be in (0, 1)"),
        ),
        (
            Some(f64::NAN),
            None,
            None,
            invalid("epsilon", "NaN", "must be in (0, 1)"),
        ),
        (
            Some(-1.0),
            None,
            None,
            invalid("epsilon", "-1.0", "must be in (0, 1)"),
        ),
        (
            None,
            Some(-0.5),
            None,
            invalid("cutoff", "-0.5", "must be in [0, 1]"),
        ),
        (
            None,
            Some(1.5),
            None,
            invalid("cutoff", "1.5", "must be in [0, 1]"),
        ),
        (
            None,
            Some(f64::NAN),
            None,
            invalid("cutoff", "NaN", "must be in [0, 1]"),
        ),
        (
            None,
            None,
            Some(0),
            invalid("max_num_svals", "0", "must be positive"),
        ),
    ];
    for (epsilon, cutoff, max_num_svals, expected) in cases {
        let err = compute_sve(kernel, epsilon, cutoff, max_num_svals, TworkType::Auto).unwrap_err();
        assert_eq!(err, expected);
        let err = compute_sve_general(kernel, epsilon, cutoff, max_num_svals, TworkType::Auto)
            .unwrap_err();
        assert_eq!(err, expected);
    }
}

#[test]
fn test_sve_result_new_checks_its_invariants() {
    let sve = compute_sve(
        ExpKernel::new(10.0),
        Some(1e-6),
        None,
        None,
        TworkType::Auto,
    )
    .unwrap();
    let (u, s, v) = (sve.u.clone(), sve.s.clone(), sve.v.clone());
    let n = s.len();
    let empty = || PiecewiseLegendrePolyVector::from_polys_unchecked(vec![]);

    assert!(SVEResult::new(u.clone(), s.clone(), v.clone(), 0.0).is_ok());
    assert_eq!(
        SVEResult::new(empty(), vec![], empty(), 1e-6).unwrap_err(),
        Error::EmptyInput { name: "s" }
    );
    assert_eq!(
        SVEResult::new(u.clone(), s[..n - 1].to_vec(), v.clone(), 1e-6).unwrap_err(),
        invalid(
            "u",
            &format!("{n} functions"),
            &format!("must have one function per singular value ({})", n - 1)
        )
    );
    let mut nan = s.clone();
    nan[1] = f64::NAN;
    assert!(matches!(
        SVEResult::new(u.clone(), nan, v.clone(), 1e-6),
        Err(Error::NonFiniteInput { name: "s", ref index, .. }) if index == &vec![1]
    ));
    let mut negative = s.clone();
    negative[n - 1] = -1.0;
    assert!(matches!(
        SVEResult::new(u.clone(), negative, v.clone(), 1e-6),
        Err(Error::InvalidParameter { name: "s", .. })
    ));
    let mut increasing = s.clone();
    increasing.swap(0, 1);
    assert!(matches!(
        SVEResult::new(u.clone(), increasing, v.clone(), 1e-6),
        Err(Error::InvalidParameter { name: "s", .. })
    ));
    assert_eq!(
        SVEResult::new(u, s, v, 1.0).unwrap_err(),
        invalid("epsilon", "1.0", "must be in [0, 1)")
    );
}

/// `part` rejects a threshold of 1 or more and max_size = 0 (both emptied
/// the result and panicked, except eps = 1, which kept one value). A
/// threshold of 0 keeps every singular value.
#[test]
fn test_sve_result_part_checks_its_parameters() {
    let sve = compute_sve(
        ExpKernel::new(10.0),
        Some(1e-6),
        None,
        None,
        TworkType::Auto,
    )
    .unwrap();
    assert_eq!(sve.part(Some(0.0), None).unwrap().1.len(), sve.s.len());
    assert_eq!(
        sve.part(Some(1.0), None).unwrap_err(),
        invalid("eps", "1.0", "must be in [0, 1)")
    );
    assert_eq!(
        sve.part(Some(f64::NAN), None).unwrap_err(),
        invalid("eps", "NaN", "must be in [0, 1)")
    );
    assert_eq!(
        sve.part(None, Some(0)).unwrap_err(),
        invalid("max_size", "0", "must be positive")
    );
    assert_eq!(sve.part(None, Some(3)).unwrap().1, sve.s[..3].to_vec());
}

/// `extend_to_full_domain` needs functions on [0, xmax]. Before the change a
/// negative first knot made the mirrored knots decrease (a panic in `new`)
/// and a positive one gave wrong functions silently.
#[test]
fn test_extend_to_full_domain_requires_the_half_domain() {
    for start in [-0.5, 0.25] {
        let data = Mat::<f64>::from_elem([1, 1], 1.0);
        let poly = PiecewiseLegendrePoly::new(data, vec![start, 1.0], 0, None, 0).unwrap();
        let err = extend_to_full_domain(vec![poly], SymmetryType::Even).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter { name: "polys", .. }),
            "{err:?}"
        );
    }
}

/// `canonicalize_signs` fixes the sign at the right end of each function's
/// own domain. Before the change it evaluated at x = 1, which panicked for
/// functions whose domain ends below 1 (e.g. from spir_sve_result_from_matrix).
#[test]
fn test_canonicalize_signs_uses_the_end_of_the_domain() {
    use super::utils::canonicalize_signs;
    let minus_one = Mat::<f64>::from_elem([1, 1], -1.0);
    let u = PiecewiseLegendrePoly::new(minus_one.clone(), vec![-1.0, 0.5], 0, None, 0).unwrap();
    let v = PiecewiseLegendrePoly::new(minus_one, vec![-1.0, 1.0], 0, None, 0).unwrap();
    let (u, v) = canonicalize_signs(vec![u], vec![v]);
    assert!(u[0].evaluate(0.5) > 0.0);
    assert!(v[0].evaluate(1.0) > 0.0);
}

/// `svd_to_polynomials` needs at least one segment; before, an empty segment
/// list underflowed `segments.len() - 1`.
#[test]
fn test_svd_to_polynomials_needs_a_segment() {
    use super::utils::svd_to_polynomials;
    let rule = crate::gauss::legendre::<f64>(2);
    let u = Mat::<f64>::zeros([2, 1]);
    let err = svd_to_polynomials(&u, &[0.0_f64], &rule, 2).unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidParameter {
                name: "segments",
                ..
            }
        ),
        "{err:?}"
    );
}

/// `compute_sve_for_basis` truncates at max(2·ε_work, BASIS_CUTOFF_FACTOR·ε),
/// dropping the singular values far below ε, but everything a basis built
/// for ε uses agrees with the full spectrum (`compute_sve`)
#[test]
fn test_compute_sve_for_basis_keeps_what_a_basis_uses() {
    use super::{BASIS_CUTOFF_FACTOR, compute_sve_for_basis};

    for (twork, eps) in [(TworkType::Float64, 1e-6), (TworkType::Float64X2, 1e-10)] {
        let full_cutoff = 2.0 * working_machine_epsilon(eps, twork);
        let kernel = ExpKernel::new(20.0);
        let full = compute_sve(kernel, Some(eps), None, None, twork).unwrap();
        let part = compute_sve_for_basis(kernel, eps, None, twork).unwrap();
        let s0 = full.s[0];
        let cutoff = BASIS_CUTOFF_FACTOR * eps;
        if cutoff > full_cutoff {
            assert!(part.s.len() < full.s.len(), "{twork:?}: nothing truncated");
            assert!(*part.s.last().unwrap() >= cutoff * s0 * 0.99);
            assert!(full.s[part.s.len()] < cutoff * s0 * 1.01);
        } else {
            // The cutoff is floored at 2·ε_work: the full spectrum.
            assert_eq!(part.s, full.s, "{twork:?}");
        }
        // The values a basis for ε uses (those >= ε·s₀ plus one more)
        let l = full.s.iter().take_while(|&&s| s >= eps * s0).count() + 1;
        for i in 0..l {
            assert!(
                (part.s[i] - full.s[i]).abs() < 1e-5 * eps * s0,
                "{twork:?}: s[{i}] = {} vs {}",
                part.s[i],
                full.s[i]
            );
        }
    }
}

/// The SVE runs on the current rayon pool (its blocks, the ACA steps and the
/// double-double products), but each value is computed by the same
/// operations whatever the threading: the result on four threads is bitwise
/// that on one, in f64 and Df64, with and without centrosymmetry.
#[cfg(feature = "parallel")]
#[test]
fn test_sve_does_not_depend_on_the_number_of_threads() {
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    };
    let (one, four) = (pool(1), pool(4));
    let run = |eps: f64, twork: TworkType, general: bool| {
        let kernel = ExpKernel { a: 30.0, nseg: 8 };
        let sve = || {
            if general {
                compute_sve_general(kernel, Some(eps), None, None, twork).unwrap()
            } else {
                compute_sve(kernel, Some(eps), None, None, twork).unwrap()
            }
        };
        (one.install(sve), four.install(sve))
    };
    for (eps, twork, general) in [
        (1e-6, TworkType::Float64, false),
        (1e-10, TworkType::Float64X2, false),
        (1e-6, TworkType::Float64, true),
    ] {
        let (a, b) = run(eps, twork, general);
        assert_eq!(a.s, b.s, "{twork:?}, general = {general}");
        for (p, q) in a.u.get_polys().iter().zip(b.u.get_polys()) {
            assert_eq!(
                p.data.as_slice(),
                q.data.as_slice(),
                "{twork:?}, general = {general}"
            );
        }
    }
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
        &reference.u.get_polys()[0].knots,
        &reference.v.get_polys()[0].knots,
    );
    let xs = grid(u_knots[0], *u_knots.last().unwrap());
    let ys = grid(v_knots[0], *v_knots.last().unwrap());
    let mut ds: f64 = 0.0;
    let mut df: f64 = 0.0;
    for l in ls {
        ds = ds.max((sve.s[l] - reference.s[l]).abs() / reference.s[l]);
        for (a, b, pts) in [(&sve.u, &reference.u, &xs), (&sve.v, &reference.v, &ys)] {
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

/// Checks the SVE of `compute_sve` (or, with `full_domain`,
/// `compute_sve_general`) against the SVE by the dense CPQR-based SVD of the
/// fully formed blocks, the method of both before the ACA
///
/// In f64, and in `compute_sve_general` of a centrosymmetric kernel, they use
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
    K: SVEStrategyKernel,
{
    use super::compute_sve_dense;

    let sve = if full_domain {
        compute_sve_general(kernel.clone(), Some(eps), None, None, twork).unwrap()
    } else {
        compute_sve(kernel.clone(), Some(eps), None, None, twork).unwrap()
    };
    let identical = twork == TworkType::Float64 || (full_domain && kernel.is_centrosymmetric());
    let dense = compute_sve_dense(kernel, Some(eps), twork, full_domain).unwrap();
    let case = format!("{twork:?}, full_domain = {full_domain}");
    if identical {
        assert_eq!(sve.s, dense.s, "{case}");
        for (a, b) in [(&sve.u, &dense.u), (&sve.v, &dense.v)] {
            for (p, q) in a.get_polys().iter().zip(b.get_polys()) {
                assert_eq!(p.data.as_slice(), q.data.as_slice(), "{case}");
            }
        }
        return;
    }

    let eps_w = working_machine_epsilon(eps, twork);
    let s0 = dense.s[0];
    let n = sve.s.len().min(dense.s.len());
    for extra in sve.s[n..].iter().chain(&dense.s[n..]) {
        assert!(
            *extra < 8.0 * eps_w * s0,
            "{case}: {} vs {} singular values, the last {extra:e}",
            sve.s.len(),
            dense.s.len()
        );
    }
    for (l, (a, d)) in sve.s.iter().zip(&dense.s).enumerate() {
        assert!(
            (a - d).abs() <= 64.0 * eps_w * s0,
            "{case}: s[{l}] = {a:e} vs {d:e}"
        );
    }
    let l_basis = dense.s.iter().take_while(|&&s| s >= eps * s0).count() + 1;
    let (ds, df) = sve_deviation(&sve, &dense, 0..l_basis);
    assert!(ds <= 1e-15, "{case}: s_l relative error {ds:e}");
    assert!(df <= 1e-14, "{case}: u_l, v_l error {df:e}");
}

/// `compute_sve` agrees with the SVE by dense CPQR in f64 (bitwise) and in
/// Df64 (by ACA, to the working precision), for a centrosymmetric kernel and
/// one that is not; `compute_sve_general` of the centrosymmetric kernel, which
/// does not use the ACA, agrees bitwise. The kernels of the bases built on
/// this crate are checked the same way there, at larger sizes.
#[test]
fn test_sve_matches_dense_cpqr() {
    for (eps, twork) in [(1e-6, TworkType::Float64), (1e-15, TworkType::Float64X2)] {
        for full_domain in [false, true] {
            check_sve_against_dense_cpqr(ExpKernel { a: 30.0, nseg: 8 }, eps, twork, full_domain);
        }
        check_sve_against_dense_cpqr(
            TiltedExpKernel(ExpKernel { a: 30.0, nseg: 4 }),
            eps,
            twork,
            false,
        );
    }
}

/// The block SVDs of exp(50 x y) in double-double precision, whose ACA
/// factors make the Cholesky factor of the recompression NaN, finish (they
/// looped forever in the SVD of that factor) and agree with the SVE of the
/// same discretization in f64.
#[test]
fn test_compute_sve_of_a_kernel_whose_cholesky_recompression_fails() {
    let kernel = ExpKernel { a: 50.0, nseg: 2 };
    let df64 = compute_sve(kernel, Some(1e-12), None, None, TworkType::Float64X2).unwrap();
    let f64 = compute_sve(kernel, Some(1e-12), None, None, TworkType::Float64).unwrap();
    let (s, s_f64) = (df64.s(), f64.s());
    assert!(s.len() > s_f64.len(), "{} vs {}", s.len(), s_f64.len());
    for (l, (a, b)) in s.iter().zip(s_f64).enumerate() {
        // Measured: at most 2e-15 s_0
        assert!((a - b).abs() <= 1e-13 * s[0], "s_{l}: {a:e} vs {b:e}");
    }
}
