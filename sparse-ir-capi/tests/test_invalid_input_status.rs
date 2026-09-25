//! Known-invalid input must come back from the C API as `SPIR_INVALID_ARGUMENT`
//! (`SPIR_INVALID_DIMENSION` for an array too large to be addressed), with a
//! null handle or untouched output, instead of reaching a Rust panic that the
//! boundary reports as `SPIR_INTERNAL_ERROR`, or a crash.
//!
//! Each test states the status the library returned before the fix.

use sparse_ir_capi::*;
use std::ptr;

// Setup of the reports in #266: beta = 1, wmax = 10, eps = 1e-6.
const BETA: f64 = 1.0;
const WMAX: f64 = 10.0;
const EPS: f64 = 1e-6;

const STATISTICS: [i32; 2] = [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC];

/// A kernel, an IR basis and its DLR, released on drop.
struct Fixture {
    kernel: *mut spir_kernel,
    basis: *mut spir_basis,
    dlr: *mut spir_basis,
}

impl Fixture {
    fn new(statistics: i32) -> Self {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(BETA * WMAX, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let basis = spir_basis_new(
            statistics,
            BETA,
            WMAX,
            EPS,
            kernel,
            ptr::null(),
            -1,
            &mut status,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        let dlr = spir_dlr_new(basis, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        Self { kernel, basis, dlr }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        spir_basis_release(self.dlr);
        spir_basis_release(self.basis);
        spir_kernel_release(self.kernel);
    }
}

/// An owned `spir_funcs` handle, released on drop.
struct Funcs(*mut spir_funcs);

impl Drop for Funcs {
    fn drop(&mut self) {
        spir_funcs_release(self.0);
    }
}

impl Funcs {
    fn size(&self) -> i32 {
        let mut size = -1;
        assert_eq!(
            spir_funcs_get_size(self.0, &mut size),
            SPIR_COMPUTATION_SUCCESS
        );
        size
    }
}

type FuncsGetter = unsafe extern "C" fn(*const spir_basis, *mut StatusCode) -> *mut spir_funcs;

fn get_funcs(basis: *const spir_basis, getter: FuncsGetter) -> Funcs {
    let mut status = SPIR_INTERNAL_ERROR;
    let funcs = unsafe { getter(basis, &mut status) };
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    assert!(!funcs.is_null());
    Funcs(funcs)
}

/// Which argument a function set is evaluated at.
#[derive(Clone, Copy, Debug)]
enum Domain {
    Tau,
    Omega,
    Matsubara,
}

/// Every function set the C API hands out for a basis of the given statistics.
fn all_function_sets(fx: &Fixture) -> Vec<(&'static str, Domain, Funcs)> {
    vec![
        ("u", Domain::Tau, get_funcs(fx.basis, spir_basis_get_u)),
        ("v", Domain::Omega, get_funcs(fx.basis, spir_basis_get_v)),
        (
            "uhat",
            Domain::Matsubara,
            get_funcs(fx.basis, spir_basis_get_uhat),
        ),
        (
            "uhat_full",
            Domain::Matsubara,
            get_funcs(fx.basis, spir_basis_get_uhat_full),
        ),
        ("dlr u", Domain::Tau, get_funcs(fx.dlr, spir_basis_get_u)),
        (
            "dlr uhat",
            Domain::Matsubara,
            get_funcs(fx.dlr, spir_basis_get_uhat),
        ),
    ]
}

/// A Matsubara index that is valid for the statistics.
fn valid_matsubara_index(statistics: i32) -> i64 {
    if statistics == SPIR_STATISTICS_FERMIONIC {
        1
    } else {
        2
    }
}

/// Values of every function of `funcs` at one representative point.
fn values_at_sample_point(funcs: &Funcs, domain: Domain, statistics: i32) -> Vec<(f64, f64)> {
    let n = funcs.size() as usize;
    match domain {
        Domain::Tau | Domain::Omega => {
            let x = match domain {
                Domain::Tau => 0.3 * BETA,
                _ => 0.3 * WMAX,
            };
            let mut out = vec![f64::NAN; n];
            assert_eq!(
                spir_funcs_eval(funcs.0, x, out.as_mut_ptr()),
                SPIR_COMPUTATION_SUCCESS
            );
            out.into_iter().map(|v| (v, 0.0)).collect()
        }
        Domain::Matsubara => {
            let mut out = vec![num_complex::Complex64::new(f64::NAN, f64::NAN); n];
            assert_eq!(
                spir_funcs_eval_matsu(funcs.0, valid_matsubara_index(statistics), out.as_mut_ptr()),
                SPIR_COMPUTATION_SUCCESS
            );
            out.into_iter().map(|z| (z.re, z.im)).collect()
        }
    }
}

/// Calls `spir_funcs_get_slice` and returns (status, handle).
fn get_slice(funcs: &Funcs, indices: &[i32]) -> (StatusCode, *mut spir_funcs) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sliced = spir_funcs_get_slice(funcs.0, indices.len() as i32, indices.as_ptr(), &mut status);
    (status, sliced)
}

// ---------------------------------------------------------------------------
// spir_funcs_get_slice (#269)
// ---------------------------------------------------------------------------

/// Before the fix, an empty selection panicked for u and v (-7) and returned
/// an empty set (0) for the Matsubara and DLR function sets.
#[test]
fn get_slice_rejects_empty_selection_for_every_function_type() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, _, funcs) in all_function_sets(&fx) {
            // A non-null pointer with nslice = 0 reaches the size check, not
            // the null-pointer check.
            let (status, sliced) = get_slice(&funcs, &[]);
            assert_eq!(
                status, SPIR_INVALID_ARGUMENT,
                "{name}, statistics {statistics}"
            );
            assert!(sliced.is_null(), "{name}, statistics {statistics}");
        }
    }
}

/// Before the fix, repeated indices were accepted (0) and the function was
/// repeated in the result; libsparseir rejected them.
#[test]
fn get_slice_rejects_duplicate_indices() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, _, funcs) in all_function_sets(&fx) {
            for indices in [[0, 0], [1, 1]] {
                let (status, sliced) = get_slice(&funcs, &indices);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name} {indices:?}, statistics {statistics}"
                );
                assert!(sliced.is_null());
            }
            let (status, sliced) = get_slice(&funcs, &[1, 0, 1]);
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "{name}");
            assert!(sliced.is_null());
        }
    }
}

/// Negative and out-of-range indices were already rejected; keep it so.
#[test]
fn get_slice_rejects_indices_out_of_range() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, _, funcs) in all_function_sets(&fx) {
            let size = funcs.size();
            for indices in [
                vec![-1],
                vec![i32::MIN],
                vec![size],
                vec![i32::MAX],
                vec![0, size],
                vec![0, -1],
            ] {
                let (status, sliced) = get_slice(&funcs, &indices);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name} {indices:?}, size {size}, statistics {statistics}"
                );
                assert!(sliced.is_null());
            }
        }
    }
}

/// The first and last index are valid, a selection keeps the caller's order,
/// and the selected functions keep their values.
#[test]
fn get_slice_selects_functions_in_the_given_order() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, domain, funcs) in all_function_sets(&fx) {
            let size = funcs.size();
            assert!(size >= 2, "{name}");
            let all = values_at_sample_point(&funcs, domain, statistics);

            for indices in [vec![0], vec![size - 1], vec![size - 1, 0]] {
                let (status, sliced) = get_slice(&funcs, &indices);
                assert_eq!(
                    status, SPIR_COMPUTATION_SUCCESS,
                    "{name} {indices:?}, statistics {statistics}"
                );
                assert!(!sliced.is_null());
                let sliced = Funcs(sliced);
                assert_eq!(sliced.size(), indices.len() as i32);

                let got = values_at_sample_point(&sliced, domain, statistics);
                for (k, &i) in indices.iter().enumerate() {
                    assert_eq!(got[k], all[i as usize], "{name} {indices:?}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// spir_matsu_sampling_new / spir_matsu_sampling_new_with_matrix (#247)
// ---------------------------------------------------------------------------

/// The default Matsubara sampling points of `basis`, in ascending order.
fn default_matsus(basis: *const spir_basis, positive_only: bool) -> Vec<i64> {
    let mut n = -1;
    assert_eq!(
        spir_basis_get_n_default_matsus(basis, positive_only, &mut n),
        SPIR_COMPUTATION_SUCCESS
    );
    let mut points = vec![i64::MIN; n as usize];
    assert_eq!(
        spir_basis_get_default_matsus(basis, positive_only, points.as_mut_ptr()),
        SPIR_COMPUTATION_SUCCESS
    );
    assert!(points.windows(2).all(|w| w[0] < w[1]));
    points
}

/// `points` with one index moved to the wrong parity; the order is kept.
fn with_wrong_parity(points: &[i64]) -> Vec<i64> {
    let mut bad = points.to_vec();
    let k = bad.len() / 2;
    // Neighbours of the same parity differ by at least 2, so +1 stays sorted.
    bad[k] += 1;
    bad
}

/// Positive-only `points` whose smallest index is replaced by a negative
/// index of the right parity; the order is kept.
fn with_negative_index(points: &[i64], statistics: i32) -> Vec<i64> {
    let mut bad = points.to_vec();
    bad[0] = if statistics == SPIR_STATISTICS_FERMIONIC {
        -1
    } else {
        -2
    };
    bad
}

struct Sampling(*mut spir_sampling);

impl Drop for Sampling {
    fn drop(&mut self) {
        spir_sampling_release(self.0);
    }
}

impl Sampling {
    fn matsus(&self) -> Vec<i64> {
        let mut n = -1;
        assert_eq!(
            spir_sampling_get_npoints(self.0, &mut n),
            SPIR_COMPUTATION_SUCCESS
        );
        let mut points = vec![i64::MIN; n as usize];
        assert_eq!(
            spir_sampling_get_matsus(self.0, points.as_mut_ptr()),
            SPIR_COMPUTATION_SUCCESS
        );
        points
    }

    fn npoints(&self) -> usize {
        let mut n = -1;
        assert_eq!(
            spir_sampling_get_npoints(self.0, &mut n),
            SPIR_COMPUTATION_SUCCESS
        );
        n as usize
    }

    fn taus(&self) -> Vec<f64> {
        let mut points = vec![f64::NAN; self.npoints()];
        assert_eq!(
            spir_sampling_get_taus(self.0, points.as_mut_ptr()),
            SPIR_COMPUTATION_SUCCESS
        );
        points
    }

    /// Values at the sampling points of the real coefficients `coeffs`.
    fn eval_dd(&self, coeffs: &[f64]) -> Vec<f64> {
        let dims = [coeffs.len() as i32];
        let mut out = vec![f64::NAN; self.npoints()];
        assert_eq!(
            spir_sampling_eval_dd(
                self.0,
                ptr::null(),
                SPIR_ORDER_ROW_MAJOR,
                1,
                dims.as_ptr(),
                0,
                coeffs.as_ptr(),
                out.as_mut_ptr(),
            ),
            SPIR_COMPUTATION_SUCCESS
        );
        out
    }
}

fn matsu_sampling_new(
    basis: *const spir_basis,
    positive_only: bool,
    points: &[i64],
) -> (StatusCode, *mut spir_sampling) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sampling = spir_matsu_sampling_new(
        basis,
        positive_only,
        points.len() as i32,
        points.as_ptr(),
        &mut status,
    );
    (status, sampling)
}

fn matsu_sampling_new_with_matrix(
    statistics: i32,
    basis_size: i32,
    positive_only: bool,
    points: &[i64],
    matrix: &[num_complex::Complex64],
) -> (StatusCode, *mut spir_sampling) {
    assert_eq!(matrix.len(), points.len() * basis_size as usize);
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sampling = spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR,
        statistics,
        basis_size,
        positive_only,
        points.len() as i32,
        points.as_ptr(),
        matrix.as_ptr(),
        &mut status,
    );
    (status, sampling)
}

/// A dense stand-in sampling matrix; validation must not depend on it.
fn stand_in_matrix(n_points: usize, basis_size: usize) -> Vec<num_complex::Complex64> {
    (0..n_points * basis_size)
        .map(|k| num_complex::Complex64::new(1.0 + k as f64, 0.5))
        .collect()
}

/// Before the fix, a parity-invalid index reached
/// `MatsubaraFreq::new(n).expect(...)` and returned -7.
#[test]
fn matsu_sampling_new_rejects_wrong_parity() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, basis) in [("ir", fx.basis), ("dlr", fx.dlr)] {
            for positive_only in [false, true] {
                let points = with_wrong_parity(&default_matsus(fx.basis, positive_only));
                let (status, sampling) = matsu_sampling_new(basis, positive_only, &points);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}, statistics {statistics}, positive_only {positive_only}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// Before the fix, a negative index with positive_only reached the core
/// `assert!` in `MatsubaraSamplingPositiveOnly::with_sampling_points` (-7).
#[test]
fn matsu_sampling_new_rejects_negative_index_with_positive_only() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, basis) in [("ir", fx.basis), ("dlr", fx.dlr)] {
            let points = with_negative_index(&default_matsus(fx.basis, true), statistics);
            let (status, sampling) = matsu_sampling_new(basis, true, &points);
            assert_eq!(
                status, SPIR_INVALID_ARGUMENT,
                "{name}, statistics {statistics}, points {points:?}"
            );
            assert!(sampling.is_null());
        }
    }
}

/// Valid indices still build a sampling object: negative indices on the
/// full grid, and n = 0 for bosons with positive_only.
#[test]
fn matsu_sampling_new_accepts_valid_indices() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, basis) in [("ir", fx.basis), ("dlr", fx.dlr)] {
            for positive_only in [false, true] {
                let points = default_matsus(fx.basis, positive_only);
                if positive_only {
                    assert!(points.iter().all(|&n| n >= 0));
                } else {
                    assert!(points.iter().any(|&n| n < 0));
                }
                if statistics == SPIR_STATISTICS_BOSONIC {
                    assert!(points.contains(&0));
                }
                let (status, sampling) = matsu_sampling_new(basis, positive_only, &points);
                assert_eq!(
                    status, SPIR_COMPUTATION_SUCCESS,
                    "{name}, statistics {statistics}, positive_only {positive_only}"
                );
                assert!(!sampling.is_null());
                assert_eq!(Sampling(sampling).matsus(), points);
            }
        }
    }
}

/// Before the fix, a parity-invalid index returned -7, as in
/// `spir_matsu_sampling_new`.
#[test]
fn matsu_sampling_new_with_matrix_rejects_wrong_parity() {
    let basis_size = 3;
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for positive_only in [false, true] {
            let points = with_wrong_parity(&default_matsus(fx.basis, positive_only));
            let matrix = stand_in_matrix(points.len(), basis_size as usize);
            let (status, sampling) = matsu_sampling_new_with_matrix(
                statistics,
                basis_size,
                positive_only,
                &points,
                &matrix,
            );
            assert_eq!(
                status, SPIR_INVALID_ARGUMENT,
                "statistics {statistics}, positive_only {positive_only}"
            );
            assert!(sampling.is_null());
        }
    }
}

/// Before the fix, a negative index with positive_only was accepted (0),
/// unlike in `spir_matsu_sampling_new`.
#[test]
fn matsu_sampling_new_with_matrix_rejects_negative_index_with_positive_only() {
    let basis_size = 3;
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let points = with_negative_index(&default_matsus(fx.basis, true), statistics);
        let matrix = stand_in_matrix(points.len(), basis_size as usize);
        let (status, sampling) =
            matsu_sampling_new_with_matrix(statistics, basis_size, true, &points, &matrix);
        assert_eq!(
            status, SPIR_INVALID_ARGUMENT,
            "statistics {statistics}, points {points:?}"
        );
        assert!(sampling.is_null());
    }
}

/// Unknown statistics were already rejected; keep it so.
#[test]
fn matsu_sampling_new_with_matrix_rejects_unknown_statistics() {
    let points = [1i64, 3];
    let matrix = stand_in_matrix(points.len(), 2);
    for statistics in [-1, 2] {
        for positive_only in [false, true] {
            let (status, sampling) =
                matsu_sampling_new_with_matrix(statistics, 2, positive_only, &points, &matrix);
            assert_eq!(status, SPIR_INVALID_ARGUMENT, "statistics {statistics}");
            assert!(sampling.is_null());
        }
    }
}

/// Valid indices with the basis' own sampling matrix still build a sampling
/// object that reports the given points.
#[test]
fn matsu_sampling_new_with_matrix_accepts_valid_indices() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let uhat = get_funcs(fx.basis, spir_basis_get_uhat);
        let basis_size = uhat.size();
        for positive_only in [false, true] {
            let points = default_matsus(fx.basis, positive_only);
            let mut matrix = vec![
                num_complex::Complex64::new(f64::NAN, f64::NAN);
                points.len() * basis_size as usize
            ];
            assert_eq!(
                spir_funcs_batch_eval_matsu(
                    uhat.0,
                    SPIR_ORDER_ROW_MAJOR,
                    points.len() as i32,
                    points.as_ptr(),
                    matrix.as_mut_ptr(),
                ),
                SPIR_COMPUTATION_SUCCESS
            );
            let (status, sampling) = matsu_sampling_new_with_matrix(
                statistics,
                basis_size,
                positive_only,
                &points,
                &matrix,
            );
            assert_eq!(
                status, SPIR_COMPUTATION_SUCCESS,
                "statistics {statistics}, positive_only {positive_only}"
            );
            assert!(!sampling.is_null());
            assert_eq!(Sampling(sampling).matsus(), points);
        }
    }
}

// ---------------------------------------------------------------------------
// spir_funcs_eval / spir_funcs_batch_eval, spir_funcs_eval_matsu /
// spir_funcs_batch_eval_matsu and spir_tau_sampling_new (#266)
// ---------------------------------------------------------------------------

/// The next representable value above `x` (finite `x`).
fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// The next representable value below `x` (finite `x`).
fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

const OUT_SENTINEL: f64 = -12345.5;

fn knots(funcs: &Funcs) -> Vec<f64> {
    let mut n = -1;
    assert_eq!(
        spir_funcs_get_n_knots(funcs.0, &mut n),
        SPIR_COMPUTATION_SUCCESS
    );
    let mut knots = vec![f64::NAN; n as usize];
    assert_eq!(
        spir_funcs_get_knots(funcs.0, knots.as_mut_ptr()),
        SPIR_COMPUTATION_SUCCESS
    );
    knots
}

/// Two functions on the segments [0.5, 1] and [1, 2], built from Legendre
/// coefficients; their domain differs from [-beta, beta] of their handle.
fn piecewise_funcs() -> Funcs {
    let segments = [0.5, 1.0, 2.0];
    let coeffs = [1.0, 0.5, -0.25, 2.0];
    let mut status = SPIR_INTERNAL_ERROR;
    let funcs = spir_funcs_from_piecewise_legendre(
        segments.as_ptr(),
        2,
        coeffs.as_ptr(),
        2,
        0,
        &mut status,
    );
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    Funcs(funcs)
}

/// Every τ- or ω-domain function set with its closed domain.
fn continuous_function_sets(fx: &Fixture) -> Vec<(&'static str, Funcs, (f64, f64))> {
    let u = get_funcs(fx.basis, spir_basis_get_u);
    let mut status = SPIR_INTERNAL_ERROR;
    let du = Funcs(spir_funcs_deriv(u.0, 1, &mut status));
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    let v = get_funcs(fx.basis, spir_basis_get_v);
    // omega_max = lambda / beta is exact for this setup.
    let v_knots = knots(&v);
    assert_eq!((v_knots[0], v_knots[v_knots.len() - 1]), (-WMAX, WMAX));
    vec![
        ("u", u, (-BETA, BETA)),
        ("u'", du, (-BETA, BETA)),
        ("v", v, (-WMAX, WMAX)),
        ("dlr u", get_funcs(fx.dlr, spir_basis_get_u), (-BETA, BETA)),
        ("piecewise", piecewise_funcs(), (0.5, 2.0)),
    ]
}

/// Points just outside, far outside and not on the real line.
fn points_outside(lo: f64, hi: f64) -> Vec<f64> {
    vec![
        next_up(hi),
        next_down(lo),
        hi + 0.5,
        lo - 0.5,
        2.0 * hi - lo,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
}

/// The domain endpoints, its midpoint and both zeros when they lie inside.
fn points_inside(lo: f64, hi: f64) -> Vec<f64> {
    let mut points = vec![lo, hi, 0.5 * (lo + hi)];
    if lo <= 0.0 && 0.0 <= hi {
        points.extend([0.0, -0.0]);
    }
    points
}

fn eval(funcs: &Funcs, x: f64) -> (StatusCode, Vec<f64>) {
    let mut out = vec![OUT_SENTINEL; funcs.size() as usize];
    let status = spir_funcs_eval(funcs.0, x, out.as_mut_ptr());
    (status, out)
}

fn batch_eval(funcs: &Funcs, order: i32, xs: &[f64]) -> (StatusCode, Vec<f64>) {
    let mut out = vec![OUT_SENTINEL; funcs.size() as usize * xs.len()];
    let status = spir_funcs_batch_eval(
        funcs.0,
        order,
        xs.len() as i32,
        xs.as_ptr(),
        out.as_mut_ptr(),
    );
    (status, out)
}

/// Before the fix, a point outside the domain panicked in `normalize_tau`
/// or `PiecewiseLegendrePoly::split` (-7), and NaN was accepted (0) with NaN
/// values.
#[test]
fn eval_rejects_points_outside_the_domain() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, funcs, (lo, hi)) in continuous_function_sets(&fx) {
            for x in points_outside(lo, hi) {
                let (status, out) = eval(&funcs, x);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}({x:?}), domain [{lo}, {hi}], statistics {statistics}"
                );
                assert!(out.iter().all(|&v| v == OUT_SENTINEL), "{name}({x:?})");
            }
        }
    }
}

/// Before the fix, one invalid point in a batch panicked (-7) or, for NaN,
/// was accepted (0).
#[test]
fn batch_eval_rejects_any_point_outside_the_domain() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, funcs, (lo, hi)) in continuous_function_sets(&fx) {
            for x in points_outside(lo, hi) {
                let xs = [lo, 0.5 * (lo + hi), x, hi];
                for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
                    let (status, out) = batch_eval(&funcs, order, &xs);
                    assert_eq!(
                        status, SPIR_INVALID_ARGUMENT,
                        "{name} at {xs:?}, statistics {statistics}"
                    );
                    assert!(out.iter().all(|&v| v == OUT_SENTINEL), "{name}");
                }
            }
        }
    }
}

/// The endpoints stay valid, and a batch gives the single-point values.
#[test]
fn eval_accepts_the_closed_domain() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, funcs, (lo, hi)) in continuous_function_sets(&fx) {
            let xs = points_inside(lo, hi);
            let n_funcs = funcs.size() as usize;
            let (status, row_major) = batch_eval(&funcs, SPIR_ORDER_ROW_MAJOR, &xs);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name} at {xs:?}");
            let (status, col_major) = batch_eval(&funcs, SPIR_ORDER_COLUMN_MAJOR, &xs);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name} at {xs:?}");
            for (i, &x) in xs.iter().enumerate() {
                let (status, values) = eval(&funcs, x);
                assert_eq!(
                    status, SPIR_COMPUTATION_SUCCESS,
                    "{name}({x:?}), statistics {statistics}"
                );
                assert!(values.iter().all(|v| v.is_finite()), "{name}({x:?})");
                assert!(values.iter().any(|&v| v != 0.0), "{name}({x:?})");
                for (l, &value) in values.iter().enumerate() {
                    assert_eq!(row_major[i * n_funcs + l], value, "{name}({x:?})");
                    assert_eq!(col_major[l * xs.len() + i], value, "{name}({x:?})");
                }
            }
        }
    }
}

/// Negative τ follows the (anti)periodicity: -beta folds onto 0 and -0.0
/// onto beta, with a sign flip for fermions only.
#[test]
fn eval_folds_negative_tau_with_the_statistics_sign() {
    for statistics in STATISTICS {
        let sign = if statistics == SPIR_STATISTICS_FERMIONIC {
            -1.0
        } else {
            1.0
        };
        let fx = Fixture::new(statistics);
        for (name, funcs) in [
            ("u", get_funcs(fx.basis, spir_basis_get_u)),
            ("dlr u", get_funcs(fx.dlr, spir_basis_get_u)),
        ] {
            for (x, folded) in [(-BETA, 0.0), (-0.0, BETA)] {
                let (status, values) = eval(&funcs, x);
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name}({x:?})");
                let (status, expected) = eval(&funcs, folded);
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name}({folded:?})");
                for (value, expected) in values.iter().zip(&expected) {
                    assert_eq!(
                        *value,
                        sign * expected,
                        "{name}({x:?}), statistics {statistics}"
                    );
                }
            }
        }
    }
}

/// Matsubara functions still report SPIR_NOT_SUPPORTED for a real argument,
/// whatever its value, and τ functions for a Matsubara index.
#[test]
fn eval_on_the_wrong_function_type_is_not_supported() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let uhat = get_funcs(fx.basis, spir_basis_get_uhat);
        for x in [0.5, f64::NAN, 1.5] {
            assert_eq!(eval(&uhat, x).0, SPIR_NOT_SUPPORTED, "uhat({x:?})");
            assert_eq!(
                batch_eval(&uhat, SPIR_ORDER_ROW_MAJOR, &[x]).0,
                SPIR_NOT_SUPPORTED
            );
        }
        let u = get_funcs(fx.basis, spir_basis_get_u);
        for n in [0, 1, 2] {
            assert_eq!(eval_matsu(&u, n).0, SPIR_NOT_SUPPORTED, "u(n = {n})");
            assert_eq!(
                batch_eval_matsu(&u, SPIR_ORDER_ROW_MAJOR, &[n]).0,
                SPIR_NOT_SUPPORTED
            );
        }
    }
}

const OUT_SENTINEL_Z: num_complex::Complex64 = num_complex::Complex64::new(-12345.5, 678.25);

fn eval_matsu(funcs: &Funcs, n: i64) -> (StatusCode, Vec<num_complex::Complex64>) {
    let mut out = vec![OUT_SENTINEL_Z; funcs.size() as usize];
    let status = spir_funcs_eval_matsu(funcs.0, n, out.as_mut_ptr());
    (status, out)
}

fn batch_eval_matsu(
    funcs: &Funcs,
    order: i32,
    ns: &[i64],
) -> (StatusCode, Vec<num_complex::Complex64>) {
    let mut out = vec![OUT_SENTINEL_Z; funcs.size() as usize * ns.len()];
    let status = spir_funcs_batch_eval_matsu(
        funcs.0,
        order,
        ns.len() as i32,
        ns.as_ptr(),
        out.as_mut_ptr(),
    );
    (status, out)
}

fn matsubara_function_sets(fx: &Fixture) -> Vec<(&'static str, Funcs)> {
    vec![
        ("uhat", get_funcs(fx.basis, spir_basis_get_uhat)),
        ("uhat_full", get_funcs(fx.basis, spir_basis_get_uhat_full)),
        ("dlr uhat", get_funcs(fx.dlr, spir_basis_get_uhat)),
    ]
}

/// Indices of the wrong parity, and valid ones, for the statistics.
fn matsubara_indices(statistics: i32) -> (Vec<i64>, Vec<i64>) {
    if statistics == SPIR_STATISTICS_FERMIONIC {
        (vec![0, 2, -2, i64::MIN], vec![1, -1, 3, -3])
    } else {
        (vec![1, -1, 3, i64::MAX], vec![0, 2, -2, 4])
    }
}

/// Before the fix, an index of the wrong parity was reported as
/// SPIR_NOT_SUPPORTED (-5), as if the functions were not Matsubara functions.
#[test]
fn eval_matsu_rejects_wrong_parity() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let (invalid, valid) = matsubara_indices(statistics);
        for (name, funcs) in matsubara_function_sets(&fx) {
            for &n in &invalid {
                let (status, out) = eval_matsu(&funcs, n);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}(n = {n}), statistics {statistics}"
                );
                assert!(out.iter().all(|&z| z == OUT_SENTINEL_Z));

                let ns = [valid[0], n, valid[1]];
                for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
                    let (status, out) = batch_eval_matsu(&funcs, order, &ns);
                    assert_eq!(
                        status, SPIR_INVALID_ARGUMENT,
                        "{name} at {ns:?}, statistics {statistics}"
                    );
                    assert!(out.iter().all(|&z| z == OUT_SENTINEL_Z));
                }
            }
        }
    }
}

/// Valid indices, negative ones and n = 0 for bosons included, still
/// evaluate, and a batch gives the single-index values.
#[test]
fn eval_matsu_accepts_valid_indices() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let (_, valid) = matsubara_indices(statistics);
        for (name, funcs) in matsubara_function_sets(&fx) {
            let n_funcs = funcs.size() as usize;
            let (status, batch) = batch_eval_matsu(&funcs, SPIR_ORDER_ROW_MAJOR, &valid);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name} at {valid:?}");
            for (i, &n) in valid.iter().enumerate() {
                let (status, values) = eval_matsu(&funcs, n);
                assert_eq!(
                    status, SPIR_COMPUTATION_SUCCESS,
                    "{name}(n = {n}), statistics {statistics}"
                );
                assert!(values.iter().any(|z| z.norm() > 0.0), "{name}(n = {n})");
                assert_eq!(&batch[i * n_funcs..(i + 1) * n_funcs], &values[..]);
            }
        }
    }
}

fn default_taus(basis: *const spir_basis) -> Vec<f64> {
    let mut n = -1;
    assert_eq!(
        spir_basis_get_n_default_taus(basis, &mut n),
        SPIR_COMPUTATION_SUCCESS
    );
    let mut points = vec![f64::NAN; n as usize];
    assert_eq!(
        spir_basis_get_default_taus(basis, points.as_mut_ptr()),
        SPIR_COMPUTATION_SUCCESS
    );
    points
}

fn tau_sampling_new(basis: *const spir_basis, points: &[f64]) -> (StatusCode, *mut spir_sampling) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sampling = spir_tau_sampling_new(basis, points.len() as i32, points.as_ptr(), &mut status);
    (status, sampling)
}

/// Before the fix, a point outside [-beta, beta], NaN or an infinity reached
/// the `assert!` in `TauSampling::with_sampling_points` (-7).
#[test]
fn tau_sampling_new_rejects_points_outside_the_domain() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let taus = default_taus(fx.basis);
        for (name, basis) in [("ir", fx.basis), ("dlr", fx.dlr)] {
            // 1.1666666666666665 is the point reported in #266.
            let mut outside = points_outside(-BETA, BETA);
            outside.push(1.1666666666666665);
            for x in outside {
                let mut points = taus.clone();
                points[taus.len() / 2] = x;
                let (status, sampling) = tau_sampling_new(basis, &points);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}, point {x:?}, statistics {statistics}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// The endpoints and both zeros stay valid sampling points.
#[test]
fn tau_sampling_new_accepts_the_closed_domain() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let mut points = default_taus(fx.basis);
        points.extend([-BETA, -0.0, 0.0, BETA]);
        for (name, basis) in [("ir", fx.basis), ("dlr", fx.dlr)] {
            let (status, sampling) = tau_sampling_new(basis, &points);
            assert_eq!(
                status, SPIR_COMPUTATION_SUCCESS,
                "{name}, statistics {statistics}"
            );
            assert!(!sampling.is_null());
            let sampling = Sampling(sampling);
            let mut n = -1;
            assert_eq!(
                spir_sampling_get_npoints(sampling.0, &mut n),
                SPIR_COMPUTATION_SUCCESS
            );
            let mut got = vec![f64::NAN; n as usize];
            assert_eq!(
                spir_sampling_get_taus(sampling.0, got.as_mut_ptr()),
                SPIR_COMPUTATION_SUCCESS
            );
            assert_eq!(got, points, "{name}");
        }
    }
}

// ---------------------------------------------------------------------------
// spir_tau_sampling_new_with_matrix / spir_matsu_sampling_new_with_matrix:
// shape and entries of the matrix (#245)
// ---------------------------------------------------------------------------

/// Calls `spir_tau_sampling_new_with_matrix` with an explicit shape, which
/// need not match the buffers: the shape must be rejected before either
/// buffer is read.
fn tau_sampling_new_with_matrix_raw(
    order: i32,
    statistics: i32,
    basis_size: i32,
    num_points: i32,
    points: &[f64],
    matrix: &[f64],
) -> (StatusCode, *mut spir_sampling) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sampling = spir_tau_sampling_new_with_matrix(
        order,
        statistics,
        basis_size,
        num_points,
        points.as_ptr(),
        matrix.as_ptr(),
        &mut status,
    );
    (status, sampling)
}

/// As [`tau_sampling_new_with_matrix_raw`] for `spir_matsu_sampling_new_with_matrix`.
fn matsu_sampling_new_with_matrix_raw(
    order: i32,
    statistics: i32,
    basis_size: i32,
    positive_only: bool,
    num_points: i32,
    points: &[i64],
    matrix: &[num_complex::Complex64],
) -> (StatusCode, *mut spir_sampling) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let sampling = spir_matsu_sampling_new_with_matrix(
        order,
        statistics,
        basis_size,
        positive_only,
        num_points,
        points.as_ptr(),
        matrix.as_ptr(),
        &mut status,
    );
    (status, sampling)
}

/// `[num_points, basis_size]` shapes of a matrix whose byte size exceeds
/// `isize::MAX` for elements of `elem_size` bytes, although both extents fit
/// in a `c_int`. No buffer of such a size can exist.
fn unaddressable_matrix_shapes(elem_size: usize) -> Vec<(i32, i32)> {
    let shapes = match elem_size {
        // 2^60 f64 = 2^63 bytes
        8 => vec![(i32::MAX, i32::MAX), (1 << 30, 1 << 30)],
        // 2^59 Complex64 = 2^63 bytes
        16 => vec![(i32::MAX, i32::MAX), (1 << 30, 1 << 29), (1 << 29, 1 << 30)],
        _ => unreachable!(),
    };
    for &(n, l) in &shapes {
        let bytes = (n as u128) * (l as u128) * elem_size as u128;
        assert!(bytes > isize::MAX as u128, "({n}, {l})");
    }
    shapes
}

/// Before the fix, the shape was never checked: the points were read at the
/// claimed length (a segfault for these small buffers), and a matrix of more
/// than `isize::MAX` bytes was then viewed as a slice.
#[test]
fn tau_sampling_new_with_matrix_rejects_unaddressable_matrix() {
    let points = [0.25, 0.5, 0.75];
    let matrix = [1.0; 9];
    for (num_points, basis_size) in unaddressable_matrix_shapes(8) {
        for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
            for statistics in STATISTICS {
                let (status, sampling) = tau_sampling_new_with_matrix_raw(
                    order, statistics, basis_size, num_points, &points, &matrix,
                );
                assert_eq!(
                    status, SPIR_INVALID_DIMENSION,
                    "shape [{num_points}, {basis_size}], order {order}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// As for τ; the Matsubara indices were read before the shape (a segfault).
#[test]
fn matsu_sampling_new_with_matrix_rejects_unaddressable_matrix() {
    let points = [1i64, 3, 5];
    let matrix = stand_in_matrix(points.len(), 3);
    for (num_points, basis_size) in unaddressable_matrix_shapes(16) {
        for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
            for positive_only in [false, true] {
                let (status, sampling) = matsu_sampling_new_with_matrix_raw(
                    order,
                    SPIR_STATISTICS_FERMIONIC,
                    basis_size,
                    positive_only,
                    num_points,
                    &points,
                    &matrix,
                );
                assert_eq!(
                    status, SPIR_INVALID_DIMENSION,
                    "shape [{num_points}, {basis_size}], order {order}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// A zero or negative extent was already SPIR_INVALID_ARGUMENT; keep it so.
#[test]
fn with_matrix_rejects_non_positive_extents() {
    let taus = [0.25, 0.5, 0.75];
    let tau_matrix = [1.0; 9];
    let matsus = [1i64, 3, 5];
    let matsu_matrix = stand_in_matrix(3, 3);
    for (num_points, basis_size) in [(0, 3), (3, 0), (-1, 3), (3, -1), (i32::MIN, i32::MIN)] {
        let (status, sampling) = tau_sampling_new_with_matrix_raw(
            SPIR_ORDER_ROW_MAJOR,
            SPIR_STATISTICS_FERMIONIC,
            basis_size,
            num_points,
            &taus,
            &tau_matrix,
        );
        assert_eq!(
            status, SPIR_INVALID_ARGUMENT,
            "[{num_points}, {basis_size}]"
        );
        assert!(sampling.is_null());
        let (status, sampling) = matsu_sampling_new_with_matrix_raw(
            SPIR_ORDER_ROW_MAJOR,
            SPIR_STATISTICS_FERMIONIC,
            basis_size,
            false,
            num_points,
            &matsus,
            &matsu_matrix,
        );
        assert_eq!(
            status, SPIR_INVALID_ARGUMENT,
            "[{num_points}, {basis_size}]"
        );
        assert!(sampling.is_null());
    }
}

/// Unknown statistics or memory order were already rejected; keep it so.
#[test]
fn tau_sampling_new_with_matrix_rejects_unknown_constants() {
    let points = [0.25, 0.5];
    let matrix = [1.0, 0.0, 0.0, 1.0];
    for (order, statistics) in [
        (SPIR_ORDER_ROW_MAJOR, -1),
        (SPIR_ORDER_ROW_MAJOR, 2),
        (-1, SPIR_STATISTICS_FERMIONIC),
        (2, SPIR_STATISTICS_BOSONIC),
    ] {
        let (status, sampling) =
            tau_sampling_new_with_matrix_raw(order, statistics, 2, 2, &points, &matrix);
        assert_eq!(
            status, SPIR_INVALID_ARGUMENT,
            "order {order}, statistics {statistics}"
        );
        assert!(sampling.is_null());
    }
}

/// Real entries that are not finite.
const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

/// Before the fix, a NaN or infinite entry was accepted (0); the fit then
/// factorized a non-finite matrix.
#[test]
fn tau_sampling_new_with_matrix_rejects_non_finite_entries() {
    let (n, l) = (3usize, 2usize);
    let points = [0.25, 0.5, 0.75];
    for bad in NON_FINITE {
        for k in [0, n * l / 2, n * l - 1] {
            let mut matrix: Vec<f64> = (0..n * l).map(|k| 1.0 + k as f64).collect();
            matrix[k] = bad;
            for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
                let (status, sampling) = tau_sampling_new_with_matrix_raw(
                    order,
                    SPIR_STATISTICS_FERMIONIC,
                    l as i32,
                    n as i32,
                    &points,
                    &matrix,
                );
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "entry {k} = {bad:?}, order {order}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// As for τ, in either part of a complex entry.
#[test]
fn matsu_sampling_new_with_matrix_rejects_non_finite_entries() {
    let (n, l) = (3usize, 2usize);
    let points = [-1i64, 1, 3];
    for bad in NON_FINITE {
        for k in [0, n * l / 2, n * l - 1] {
            for bad_entry in [
                num_complex::Complex64::new(bad, 0.5),
                num_complex::Complex64::new(0.5, bad),
            ] {
                let mut matrix = stand_in_matrix(n, l);
                matrix[k] = bad_entry;
                for order in [SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR] {
                    let (status, sampling) = matsu_sampling_new_with_matrix_raw(
                        order,
                        SPIR_STATISTICS_FERMIONIC,
                        l as i32,
                        false,
                        n as i32,
                        &points,
                        &matrix,
                    );
                    assert_eq!(
                        status, SPIR_INVALID_ARGUMENT,
                        "entry {k} = {bad_entry:?}, order {order}"
                    );
                    assert!(sampling.is_null());
                }
            }
        }
    }
}

/// `n × l` values of `u` at `taus`, row-major (`[point][function]`).
fn u_rows(u: &Funcs, taus: &[f64]) -> Vec<f64> {
    let (status, rows) = batch_eval(u, SPIR_ORDER_ROW_MAJOR, taus);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    rows
}

/// `a` (`n × l`, row-major) in column-major order.
fn to_column_major<T: Copy>(a: &[T], n: usize, l: usize) -> Vec<T> {
    (0..n * l).map(|k| a[(k % n) * l + k / n]).collect()
}

/// A valid matrix still builds a sampling object in both memory orders; it
/// keeps the points and evaluates with the given matrix.
#[test]
fn tau_sampling_new_with_matrix_accepts_a_valid_matrix() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let u = get_funcs(fx.basis, spir_basis_get_u);
        let l = u.size() as usize;
        let taus = default_taus(fx.basis);
        let n = taus.len();
        let rows = u_rows(&u, &taus);
        let coeffs: Vec<f64> = (0..l).map(|i| 1.0 / (1.0 + i as f64)).collect();
        for (order, matrix) in [
            (SPIR_ORDER_ROW_MAJOR, rows.clone()),
            (SPIR_ORDER_COLUMN_MAJOR, to_column_major(&rows, n, l)),
        ] {
            let (status, sampling) = tau_sampling_new_with_matrix_raw(
                order, statistics, l as i32, n as i32, &taus, &matrix,
            );
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "order {order}");
            let sampling = Sampling(sampling);
            assert_eq!(sampling.taus(), taus);

            let values = sampling.eval_dd(&coeffs);
            for i in 0..n {
                let expected: f64 = (0..l).map(|j| rows[i * l + j] * coeffs[j]).sum();
                let scale: f64 = (0..l).map(|j| (rows[i * l + j] * coeffs[j]).abs()).sum();
                assert!(
                    (values[i] - expected).abs() <= 1e-13 * scale,
                    "row {i}: {} vs {expected}, order {order}",
                    values[i]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// spir_tau_sampling_new_with_matrix: sampling points (#266)
// ---------------------------------------------------------------------------

/// A `n × l` stand-in for a real sampling matrix; the points are only labels.
fn stand_in_real_matrix(n: usize, l: usize) -> Vec<f64> {
    (0..n * l).map(|k| 1.0 + (k * k) as f64).collect()
}

/// Before the fix, a NaN or infinite point was accepted (0) and reported
/// back by `spir_sampling_get_taus`.
#[test]
fn tau_sampling_new_with_matrix_rejects_non_finite_points() {
    let (n, l) = (3usize, 2usize);
    let matrix = stand_in_real_matrix(n, l);
    for bad in NON_FINITE {
        for k in 0..n {
            let mut points = vec![0.25, 0.5, 0.75];
            points[k] = bad;
            for statistics in STATISTICS {
                let (status, sampling) = tau_sampling_new_with_matrix_raw(
                    SPIR_ORDER_ROW_MAJOR,
                    statistics,
                    l as i32,
                    n as i32,
                    &points,
                    &matrix,
                );
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "points {points:?}, statistics {statistics}"
                );
                assert!(sampling.is_null());
            }
        }
    }
}

/// Without β the τ domain cannot be checked: any finite point is accepted,
/// kept in the given order and reported back unchanged (documented).
#[test]
fn tau_sampling_new_with_matrix_does_not_check_the_tau_domain() {
    let (n, l) = (4usize, 2usize);
    let matrix = stand_in_real_matrix(n, l);
    let points = vec![1e300, -7.5, -0.0, f64::MIN_POSITIVE];
    let (status, sampling) = tau_sampling_new_with_matrix_raw(
        SPIR_ORDER_ROW_MAJOR,
        SPIR_STATISTICS_FERMIONIC,
        l as i32,
        n as i32,
        &points,
        &matrix,
    );
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    let got = Sampling(sampling).taus();
    assert_eq!(got.len(), n);
    for (g, p) in got.iter().zip(&points) {
        assert_eq!(g.to_bits(), p.to_bits(), "{got:?} vs {points:?}");
    }
}

// ---------------------------------------------------------------------------
// spir_funcs_from_piecewise_legendre: array sizes (#245)
// ---------------------------------------------------------------------------

/// Calls `spir_funcs_from_piecewise_legendre` with explicit sizes, which need
/// not match the buffers: an invalid size must be rejected before either
/// buffer is read.
fn from_piecewise_legendre_raw(
    segments: &[f64],
    n_segments: i32,
    coeffs: &[f64],
    nfuncs: i32,
) -> (StatusCode, *mut spir_funcs) {
    let mut status = SPIR_COMPUTATION_SUCCESS - 100;
    let funcs = spir_funcs_from_piecewise_legendre(
        segments.as_ptr(),
        n_segments,
        coeffs.as_ptr(),
        nfuncs,
        0,
        &mut status,
    );
    (status, funcs)
}

/// Before the fix, `n_segments + 1` and `n_segments * nfuncs` were computed
/// in `c_int` and the segments were read before any size check: with
/// `n_segments = INT_MAX` the knot count wrapped to a negative `c_int`, and
/// the segments were read at the wrapped length, which is undefined behavior
/// (seen as -7, from a capacity-overflow panic); for 2^30 segments they were
/// read past these small buffers (a segfault).
#[test]
fn from_piecewise_legendre_rejects_sizes_that_overflow() {
    let segments = [-1.0, 0.0, 1.0];
    let coeffs = [1.0; 4];
    // A knot count of 2^31, which `spir_funcs_get_n_knots` cannot report,
    // then 2^62 - 2^32 + 1 and 2^60 coefficients, more than isize::MAX bytes.
    for (n_segments, nfuncs) in [(i32::MAX, 1), (i32::MAX, i32::MAX), (1 << 30, 1 << 30)] {
        let (status, funcs) = from_piecewise_legendre_raw(&segments, n_segments, &coeffs, nfuncs);
        assert_eq!(
            status, SPIR_INVALID_DIMENSION,
            "n_segments {n_segments}, nfuncs {nfuncs}"
        );
        assert!(funcs.is_null());
    }
}

/// Sizes below 1 were already SPIR_INVALID_ARGUMENT; keep it so.
#[test]
fn from_piecewise_legendre_rejects_sizes_below_one() {
    let segments = [-1.0, 0.0, 1.0];
    let coeffs = [1.0; 4];
    for (n_segments, nfuncs) in [(0, 1), (1, 0), (-1, 1), (1, -1), (i32::MIN, i32::MIN)] {
        let (status, funcs) = from_piecewise_legendre_raw(&segments, n_segments, &coeffs, nfuncs);
        assert_eq!(
            status, SPIR_INVALID_ARGUMENT,
            "n_segments {n_segments}, nfuncs {nfuncs}"
        );
        assert!(funcs.is_null());
    }
}

/// The smallest sizes, one segment with one coefficient, still build the
/// constant function: P_0 normalized on [-1, 1] is 1/sqrt(2) * sqrt(2) = 1.
#[test]
fn from_piecewise_legendre_accepts_the_smallest_sizes() {
    let (status, funcs) = from_piecewise_legendre_raw(&[-1.0, 1.0], 1, &[1.0], 1);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    let funcs = Funcs(funcs);
    assert_eq!(funcs.size(), 1);
    for x in [-1.0, 0.25, 1.0] {
        let (status, values) = eval(&funcs, x);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!((values[0] - 1.0).abs() < 1e-14, "f({x}) = {}", values[0]);
    }
}

// ---------------------------------------------------------------------------
// spir_funcs_from_piecewise_legendre: segment boundaries (#266)
// ---------------------------------------------------------------------------

/// Builds one function on `segments` with the constant coefficient 1 on
/// every segment.
fn from_piecewise_legendre_constant(segments: &[f64]) -> (StatusCode, *mut spir_funcs) {
    let n_segments = segments.len() - 1;
    let coeffs = vec![1.0; n_segments];
    from_piecewise_legendre_raw(segments, n_segments as i32, &coeffs, 1)
}

/// Before the fix, the check `knots[i] <= knots[i - 1]` let NaN through
/// (every comparison with NaN is false), and infinite boundaries passed it:
/// both were accepted (0). The functions then evaluated, with status 0, to
/// NaN on a NaN segment, and to 0.0 instead of a positive constant on an
/// infinite segment. Equal or decreasing boundaries were already rejected
/// (-6).
#[test]
fn from_piecewise_legendre_rejects_non_finite_or_unordered_segments() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    for segments in [
        vec![nan, 1.0],
        vec![-1.0, nan],
        vec![nan, 0.0, 1.0],
        vec![-1.0, nan, 1.0],
        vec![-1.0, 0.0, nan],
        vec![nan, nan],
        vec![-inf, 1.0],
        vec![-1.0, inf],
        vec![-inf, inf],
        vec![-1.0, 0.0, inf],
        vec![-1.0, 0.0, 0.0, 1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0, 0.5],
    ] {
        let (status, funcs) = from_piecewise_legendre_constant(&segments);
        assert_eq!(status, SPIR_INVALID_ARGUMENT, "segments {segments:?}");
        assert!(funcs.is_null(), "segments {segments:?}");
    }
}

/// Before the fix, finite boundaries were accepted (0) even when a segment
/// length is not a normal double: the length of [-DBL_MAX, DBL_MAX]
/// overflows, and the function evaluated to 0.0 everywhere; a subnormal
/// length made every value infinite.
#[test]
fn from_piecewise_legendre_rejects_segment_lengths_that_are_not_normal() {
    let max = f64::MAX;
    for segments in [
        vec![-max, max],
        vec![-0.6 * max, 0.6 * max, max],
        vec![0.0, f64::from_bits(1)],
        vec![0.0, 1e-308],
        vec![-1.0, 0.0, next_down(f64::MIN_POSITIVE)],
    ] {
        let (status, funcs) = from_piecewise_legendre_constant(&segments);
        assert_eq!(status, SPIR_INVALID_ARGUMENT, "segments {segments:?}");
        assert!(funcs.is_null(), "segments {segments:?}");
    }
}

/// Boundaries one ULP apart, and the shortest (DBL_MIN) and longest (DBL_MAX)
/// normal segment lengths, are still accepted, and the constant function is
/// finite and positive at both ends.
#[test]
fn from_piecewise_legendre_accepts_the_extreme_valid_segments() {
    let max = f64::MAX;
    for segments in [
        vec![1.0, next_up(1.0)],
        vec![-1.0, next_up(-1.0), 0.0, 1.0],
        vec![0.0, f64::MIN_POSITIVE],
        vec![-max / 2.0, max / 2.0],
        vec![-1.0, 0.0, max],
    ] {
        let (status, funcs) = from_piecewise_legendre_constant(&segments);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "segments {segments:?}");
        let funcs = Funcs(funcs);
        let (lo, hi) = (segments[0], segments[segments.len() - 1]);
        for x in [lo, hi] {
            let (status, values) = eval(&funcs, x);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "f({x:?}) on {segments:?}");
            assert!(
                values[0].is_finite() && values[0] > 0.0,
                "f({x:?}) = {} on {segments:?}",
                values[0]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// spir_funcs_batch_eval / spir_funcs_batch_eval_matsu: memory order (#266)
// ---------------------------------------------------------------------------

/// Values of `order` that are neither SPIR_ORDER_ROW_MAJOR nor
/// SPIR_ORDER_COLUMN_MAJOR.
const UNKNOWN_ORDERS: [i32; 4] = [-1, 2, i32::MIN, i32::MAX];

/// Before the fix, every `order` other than 0 was taken as column-major: the
/// call succeeded (0) and wrote a column-major result.
#[test]
fn batch_eval_rejects_unknown_order() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        for (name, funcs, (lo, hi)) in continuous_function_sets(&fx) {
            let xs = points_inside(lo, hi);
            for order in UNKNOWN_ORDERS {
                let (status, out) = batch_eval(&funcs, order, &xs);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}, order {order}, statistics {statistics}"
                );
                assert!(out.iter().all(|&v| v == OUT_SENTINEL), "{name}");
            }
        }
    }
}

/// As for `spir_funcs_batch_eval`. Both valid orders still give the same
/// values, as transposes of each other.
#[test]
fn batch_eval_matsu_rejects_unknown_order() {
    for statistics in STATISTICS {
        let fx = Fixture::new(statistics);
        let (_, ns) = matsubara_indices(statistics);
        for (name, funcs) in matsubara_function_sets(&fx) {
            for order in UNKNOWN_ORDERS {
                let (status, out) = batch_eval_matsu(&funcs, order, &ns);
                assert_eq!(
                    status, SPIR_INVALID_ARGUMENT,
                    "{name}, order {order}, statistics {statistics}"
                );
                assert!(out.iter().all(|&z| z == OUT_SENTINEL_Z), "{name}");
            }

            let n_funcs = funcs.size() as usize;
            let (status, row_major) = batch_eval_matsu(&funcs, SPIR_ORDER_ROW_MAJOR, &ns);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name}");
            let (status, col_major) = batch_eval_matsu(&funcs, SPIR_ORDER_COLUMN_MAJOR, &ns);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "{name}");
            for i in 0..ns.len() {
                for l in 0..n_funcs {
                    assert_eq!(
                        row_major[i * n_funcs + l],
                        col_major[l * ns.len() + i],
                        "{name}, n = {}, l = {l}",
                        ns[i]
                    );
                }
            }
        }
    }
}
