//! Known-invalid input must come back from the C API as `SPIR_INVALID_ARGUMENT`,
//! with a null handle or untouched output, instead of reaching a Rust panic
//! that the boundary reports as `SPIR_INTERNAL_ERROR`.
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
