//! Basis construction at the edges of its valid input, through the C API.
//!
//! Each test states the status the library returned before the fix.

use sparse_ir_capi::*;
use std::ptr;

const BETA: f64 = 10.0;
const WMAX: f64 = 1.0;
const EPS: f64 = 1e-6;

const STATISTICS: [i32; 2] = [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC];

fn new_basis(statistics: i32, kernel: *const spir_kernel, max_size: i32) -> *mut spir_basis {
    let mut status = SPIR_INTERNAL_ERROR;
    let basis = spir_basis_new(
        statistics,
        BETA,
        WMAX,
        EPS,
        kernel,
        ptr::null(),
        max_size,
        &mut status,
    );
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS, "max_size = {max_size}");
    assert!(!basis.is_null());
    basis
}

fn singular_values(basis: *const spir_basis) -> Vec<f64> {
    let mut size = 0;
    assert_eq!(
        spir_basis_get_size(basis, &mut size),
        SPIR_COMPUTATION_SUCCESS
    );
    let mut svals = vec![0.0; size as usize];
    assert_eq!(
        spir_basis_get_svals(basis, svals.as_mut_ptr()),
        SPIR_COMPUTATION_SUCCESS
    );
    svals
}

/// Until #285, `max_size = 1` truncated the SVE to its largest singular
/// value, which is even, so the odd block of the SVE was empty. The empty
/// block made the core panic and `spir_basis_new` returned
/// SPIR_INTERNAL_ERROR (-7) with a NULL basis. The SVE is now kept in full
/// and only the basis is truncated; the SVE truncation itself is tested in
/// the core (`compute_sve` with `max_num_svals = Some(1)`).
#[test]
fn basis_new_accepts_max_size_one() {
    let mut status = SPIR_INTERNAL_ERROR;
    let kernel = spir_logistic_kernel_new(BETA * WMAX, &mut status);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

    for statistics in STATISTICS {
        let full = new_basis(statistics, kernel, -1);
        let full_svals = singular_values(full);
        assert!(full_svals.len() > 2);
        for max_size in [1, 2] {
            let basis = new_basis(statistics, kernel, max_size);
            // The same SVE computation, truncated: the leading values agree exactly.
            assert_eq!(
                singular_values(basis),
                full_svals[..max_size as usize].to_vec()
            );
            spir_basis_release(basis);
        }
        spir_basis_release(full);
    }
    spir_kernel_release(kernel);
}

// ---------------------------------------------------------------------------
// Default Matsubara points of a basis without parity (#183)
// ---------------------------------------------------------------------------

/// A sentinel that no status or point count takes
const UNTOUCHED: i32 = 12345;

/// SVE of the logistic kernel (Λ = β ωmax) computed by
/// `spir_sve_result_from_matrix` from its full-domain discretization. Its
/// singular functions carry no parity (`symm = 0`), unlike those of
/// `spir_sve_result_new` or the centrosymmetric variant.
fn sve_without_parity() -> *mut spir_sve_result {
    use sparse_ir::gauss::legendre;
    use sparse_ir::kernel::{KernelProperties, LogisticKernel, SVEHints};
    use sparse_ir::kernelmatrix::matrix_from_gauss_noncentrosymmetric;

    let kernel = LogisticKernel::new(BETA * WMAX);
    let hints = kernel.sve_hints::<f64>(EPS);
    let mirror = |half: Vec<f64>| -> Vec<f64> {
        let mut full: Vec<f64> = half.iter().rev().map(|&s| -s).collect();
        full.extend_from_slice(&half[1..]);
        full
    };
    let (segs_x, segs_y) = (mirror(hints.segments_x()), mirror(hints.segments_y()));
    let rule = legendre::<f64>(hints.ngauss());
    let matrix = matrix_from_gauss_noncentrosymmetric(
        &kernel,
        &rule.piecewise(&segs_x),
        &rule.piecewise(&segs_y),
        &hints,
    )
    .apply_weights_for_sve();
    let (nx, ny) = *matrix.shape();
    let k_high: Vec<f64> = (0..nx * ny).map(|k| matrix[[k / ny, k % ny]]).collect();

    let mut status = SPIR_INTERNAL_ERROR;
    let sve = spir_sve_result_from_matrix(
        k_high.as_ptr(),
        ptr::null(),
        nx as i32,
        ny as i32,
        SPIR_ORDER_ROW_MAJOR,
        segs_x.as_ptr(),
        (segs_x.len() - 1) as i32,
        segs_y.as_ptr(),
        (segs_y.len() - 1) as i32,
        hints.ngauss() as i32,
        EPS,
        &mut status,
    );
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    assert!(!sve.is_null());
    sve
}

/// Status of each default-Matsubara getter for `basis`, and whether the
/// getter left its outputs untouched
fn default_matsus_statuses(basis: *const spir_basis) -> Vec<(&'static str, i32, bool)> {
    let mut rows = Vec::new();
    for positive_only in [false, true] {
        let mut n = UNTOUCHED;
        let status = spir_basis_get_n_default_matsus(basis, positive_only, &mut n);
        rows.push(("spir_basis_get_n_default_matsus", status, n == UNTOUCHED));

        let mut points = vec![i64::from(UNTOUCHED); 64];
        let status = spir_basis_get_default_matsus(basis, positive_only, points.as_mut_ptr());
        let untouched = points.iter().all(|&p| p == i64::from(UNTOUCHED));
        rows.push(("spir_basis_get_default_matsus", status, untouched));

        for fence in [false, true] {
            let mut n = UNTOUCHED;
            let status =
                spir_basis_get_n_default_matsus_ext(basis, positive_only, fence, 10, &mut n);
            rows.push((
                "spir_basis_get_n_default_matsus_ext",
                status,
                n == UNTOUCHED,
            ));

            let (mut points, mut n) = (vec![i64::from(UNTOUCHED); 64], UNTOUCHED);
            let status = spir_basis_get_default_matsus_ext(
                basis,
                positive_only,
                fence,
                10,
                points.len() as i32,
                points.as_mut_ptr(),
                &mut n,
            );
            let untouched = n == UNTOUCHED && points.iter().all(|&p| p == i64::from(UNTOUCHED));
            rows.push(("spir_basis_get_default_matsus_ext", status, untouched));

            for getter in [spir_basis_get_uhat, spir_basis_get_uhat_full] {
                let mut status = SPIR_INTERNAL_ERROR;
                let uhat = unsafe { getter(basis, &mut status) };
                assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
                let (mut points, mut n) = (vec![i64::from(UNTOUCHED); 64], UNTOUCHED);
                let status = spir_uhat_get_default_matsus(
                    uhat,
                    positive_only,
                    fence,
                    10,
                    points.len() as i32,
                    points.as_mut_ptr(),
                    &mut n,
                );
                let untouched = n == UNTOUCHED && points.iter().all(|&p| p == i64::from(UNTOUCHED));
                rows.push(("spir_uhat_get_default_matsus", status, untouched));
                spir_funcs_release(uhat);
            }
        }
    }
    rows
}

/// The default Matsubara points are sign changes of the real or imaginary
/// part of a basis function, chosen by its parity. A basis built on an SVE
/// without parity (`spir_sve_result_from_matrix`) has none: every getter
/// must report SPIR_NOT_SUPPORTED and write nothing. Before the fix the core
/// panicked ("Cannot detect parity") and the getters returned
/// SPIR_INTERNAL_ERROR (-7).
#[test]
fn default_matsus_of_a_basis_without_parity_are_not_supported() {
    let mut status = SPIR_INTERNAL_ERROR;
    let kernel = spir_logistic_kernel_new(BETA * WMAX, &mut status);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    let sve = sve_without_parity();

    for statistics in STATISTICS {
        let basis = spir_basis_new(statistics, BETA, WMAX, EPS, kernel, sve, -1, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        for (getter, status, untouched) in default_matsus_statuses(basis) {
            assert_eq!(
                status, SPIR_NOT_SUPPORTED,
                "{getter}, statistics {statistics}"
            );
            assert!(untouched, "{getter} wrote its output");
        }

        // The default tau points do not need the parity
        let mut n_taus = 0;
        assert_eq!(
            spir_basis_get_n_default_taus(basis, &mut n_taus),
            SPIR_COMPUTATION_SUCCESS
        );
        assert!(n_taus > 0);
        spir_basis_release(basis);
    }
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}

/// The same getters succeed for a basis of the same kernel with parity
#[test]
fn default_matsus_of_a_basis_with_parity_are_supported() {
    let mut status = SPIR_INTERNAL_ERROR;
    let kernel = spir_logistic_kernel_new(BETA * WMAX, &mut status);
    assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
    for statistics in STATISTICS {
        let basis = new_basis(statistics, kernel, -1);
        for (getter, status, untouched) in default_matsus_statuses(basis) {
            assert_eq!(
                status, SPIR_COMPUTATION_SUCCESS,
                "{getter}, statistics {statistics}"
            );
            assert!(!untouched, "{getter} wrote nothing");
        }
        spir_basis_release(basis);
    }
    spir_kernel_release(kernel);
}
