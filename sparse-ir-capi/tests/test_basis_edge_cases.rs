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

/// `max_size = 1` truncates the SVE to its largest singular value, which is
/// even, so the odd block of the SVE is empty. Before the fix the empty block
/// made the core panic and `spir_basis_new` returned SPIR_INTERNAL_ERROR (-7)
/// with a NULL basis.
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
