//! Checks the numbers the examples produced against reference values.
//!
//! The references come from the published Python implementation
//! (`scripts/make_reference.py`); each file's first line says which version
//! wrote it. A failure here means the Rust library, the example, or the
//! reference moved — and the tolerances below are chosen tightly enough that
//! it can only be one of those three, not noise.
//!
//! Skipped unless `SPARSEIR_TUTORIAL_RUN` is set, because it needs an example
//! to have run first.

mod common;

use common::{
    assert_close, assert_exact_integers, assert_negligible, examples_requested, output, reference,
};

/// `sparse_sampling_demo`: a semicircular spectral function at β = 10⁴,
/// ωmax = 1, ε = 10⁻¹⁵.
///
/// Every tolerance below is quoted with what the deviation actually was on the
/// machine the reference was written on, so that a later tightening or
/// loosening is a visible decision rather than a guess. The headroom covers a
/// differently compiled `libsparseir` behind the Python reference, not a
/// different algorithm: a change of algorithm would move these by orders of
/// magnitude, not by a factor of ten.
#[test]
fn sparse_sampling_demo_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "sparse_sampling_demo";

    // --- the basis itself ---------------------------------------------------
    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "beta",
        "wmax",
        "eps",
        "basis_size",
        "n_tau_points",
        "n_matsubara_points",
    ] {
        // Exact: these are the inputs and the counts they produced. A basis of
        // a different size is a different tutorial.
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "accuracy", 1e-14); // measured 0
    assert_close(&actual, &expected, "cond_tau", 1e-12); // measured 2.2e-15
    assert_close(&actual, &expected, "cond_matsubara", 1e-12); // measured 0

    // --- the exact coefficients ---------------------------------------------
    let (actual, expected) = (
        output(example, "coefficients"),
        reference(example, "coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 0
    // ρₗ is an integral the two implementations evaluate by different
    // quadratures — an adaptive one in Python, a substitution that removes the
    // semicircle's square-root edge here — so agreeing to 2e-15 is agreeing as
    // well as two quadratures of a function with an edge singularity can.
    assert_close(&actual, &expected, "rho_l", 1e-13); // measured 2.1e-15
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 1.3e-15

    // --- the sampling points, and G on them ---------------------------------
    // The τ points are the roots of the first discarded basis function, found
    // by the same bisection in both implementations; the reference file has
    // them folded onto [-β/2, β/2] to match what this example writes.
    let (actual, expected) = (
        output(example, "tau_sampling"),
        reference(example, "tau_sampling"),
    );
    assert_close(&actual, &expected, "tau", 1e-13); // measured 0
    assert_close(&actual, &expected, "g_tau", 1e-13); // measured 3.4e-16

    let (actual, expected) = (
        output(example, "matsubara_sampling"),
        reference(example, "matsubara_sampling"),
    );
    // Matsubara points are integers, so there is nothing to round: they agree
    // or the point selection changed.
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "nu", 1e-15);
    assert_close(&actual, &expected, "g_iv_im", 1e-12); // measured 3.0e-14
    // A spectral function even in ω makes G(iν) purely imaginary, so the real
    // part is rounding error in both implementations — of the same size, but
    // with no reason to be the same rounding error.
    for table in [&actual, &expected] {
        assert_negligible(table, "g_iv_re", "g_iv_im", 1e-14);
    }

    // --- the round trip -----------------------------------------------------
    let (actual, expected) = (
        output(example, "reconstruction"),
        reference(example, "reconstruction"),
    );
    assert_close(&actual, &expected, "g_l_from_tau", 1e-13); // measured 5.5e-15
    assert_close(&actual, &expected, "g_l_from_matsubara", 1e-13); // measured 1.3e-15

    // The point of the whole example: fitting from the sampling points alone
    // recovers the coefficients to the accuracy of the basis. This is a claim
    // about the library, not a comparison, so it is asserted outright.
    let g_l = actual.expect_column("g_l");
    let largest = g_l.iter().fold(0.0_f64, |acc, g| acc.max(g.abs()));
    for column in ["error_tau", "error_matsubara"] {
        let worst = actual
            .expect_column(column)
            .iter()
            .fold(0.0_f64, |acc, e| acc.max(*e));
        assert!(
            worst < 1e-13 * largest,
            "{column}: worst error {worst:.3e} is not small against |G|max = {largest:.3e}"
        );
    }
}
