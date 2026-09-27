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

/// `transformation`: the routes into the basis and back out of it, at β = 10,
/// ωmax = 10, ε = 10⁻¹⁰ (and a bosonic basis at β = 15 for the pole section).
///
/// As above, every tolerance carries the deviation that was actually measured.
/// The whole example agrees with Python to within a few units in the last
/// place, which is what it should: the two implementations run the same
/// algorithms on the same basis.
#[test]
fn transformation_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "transformation";

    // --- the two bases ------------------------------------------------------
    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "pole_beta",
        "pole_wmax",
        "pole_basis_size",
        "beta",
        "wmax",
        "eps",
        "basis_size",
        "narrow_wmax",
        "narrow_basis_size",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "accuracy", 1e-14); // measured 0

    // --- from poles ---------------------------------------------------------
    // No quadrature is involved: ρₗ is vₗ at the pole times a weight, so this
    // is the basis itself being compared, and it agrees bit for bit.
    let (actual, expected) = (
        output(example, "pole_coefficients"),
        reference(example, "pole_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "rho_l", 1e-14); // measured 0
    assert_close(&actual, &expected, "g_l", 1e-14); // measured 0
    // The DLR must reproduce the same Gₗ from the same pole weights; that it
    // does so to the last bit says the two take the same route.
    assert_close(&actual, &expected, "g_l_dlr", 1e-14); // measured 0

    // --- from a smooth spectral function ------------------------------------
    let (actual, expected) = (
        output(example, "smooth_coefficients"),
        reference(example, "smooth_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 0
    // Python integrates adaptively, this example by a fixed high-order rule on
    // the same segments; both are exact to rounding for a smooth ρ.
    assert_close(&actual, &expected, "rho_l", 1e-13); // measured 7.6e-16
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 4.0e-16

    let (actual, expected) = (output(example, "spectrum"), reference(example, "spectrum"));
    assert_close(&actual, &expected, "omega", 1e-15); // measured 0
    assert_close(&actual, &expected, "rho_exact", 1e-15); // measured 0
    assert_close(&actual, &expected, "rho_reconstructed", 1e-13); // measured 9.8e-16

    // --- from IR to imaginary time ------------------------------------------
    let (actual, expected) = (output(example, "gtau"), reference(example, "gtau"));
    assert_close(&actual, &expected, "tau", 1e-15); // measured 0
    assert_close(&actual, &expected, "g_tau_direct", 1e-13); // measured 4.4e-16
    assert_close(&actual, &expected, "g_tau_sampling", 1e-13); // measured 5.6e-16

    // The two routes to G(τ) are the same matrix applied the same way, so they
    // must agree far more closely than either agrees with Python.
    let direct = actual.expect_column("g_tau_direct");
    let sampled = actual.expect_column("g_tau_sampling");
    let largest = direct.iter().fold(0.0_f64, |acc, g| acc.max(g.abs()));
    let worst = direct
        .iter()
        .zip(sampled)
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        worst < 1e-14 * largest,
        "the direct and sampled G(τ) differ by {worst:.3e}, which is more than rounding"
    );

    // --- and back again -----------------------------------------------------
    let (actual, expected) = (
        output(example, "roundtrip"),
        reference(example, "roundtrip"),
    );
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 4.0e-16
    assert_close(&actual, &expected, "g_l_reconstructed", 1e-13); // measured 1.1e-15
    // The error column is the residual of the round trip: rounding error in
    // both implementations, with no reason for the two to agree digit by
    // digit. What matters is that it is rounding error.
    for table in [&actual, &expected] {
        assert_negligible(table, "error", "g_l", 1e-14);
    }

    // --- what a too-small ωmax looks like -----------------------------------
    let (actual, expected) = (
        output(example, "narrow_basis"),
        reference(example, "narrow_basis"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 0
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 9.8e-16
}

/// `dlr`: the semicircle of `sparse_sampling_demo` again, this time turned
/// into a sum of poles and back.
#[test]
fn dlr_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "dlr";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in ["beta", "wmax", "eps", "lambda", "basis_size", "n_poles"] {
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "accuracy", 1e-14); // measured 0

    let (actual, expected) = (
        output(example, "coefficients"),
        reference(example, "coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 0
    assert_close(&actual, &expected, "rho_l", 1e-13); // measured 2.1e-15
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 1.3e-15

    // --- the poles and the coefficients on them -----------------------------
    let (actual, expected) = (
        output(example, "dlr_coefficients"),
        reference(example, "dlr_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "p");
    // The poles are roots of a basis function, found by the same bisection in
    // both implementations, so they agree bit for bit — as the sampling points
    // of the other examples do.
    assert_exact_integers(&actual, &expected, "pole");
    // cₚ comes out of a least-squares solve against the well-conditioned
    // matrix vₗ(ω̄ₚ); the DLR is only useful because that stays at rounding.
    assert_close(&actual, &expected, "c_p", 1e-13); // measured 2.0e-15

    let (actual, expected) = (
        output(example, "reconstruction"),
        reference(example, "reconstruction"),
    );
    assert_close(&actual, &expected, "g_l", 1e-13); // measured 1.3e-15
    assert_close(&actual, &expected, "g_l_from_dlr", 1e-13); // measured 1.2e-15
    // The residual of the IR → DLR → IR round trip: rounding error in both
    // implementations, with no reason to agree digit by digit.
    for table in [&actual, &expected] {
        assert_negligible(table, "error", "g_l", 1e-14);
    }

    // --- on the Matsubara axis ----------------------------------------------
    let (actual, expected) = (
        output(example, "matsubara"),
        reference(example, "matsubara"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "nu", 1e-15); // measured 1.8e-16
    assert_close(&actual, &expected, "g_iv_exact_im", 1e-12); // measured 3.0e-14
    assert_close(&actual, &expected, "g_iv_dlr_im", 1e-12); // measured 3.7e-14
    // The semicircle is even in ω, so G(iν) is purely imaginary and the real
    // parts are rounding error.
    for table in [&actual, &expected] {
        assert_negligible(table, "g_iv_exact_re", "g_iv_exact_im", 1e-13);
        assert_negligible(table, "g_iv_dlr_re", "g_iv_dlr_im", 1e-13);
    }

    // The point of the example: the two routes to G(iν) — through the basis
    // functions, and through the poles — give the same Green's function, at
    // frequencies far outside the sampling set.
    let exact = actual.expect_column("g_iv_exact_im");
    let from_dlr = actual.expect_column("g_iv_dlr_im");
    let largest = exact.iter().fold(0.0_f64, |acc, g| acc.max(g.abs()));
    let worst = exact
        .iter()
        .zip(from_dlr)
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        worst < 1e-12 * largest,
        "the basis and the poles disagree by {worst:.3e}, which is more than rounding"
    );
}
