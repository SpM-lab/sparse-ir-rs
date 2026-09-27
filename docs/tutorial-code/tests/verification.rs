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
    scans_requested,
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

/// `spm`: sparse modeling — an L1-regularised fit of the IR coefficients of a
/// spectral function to a noisy `G(τ)`, at β = 100, ωmax = 4, ε = 10⁻¹⁰.
///
/// Both sides read the same committed input and run the same fixed number of
/// FISTA steps from the same starting point, so the comparison is between two
/// arithmetics rather than between two solvers — the deviations below are at
/// rounding level, four orders of magnitude tighter than the 1e-6 the plan
/// allowed for an iterative method. That is the point of fixing the iteration
/// count: an iteration that stopped on a tolerance would have made this test
/// blind to anything smaller than the tolerance.
#[test]
fn spm_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "spm";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    // A basis of a different size, or a different number of input times, is a
    // different problem — there is nothing to compare after that.
    assert_exact_integers(&actual, &expected, "basis_size");
    assert_exact_integers(&actual, &expected, "n_tau");
    assert_close(&actual, &expected, "lambda", 1e-15); // measured 0
    // The Lipschitz bound scales every FISTA step, so the two runs only stay
    // on the same trajectory because this agrees to rounding.
    assert_close(&actual, &expected, "lipschitz", 1e-13); // measured 1.7e-16

    // --- what the sweep over λ found ----------------------------------------
    let (actual, expected) = (
        output(example, "lambda_scan"),
        reference(example, "lambda_scan"),
    );
    assert_close(&actual, &expected, "lambda", 1e-15); // measured 0
    // The number of surviving coefficients is a count, and it is the headline
    // of the method: if L1 stops producing exact zeros, the page is wrong.
    assert_exact_integers(&actual, &expected, "nonzero");
    assert_close(&actual, &expected, "residual", 1e-12); // measured 1.3e-14
    assert_close(&actual, &expected, "l1_norm", 1e-12); // measured 2.8e-14
    assert_close(&actual, &expected, "l2_error", 1e-12); // measured 5.3e-14
    assert_close(&actual, &expected, "sum_rule", 1e-13); // measured 6.5e-16
    assert_close(&actual, &expected, "min_rho", 1e-12); // measured 6.3e-14

    // --- the recovered spectrum ---------------------------------------------
    let (actual, expected) = (output(example, "spectrum"), reference(example, "spectrum"));
    assert_close(&actual, &expected, "omega", 1e-15); // measured 0
    assert_close(&actual, &expected, "rho_exact", 1e-15); // measured 0
    assert_close(&actual, &expected, "rho_recovered", 1e-12); // measured 5.3e-15

    let (actual, expected) = (
        output(example, "coefficients"),
        reference(example, "coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 0
    assert_close(&actual, &expected, "rho_l_exact", 1e-13); // measured 1.3e-15
    assert_close(&actual, &expected, "rho_l_recovered", 1e-12); // measured 5.5e-15

    // --- the data, and what the fit says it should have been ----------------
    let (actual, expected) = (output(example, "gtau"), reference(example, "gtau"));
    // These two come from the committed input file, so anything but an exact
    // match means one side read it wrong.
    assert_close(&actual, &expected, "tau", 1e-15); // measured 0
    assert_close(&actual, &expected, "g_tau_input", 1e-15); // measured 0
    assert_close(&actual, &expected, "g_tau_clean", 1e-15); // measured 0
    assert_close(&actual, &expected, "g_tau_fit", 1e-13); // measured 4.4e-16
}

/// `analytic_continuation`: two semicircle-based models at β = 40, ωmax = 2,
/// ε = 2×10⁻⁸ — a basis of 24 functions whose last singular value is 2.5×10⁻⁸
/// of the first.
///
/// The two implementations integrate differently everywhere a `vₗ` overlap
/// appears: Python's `basis.v.overlap` is adaptive, while this example
/// substitutes the singularity away and integrates the resulting polynomial in
/// closed form. Everything that is not an overlap agrees to rounding.
#[test]
fn analytic_continuation_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "analytic_continuation";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in ["beta", "wmax", "basis_size", "n_lorentz"] {
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "eps", 1e-15); // measured 0
    assert_close(&actual, &expected, "eta", 1e-15); // measured 0
    // The noise level is `0.3 s_{L−1}/s_0`, so it inherits the accuracy of the
    // very last singular value — the one the truncation was about to drop.
    assert_close(&actual, &expected, "noise", 1e-9); // measured 1.8e-11
    assert_close(&actual, &expected, "alpha", 1e-9); // measured 1.8e-11

    // --- the models and the noise put on them -------------------------------
    let (actual, expected) = (
        output(example, "coefficients"),
        reference(example, "coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-14); // measured 6.2e-16
    assert_close(&actual, &expected, "s_ratio", 1e-14); // measured 7.8e-16
    // ρₗ of a semicircle: one quadrature against another.
    assert_close(&actual, &expected, "rho_semielliptic", 1e-12); // measured 1.2e-14
    // The insulating model's two semicircles are a quarter of the band wide,
    // so their edges sit where the basis has few knots and the quadratures
    // have the least in common.
    assert_close(&actual, &expected, "rho_insulating", 1e-11); // measured 9.8e-13
    assert_close(&actual, &expected, "g_semielliptic", 1e-14); // measured 7.2e-16
    assert_close(&actual, &expected, "g_insulating", 1e-14); // measured 5.9e-16
    assert_close(&actual, &expected, "g_semielliptic_noisy", 1e-14); // measured 7.2e-16
    assert_close(&actual, &expected, "g_insulating_noisy", 1e-14); // measured 5.9e-16

    // --- truncated-SVD regularisation ---------------------------------------
    let (actual, expected) = (output(example, "tsvd"), reference(example, "tsvd"));
    assert_close(&actual, &expected, "omega", 1e-15); // measured 0
    assert_close(&actual, &expected, "semielliptic_exact", 1e-15); // measured 0
    assert_close(&actual, &expected, "insulating_exact", 1e-15); // measured 0
    assert_close(&actual, &expected, "semielliptic_half", 1e-12); // measured 6.2e-15
    assert_close(&actual, &expected, "insulating_half", 1e-12); // measured 3.8e-15
    // Dividing by the last singular value multiplies a 10⁻¹⁶ disagreement in
    // `sₗ` by 4×10⁷. That amplification is the point of the whole page, so the
    // tolerance here has to allow it — and it still pins the curve to eight
    // digits.
    assert_close(&actual, &expected, "semielliptic_full", 1e-8); // measured 3.4e-10
    assert_close(&actual, &expected, "insulating_full", 1e-8); // measured 2.3e-10

    // --- ridge regression ---------------------------------------------------
    // The ridge filter rolls the small singular values off instead of dividing
    // by them, which is why this is four orders of magnitude tighter than the
    // untruncated inversion above.
    let (actual, expected) = (output(example, "ridge"), reference(example, "ridge"));
    assert_close(&actual, &expected, "omega", 1e-15); // measured 0
    assert_close(&actual, &expected, "semielliptic_ridge", 1e-10); // measured 1.3e-12
    assert_close(&actual, &expected, "insulating_ridge", 1e-10); // measured 2.9e-12

    // --- the coefficients of a discrete spectrum ----------------------------
    let (actual, expected) = (output(example, "discrete"), reference(example, "discrete"));
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "g_semielliptic", 1e-14); // measured 7.2e-16
    assert_close(&actual, &expected, "rho_semielliptic", 1e-12); // measured 1.2e-14
    assert_close(&actual, &expected, "g_discrete", 1e-14); // measured 1.1e-15
    // ρₗ of four delta peaks does not decay, so the largest entry of this
    // column is at the end, where `vₗ` itself is evaluated least accurately.
    assert_close(&actual, &expected, "rho_discrete", 1e-8); // measured 1.8e-10

    // --- the real-axis basis ------------------------------------------------
    let (actual, expected) = (output(example, "lorentz"), reference(example, "lorentz"));
    assert_close(&actual, &expected, "omega", 1e-15); // measured 0
    assert_close(&actual, &expected, "f", 1e-15); // measured 0

    let (actual, expected) = (
        output(example, "lorentz_kernel"),
        reference(example, "lorentz_kernel"),
    );
    assert_exact_integers(&actual, &expected, "l");
    for m in 0..21 {
        // η is a hundredth of the knot spacing, so both sides are integrating a
        // near-delta: Python by telling the adaptive rule where the peak is,
        // this example by mapping the peak onto the whole interval. The worst
        // column is the one whose centre falls nearest a knot.
        assert_close(&actual, &expected, &format!("k_{m}"), 1e-10); // measured 3.0e-12
    }
}

/// `second_order_perturbation`: the Hubbard model on a 256 × 256 square
/// lattice at β = 10³, Λ = 10⁵, ε = 10⁻⁷, U = 2.
///
/// This is the first example whose answer depends on the τ convention and on
/// the momentum Fourier convention at once. Getting either wrong changes Σ by
/// a sign or by a power of nk, so the agreement below is what says both are
/// right — a picture would not have said it.
#[test]
fn second_order_perturbation_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "second_order_perturbation";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "beta",
        "wmax",
        "eps",
        "lambda",
        "u",
        "nk_lin",
        "basis_size",
        "n_tau",
        "n_matsubara",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "cond_tau", 1e-8); // measured 3.6e-10
    assert_close(&actual, &expected, "cond_matsubara", 1e-8); // measured 3.4e-11

    // --- G₀ on the Matsubara axis: a formula, so this is only arithmetic ----
    let (actual, expected) = (
        output(example, "green_matsubara"),
        reference(example, "green_matsubara"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "nu", 1e-15);
    assert_close(&actual, &expected, "g_gamma_im", 1e-14); // measured 1.1e-16
    assert_close(&actual, &expected, "g_gamma_re", 1e-14); // measured 1.1e-16

    let (actual, expected) = (
        output(example, "green_coefficients"),
        reference(example, "green_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "s_l", 1e-12); // measured 5.4e-14
    assert_close(&actual, &expected, "g_gamma_abs", 1e-11); // measured 3.7e-13

    // --- on the sampling times, and in real space ---------------------------
    // The sampling times are roots of the first discarded basis function,
    // located by bisection; at Λ = 10⁵ the two implementations place them to
    // about 10⁻¹¹ relative, and everything sampled there inherits that. The
    // reference file has them folded onto [−β/2, β/2] to match this example.
    let (actual, expected) = (
        output(example, "green_tau"),
        reference(example, "green_tau"),
    );
    assert_close(&actual, &expected, "tau", 1e-10); // measured 2.6e-11
    assert_close(&actual, &expected, "g_gamma", 1e-10); // measured 1.1e-11
    assert_close(&actual, &expected, "g_m", 1e-10); // measured 1.1e-11
    assert_close(&actual, &expected, "g_origin", 1e-10); // measured 9.9e-12

    // --- the self-energy ----------------------------------------------------
    // Σ(τ, r) = U² G(τ, r)² G(β − τ, r). A wrong sign in the reversal, or a
    // reversal that permuted the rows without it, would show up here as a
    // deviation of order one rather than of order 10⁻¹¹.
    let (actual, expected) = (
        output(example, "self_energy_tau"),
        reference(example, "self_energy_tau"),
    );
    assert_close(&actual, &expected, "tau", 1e-10); // measured 2.6e-11
    assert_close(&actual, &expected, "sigma_origin", 1e-10); // measured 1.0e-11

    let (actual, expected) = (
        output(example, "self_energy_coefficients"),
        reference(example, "self_energy_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "sigma_origin_abs", 1e-11); // measured 3.4e-13
    // The momentum transform is the one place a factor of nk = 65536 could
    // hide; it would be visible here, not subtle.
    assert_close(&actual, &expected, "sigma_gamma_abs", 1e-11); // measured 3.4e-13

    let (actual, expected) = (
        output(example, "self_energy_matsubara"),
        reference(example, "self_energy_matsubara"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "sigma_gamma_im", 1e-11); // measured 5.3e-13
    assert_close(&actual, &expected, "sigma_gamma_re", 1e-11); // measured 1.8e-12

    // --- and on frequencies nobody sampled ----------------------------------
    let (actual, expected) = (
        output(example, "self_energy_far"),
        reference(example, "self_energy_far"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "sigma_gamma_im", 1e-11); // measured 6.3e-14
    assert_close(&actual, &expected, "sigma_gamma_re", 1e-11); // measured 1.3e-12
}

/// `gw`: one self-consistent GW loop for the single-site Hubbard atom at
/// β = 10, U = 0.5.
///
/// Nothing here is a formula: every number is the end of twenty iterations
/// that pass through both statistics four times each. The fermionic and
/// bosonic sampling grids of this basis each carry a point at exactly τ = β/2,
/// so the τ-reversal in `P(τ) = G(τ) G(β − τ)` has to handle the wrap; get it
/// wrong and the loop converges to a different fixed point, not to a slightly
/// different number. That the tolerances below sit at machine precision is
/// therefore a strong statement, and they are kept there deliberately.
///
/// The atom is particle-hole symmetric, so several columns are zero by
/// symmetry and hold nothing but rounding error; those are checked to be
/// negligible against the part that carries the signal rather than compared
/// with Python, which has no reason to round the same way.
#[test]
fn gw_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "gw";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "beta",
        "wmax",
        "iterations",
        "basis_size_f",
        "basis_size_b",
        "n_tau_f",
        "n_tau_b",
        "n_wn_f",
        "n_wn_b",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }
    assert_close(&actual, &expected, "t", 1e-15); // measured 0
    assert_close(&actual, &expected, "u", 1e-15); // measured 0

    // --- the starting point, which is the atomic Green's function ----------
    let (actual, expected) = (
        output(example, "green_initial"),
        reference(example, "green_initial"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "g_im", 1e-14); // measured 4.5e-16
    for table in [&actual, &expected] {
        assert_negligible(table, "g_re", "g_im", 1e-14);
    }

    // --- G on both grids ----------------------------------------------------
    // `green_tau_bosonic` is G, a fermionic function, evaluated at the
    // *bosonic* sampling times, and `green_tau_reversed` is the same function
    // at `β − τ`. Those two are where a cross-statistics evaluation that
    // ignored the statistics of the function, or a reversal that permuted the
    // rows without their sign, would show up — as a deviation of order one.
    // At β = 10 with this ε the fermionic and bosonic grids happen to hold the
    // same times, so the abscissa alone would not give such a mistake away.
    for name in ["green_tau", "green_tau_bosonic", "green_tau_reversed"] {
        let (actual, expected) = (output(example, name), reference(example, name));
        assert_close(&actual, &expected, "g_re", 1e-13); // measured 7.9e-16
        for table in [&actual, &expected] {
            assert_negligible(table, "g_im", "g_re", 1e-14);
        }
    }

    // --- P = G(τ)G(β − τ), the bosonic polarization -------------------------
    let (actual, expected) = (
        output(example, "polarization_tau"),
        reference(example, "polarization_tau"),
    );
    assert_close(&actual, &expected, "tau_b", 1e-13); // measured 0
    assert_close(&actual, &expected, "p_re", 1e-13); // measured 1.6e-15
    for table in [&actual, &expected] {
        assert_negligible(table, "p_im", "p_re", 1e-14);
    }

    let (actual, expected) = (
        output(example, "polarization_coefficients"),
        reference(example, "polarization_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "p_l_abs", 1e-13); // measured 1.2e-15

    let (actual, expected) = (
        output(example, "polarization_matsubara"),
        reference(example, "polarization_matsubara"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "p_re", 1e-13); // measured 1.2e-15
    for table in [&actual, &expected] {
        assert_negligible(table, "p_im", "p_re", 1e-14);
    }

    // --- W = U/(1 − UP) − U, back onto the fermionic times ------------------
    for (name, real, imaginary) in [
        ("screened_matsubara", "w_re", Some("w_im")),
        ("screened_tau", "w_re", Some("w_im")),
        ("screened_coefficients", "w_l_abs", None),
    ] {
        let (actual, expected) = (output(example, name), reference(example, name));
        assert_close(&actual, &expected, real, 1e-13); // measured ≤ 7.8e-15
        if let Some(imaginary) = imaginary {
            for table in [&actual, &expected] {
                assert_negligible(table, imaginary, real, 1e-14);
            }
        }
    }

    // --- Σ = G W, and the Hartree term kept out of the Dyson equation -------
    let (actual, expected) = (
        output(example, "self_energy_tau"),
        reference(example, "self_energy_tau"),
    );
    assert_close(&actual, &expected, "e_re", 1e-13); // measured 8.1e-15
    for table in [&actual, &expected] {
        assert_negligible(table, "e_im", "e_re", 1e-14);
    }

    let (actual, expected) = (
        output(example, "self_energy_coefficients"),
        reference(example, "self_energy_coefficients"),
    );
    assert_exact_integers(&actual, &expected, "l");
    assert_close(&actual, &expected, "e_l_abs", 1e-13); // measured 2.7e-15

    let (actual, expected) = (
        output(example, "self_energy_matsubara"),
        reference(example, "self_energy_matsubara"),
    );
    assert_exact_integers(&actual, &expected, "n");
    for column in ["e_re", "e_im", "hartree"] {
        assert_close(&actual, &expected, column, 1e-13); // measured ≤ 4.0e-15
    }

    // --- the converged fixed point ------------------------------------------
    let (actual, expected) = (
        output(example, "self_energy_final"),
        reference(example, "self_energy_final"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "e_re", 1e-13); // measured 3.3e-16
    assert_close(&actual, &expected, "e_im", 1e-13); // measured 2.8e-15

    let (actual, expected) = (
        output(example, "green_final"),
        reference(example, "green_final"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "g_im", 1e-13); // measured 5.6e-16
    for table in [&actual, &expected] {
        assert_negligible(table, "g_re", "g_im", 1e-14);
    }

    // --- and the way it got there -------------------------------------------
    // The iteration-by-iteration change, which says the two implementations
    // took the same path and not merely arrived at the same place.
    let (actual, expected) = (
        output(example, "convergence"),
        reference(example, "convergence"),
    );
    assert_exact_integers(&actual, &expected, "iteration");
    assert_close(&actual, &expected, "difference", 1e-13); // measured 2.2e-14
}

/// `liechtenstein`: exchange interactions of a square-lattice ferromagnet at
/// β = 50 on a 36 × 36 momentum grid, with an effective field B = 3.
///
/// The interesting comparison here is internal as much as external: `J₀`
/// through the basis against `J₀` summed over 200 to 3200 Matsubara
/// frequencies. Those columns are checked against Python too, because a
/// truncated sum is a well-defined number and the two implementations should
/// arrive at the same one.
#[test]
fn liechtenstein_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "liechtenstein";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "t",
        "beta",
        "wmax",
        "eps",
        "lambda",
        "b_eff",
        "nk_lin",
        "basis_size",
        "n_matsubara",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }

    // --- J₀(μ), both ways ---------------------------------------------------
    let (actual, expected) = (output(example, "j0"), reference(example, "j0"));
    assert_close(&actual, &expected, "mu", 1e-15); // measured 0
    // Through the basis. The fit is the one step with any conditioning in it,
    // and at ε = 10⁻⁷ it still agrees to 13 digits.
    assert_close(&actual, &expected, "j0", 1e-11); // measured 4.5e-13
    for nm in [100, 200, 400, 800, 1600] {
        // A truncated sum is plain arithmetic in a different order; the
        // deviation is the reassociation, nothing else.
        assert_close(&actual, &expected, &format!("j0_nm{nm}"), 1e-12); // measured ≤ 1.5e-14
    }

    // The point of the example: the truncated sums walk towards the basis
    // answer like 1/N_M, and the basis answer is the limit. Asserted outright,
    // because it is a claim about the method rather than a comparison.
    let j0 = actual.expect_column("j0");
    let mut previous = f64::INFINITY;
    for nm in [100, 200, 400, 800, 1600] {
        let naive = actual.expect_column(&format!("j0_nm{nm}"));
        let worst = naive
            .iter()
            .zip(j0)
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            worst < 0.55 * previous,
            "doubling the Matsubara grid to {nm} should roughly halve the error, \
             but it went from {previous:.3e} to {worst:.3e}"
        );
        previous = worst;
    }

    // --- J_ij at half filling -----------------------------------------------
    let (actual, expected) = (output(example, "jij"), reference(example, "jij"));
    assert_close(&actual, &expected, "distance", 1e-15); // measured 0
    // Two FFTs and a fit. The deviation would be of order one if either
    // transform went the wrong way round.
    assert_close(&actual, &expected, "j_ij", 1e-12); // measured 5.3e-15

    // --- and the sum rule ---------------------------------------------------
    let (actual, expected) = (output(example, "sum_rule"), reference(example, "sum_rule"));
    assert_close(&actual, &expected, "j0_direct", 1e-11); // measured 4.7e-14
    assert_close(&actual, &expected, "j0_from_jij", 1e-12); // measured 2.6e-15
    // J₀ = Σ_j J_0j, computed two different ways in the same run. The k-space
    // route sums 1296 real-space terms; the direct route never leaves k. They
    // agree to the accuracy of the basis, ε = 10⁻⁷, not to machine precision.
    let direct = actual.expect_column("j0_direct")[0];
    let from_jij = actual.expect_column("j0_from_jij")[0];
    assert!(
        (direct - from_jij).abs() < 1e-7 * direct.abs().max(1.0),
        "the sum rule is broken: {direct} from J₀ against {from_jij} from J_ij"
    );
}

/// T = 0.1 on a 200 × 200 momentum grid, ε = 10⁻¹⁰, for both lattices.
///
/// The susceptibility is a Matsubara sum of a product of three or four
/// Green's functions, so it falls off as 1/ν³ and the basis does the sum in
/// one evaluation at τ = 0. Graphene additionally goes through a 2×2
/// eigendecomposition at every momentum; the closed form in `linalg` and
/// numpy's `eigh` are free to disagree on the phase of each eigenvector, and
/// the agreement below is what says the traces do not care.
#[test]
fn orbital_magnetic_susceptibility_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "orbital_magnetic_susceptibility";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "t",
        "a",
        "beta",
        "wmax",
        "eps",
        "nk_lin",
        "n_mu",
        "basis_size",
        "n_matsubara",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }

    for lattice in ["square", "graphene"] {
        // χ(iν) at one chemical potential, before the sum: this is where a
        // wrong velocity matrix or a wrong band energy would show up.
        let name = format!("{lattice}_matsubara");
        let (actual, expected) = (output(example, &name), reference(example, &name));
        assert_close(&actual, &expected, "mu", 1e-15); // measured 0
        assert_exact_integers(&actual, &expected, "wn");
        assert_close(&actual, &expected, "chi_re", 1e-12); // measured ≤ 5.9e-14
        assert_close(&actual, &expected, "chi_im", 1e-12); // measured ≤ 1.3e-14

        // and after it, for every chemical potential.
        let (actual, expected) = (output(example, lattice), reference(example, lattice));
        assert_close(&actual, &expected, "mu", 1e-15); // measured 0
        assert_close(&actual, &expected, "chi", 1e-12); // measured ≤ 1.7e-14
    }

    // The closed form is arithmetic on two elliptic integrals, computed here
    // by the arithmetic-geometric mean and in the reference by scipy.
    let (actual, expected) = (
        output(example, "square_analytic"),
        reference(example, "square_analytic"),
    );
    assert_close(&actual, &expected, "mu", 1e-15); // measured 0
    assert_close(&actual, &expected, "chi", 1e-14); // measured 4.0e-16

    // --- and what the numbers say -------------------------------------------
    let square = output(example, "square");
    let (mu, chi) = (square.expect_column("mu"), square.expect_column("chi"));
    let analytic = output(example, "square_analytic");
    let closed: std::collections::HashMap<u64, f64> = analytic
        .expect_column("mu")
        .iter()
        .zip(analytic.expect_column("chi"))
        .map(|(mu, chi)| (mu.to_bits(), *chi))
        .collect();

    // Away from the van Hove filling and from the band edge, T = 0.1 is cold
    // enough that the finite-temperature curve is the zero-temperature one.
    let worst = mu
        .iter()
        .zip(chi)
        .filter(|(mu, _)| (1.0..=3.0).contains(&mu.abs()))
        .fold(0.0_f64, |acc, (mu, chi)| {
            acc.max((chi - closed[&mu.to_bits()]).abs())
        });
    assert!(
        worst < 1e-3,
        "at 1 ≤ |μ| ≤ 3 the T = 0.1 susceptibility should follow the T = 0 \
         closed form, but it is off by {worst:.3e}"
    );

    // The van Hove singularity at the centre of the band is paramagnetic, and
    // it is the largest the square lattice gets.
    let peak = chi.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
    let centre = chi[mu
        .iter()
        .position(|&mu| mu == 0.0)
        .expect("μ = 0 is on the grid")];
    assert!(
        centre > 0.0 && centre == peak,
        "the square lattice should peak at μ = 0, but χ(0) = {centre:.6} against a \
         maximum of {peak:.6}"
    );

    // Graphene is the opposite: a strong diamagnetic peak at the Dirac point,
    // paramagnetic away from it, and nothing at all outside the band.
    let graphene = output(example, "graphene");
    let (mu, chi) = (graphene.expect_column("mu"), graphene.expect_column("chi"));
    let dirac = chi[mu
        .iter()
        .position(|&mu| mu == 0.0)
        .expect("μ = 0 is on the grid")];
    let trough = chi.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
    assert!(
        dirac < 0.0 && dirac == trough,
        "graphene should be most diamagnetic at the Dirac point, but χ(0) = {dirac:.6} \
         against a minimum of {trough:.6}"
    );
    for (mu, chi) in mu.iter().zip(chi) {
        if mu.abs() > 4.0 {
            assert!(
                chi.abs() < 1e-6,
                "graphene has no states at μ = {mu}, so χ should vanish there, not be {chi:.3e}"
            );
        }
    }

    // The summand falls off as a power of ν: that power is why summing the
    // series directly is slow, and why thirty basis coefficients hold all of
    // it. The square lattice loses its three-Green's-function term, because
    // a dispersion that separates has no γxy; graphene keeps it but its
    // velocity matrices are purely off-diagonal, and the slope comes out
    // steeper still.
    for (lattice, power) in [("square", 4.0_f64), ("graphene", 6.0)] {
        let table = output(example, &format!("{lattice}_matsubara"));
        let mut points: Vec<(f64, f64)> = table
            .expect_column("wn")
            .iter()
            .zip(table.expect_column("chi_re"))
            .zip(table.expect_column("chi_im"))
            .filter(|((n, _), _)| **n > 0.0)
            .map(|((n, re), im)| (*n, re.hypot(*im)))
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let [(n0, chi0), (n1, chi1)] = points[points.len() - 2..] else {
            unreachable!("there are at least two positive sampling frequencies")
        };
        let slope = (chi1 / chi0).ln() / (n1 / n0).ln();
        assert!(
            (slope + power).abs() < 0.1,
            "the {lattice} summand should fall off as ν^-{power}, but between \
             n = {n0} and n = {n1} it falls off as ν^{slope:.2}"
        );
    }
}

/// `dmft_ipt`: the Bethe lattice at `D = 2`, `β = 20`, `U = 5`, `ε = 10⁻¹⁵`,
/// stopped where the notebook stops it.
///
/// The tolerances here are loose by the standards of this file — parts in
/// `10⁷` rather than parts in `10¹³` — and that is the example's subject
/// rather than a concession. The notebook stops when the *change* of `Σ` from
/// one iteration to the next falls below `10⁻⁵`, which happens while the
/// trajectory is still passing the unstable fixed point that separates the
/// metal from the insulator. Near that point a difference grows by about 40%
/// per iteration, so the `10⁻¹⁶` by which two implementations differ in their
/// starting `G⁰` has become `10⁻⁷` by iteration 65. The fixed points
/// themselves are reproducible to machine precision; this snapshot of the walk
/// towards one is not, and `dmft_ipt_scan` is where the converged numbers are.
#[test]
fn dmft_ipt_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples' output");
        return;
    }
    let example = "dmft_ipt";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "d",
        "wmax",
        "beta",
        "u",
        "eps",
        "mix",
        "sfc_tol",
        "maxiter",
        "basis_size",
        "n_tau",
        "n_wn",
        // The iteration count is exact because the residual crosses `10⁻⁵`
        // with 1% to spare, far more than the trajectories differ by there.
        "iterations",
        // The unconstrained run has no criterion to cross at all, and it ends
        // on the insulator from either implementation.
        "long_iterations",
        "long_z",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }
    // measured 1.1e-7
    assert_close(&actual, &expected, "z", 1e-6);

    let (actual, expected) = (output(example, "green"), reference(example, "green"));
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "g_im", 5e-6); // measured 3.3e-7
    // Half filling with a symmetric density of states: the real part vanishes
    // by particle-hole symmetry, which the loop preserves exactly. What is
    // left of it is the same amplified rounding error as above.
    assert_negligible(&actual, "g_re", "g_im", 5e-6); // measured 4.9e-7

    let (actual, expected) = (
        output(example, "self_energy"),
        reference(example, "self_energy"),
    );
    assert_exact_integers(&actual, &expected, "n");
    assert_close(&actual, &expected, "sigma_im", 5e-6); // measured 3.2e-7
    assert_negligible(&actual, "sigma_re", "sigma_im", 5e-6); // measured 4.7e-7

    // `Σ(τ)` is real, and the Python reference reports it on `[0, β)`: the
    // reference script folds its grid onto `[−β/2, β/2]` with the fermionic
    // sign before the comparison.
    let (actual, expected) = (
        output(example, "self_energy_tau"),
        reference(example, "self_energy_tau"),
    );
    assert_close(&actual, &expected, "tau", 1e-15); // measured 0
    assert_close(&actual, &expected, "sigma_re", 5e-6); // measured 3.1e-7
    assert_negligible(&actual, "sigma_im", "sigma_re", 5e-6); // measured 3.0e-7

    let (actual, expected) = (
        output(example, "convergence"),
        reference(example, "convergence"),
    );
    assert_exact_integers(&actual, &expected, "iteration");
    assert_close(&actual, &expected, "residual", 1e-7); // measured 1.1e-9

    // The long run's trajectory is not compared value by value, and cannot be:
    // it leaves an unstable fixed point at a moment that rounding decides, so
    // the two implementations peak two iterations and a factor of two apart.
    // What both do is climb back to order one after the criterion above was
    // satisfied, and then converge for real.
    let (actual, expected) = (
        output(example, "long_convergence"),
        reference(example, "long_convergence"),
    );
    for table in [&actual, &expected] {
        let residual = table.expect_column("residual");
        let rebound = residual[65..].iter().fold(0.0_f64, |acc, &r| acc.max(r));
        assert!(
            rebound > 0.1,
            "the loop must leave the unstable fixed point it was stopped next to, \
             but the change never rose above {rebound:.1e} again"
        );
        assert!(
            residual[3000..].iter().all(|&r| r < 1e-15),
            "the long run must actually converge"
        );
    }

    // --- and the physics ----------------------------------------------------
    let summary = output(example, "summary");
    let z = summary.expect_column("z")[0];
    assert!(
        (0.0..=1.0).contains(&z),
        "the quasiparticle weight must lie in [0, 1], got {z}"
    );

    let green = output(example, "green");
    let index = green
        .expect_column("n")
        .iter()
        .position(|&n| n == 1.0)
        .expect("the fermionic sampling frequencies include n = 1");
    let g_im = green.expect_column("g_im")[index];
    assert!(
        g_im < 0.0,
        "Im G(iν) must be negative at positive frequency, got {g_im}"
    );

    // The criterion stopped the loop the first time it was satisfied, and the
    // residual fell every single step on the way there.
    let convergence = output(example, "convergence");
    let residual = convergence.expect_column("residual");
    let tolerance = summary.expect_column("sfc_tol")[0];
    assert!(
        residual[residual.len() - 1] < tolerance && residual[residual.len() - 2] >= tolerance,
        "the loop must stop at the first iteration below {tolerance:.0e}"
    );
    assert!(
        residual.windows(2).all(|w| w[1] < w[0]),
        "the residual must fall at every iteration"
    );
}

/// `dmft_ipt_scan`: `Z(U)` over 66 interaction strengths from three starting
/// points, each run for a fixed 5000 iterations.
///
/// These are fixed points rather than snapshots of a walk towards one, and
/// they agree to parts in `10¹⁵` — the contrast with `dmft_ipt` above is the
/// whole reason the scan does not stop on a threshold.
#[test]
fn dmft_ipt_scan_matches_the_python_reference() {
    if !scans_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_SCANS=1 to check the parameter scans");
        return;
    }
    let example = "dmft_ipt_scan";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "d",
        "beta",
        "eps",
        "u_max",
        "u_num",
        "iterations",
        "basis_size",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }

    let (actual, expected) = (
        output(example, "renormalisation"),
        reference(example, "renormalisation"),
    );
    assert_exact_integers(&actual, &expected, "u");
    for column in ["from_g0", "metal", "insulator"] {
        assert_close(&actual, &expected, column, 1e-12); // measured ≤ 3.9e-15
    }

    let (actual, expected) = (
        output(example, "self_energy"),
        reference(example, "self_energy"),
    );
    assert_exact_integers(&actual, &expected, "n");
    for column in [
        "sigma_im_u30",
        "sigma_im_u32",
        "sigma_im_u34",
        "sigma_im_u35",
        "sigma_im_u40",
    ] {
        assert_close(&actual, &expected, column, 1e-12); // measured ≤ 1.4e-14
    }

    // --- and the physics ----------------------------------------------------
    let table = output(example, "renormalisation");
    let u = table.expect_column("u");
    let from_g0 = table.expect_column("from_g0");
    let metal = table.expect_column("metal");
    let insulator = table.expect_column("insulator");

    // Starting from `G⁰` is starting in the metal's basin, so that sweep has
    // to land on the metallic branch wherever the metallic branch exists —
    // and where it does not, both have to fall to the same insulator. The two
    // sweeps take entirely different routes to each point, so their agreeing
    // to the last few bits is a statement about the fixed points and not about
    // the arithmetic.
    for (index, (&a, &b)) in from_g0.iter().zip(metal).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "row {index} (U = {}): the run from G⁰ gives Z = {a} but the metallic \
             sweep gives {b}",
            u[index]
        );
    }

    for name in ["from_g0", "metal", "insulator"] {
        let z = table.expect_column(name);
        assert!(
            (z[0] - 1.0).abs() < 1e-12,
            "`{name}`: the non-interacting limit must have Z = 1, got {}",
            z[0]
        );
        assert!(
            z.iter().all(|z| (0.0..=1.0).contains(z)),
            "`{name}`: every quasiparticle weight must lie in [0, 1]"
        );
    }

    // The metal is destroyed continuously in `U` until it stops existing, and
    // the insulator appears first: a coexistence window, which is what makes
    // the transition first order.
    let collapse = |z: &[f64]| z.iter().position(|&z| z == 0.0).expect("Z reaches zero");
    let (uc2, uc1) = (collapse(metal), collapse(insulator));
    assert!(
        uc1 < uc2,
        "the insulator must appear below U_c2 = {}, but it appears at {}",
        u[uc2],
        u[uc1]
    );
    assert!(
        metal[..uc2].windows(2).all(|w| w[1] < w[0]),
        "the metal's quasiparticle weight must fall with U"
    );
    assert!(
        insulator[uc1..].iter().all(|&z| z == 0.0),
        "the insulating branch must stay insulating above U_c1"
    );
}

/// The TPSC solution at `U = 4`, `n = 0.85`, `T = 0.1`.
///
/// Nothing here iterates to a fixed point: the calculation is three root
/// searches and a fixed sequence of transforms, so the two implementations
/// follow the same path and agree to the last couple of digits. The
/// tolerances are set about a hundred times above what was measured.
#[test]
fn tpsc_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples");
        return;
    }
    let example = "tpsc";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "t",
        "beta",
        "wmax",
        "n",
        "u",
        "nk_lin",
        "eps",
        "basis_size_f",
        "basis_size_b",
        "n_tau",
        "n_wn_f",
        "n_wn_b",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }
    // measured ≤ 1.8e-14
    for column in ["mu_0", "mu", "u_crit", "u_sp", "u_ch", "docc"] {
        assert_close(&actual, &expected, column, 1e-12);
    }

    let (actual, expected) = (output(example, "momentum"), reference(example, "momentum"));
    for column in ["kx", "ky"] {
        assert_exact_integers(&actual, &expected, column);
    }
    // measured ≤ 4.0e-14
    for column in ["ek", "g_re", "sigma_im", "chi_0", "chi_spin", "chi_charge"] {
        assert_close(&actual, &expected, column, 1e-12);
    }

    let (actual, expected) = (output(example, "path"), reference(example, "path"));
    assert_exact_integers(&actual, &expected, "distance");
    for column in ["chi_spin", "chi_charge", "chi_0"] {
        assert_close(&actual, &expected, column, 1e-12); // measured ≤ 3.8e-14
    }

    let (actual, expected) = (
        output(example, "self_energy"),
        reference(example, "self_energy"),
    );
    assert_exact_integers(&actual, &expected, "n");
    for column in ["nu", "sigma_im", "sigma_re"] {
        assert_close(&actual, &expected, column, 1e-12); // measured ≤ 1.3e-14
    }

    // --- and the physics ----------------------------------------------------
    let summary = output(example, "summary");
    let value = |name: &str| summary.expect_column(name)[0];
    let (u, u_sp, u_ch, u_crit) = (value("u"), value("u_sp"), value("u_ch"), value("u_crit"));
    // The whole point of TPSC: the spin vertex is screened below the bare
    // interaction and stays below the value at which the spin susceptibility
    // would diverge, while the charge vertex is pushed the other way.
    assert!(
        0.0 < u_sp && u_sp < u.min(u_crit),
        "U_sp = {u_sp} must lie between 0 and min(U, U_crit) = {}",
        u.min(u_crit)
    );
    assert!(u_ch > u, "U_ch = {u_ch} must exceed the bare U = {u}");
    // Double occupancy of an uncorrelated state at this filling would be
    // (n/2)²; the interaction can only suppress it.
    let n = value("n");
    assert!(
        0.0 < value("docc") && value("docc") < 0.25 * n * n,
        "the double occupancy {} must be positive and below (n/2)² = {}",
        value("docc"),
        0.25 * n * n
    );

    // χ_sp is peaked at M = (π, π), the antiferromagnetic wave vector, and
    // enhanced over χ⁰ there; χ_ch is suppressed.
    let momentum = output(example, "momentum");
    let chi_spin = momentum.expect_column("chi_spin");
    let chi_charge = momentum.expect_column("chi_charge");
    let chi_0 = momentum.expect_column("chi_0");
    let m_point = chi_spin
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("the zone is not empty")
        .0;
    let (kx, ky) = (
        momentum.expect_column("kx")[m_point],
        momentum.expect_column("ky")[m_point],
    );
    // At half filling the peak would sit exactly on M = (π, π); this is a
    // doped system, so it is pushed off M along the zone boundary — here by
    // a single grid step, 2π/24.
    let step = 2.0 / summary.expect_column("nk_lin")[0];
    assert!(
        (kx - 1.0).abs() < 1e-12 && (ky - 1.0).abs() <= step + 1e-12,
        "the spin susceptibility must peak on the zone boundary near M, \
         got ({kx}π, {ky}π)"
    );
    assert!(
        chi_spin[m_point] > chi_0[m_point] && chi_0[m_point] > chi_charge[m_point],
        "at M: χ_sp = {}, χ⁰ = {}, χ_ch = {} must be in that order",
        chi_spin[m_point],
        chi_0[m_point],
        chi_charge[m_point]
    );
}

/// The interaction dependence of the two vertices at half filling, which is
/// Fig. 2 of Vilk and Tremblay (1997).
#[test]
fn tpsc_scan_matches_the_python_reference() {
    if !examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to check the examples");
        return;
    }
    let example = "tpsc_scan";

    let (actual, expected) = (output(example, "summary"), reference(example, "summary"));
    for column in [
        "t",
        "beta",
        "wmax",
        "n",
        "nk_lin",
        "eps",
        "u_num",
        "u_min",
        "u_max",
        "basis_size_f",
        "probe_u_0",
        "probe_u_25",
        "probe_u_50",
    ] {
        assert_exact_integers(&actual, &expected, column);
    }

    let (actual, expected) = (output(example, "vertices"), reference(example, "vertices"));
    assert_exact_integers(&actual, &expected, "u");
    for column in ["u_sp", "u_ch", "u_crit", "docc"] {
        assert_close(&actual, &expected, column, 1e-11); // measured ≤ 1.2e-13
    }

    let (actual, expected) = (output(example, "chi_spin"), reference(example, "chi_spin"));
    assert_exact_integers(&actual, &expected, "distance");
    for column in ["u_0", "u_25", "u_50"] {
        assert_close(&actual, &expected, column, 1e-11); // measured ≤ 1.1e-13
    }

    // --- and the physics ----------------------------------------------------
    let table = output(example, "vertices");
    let u = table.expect_column("u");
    let u_sp = table.expect_column("u_sp");
    let u_ch = table.expect_column("u_ch");
    let u_crit = table.expect_column("u_crit");
    let docc = table.expect_column("docc");

    // `U_crit = 1/max χ⁰` is built from the non-interacting Green's function,
    // so it is the same number at every point of the scan.
    assert!(
        u_crit.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12),
        "U_crit must not depend on U"
    );
    for (index, (&u, (&u_sp, &u_ch))) in u.iter().zip(u_sp.iter().zip(u_ch)).enumerate() {
        assert!(
            u_sp < u && u < u_ch,
            "row {index}: U_sp = {u_sp} < U = {u} < U_ch = {u_ch} must hold"
        );
        assert!(
            u_sp < u_crit[index],
            "row {index}: U_sp = {u_sp} must stay below U_crit = {}",
            u_crit[index]
        );
    }
    // Both vertices grow with `U`, but `U_sp` saturates against `U_crit`
    // while `U_ch` runs away — which is the figure.
    assert!(
        u_sp.windows(2).all(|w| w[1] > w[0]) && u_ch.windows(2).all(|w| w[1] > w[0]),
        "both vertices must increase with U"
    );
    assert!(
        u_sp[u_sp.len() - 1] > 0.8 * u_crit[0],
        "U_sp must approach U_crit by the end of the scan: {} against {}",
        u_sp[u_sp.len() - 1],
        u_crit[0]
    );
    // At half filling an uncorrelated state has double occupancy ¼, and the
    // interaction suppresses it monotonically.
    assert!(
        (docc[0] - 0.25).abs() < 1e-3 && docc.windows(2).all(|w| w[1] < w[0]),
        "the double occupancy must start near ¼ and fall with U"
    );
}
