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
