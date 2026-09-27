"""Reference values for the `gw` example.

This is the computation of `GW_py.ipynb` of sparse-ir-tutorial, with the
plotting removed and the numbers written out. The notebook performs one
iteration per run of its cells; here a fixed number of iterations is run, so
that the reference and the Rust example compare the same state.

Every quantity that lives on a sampling time is carried onto the `[−β/2, β/2]`
grid the Rust example reports on. Which sign it picks up there is decided by
the *function*, not by the grid it sits on: `G` is anti-periodic even when it
is evaluated at the bosonic sampling times, and `W` is periodic even at the
fermionic ones. Getting that backwards is the mistake this example exists to
rule out.
"""

from __future__ import annotations

import numpy as np
import sparse_ir

from reference_common import fold_tau, provenance, write_table

EXAMPLE = "gw"
T = 0.1
BETA = 1 / T
WMAX = 1.0
U = 0.5
ITERATIONS = 20


def rho(x: float) -> float:
    """The semicircular spectral function of full bandwidth 2ωmax."""
    return 2 / np.pi * np.sqrt(1 - (x / WMAX) ** 2)


def signs(tau: np.ndarray, zeta: float) -> np.ndarray:
    """The sign a function of statistics `zeta` picks up when its sampling
    times are folded from `[0, β)` onto `[−β/2, β/2]`."""
    return np.where(tau > BETA / 2, zeta, 1.0)


def write() -> None:
    comment = provenance(EXAMPLE)

    basisf = sparse_ir.FiniteTempBasis("F", BETA, WMAX)
    basisb = sparse_ir.FiniteTempBasis("B", BETA, WMAX)
    matsf = sparse_ir.MatsubaraSampling(basisf)
    tausf = sparse_ir.TauSampling(basisf)
    matsb = sparse_ir.MatsubaraSampling(basisb)
    tausb = sparse_ir.TauSampling(basisb)

    tau_f, _, order_f = fold_tau(tausf.sampling_points, BETA, "F")
    tau_b, _, order_b = fold_tau(tausb.sampling_points, BETA, "B")
    # A fermionic function sampled at the fermionic times, and at the bosonic
    # ones; a bosonic function sampled at the fermionic times.
    sign_ff = signs(tausf.sampling_points, -1.0)[order_f]
    sign_fb = signs(tausb.sampling_points, -1.0)[order_b]
    sign_bf = signs(tausf.sampling_points, +1.0)[order_f]

    rho_l = basisf.v.overlap(rho)
    g_l_0 = -basisf.s * rho_l
    g_iw_0 = matsf.evaluate(g_l_0)

    write_table(
        EXAMPLE,
        "summary",
        {
            "t": np.array([T]),
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "u": np.array([U]),
            "iterations": np.array([float(ITERATIONS)]),
            "basis_size_f": np.array([float(basisf.size)]),
            "basis_size_b": np.array([float(basisb.size)]),
            "n_tau_f": np.array([float(len(tau_f))]),
            "n_tau_b": np.array([float(len(tau_b))]),
            "n_wn_f": np.array([float(matsf.sampling_points.size)]),
            "n_wn_b": np.array([float(matsb.sampling_points.size)]),
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "green_initial",
        {
            "n": matsf.sampling_points.astype(float),
            "g_re": g_iw_0.real,
            "g_im": g_iw_0.imag,
        },
        comment,
    )

    g_iw_f = g_iw_0.copy()
    differences = []
    previous = None

    for iteration in range(ITERATIONS):
        g_l_f = matsf.fit(g_iw_f)
        g_tau_f = tausf.evaluate(g_l_f)

        # Into bosonic statistics: the fermionic basis functions, evaluated at
        # the bosonic sampling times.
        g_tau_b = basisf.u(tausb.sampling_points).T @ g_l_f
        # P(τ) = G(τ) G(β − τ), the random-phase approximation.
        g_beta_minus_tau = basisf.u(BETA - tausb.sampling_points).T @ g_l_f
        p_tau_b = g_tau_b * g_beta_minus_tau
        p_l_b = tausb.fit(p_tau_b)
        p_iw_b = matsb.evaluate(p_l_b)

        # The screened interaction, with its constant part split off: only the
        # frequency-dependent piece goes any further.
        w_iw_b = U / (1 - U * p_iw_b) - U
        w_l_b = matsb.fit(w_iw_b)
        # And back into fermionic statistics.
        w_tau_f = basisb.u(tausf.sampling_points).T @ w_l_b

        e_tau_f = g_tau_f * w_tau_f
        e_l_f = tausf.fit(e_tau_f)
        hartree = U * (basisf.u(BETA).T @ g_l_f)
        e_iw_f = matsf.evaluate(e_l_f)
        e_iw_f_hartree = e_iw_f - hartree

        if previous is not None:
            differences.append(np.abs(e_iw_f_hartree - previous).max())
        previous = e_iw_f_hartree

        if iteration == 0:
            write_table(
                EXAMPLE,
                "green_tau",
                {
                    "tau_f": tau_f,
                    "g_re": (sign_ff * g_tau_f[order_f]).real,
                    "g_im": (sign_ff * g_tau_f[order_f]).imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "green_tau_bosonic",
                {
                    "tau_b": tau_b,
                    "g_re": (sign_fb * g_tau_b[order_b]).real,
                    "g_im": (sign_fb * g_tau_b[order_b]).imag,
                },
                comment,
            )
            # G(β − τ) carries the same wrap sign as G(τ_B) does: on the Rust
            # grid the time is τ or τ − β, and shifting the argument of a
            # fermionic function by a period costs the same ζ either way.
            write_table(
                EXAMPLE,
                "green_tau_reversed",
                {
                    "tau_b": tau_b,
                    "g_re": (sign_fb * g_beta_minus_tau[order_b]).real,
                    "g_im": (sign_fb * g_beta_minus_tau[order_b]).imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "polarization_tau",
                {
                    "tau_b": tau_b,
                    "p_re": p_tau_b[order_b].real,
                    "p_im": p_tau_b[order_b].imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "polarization_coefficients",
                {
                    "l": np.arange(basisb.size, dtype=float),
                    "p_l_abs": np.abs(p_l_b),
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "polarization_matsubara",
                {
                    "n": matsb.sampling_points.astype(float),
                    "p_re": p_iw_b.real,
                    "p_im": p_iw_b.imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "screened_matsubara",
                {
                    "n": matsb.sampling_points.astype(float),
                    "w_re": w_iw_b.real,
                    "w_im": w_iw_b.imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "screened_coefficients",
                {
                    "l": np.arange(basisb.size, dtype=float),
                    "w_l_abs": np.abs(w_l_b),
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "screened_tau",
                {
                    "tau_f": tau_f,
                    "w_re": (sign_bf * w_tau_f[order_f]).real,
                    "w_im": (sign_bf * w_tau_f[order_f]).imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "self_energy_tau",
                {
                    "tau_f": tau_f,
                    "e_re": (sign_ff * e_tau_f[order_f]).real,
                    "e_im": (sign_ff * e_tau_f[order_f]).imag,
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "self_energy_coefficients",
                {
                    "l": np.arange(basisf.size, dtype=float),
                    "e_l_abs": np.abs(e_l_f),
                },
                comment,
            )
            write_table(
                EXAMPLE,
                "self_energy_matsubara",
                {
                    "n": matsf.sampling_points.astype(float),
                    "e_re": e_iw_f_hartree.real,
                    "e_im": e_iw_f_hartree.imag,
                    "hartree": np.full(matsf.sampling_points.size, hartree.real),
                },
                comment,
            )

        # The Dyson equation. As in the notebook, the Hartree term is left out
        # of it: it is a constant shift of the chemical potential.
        g_iw_f = 1 / (1 / g_iw_0 - e_iw_f)

    write_table(
        EXAMPLE,
        "self_energy_final",
        {
            "n": matsf.sampling_points.astype(float),
            "e_re": previous.real,
            "e_im": previous.imag,
        },
        comment,
    )
    write_table(
        EXAMPLE,
        "green_final",
        {
            "n": matsf.sampling_points.astype(float),
            "g_re": g_iw_f.real,
            "g_im": g_iw_f.imag,
        },
        comment,
    )
    write_table(
        EXAMPLE,
        "convergence",
        {
            "iteration": np.arange(1, ITERATIONS, dtype=float),
            "difference": np.array(differences),
        },
        comment,
    )
