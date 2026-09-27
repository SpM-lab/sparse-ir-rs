"""Reference values for the `transformation` example.

This is the computation of `transformation_py.ipynb` of sparse-ir-tutorial,
with the plotting removed and the numbers written out.
"""

from __future__ import annotations

import numpy as np
import sparse_ir
from sparse_ir.dlr import DiscreteLehmannRepresentation

from reference_common import provenance, write_table

EXAMPLE = "transformation"

# The pole section: one bosonic pole slightly off zero.
POLE_BETA = 15.0
POLE_WMAX = 10.0
POLE_EPS = 1e-10
POLE_POSITION = 0.1
POLE_WEIGHT = 1.0

# The smooth section: three Gaussian peaks.
BETA = 10.0
WMAX = 10.0
EPS = 1e-10

# The remark at the end: a basis whose ωmax is far too small.
NARROW_WMAX = 0.5

N_OMEGA = 1000
N_TAU = 1000


def gaussian(omega: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-(((omega - mu) / sigma) ** 2)) / (np.sqrt(np.pi) * sigma)


def rho(omega: np.ndarray) -> np.ndarray:
    """Three Gaussian peaks, normalized to one."""
    return (
        0.2 * gaussian(omega, 0.0, 0.15)
        + 0.4 * gaussian(omega, 1.0, 0.8)
        + 0.4 * gaussian(omega, -1.0, 0.8)
    )


def write() -> None:
    comment = provenance(EXAMPLE)

    # --- poles -----------------------------------------------------------
    basis_b = sparse_ir.FiniteTempBasis("B", POLE_BETA, POLE_WMAX, eps=POLE_EPS)
    poles = np.array([POLE_POSITION])
    weights = np.array([POLE_WEIGHT])
    # The logistic kernel absorbs the bosonic tanh into the spectral function.
    regularized = weights / np.tanh(0.5 * POLE_BETA * poles)

    v_at_poles = basis_b.v(poles).reshape(-1, poles.size)
    rho_l_pole = np.einsum("lp,p->l", v_at_poles, regularized)
    g_l_pole = -basis_b.s * rho_l_pole

    dlr = DiscreteLehmannRepresentation(basis_b, poles)
    g_l_pole_dlr = dlr.to_IR(regularized)

    write_table(
        EXAMPLE,
        "pole_coefficients",
        {
            "l": np.arange(basis_b.size, dtype=float),
            "rho_l": rho_l_pole,
            "g_l": g_l_pole,
            "g_l_dlr": g_l_pole_dlr,
        },
        comment,
    )

    # --- a smooth spectral function --------------------------------------
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    rho_l = basis.v.overlap(rho)
    g_l = -basis.s * rho_l

    write_table(
        EXAMPLE,
        "smooth_coefficients",
        {
            "l": np.arange(basis.size, dtype=float),
            "s_l": basis.s,
            "rho_l": rho_l,
            "g_l": g_l,
        },
        comment,
    )

    omegas = np.linspace(-5.0, 5.0, N_OMEGA)
    write_table(
        EXAMPLE,
        "spectrum",
        {
            "omega": omegas,
            "rho_exact": rho(omegas),
            "rho_reconstructed": basis.v(omegas).T @ rho_l,
        },
        comment,
    )

    # --- from IR to imaginary time ---------------------------------------
    taus = np.linspace(0.0, BETA, N_TAU)
    g_tau_direct = basis.u(taus).T @ g_l
    g_tau_sampling = sparse_ir.TauSampling(basis, taus).evaluate(g_l)
    write_table(
        EXAMPLE,
        "gtau",
        {"tau": taus, "g_tau_direct": g_tau_direct, "g_tau_sampling": g_tau_sampling},
        comment,
    )

    # --- back from the full imaginary-time data ---------------------------
    def eval_gtau(tau):
        return basis.u(tau).T @ g_l

    g_l_reconstructed = basis.u.overlap(eval_gtau)
    write_table(
        EXAMPLE,
        "roundtrip",
        {
            "l": np.arange(basis.size, dtype=float),
            "g_l": g_l,
            "g_l_reconstructed": g_l_reconstructed,
            "error": np.abs(g_l_reconstructed - g_l),
        },
        comment,
    )

    # --- what a too-small ωmax looks like ---------------------------------
    basis_narrow = sparse_ir.FiniteTempBasis("F", BETA, NARROW_WMAX, eps=EPS)
    g_l_narrow = basis_narrow.u.overlap(eval_gtau)
    write_table(
        EXAMPLE,
        "narrow_basis",
        {
            "l": np.arange(basis_narrow.size, dtype=float),
            "s_l": basis_narrow.s,
            "g_l": g_l_narrow,
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "summary",
        {
            "pole_beta": np.array([POLE_BETA]),
            "pole_wmax": np.array([POLE_WMAX]),
            "pole_basis_size": np.array([float(basis_b.size)]),
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "eps": np.array([EPS]),
            "basis_size": np.array([float(basis.size)]),
            "accuracy": np.array([basis.accuracy]),
            "narrow_wmax": np.array([NARROW_WMAX]),
            "narrow_basis_size": np.array([float(basis_narrow.size)]),
        },
        comment,
    )
