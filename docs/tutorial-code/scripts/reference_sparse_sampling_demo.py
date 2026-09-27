"""Reference values for the `sparse_sampling_demo` example.

This is the computation of `sparse_sampling_demo_py.ipynb` of
sparse-ir-tutorial, with the plotting removed and the numbers written out.
"""

from __future__ import annotations

import numpy as np
import sparse_ir

from reference_common import fold_tau, provenance, write_table

EXAMPLE = "sparse_sampling_demo"
BETA = 10_000.0
WMAX = 1.0
EPS = 1e-15


def rho(omega: float) -> float:
    """The semicircular spectral function of full bandwidth 2."""
    if abs(omega) < 1:
        return (2 / np.pi) * np.sqrt(1 - omega**2)
    return 0.0


def write() -> None:
    comment = provenance(EXAMPLE)
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)

    rho_l = basis.v.overlap(rho)
    g_l = -basis.s * rho_l
    ls = np.arange(basis.size, dtype=float)

    write_table(
        EXAMPLE,
        "coefficients",
        {"l": ls, "s_l": basis.s, "rho_l": rho_l, "g_l": g_l},
        comment,
    )

    smpl_tau = sparse_ir.TauSampling(basis)
    g_tau = smpl_tau.evaluate(g_l)
    g_l_from_tau = smpl_tau.fit(g_tau)
    # The Rust example reports the sampling times folded onto [-β/2, β/2];
    # carry `G(τ)` across with the antiperiodicity so the two line up.
    tau, sign, order = fold_tau(smpl_tau.sampling_points, BETA, "F")
    write_table(EXAMPLE, "tau_sampling", {"tau": tau, "g_tau": sign * g_tau[order]}, comment)

    smpl_matsu = sparse_ir.MatsubaraSampling(basis)
    g_iv = smpl_matsu.evaluate(g_l)
    g_l_from_matsu = np.real(smpl_matsu.fit(g_iv))
    n = np.asarray(smpl_matsu.wn, dtype=float)
    write_table(
        EXAMPLE,
        "matsubara_sampling",
        {"n": n, "nu": n * np.pi / BETA, "g_iv_re": g_iv.real, "g_iv_im": g_iv.imag},
        comment,
    )

    write_table(
        EXAMPLE,
        "reconstruction",
        {
            "l": ls,
            "g_l": g_l,
            "g_l_from_tau": g_l_from_tau,
            "g_l_from_matsubara": g_l_from_matsu,
            "error_tau": np.abs(g_l_from_tau - g_l),
            "error_matsubara": np.abs(g_l_from_matsu - g_l),
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "summary",
        {
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "eps": np.array([EPS]),
            "basis_size": np.array([float(basis.size)]),
            "accuracy": np.array([basis.accuracy]),
            "n_tau_points": np.array([float(len(tau))]),
            "n_matsubara_points": np.array([float(len(n))]),
            "cond_tau": np.array([smpl_tau.cond]),
            "cond_matsubara": np.array([smpl_matsu.cond]),
        },
        comment,
    )
