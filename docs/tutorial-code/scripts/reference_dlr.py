"""Reference values for the `dlr` example.

This is the computation of `DLR_py.ipynb` of sparse-ir-tutorial, with the
plotting removed and the numbers written out.
"""

from __future__ import annotations

import numpy as np
from sparse_ir import FiniteTempBasis, MatsubaraSampling
from sparse_ir.dlr import DiscreteLehmannRepresentation

from reference_common import provenance, write_table

EXAMPLE = "dlr"
WMAX = 1.0
LAMBDA = 1e4
BETA = LAMBDA / WMAX
EPS = 1e-15

# The Matsubara frequencies the last section evaluates on: every tenth
# fermionic frequency out to |n| = 2000.
FREQUENCIES = 2 * np.arange(-1000, 1000, 10) + 1


def rho(omega: float) -> float:
    """The semicircular spectral function of full bandwidth 2."""
    if abs(omega) < 1:
        return (2 / np.pi) * np.sqrt(1 - omega**2)
    return 0.0


def write() -> None:
    comment = provenance(EXAMPLE)
    basis = FiniteTempBasis("F", BETA, WMAX, eps=EPS)

    rho_l = basis.v.overlap(rho)
    g_l = -basis.s * rho_l
    write_table(
        EXAMPLE,
        "coefficients",
        {
            "l": np.arange(basis.size, dtype=float),
            "s_l": basis.s,
            "rho_l": rho_l,
            "g_l": g_l,
        },
        comment,
    )

    dlr = DiscreteLehmannRepresentation(basis)
    g_dlr = dlr.from_IR(g_l)
    write_table(
        EXAMPLE,
        "dlr_coefficients",
        {
            "p": np.arange(len(dlr.sampling_points), dtype=float),
            "pole": np.asarray(dlr.sampling_points, dtype=float),
            "c_p": g_dlr,
        },
        comment,
    )

    g_l_reconstructed = dlr.to_IR(g_dlr)
    write_table(
        EXAMPLE,
        "reconstruction",
        {
            "l": np.arange(basis.size, dtype=float),
            "g_l": g_l,
            "g_l_from_dlr": g_l_reconstructed,
            "error": np.abs(g_l - g_l_reconstructed),
        },
        comment,
    )

    # The DLR evaluates on the Matsubara axis by its own definition: a sum of
    # poles, with no basis functions in sight.
    i_nu = 1j * FREQUENCIES * (np.pi / BETA)
    g_iv_dlr = (1 / (i_nu[:, None] - dlr.sampling_points[None, :])) @ g_dlr
    g_iv_exact = MatsubaraSampling(basis, FREQUENCIES).evaluate(g_l)
    write_table(
        EXAMPLE,
        "matsubara",
        {
            "n": FREQUENCIES.astype(float),
            "nu": i_nu.imag,
            "g_iv_exact_im": g_iv_exact.imag,
            "g_iv_dlr_im": g_iv_dlr.imag,
            "g_iv_exact_re": g_iv_exact.real,
            "g_iv_dlr_re": g_iv_dlr.real,
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
            "lambda": np.array([LAMBDA]),
            "basis_size": np.array([float(basis.size)]),
            "n_poles": np.array([float(len(dlr.sampling_points))]),
            "accuracy": np.array([basis.accuracy]),
        },
        comment,
    )
