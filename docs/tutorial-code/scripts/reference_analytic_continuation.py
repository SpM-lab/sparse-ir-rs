"""Reference values for the `analytic_continuation` example.

Everything here follows `analytic_continuation_py.ipynb` of
sparse-ir-tutorial, computed against the published `sparse-ir`. The noise is
not drawn again: both sides read the committed `input/analytic_continuation/
noise.csv`, so the two reconstructions are comparable number by number rather
than only in distribution.

The one place the two implementations genuinely differ is the quadrature.
Python uses `basis.v.overlap`, which is adaptive; Rust integrates in closed
form over the basis' own knots, with a substitution that removes the square
root of the semicircle and the peak of the Lorentzian. Agreeing to a few
times machine precision is therefore a statement about both.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import sparse_ir

from reference_common import provenance, write_table

EXAMPLE = "analytic_continuation"

# These must agree with `src/bin/analytic_continuation.rs`.
BETA = 40.0
WMAX = 2.0
EPS = 2e-8
NOISE_FRACTION = 0.3
ALPHA_IN_NOISE = 100.0
N_OMEGA = 1001
ETA_IN_PI_T = 0.1
N_LORENTZ = 21
DISCRETE_POLES = (-0.6, -0.1, 0.1, 0.6)

INPUT = Path(__file__).resolve().parents[1] / "input" / EXAMPLE / "noise.csv"


def read_noise() -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.reader(INPUT.open()))
    header = rows[1]
    columns = np.array([[float(value) for value in row] for row in rows[2:]]).T
    named = dict(zip(header, columns))
    return named["z_semielliptic"], named["z_insulating"]


def semicircle(omega: np.ndarray, center: float, half_width: float, weight: float) -> np.ndarray:
    x = (np.asarray(omega, dtype=float) - center) / half_width
    inside = np.abs(x) < 1.0
    return np.where(
        inside,
        weight * (2.0 / (np.pi * half_width)) * np.sqrt(np.where(inside, 1.0 - x * x, 0.0)),
        0.0,
    )


def semielliptic(omega: np.ndarray) -> np.ndarray:
    return semicircle(omega, 0.0, WMAX, 1.0)


def insulating(omega: np.ndarray) -> np.ndarray:
    return semicircle(omega, WMAX / 2, WMAX / 4, 0.5) + semicircle(
        omega, -WMAX / 2, WMAX / 4, 0.5
    )


def lorentz(omega: np.ndarray, eta: float) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    return eta / (np.pi * (omega * omega + eta * eta))


def write() -> None:
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    size = basis.size
    s = np.asarray(basis.s)

    noise = NOISE_FRACTION * s[-1] / s[0]
    alpha = ALPHA_IN_NOISE * noise
    eta = ETA_IN_PI_T * np.pi / BETA
    comment = provenance(EXAMPLE)

    # --- the two models and the noise on them -------------------------------
    edges = (-WMAX, -WMAX / 2, 0.0, WMAX / 2, WMAX)
    rho_semi = basis.v.overlap(semielliptic, points=edges)
    rho_insul = basis.v.overlap(insulating, points=edges)
    g_semi = -s * rho_semi
    g_insul = -s * rho_insul

    z_semi, z_insul = read_noise()
    g_semi_noisy = g_semi + noise * z_semi * np.linalg.norm(g_semi)
    g_insul_noisy = g_insul + noise * z_insul * np.linalg.norm(g_insul)

    write_table(
        EXAMPLE,
        "coefficients",
        {
            "l": np.arange(size, dtype=float),
            "s_l": s,
            "s_ratio": s / s[0],
            "rho_semielliptic": rho_semi,
            "g_semielliptic": g_semi,
            "g_semielliptic_noisy": g_semi_noisy,
            "rho_insulating": rho_insul,
            "g_insulating": g_insul,
            "g_insulating_noisy": g_insul_noisy,
        },
        comment,
    )

    # --- truncated-SVD regularisation ---------------------------------------
    omegas = np.linspace(-WMAX, WMAX, N_OMEGA)
    v_at_omegas = basis.v(omegas).T  # (points, size)
    half = size // 2

    rho_l_semi = -g_semi_noisy / s
    rho_l_insul = -g_insul_noisy / s

    write_table(
        EXAMPLE,
        "tsvd",
        {
            "omega": omegas,
            "semielliptic_exact": semielliptic(omegas),
            "semielliptic_half": v_at_omegas[:, :half] @ rho_l_semi[:half],
            "semielliptic_full": v_at_omegas @ rho_l_semi,
            "insulating_exact": insulating(omegas),
            "insulating_half": v_at_omegas[:, :half] @ rho_l_insul[:half],
            "insulating_full": v_at_omegas @ rho_l_insul,
        },
        comment,
    )

    # --- ridge regression ---------------------------------------------------
    ridge = -s / (s * s + alpha * alpha)
    write_table(
        EXAMPLE,
        "ridge",
        {
            "omega": omegas,
            "semielliptic_ridge": v_at_omegas @ (ridge * g_semi_noisy),
            "insulating_ridge": v_at_omegas @ (ridge * g_insul_noisy),
        },
        comment,
    )

    # --- why ρₗ is the wrong thing to solve for -----------------------------
    poles = np.array(DISCRETE_POLES) * WMAX
    rho_discrete = basis.v(poles).sum(axis=1)
    g_discrete = -s * rho_discrete

    write_table(
        EXAMPLE,
        "discrete",
        {
            "l": np.arange(size, dtype=float),
            "g_semielliptic": g_semi,
            "rho_semielliptic": rho_semi,
            "g_discrete": g_discrete,
            "rho_discrete": rho_discrete,
        },
        comment,
    )

    # --- a real-axis basis instead ------------------------------------------
    write_table(
        EXAMPLE,
        "lorentz",
        {"omega": omegas, "f": lorentz(omegas, eta)},
        comment,
    )

    centres = np.linspace(-WMAX, WMAX, N_LORENTZ)
    columns = {"l": np.arange(size, dtype=float)}
    for m, centre in enumerate(centres):
        # `points` puts the peak on a segment boundary, which is what lets the
        # adaptive quadrature see it at all: η is a hundredth of the spacing
        # between the knots it would otherwise subdivide on.
        overlap = basis.v.overlap(
            lambda omega: lorentz(omega - centre, eta), points=(centre,)
        )
        columns[f"k_{m}"] = -s * overlap
    write_table(EXAMPLE, "lorentz_kernel", columns, comment)

    write_table(
        EXAMPLE,
        "summary",
        {
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "eps": np.array([EPS]),
            "basis_size": np.array([float(size)]),
            "noise": np.array([noise]),
            "alpha": np.array([alpha]),
            "eta": np.array([eta]),
            "n_lorentz": np.array([float(N_LORENTZ)]),
        },
        comment,
    )


if __name__ == "__main__":
    write()
