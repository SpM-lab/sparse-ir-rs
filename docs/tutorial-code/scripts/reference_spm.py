"""Reference values for the `spm` example.

The Python notebook `spm_py.ipynb` of sparse-ir-tutorial solves the
sparse-modeling problem with ADMM (the `admmsolver` package), because it
imposes `ρ(ω) ≥ 0` and the sum rule as hard constraints. The Rust example
drops both constraints and uses plain FISTA, so this reference does the same:
it is an independent implementation of the *same* problem, written against the
published `sparse-ir`, and what it checks is the basis, the sampling of `u_l`
and `v_l`, and the arithmetic — not a different algorithm's answer.

Both sides read the same committed input, `input/spm/gtau.csv`, and run the
same fixed number of FISTA steps, so the two solutions agree far more closely
than the solver's own accuracy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import sparse_ir

from reference_common import provenance, write_table

EXAMPLE = "spm"

# These must agree with `src/bin/spm.rs`.
BETA = 100.0
WMAX = 4.0
EPS = 1e-10
N_OMEGA = 501
LAMBDAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
CHOSEN_LAMBDA = 3e-5
POWER_ITERATIONS = 200
FISTA_MAX_ITER = 20000
FISTA_TOL = 0.0
SETTLED = 1e-6

INPUT = Path(__file__).resolve().parents[1] / "input" / EXAMPLE / "gtau.csv"


def gaussian(omega: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-(((omega - mu) / sigma) ** 2)) / (np.sqrt(np.pi) * sigma)


def three_gaussians(omega: np.ndarray) -> np.ndarray:
    return (
        0.2 * gaussian(omega, 0.0, 0.15)
        + 0.4 * gaussian(omega, 1.0, 0.8)
        + 0.4 * gaussian(omega, -1.0, 0.8)
    )


def read_input() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.reader(INPUT.open()))
    header = rows[1]
    columns = np.array([[float(value) for value in row] for row in rows[2:]]).T
    named = dict(zip(header, columns))
    return named["tau"], named["g_tau"], named["g_tau_clean"]


def largest_eigenvalue_of_ata(a: np.ndarray) -> float:
    """Power iteration from a fixed start, matching the Rust example."""
    v = np.full(a.shape[1], 1.0 / np.sqrt(a.shape[1]))
    eigenvalue = 0.0
    for _ in range(POWER_ITERATIONS):
        w = a.T @ (a @ v)
        eigenvalue = np.linalg.norm(w)
        v = w / eigenvalue
    return float(eigenvalue)


def solve(a: np.ndarray, y: np.ndarray, lipschitz: float, lam: float) -> tuple[np.ndarray, float]:
    """FISTA on `½‖y − A x‖² + λ‖x‖₁`, step for step as `sparse_ir_tutorial::fista`."""
    step = 1.0 / lipschitz
    x = np.zeros(a.shape[1])
    previous = x.copy()
    momentum_point = x.copy()
    t = 1.0
    relative_change = np.inf
    for _ in range(FISTA_MAX_ITER):
        u = momentum_point - step * (a.T @ (a @ momentum_point - y))
        x = np.sign(u) * np.maximum(np.abs(u) - step * lam, 0.0)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum_point = x + (t - 1.0) / t_next * (x - previous)
        t = t_next
        relative_change = np.abs(x - previous).max() / max(1.0, np.abs(x).max())
        previous = x.copy()
        if relative_change <= FISTA_TOL:
            break
    return x, float(relative_change)


def write() -> None:
    taus, g_tau, g_tau_clean = read_input()
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    size = basis.size

    a = basis.u(taus).T * basis.s
    y = -g_tau
    lipschitz = largest_eigenvalue_of_ata(a)

    omegas = np.linspace(-WMAX, WMAX, N_OMEGA)
    d_omega = omegas[1] - omegas[0]
    rho_exact = three_gaussians(omegas)
    rho_l_exact = basis.v.overlap(three_gaussians)
    v_at_omegas = basis.v(omegas).T

    comment = provenance(EXAMPLE)
    scan: dict[str, list[float]] = {
        name: []
        for name in ("lambda", "residual", "l1_norm", "l2_error", "sum_rule", "min_rho", "nonzero")
    }
    chosen = None

    for lam in LAMBDAS:
        rho_l, relative_change = solve(a, y, lipschitz, lam)
        if relative_change >= SETTLED:
            raise SystemExit(f"lambda={lam:e} was still moving by {relative_change:e}")
        rho = v_at_omegas @ rho_l
        scan["lambda"].append(lam)
        scan["residual"].append(float(np.linalg.norm(y - a @ rho_l)))
        scan["l1_norm"].append(float(np.abs(rho_l).sum()))
        scan["l2_error"].append(float(np.sqrt(np.sum((rho - rho_exact) ** 2) * d_omega)))
        scan["sum_rule"].append(float(rho.sum() * d_omega))
        scan["min_rho"].append(float(rho.min()))
        scan["nonzero"].append(float(np.count_nonzero(rho_l)))
        if lam == CHOSEN_LAMBDA:
            chosen = rho_l

    write_table(EXAMPLE, "lambda_scan", {k: np.array(v) for k, v in scan.items()}, comment)

    assert chosen is not None, "CHOSEN_LAMBDA must be one of LAMBDAS"
    rho = v_at_omegas @ chosen

    write_table(
        EXAMPLE,
        "spectrum",
        {"omega": omegas, "rho_exact": rho_exact, "rho_recovered": rho},
        comment,
    )
    write_table(
        EXAMPLE,
        "coefficients",
        {
            "l": np.arange(size, dtype=float),
            "s_l": basis.s,
            "rho_l_exact": rho_l_exact,
            "rho_l_recovered": chosen,
        },
        comment,
    )
    write_table(
        EXAMPLE,
        "gtau",
        {
            "tau": taus,
            "g_tau_input": g_tau,
            "g_tau_clean": g_tau_clean,
            "g_tau_fit": -(a @ chosen),
        },
        comment,
    )
    write_table(
        EXAMPLE,
        "summary",
        {
            "basis_size": np.array([float(size)]),
            "n_tau": np.array([float(len(taus))]),
            "lipschitz": np.array([lipschitz]),
            "lambda": np.array([CHOSEN_LAMBDA]),
        },
        comment,
    )


if __name__ == "__main__":
    write()
