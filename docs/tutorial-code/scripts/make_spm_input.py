"""Writes the committed input of the sparse-modeling example.

    uv run --project . make_spm_input.py

The Python notebook this example is ported from downloads `Gtau.in` from the
SpM repository at run time. A tutorial that is built in CI must not depend on
the network, so the input is generated once, here, and committed under
`docs/tutorial-code/input/spm/`.

The data is `G(τ)` of a known three-Gaussian spectral function, sampled on a
uniform grid and given independent Gaussian noise — the same shape of problem
the downloaded file poses, but with an exact answer to compare against. The
seed is fixed, so re-running this script reproduces the committed file
byte for byte.

Both `src/bin/spm.rs` and `reference_spm.py` read the committed file; neither
regenerates it. That way the two implementations start from bit-identical
data and any difference between them is a difference in the solver, not in
the input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sparse_ir

import reference_common

BETA = 100.0
WMAX = 4.0
EPS = 1e-10
N_TAU = 401
NOISE = 1e-3
SEED = 20260927

INPUT_DIR = Path(__file__).resolve().parents[1] / "input"


def three_gaussians(omega: np.ndarray) -> np.ndarray:
    """Matches `sparse_ir_tutorial::three_gaussians`."""

    def gaussian(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-(((omega - mu) / sigma) ** 2)) / (np.sqrt(np.pi) * sigma)

    return 0.2 * gaussian(0.0, 0.15) + 0.4 * gaussian(1.0, 0.8) + 0.4 * gaussian(-1.0, 0.8)


def write() -> Path:
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    taus = np.linspace(0.0, BETA, N_TAU)
    g_l = -basis.s * basis.v.overlap(three_gaussians)
    clean = basis.u(taus).T @ g_l
    noise = NOISE * np.random.default_rng(SEED).standard_normal(N_TAU)

    directory = INPUT_DIR / "spm"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gtau.csv"
    comment = (
        f"{reference_common.provenance('spm')}"
        f" beta={BETA!r} wmax={WMAX!r} eps={EPS!r} noise={NOISE!r} seed={SEED}"
    )
    lines = [comment, "tau,g_tau,g_tau_clean"]
    for tau, noisy, exact in zip(taus, clean + noise, clean):
        lines.append(",".join(repr(float(value)) for value in (tau, noisy, exact)))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    write()
