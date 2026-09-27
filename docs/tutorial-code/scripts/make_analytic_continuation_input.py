"""Writes the committed noise of the analytic-continuation example.

    uv run --project . make_analytic_continuation_input.py

The notebook draws its noise from `numpy.random.RandomState(4711)`, a Mersenne
Twister stream no Rust program is going to reproduce. Rather than invent a
different noise — which would make the Rust and Python results incomparable,
and change the figures — the draws are taken once, here, and committed.

`RandomState.normal(0, sigma, n)` is `sigma * standard_normal(n)` on the same
stream, so storing the standard normal draws and scaling them in the example
gives exactly the numbers the notebook used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sparse_ir

import reference_common

BETA = 40.0
WMAX = 2.0
EPS = 2e-8
SEED = 4711

INPUT_DIR = Path(__file__).resolve().parents[1] / "input"


def write() -> Path:
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    # Two blocks of `size` draws: the notebook makes one call per model.
    draws = np.random.RandomState(SEED).standard_normal(2 * basis.size)

    directory = INPUT_DIR / "analytic_continuation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "noise.csv"
    comment = (
        f"{reference_common.provenance('analytic_continuation')}"
        f" beta={BETA!r} wmax={WMAX!r} eps={EPS!r} seed={SEED}"
        " source=numpy.random.RandomState.standard_normal"
    )
    lines = [comment, "l,z_semielliptic,z_insulating"]
    for l, (first, second) in enumerate(zip(draws[: basis.size], draws[basis.size :])):
        lines.append(",".join(repr(float(value)) for value in (l, first, second)))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    write()
