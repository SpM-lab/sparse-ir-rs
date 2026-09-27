"""Writes the committed starting point of the Eliashberg example.

    uv run --project . make_eliashberg_holstein_input.py

The notebook this example is ported from seeds its self-energy with Gaussian
noise drawn from `numpy.random.default_rng(0)`. Reproducing NumPy's generator
in Rust is not the lesson of the page, and the fixed point the solver walks to
would be the same from any small perturbation, so the draw is made once, here,
and committed under `docs/tutorial-code/input/eliashberg_holstein/`.

Both `src/bin/eliashberg_holstein.rs` and its scan read the committed file;
neither regenerates it. That way the two implementations start from
bit-identical data and any difference between them is a difference in the
solver, not in the seed. The two runs need different lengths, because the
first is at a single low temperature and the scan holds `Λ` fixed instead, so
the file carries one column per run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sparse_ir

import reference_common

D = 0.5
WMAX = 10 * D
EPS = 1e-7
SEED = 0

# The single solve of `eliashberg_holstein`.
BETA = 500.0
NOISE = 1e-2

# The temperature sweep of `eliashberg_holstein_scan`, whose basis is set by
# the common `Λ` rather than by any one temperature.
TEMPERATURES = np.linspace(0.009, 0.013, 10)
LAMBDA = WMAX / TEMPERATURES.min()
SCAN_NOISE = 1e-5

INPUT_DIR = Path(__file__).resolve().parents[1] / "input"


def draw(size: int, noise: float) -> np.ndarray:
    """`add_noise` of the notebook, on an array of zeros."""
    rng = np.random.default_rng(SEED)
    return noise * rng.standard_normal(size) + noise * 1j * rng.standard_normal(size)


def write_one(name: str, values: np.ndarray, detail: str) -> Path:
    directory = INPUT_DIR / "eliashberg_holstein"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    comment = (
        f"{reference_common.provenance('eliashberg_holstein')}"
        f" wmax={WMAX!r} eps={EPS!r} seed={SEED} {detail}"
    )
    lines = [comment, "sigma_re,sigma_im"]
    for value in values:
        lines.append(f"{float(value.real)!r},{float(value.imag)!r}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


def write() -> list[Path]:
    single = sparse_ir.FiniteTempBasisSet(BETA, WMAX, EPS)
    beta = 1.0 / TEMPERATURES.min()
    scan = sparse_ir.FiniteTempBasisSet(beta, LAMBDA / beta, EPS)
    return [
        write_one("noise", draw(single.wn_f.size, NOISE), f"beta={BETA!r} noise={NOISE!r}"),
        write_one(
            "scan_noise",
            draw(scan.wn_f.size, SCAN_NOISE),
            f"lambda={LAMBDA!r} noise={SCAN_NOISE!r}",
        ),
    ]


if __name__ == "__main__":
    write()
