"""Shared plumbing for the scripts that produce reference values.

A reference file is written by the published Python implementation and
committed. The Rust examples are then checked against it, so a change in the
Rust library that moves a number shows up as a failing test rather than as a
quietly different figure in the book.

The files use the same format the Rust examples write (see
`docs/tutorial-code/src/csv.rs`): a provenance comment, a header of column
names, then rows. Numbers are written with `repr`, which is the shortest text
that parses back to the same double, so a file never loses a bit and two runs
that computed the same numbers produce identical files.
"""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def provenance(example: str) -> str:
    """The first line of every reference file: which code produced it."""
    versions = " ".join(
        f"{name}={metadata.version(name)}" for name in ("sparse-ir", "pylibsparseir", "numpy")
    )
    return f"# example={example} {versions}"


def write_table(example: str, name: str, columns: dict[str, np.ndarray], comment: str) -> Path:
    """Writes one reference file and returns where it went."""
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise SystemExit(f"{example}/{name}: the columns have different lengths: {lengths}")

    directory = REFERENCE_DIR / example
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"

    lines = [comment, ",".join(columns)]
    for row in range(lengths.pop()):
        lines.append(",".join(repr(float(values[row])) for values in columns.values()))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(REFERENCE_DIR.parents[2])}")
    return path


def fold_tau(
    tau: np.ndarray, beta: float, statistics: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moves sampling times from `[0, β]` onto `[−β/2, β/2]`.

    The Python implementation reports the default sampling times on `[0, β]`;
    the Rust one reports them folded around `β/2`, which is the same set of
    points seen through `G(τ − β) = ∓G(τ)`.

    Returns the folded times, the sign each one picked up, and the permutation
    that sorted them — apply the last two to a sampled `G` to carry it across:

        tau, sign, order = fold_tau(smpl.sampling_points, beta, "F")
        g_folded = sign * g_tau[order]
    """
    if statistics not in ("F", "B"):
        raise ValueError(f"statistics must be 'F' or 'B', got {statistics!r}")
    tau = np.asarray(tau, dtype=float)
    beyond_half = tau > beta / 2
    folded = np.where(beyond_half, tau - beta, tau)
    sign = np.where(beyond_half, -1.0 if statistics == "F" else 1.0, 1.0)
    order = np.argsort(folded)
    return folded[order], sign[order], order
