"""Shared plumbing for the tutorial's plotting scripts.

A script here never computes physics. It reads a CSV that one of the Rust
examples wrote and draws it, so that the figure in the book and the numbers the
verification tests check are always the same numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

__all__ = ["data_dir", "figure_dir", "plt", "read_table", "save"]
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Where the Rust examples wrote their CSV files."""
    override = os.environ.get("SPARSEIR_TUTORIAL_DATA_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "docs" / "tutorial-code" / "data"


def figure_dir() -> Path:
    """Where the book expects the figures."""
    override = os.environ.get("SPARSEIR_TUTORIAL_FIGURE_DIR")
    directory = Path(override) if override else REPO_ROOT / "docs" / "book" / "src" / "tutorials"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_table(example: str, name: str) -> dict[str, np.ndarray]:
    """Reads `data/<example>/<name>.csv` into a dict of columns."""
    path = data_dir() / example / f"{name}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist; run the `{example}` example first "
            f"(cargo run --release --bin {example})"
        )
    with path.open() as handle:
        handle.readline()  # the provenance comment
        header = handle.readline().strip().split(",")
        values = np.loadtxt(handle, delimiter=",", ndmin=2)
    if values.shape[1] != len(header):
        raise SystemExit(f"{path}: {values.shape[1]} columns but {len(header)} names")
    return {name: values[:, index] for index, name in enumerate(header)}


def save(fig: plt.Figure, name: str) -> Path:
    """Writes the figure where the book looks for it."""
    path = figure_dir() / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return path
