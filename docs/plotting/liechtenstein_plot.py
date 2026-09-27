"""Figures for the Liechtenstein page.

    uv run --project . liechtenstein_plot.py

Reads what `cargo run --profile ci --bin liechtenstein` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "liechtenstein"
NAIVE = (100, 200, 400, 800, 1600)


def main() -> None:
    plt.rcParams["font.size"] = 13

    j0 = read_table(EXAMPLE, "j0")
    jij = read_table(EXAMPLE, "jij")

    # The figure the notebook is built around: the truncated Matsubara sums
    # creeping towards the answer the basis gives outright.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.cm.cool(np.linspace(0.0, 0.8, len(NAIVE)))
    for nm, color in zip(NAIVE, colors):
        ax.plot(j0["mu"], j0[f"j0_nm{nm}"], marker=".", color=color, label=str(2 * nm))
    ax.plot(j0["mu"], j0["j0"], marker=".", color=plt.cm.cool(1.0), label="IR")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$J_0$")
    ax.legend(frameon=False, title=r"$N_\mathrm{Matsubara}$", ncols=2)
    save(fig, f"{EXAMPLE}_j0")

    # How fast they creep: the error against the IR answer, halving with each
    # doubling of the grid.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sizes = np.array([2 * nm for nm in NAIVE], dtype=float)
    errors = np.array(
        [np.max(np.abs(j0[f"j0_nm{nm}"] - j0["j0"])) for nm in NAIVE]
    )
    ax.loglog(sizes, errors, marker="x", label="measured")
    ax.loglog(sizes, errors[0] * sizes[0] / sizes, ls="--", label=r"$\propto 1/N$")
    ax.set_xlabel(r"$N_\mathrm{Matsubara}$")
    ax.set_ylabel(r"$\max_\mu |J_0 - J_0^\mathrm{IR}|$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_convergence")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(jij["distance"], jij["j_ij"], marker=".")
    ax.set_xlim(0, 5)
    ax.set_xlabel("distance")
    ax.set_ylabel(r"$J_{ij}$")
    save(fig, f"{EXAMPLE}_jij")


if __name__ == "__main__":
    main()
