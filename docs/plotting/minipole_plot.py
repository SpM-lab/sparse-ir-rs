"""Figures from `cargo run --profile ci --bin minipole` in docs/tutorial-code.

    uv run --project docs/plotting python docs/plotting/minipole_plot.py
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "minipole"


def main() -> None:
    plt.rcParams["font.size"] = 12
    exact = read_table(EXAMPLE, "exact_poles")
    beta = read_table(EXAMPLE, "summary")["beta"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, label, marker in [
        ("default", r"$n_0=5,\ n_{\max}=\beta=20$", "x"),
        ("extended", r"$n_0=5,\ n_{\max}=50$", "+"),
    ]:
        poles = read_table(EXAMPLE, f"poles_{name}")
        axes[0].scatter(beta * poles["pole_re"], beta * poles["pole_im"], marker=marker,
                        s=90, label=label)
        values = read_table(EXAMPLE, f"values_{name}")
        axes[1].semilogy(values["nu"], np.maximum(values["relative_error"], 1e-16),
                         label=label)
    axes[0].scatter(beta * exact["pole"], np.zeros_like(exact["pole"]), facecolors="none",
                    edgecolors="black", s=110, label="analytic poles")
    axes[0].set(xlabel=r"$\beta\,\mathrm{Re}\,\xi$", ylabel=r"$\beta\,\mathrm{Im}\,\xi$",
                xlim=(-5, 5), title="Low-energy pair (outer poles not shown)")
    axes[0].axhline(0, color="grey", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xlabel=r"$\nu$", ylabel=r"$|\chi_{\rm MP}(i\nu)-\chi(i\nu)|/|\chi(0)|$",
                title="Bosonic Matsubara reconstruction")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "minipole_pair")

    scan = read_table(EXAMPLE, "contours")
    n0s, nmaxs = np.unique(scan["n0"]), np.unique(scan["nmax"])
    errors = scan["max_relative_error"].reshape(len(n0s), len(nmaxs))
    counts = scan["n_poles"].reshape(errors.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    image = ax.imshow(np.log10(np.maximum(errors, 1e-16)), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(nmaxs)), [f"{n:g}" for n in nmaxs])
    ax.set_yticks(range(len(n0s)), [f"{n:g}" for n in n0s])
    for i in range(len(n0s)):
        for j in range(len(nmaxs)):
            ax.text(j, i, f"{errors[i, j]:.1e}\n{counts[i, j]:.0f} poles",
                    ha="center", va="center", fontsize=10,
                    color="white" if image.norm(image.get_array()[i, j]) > 0.55 else "black")
    ax.set(xlabel=r"$n_{\max}$", ylabel=r"$n_0$", title="Same DLR coefficients, different contours")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}\,[\max_\nu |\delta\chi(i\nu)|/|\chi(0)|]$")
    fig.tight_layout()
    save(fig, "minipole_contours")


if __name__ == "__main__":
    main()
