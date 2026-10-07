"""Figures from `cargo run --profile ci --bin minipole` in docs/tutorial-code.

    uv run --project docs/plotting python docs/plotting/minipole_plot.py
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize

from common import plt, read_table, save

EXAMPLE = "minipole"
POLE_COLOR = "#00d5ff"


def semicircle_figure() -> None:
    grid = read_table(EXAMPLE, "semicircle_grid")
    poles = read_table(EXAMPLE, "semicircle_poles")
    spectral = read_table(EXAMPLE, "semicircle_spectral")
    nx = len(np.unique(grid["x"]))
    ny = len(grid["x"]) // nx
    x = grid["x"].reshape(ny, nx)
    y = grid["y"].reshape(ny, nx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})
    levels = np.linspace(-1.2, 0.8, 41)
    for ax, column, title in [
        (axes[0], "log10_abs_exact", r"exact $G(z)$"),
        (axes[1], "log10_abs_mp", rf"MiniPole $G_{{\rm MP}}(z)$, {len(poles['pole_re'])} poles"),
    ]:
        filled = ax.contourf(x, y, grid[column].reshape(ny, nx), levels=levels,
                             cmap="magma", extend="both")
        ax.axhline(0, color="white", linewidth=0.6)
        ax.set_aspect("equal")
        ax.set(title=title, xlabel=r"$\mathrm{Re}\,z$", ylabel=r"$\mathrm{Im}\,z$")
        ax.text(-1.9, 1.3, "upper half plane: fitted", color="white", fontsize=10)
        ax.text(-1.9, -1.4, "lower half plane: not represented", color="white", fontsize=10)
    axes[1].set_ylabel("")
    axes[0].plot([-1, 1], [0, 0], color=POLE_COLOR, linewidth=3.5,
                 label="branch cut $[-1,1]$")
    axes[1].plot(poles["pole_re"], poles["pole_im"], "x", color=POLE_COLOR,
                 markersize=10, markeredgewidth=2.5, label="poles")
    for ax in axes[:2]:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=10, framealpha=0.85)
    fig.colorbar(filled, ax=axes[:2], label=r"$\log_{10}|G|$", shrink=0.9, pad=0.02)

    axes[2].plot(spectral["omega"], spectral["rho_exact"], color="black", linewidth=2.5,
                 label=r"exact $\rho(\omega)$")
    axes[2].plot(spectral["omega"], spectral["rho_mp"], "--", color="C1", linewidth=2,
                 label=r"$-\mathrm{Im}\,G_{\rm MP}(\omega+i0)/\pi$")
    axes[2].set(xlabel=r"$\omega$", title="spectral function on the real axis",
                ylim=(-0.05, 0.85))
    axes[2].legend(fontsize=10, frameon=False, loc="upper right")
    save(fig, "minipole_semicircle")


def n0_figure() -> None:
    scan = read_table(EXAMPLE, "semicircle_n0_scan")
    fig, axes = plt.subplots(1, len(scan["n0"]), figsize=(14, 4.2), sharey=True,
                             constrained_layout=True)
    for ax, n0, count, error in zip(axes, scan["n0"], scan["n_poles"], scan["max_error"]):
        poles = read_table(EXAMPLE, f"semicircle_poles_n0_{n0:.0f}")
        weight = np.hypot(poles["weight_re"], poles["weight_im"])
        upper = poles["pole_im"] > 0
        ax.axhspan(0, 0.5, color="#fde0dd", zorder=0)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.plot([-1, 1], [0, 0], color="black", linewidth=3, label="DOS support")
        ax.plot(poles["pole_re"][~upper], poles["pole_im"][~upper], "x", color="C0",
                markersize=10, markeredgewidth=2.5, label="pole, Im < 0")
        ax.plot(poles["pole_re"][upper], poles["pole_im"][upper], "o", color="#c00000",
                markersize=8, label="pole, Im > 0 (spurious)")
        if upper.any():
            low, high = weight[upper].min(), weight[upper].max()
            text = f"|A| = {low:.0e}" if upper.sum() == 1 else f"|A| = {low:.0e} to {high:.0e}"
            ax.text(0, 0.38, text, ha="center", fontsize=10, color="#c00000")
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.0, 0.5), xlabel=r"$\mathrm{Re}\,\xi$",
               title=rf"$n_0={n0:.0f}$: {count:.0f} poles, max error {error:.0e}")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\mathrm{Im}\,\xi$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, "minipole_n0")


def pair_figures() -> None:
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
    log_errors = np.log10(np.maximum(errors, 1e-16))
    # Sequential, one hue: light = small error (good), dark red = large error.
    norm = Normalize(vmin=np.floor(log_errors.min()), vmax=max(0.5, log_errors.max()))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    image = ax.imshow(log_errors, aspect="auto", cmap="Reds", norm=norm)
    ax.set_xticks(range(len(nmaxs)), [f"{n:g}" for n in nmaxs])
    ax.set_yticks(range(len(n0s)), [f"{n:g}" for n in n0s])
    for i in range(len(n0s)):
        for j in range(len(nmaxs)):
            dark = norm(log_errors[i, j]) > 0.6
            ax.text(j, i, f"{errors[i, j]:.1e}\n{counts[i, j]:.0f} poles",
                    ha="center", va="center", fontsize=10,
                    color="white" if dark else "black")
    for nmax, style in [(20.0, "--"), (50.0, "-")]:
        i, j = int(np.where(n0s == 5)[0][0]), int(np.where(nmaxs == nmax)[0][0])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, linewidth=2.5,
                                   linestyle=style, edgecolor="#08306b"))
    ax.set(xlabel=r"$n_{\max}$", ylabel=r"$n_0$",
           title="Same DLR coefficients, different contours (err $=10^{-8}$)")
    fig.colorbar(image, ax=ax,
                 label=r"$\log_{10}\,[\max_\nu |\delta\chi(i\nu)|/|\chi(0)|]$ (lighter is better)")
    fig.tight_layout()
    save(fig, "minipole_contours")


def main() -> None:
    plt.rcParams["font.size"] = 12
    semicircle_figure()
    n0_figure()
    pair_figures()


if __name__ == "__main__":
    main()
