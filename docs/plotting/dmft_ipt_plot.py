"""Figures for the DMFT / IPT pages.

    uv run --project . dmft_ipt_plot.py

Reads what `cargo run --profile ci --bin dmft_ipt` and `--bin dmft_ipt_scan`
wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "dmft_ipt"
SCAN = "dmft_ipt_scan"
BETA = 20.0
# The interaction strengths the scan wrote a self-energy for, as indices into
# its 66-point grid.
PROBES = (50, 54, 57, 58, 60)


def main() -> None:
    plt.rcParams["font.size"] = 13

    green = read_table(EXAMPLE, "green")
    self_energy = read_table(EXAMPLE, "self_energy")
    convergence = read_table(EXAMPLE, "convergence")
    summary = read_table(EXAMPLE, "summary")

    # The solution the notebook's convergence criterion stops at.
    nu = green["n"] * np.pi / BETA
    positive = nu > 0
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].plot(nu[positive], green["g_im"][positive], marker=".")
    axes[0].set_ylabel(r"$\mathrm{Im}\,G(\mathrm{i}\nu)$")
    axes[1].plot(nu[positive], self_energy["sigma_im"][positive], marker=".")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\Sigma(\mathrm{i}\nu)$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\nu$")
    save(fig, f"{EXAMPLE}_solution")

    # The residual of the stopped run, the same loop left to run with the
    # symmetry enforced, and left to run without it.
    long_run = read_table(EXAMPLE, "long_convergence")
    free_run = read_table(EXAMPLE, "unconstrained_convergence")
    stop = int(summary["iterations"][0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(
        free_run["iteration"],
        np.maximum(free_run["residual"], 1e-17),
        color="C3",
        label="symmetry not enforced",
    )
    ax.semilogy(
        long_run["iteration"],
        np.maximum(long_run["residual"], 1e-17),
        color="0.5",
        label="symmetry enforced",
    )
    ax.semilogy(
        convergence["iteration"], convergence["residual"], marker=".", ls="none", ms=4,
        color="C0", label="stopped at the threshold",
    )
    ax.axhline(summary["sfc_tol"][0], ls="--", color="0.4")
    ax.annotate(
        f"threshold, after {stop} iterations",
        xy=(stop, summary["sfc_tol"][0]),
        xytext=(0.35, 0.75),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax.set_xlim(0, 1200)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\sum|\Delta\Sigma| / \sum|\Sigma|$")
    ax.legend(frameon=False, loc="center right", fontsize=10)
    ax = axes[1]
    ax.semilogy(
        free_run["iteration"],
        np.maximum(free_run["asymmetry"], 1e-17),
        color="C3",
        label="symmetry not enforced",
    )
    ax.set_xlim(0, 300)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\max_\nu|\Sigma - \Sigma_\mathrm{sym}|$")
    ax.set_title("distance from particle-hole symmetry", fontsize=12)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    save(fig, f"{EXAMPLE}_convergence")

    # The scan: the hysteresis, and the coexistence window it encloses.
    scan = read_table(SCAN, "renormalisation")
    u, metal, insulator = scan["u"], scan["metal"], scan["insulator"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    window = (u[np.argmax(insulator == 0.0)], u[np.argmax(metal == 0.0)])
    ax.axvspan(*window, color="0.9")
    ax.annotate(
        "coexistence",
        xy=(sum(window) / 2, 0.24),
        ha="center",
        color="0.35",
        rotation=90,
    )
    ax.plot(u, metal, marker=".", label="from the metal")
    ax.plot(u, insulator, marker=".", label="from the insulator")
    ax.plot(u, scan["from_g0"], ls="none", marker="o", mfc="none", ms=9, label=r"from $G^0$")
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$Z$")
    ax.legend(frameon=False)
    save(fig, f"{SCAN}_renormalisation")

    # The self-energies behind it, on either side of both transitions.
    sigma = read_table(SCAN, "self_energy")
    nu = sigma["n"] * np.pi / BETA
    positive = nu > 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(PROBES)))
    for index, color in zip(PROBES, colors):
        ax.plot(
            nu[positive],
            sigma[f"sigma_im_u{index}"][positive],
            marker=".",
            color=color,
            label=f"{u[index]:.1f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\mathrm{Im}\,\Sigma(\mathrm{i}\nu)$")
    ax.legend(frameon=False, title=r"$U$")
    save(fig, f"{SCAN}_self_energy")


if __name__ == "__main__":
    main()
