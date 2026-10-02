"""Figures for the DLR page.

    uv run --project . dlr_plot.py

Reads what `cargo run --profile ci --bin dlr` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "dlr"


def main() -> None:
    plt.rcParams["font.size"] = 13

    coefficients = read_table(EXAMPLE, "coefficients")
    dlr = read_table(EXAMPLE, "dlr_coefficients")
    reconstruction = read_table(EXAMPLE, "reconstruction")
    matsubara = read_table(EXAMPLE, "matsubara")
    independent = read_table(EXAMPLE, "independent_matsubara")
    nodes = read_table(EXAMPLE, "independent_nodes")

    # Only even l: the semicircle is even in ω, so odd coefficients vanish.
    even = slice(None, None, 2)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(coefficients["l"][even], np.abs(coefficients["g_l"][even]), marker="x", ls="")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.set_ylim(1e-5, None)
    save(fig, f"{EXAMPLE}_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(dlr["pole"], dlr["c_p"], marker="x", ls="")
    ax.set_xlabel(r"$\omega_p$")
    ax.set_ylabel(r"$c_p$")
    save(fig, f"{EXAMPLE}_poles")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(reconstruction["l"], np.abs(reconstruction["g_l"]), marker="+", ls="", label="exact")
    ax.semilogy(
        reconstruction["l"],
        np.abs(reconstruction["g_l_from_dlr"]),
        marker="x",
        ls="",
        label="through the DLR",
    )
    ax.semilogy(reconstruction["l"], reconstruction["error"], label="error")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.set_ylim(1e-18, None)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_reconstruction")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(matsubara["nu"], matsubara["g_iv_exact_im"], marker="x", ls="", label="from the basis")
    ax.plot(matsubara["nu"], matsubara["g_iv_dlr_im"], marker="+", ls="", label="from the poles")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"Im $G(\mathrm{i}\nu)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_matsubara")

    # The independent DLR: error of G(iν) against the closed form, far beyond
    # the nodes it was fitted at (positive nodes marked along the bottom).
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(
        independent["nu"],
        independent["error"],
        marker="x",
        ls="",
        label="fitted at the Matsubara nodes",
    )
    ax.loglog(
        independent["nu"],
        independent["error_from_tau"],
        marker="+",
        ls="",
        label=r"refitted from $G(\tau)$ at the $\tau$ nodes",
    )
    positive = nodes["nu"][nodes["nu"] > 0]
    ax.plot(
        positive, np.full_like(positive, 1e-17), marker="|", ls="", color="k", label="Matsubara nodes"
    )
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$|G_\mathrm{DLR}(\mathrm{i}\nu) - G(\mathrm{i}\nu)|$")
    ax.set_ylim(3e-18, 1e-10)
    ax.legend(frameon=False, loc="upper right")
    save(fig, f"{EXAMPLE}_independent")


if __name__ == "__main__":
    main()
