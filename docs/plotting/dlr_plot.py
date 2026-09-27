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
    ax.set_xlabel(r"$\bar{\omega}_p$")
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


if __name__ == "__main__":
    main()
