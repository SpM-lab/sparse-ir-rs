"""Figures for the transformation page.

    uv run --project . transformation_plot.py

Reads what `cargo run --profile ci --bin transformation` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "transformation"


def main() -> None:
    plt.rcParams["font.size"] = 13

    poles = read_table(EXAMPLE, "pole_coefficients")
    coefficients = read_table(EXAMPLE, "smooth_coefficients")
    spectrum = read_table(EXAMPLE, "spectrum")
    gtau = read_table(EXAMPLE, "gtau")
    roundtrip = read_table(EXAMPLE, "roundtrip")
    narrow = read_table(EXAMPLE, "narrow_basis")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(poles["l"], np.abs(poles["rho_l"]), marker="o", ls="", label=r"$|\rho_l|$")
    ax.semilogy(poles["l"], np.abs(poles["g_l"]), marker="x", ls="", label=r"$|G_l|$")
    ax.semilogy(
        poles["l"], np.abs(poles["g_l_dlr"]), marker="+", ls="", label=r"$|G_l|$ from the DLR"
    )
    ax.set_xlabel(r"$l$")
    ax.set_ylim(1e-5, 1e1)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_pole_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(spectrum["omega"], spectrum["rho_exact"], label="exact")
    ax.plot(
        spectrum["omega"],
        spectrum["rho_reconstructed"],
        ls="--",
        label="from the expansion",
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_spectrum")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(coefficients["l"], np.abs(coefficients["rho_l"]), marker="o", ls="", label=r"$|\rho_l|$")
    ax.semilogy(coefficients["l"], np.abs(coefficients["g_l"]), marker="s", ls="", label=r"$|G_l|$")
    ax.semilogy(coefficients["l"], coefficients["s_l"], ls="--", label=r"$s_l$")
    ax.set_xlabel(r"$l$")
    ax.set_ylim(1e-5, 10)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_smooth_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(gtau["tau"], gtau["g_tau_direct"])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    save(fig, f"{EXAMPLE}_gtau")

    # Only even l: ρ is even in ω, so the odd coefficients vanish and their
    # "error" is the size of the numbers themselves.
    even = slice(None, None, 2)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(
        roundtrip["l"][even], np.abs(roundtrip["g_l"][even]), marker="x", ls="", label="exact"
    )
    ax.semilogy(
        roundtrip["l"][even],
        np.abs(roundtrip["g_l_reconstructed"][even]),
        marker="+",
        ls="",
        label="from $G(\\tau)$",
    )
    ax.semilogy(roundtrip["l"][even], roundtrip["error"][even], marker="p", ls="", label="error")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.set_ylim(1e-20, 1)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_roundtrip")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(narrow["l"], np.abs(narrow["g_l"]), marker="s", ls="", label=r"$|G_l|$")
    ax.semilogy(narrow["l"], narrow["s_l"], marker="x", ls="--", label=r"$s_l$")
    ax.set_xlabel(r"$l$")
    ax.set_ylim(1e-5, 10)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_narrow_basis")


if __name__ == "__main__":
    main()
