"""Figures for the sparse sampling page.

    uv run --project . sparse_sampling_demo_plot.py

Reads what `cargo run --profile ci --bin sparse_sampling_demo` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "sparse_sampling_demo"


def main() -> None:
    plt.rcParams["font.size"] = 13

    coefficients = read_table(EXAMPLE, "coefficients")
    tau_sampling = read_table(EXAMPLE, "tau_sampling")
    matsubara = read_table(EXAMPLE, "matsubara_sampling")
    reconstruction = read_table(EXAMPLE, "reconstruction")

    # Only even l is plotted: the semicircle is even in ω, so every odd
    # coefficient vanishes and would only show up as rounding noise.
    even = slice(None, None, 2)
    ls = coefficients["l"][even]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(ls, np.abs(coefficients["g_l"][even]), marker="x", ls="")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.set_ylim(1e-15, 10)
    save(fig, f"{EXAMPLE}_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(tau_sampling["tau"], tau_sampling["g_tau"], marker="x", ls="")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    save(fig, f"{EXAMPLE}_gtau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(matsubara["nu"], matsubara["g_iv_im"], marker="x", ls="")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"Im $G(\mathrm{i}\nu)$")
    save(fig, f"{EXAMPLE}_giv")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(ls, np.abs(reconstruction["g_l"][even]), ls="-", label="exact")
    ax.semilogy(
        ls, np.abs(reconstruction["g_l_from_tau"][even]), marker="x", ls="", label="from sampling times"
    )
    ax.semilogy(
        ls,
        np.abs(reconstruction["g_l_from_matsubara"][even]),
        marker="+",
        ls="",
        label="from sampling frequencies",
    )
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.set_ylim(1e-17, 10)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_comparison")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(ls, reconstruction["error_tau"][even], marker="x", ls="", label="from sampling times")
    ax.semilogy(
        ls, reconstruction["error_matsubara"][even], marker="+", ls="", label="from sampling frequencies"
    )
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"error in $G_l$")
    ax.set_ylim(1e-18, 10)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_errors")


if __name__ == "__main__":
    main()
