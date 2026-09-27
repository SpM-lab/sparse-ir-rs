"""Figures for the sparse-modeling page.

    uv run --project . spm_plot.py

Reads what `cargo run --profile ci --bin spm` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "spm"


def main() -> None:
    plt.rcParams["font.size"] = 13

    gtau = read_table(EXAMPLE, "gtau")
    scan = read_table(EXAMPLE, "lambda_scan")
    spectrum = read_table(EXAMPLE, "spectrum")
    coefficients = read_table(EXAMPLE, "coefficients")
    chosen = read_table(EXAMPLE, "summary")["lambda"][0]

    # The data. The noise is invisible on the curve and decisive for the answer.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(gtau["tau"], -gtau["g_tau_input"], lw=1, label="noisy data")
    ax.plot(gtau["tau"], -gtau["g_tau_clean"], lw=1, ls="--", label="exact")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$-G(\tau)$")
    ax.set_yscale("log")
    ax.legend()
    save(fig, f"{EXAMPLE}_gtau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(gtau["tau"], gtau["g_tau_input"] - gtau["g_tau_clean"], lw=0.8, label="noise")
    ax.plot(gtau["tau"], gtau["g_tau_input"] - gtau["g_tau_fit"], lw=0.8, label="data − fit")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta G(\tau)$")
    ax.legend()
    save(fig, f"{EXAMPLE}_residual")

    # Which coefficients survive the L1 penalty.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(
        coefficients["l"], np.abs(coefficients["rho_l_exact"]), marker="+", ls="", label="exact"
    )
    kept = coefficients["rho_l_recovered"] != 0.0
    ax.semilogy(
        coefficients["l"][kept],
        np.abs(coefficients["rho_l_recovered"][kept]),
        marker="x",
        ls="",
        label=rf"recovered, $\lambda = {chosen:g}$",
    )
    ax.semilogy(coefficients["l"], coefficients["s_l"], lw=1, ls=":", label=r"$s_l$")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|\rho_l|$")
    ax.set_ylim(1e-12, None)
    ax.legend()
    save(fig, f"{EXAMPLE}_coefficients")

    # The answer.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(spectrum["omega"], spectrum["rho_exact"], lw=1.5, ls="--", label="exact")
    ax.plot(
        spectrum["omega"], spectrum["rho_recovered"], lw=1.5, label=rf"$\lambda = {chosen:g}$"
    )
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.legend()
    save(fig, f"{EXAMPLE}_spectrum")

    # How the choice of λ was made.
    # Without the exact answer, the noise level is what tells you when λ has
    # started throwing away signal: a fit that is worse than the noise is
    # explaining less than the data contains.
    noise = np.std(gtau["g_tau_input"] - gtau["g_tau_clean"]) * np.sqrt(len(gtau["tau"]))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(scan["lambda"], scan["residual"], marker="o", label=r"$\|y - Ax\|$")
    ax.axhline(noise, color="0.7", lw=0.8, ls="--", label="noise level")
    ax.loglog(scan["lambda"], scan["l2_error"], marker="s", label=r"$\|\rho - \rho_\mathrm{exact}\|$")
    ax.axvline(chosen, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda$")
    ax.legend()
    save(fig, f"{EXAMPLE}_lambda_scan")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogx(scan["lambda"], scan["nonzero"], marker="o")
    ax.axvline(chosen, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("nonzero coefficients")
    save(fig, f"{EXAMPLE}_sparsity")


if __name__ == "__main__":
    main()
