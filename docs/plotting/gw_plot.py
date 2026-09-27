"""Figures for the GW page.

    uv run --project . gw_plot.py

Reads what `cargo run --profile ci --bin gw` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "gw"


def main() -> None:
    plt.rcParams["font.size"] = 13

    green_tau = read_table(EXAMPLE, "green_tau")
    polarization_tau = read_table(EXAMPLE, "polarization_tau")
    polarization_l = read_table(EXAMPLE, "polarization_coefficients")
    screened_tau = read_table(EXAMPLE, "screened_tau")
    screened_iv = read_table(EXAMPLE, "screened_matsubara")
    sigma_tau = read_table(EXAMPLE, "self_energy_tau")
    sigma_l = read_table(EXAMPLE, "self_energy_coefficients")
    green_initial = read_table(EXAMPLE, "green_initial")
    green_final = read_table(EXAMPLE, "green_final")
    sigma_final = read_table(EXAMPLE, "self_energy_final")
    convergence = read_table(EXAMPLE, "convergence")

    # G on the sampling times, with the jump at τ = 0 that the whole
    # construction has to carry. The particle-hole symmetric atom has
    # G(β − τ) = G(τ), so plotting the reversal on top of it would show
    # nothing; that it agrees is left to the verification test, where a
    # reversal without its fermionic sign fails loudly.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(green_tau["tau_f"], green_tau["g_re"], marker="x", ls="")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    save(fig, f"{EXAMPLE}_green_tau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(polarization_tau["tau_b"], polarization_tau["p_re"], marker="x", ls="")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$P(\tau) = G(\tau)\,G(\beta - \tau)$")
    save(fig, f"{EXAMPLE}_polarization_tau")

    # The two expansions side by side: both fall off like the singular values,
    # which is the whole reason the loop can be run in the basis at all.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(polarization_l["l"], polarization_l["p_l_abs"], marker="x", ls="",
                label=r"$|P_l|$ (bosonic)")
    ax.semilogy(sigma_l["l"], sigma_l["e_l_abs"], marker="+", ls="",
                label=r"$|\Sigma_l|$ (fermionic)")
    ax.set_xlabel(r"$l$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_coefficients")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(screened_iv["n"], screened_iv["w_re"], marker="x", ls="")
    axes[0].set_xlabel(r"$n$")
    axes[0].set_ylabel(r"$W(\mathrm{i}\omega_n)$")
    axes[1].plot(screened_tau["tau_f"], screened_tau["w_re"], marker="+", ls="")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$W(\tau_F)$")
    fig.tight_layout()
    save(fig, f"{EXAMPLE}_screened")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sigma_tau["tau_f"], sigma_tau["e_re"], marker="x", ls="")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Sigma(\tau) = G(\tau)\,W(\tau)$")
    save(fig, f"{EXAMPLE}_self_energy_tau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sigma_final["n"], sigma_final["e_re"], marker="x", ls="",
            label=r"Re $\Sigma$")
    ax.plot(sigma_final["n"], sigma_final["e_im"], marker="+", ls="",
            label=r"Im $\Sigma$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\Sigma(\mathrm{i}\nu_n)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_self_energy_matsubara")

    # What the loop did to G: the interacting propagator against the atomic one
    # it started from.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(green_initial["n"], green_initial["g_im"], marker="x", ls="",
            label=r"Im $G_0$")
    ax.plot(green_final["n"], green_final["g_im"], marker="+", ls="",
            label=r"Im $G$")
    ax.set_xlabel(r"$n$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_green_matsubara")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(convergence["iteration"], np.maximum(convergence["difference"], 1e-18),
                marker="x")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\max |G_{n} - G_{n-1}|$")
    save(fig, f"{EXAMPLE}_convergence")


if __name__ == "__main__":
    main()
