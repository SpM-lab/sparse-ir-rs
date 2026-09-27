"""Figures for the second-order perturbation page.

    uv run --project . second_order_perturbation_plot.py

Reads what `cargo run --profile ci --bin second_order_perturbation` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "second_order_perturbation"


def main() -> None:
    plt.rcParams["font.size"] = 13

    summary = read_table(EXAMPLE, "summary")
    matsubara = read_table(EXAMPLE, "green_matsubara")
    coefficients = read_table(EXAMPLE, "green_coefficients")
    green_tau = read_table(EXAMPLE, "green_tau")
    sigma_tau = read_table(EXAMPLE, "self_energy_tau")
    sigma_l = read_table(EXAMPLE, "self_energy_coefficients")
    sigma_iv = read_table(EXAMPLE, "self_energy_matsubara")
    sigma_far = read_table(EXAMPLE, "self_energy_far")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(matsubara["nu"], matsubara["g_gamma_im"], marker="x", label=r"$k=\Gamma$")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"Im $G(\mathrm{i}\nu)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_green_matsubara")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(coefficients["l"], coefficients["g_gamma_abs"], marker="x", ls="",
                label=r"$|G(l, \Gamma)|$")
    ax.semilogy(coefficients["l"], coefficients["s_l"], ls="--", label=r"$s_l$")
    ax.set_xlabel(r"$l$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_green_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(green_tau["tau"], green_tau["g_gamma"], marker="x", ls="", label=r"$k=\Gamma$")
    ax.plot(green_tau["tau"], green_tau["g_m"], marker="+", ls="", label=r"$k=M$")
    ax.plot(green_tau["tau"], green_tau["g_origin"], marker=".", ls="", label=r"$r=(0,0)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Re $G(\tau)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_green_tau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sigma_tau["tau"], sigma_tau["sigma_origin"], marker="x", ls="", label=r"$r=(0,0)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Re $\Sigma(\tau, r)$")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_self_energy_tau")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(sigma_l["l"], sigma_l["sigma_origin_abs"], marker="x", ls="",
                label=r"$|\Sigma(l, r=(0,0))|$")
    ax.semilogy(sigma_l["l"], sigma_l["sigma_gamma_abs"], marker="+", ls="",
                label=r"$|\Sigma(l, k=\Gamma)|$")
    ax.semilogy(sigma_l["l"], sigma_l["s_l"], ls="--", label=r"$s_l$")
    ax.set_xlabel(r"$l$")
    ax.set_ylim(1e-9, None)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_self_energy_coefficients")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sigma_far["nu"], sigma_far["sigma_gamma_im"], ls="-", label="evaluated everywhere")
    ax.plot(sigma_iv["nu"], sigma_iv["sigma_gamma_im"], marker="x", ls="",
            label="sampling frequencies")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"Im $\Sigma(\mathrm{i}\nu, \Gamma)$")
    # The sampling frequencies reach out to |ν| ≈ 880; the interesting
    # structure is where the evaluated curve is, so show that part.
    ax.set_xlim(sigma_far["nu"].min(), sigma_far["nu"].max())
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_self_energy_matsubara")

    print(f"U = {summary['u'][0]}, basis size {int(summary['basis_size'][0])}")


if __name__ == "__main__":
    main()
