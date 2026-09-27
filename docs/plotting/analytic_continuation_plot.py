"""Figures for the analytic-continuation page.

    uv run --project . analytic_continuation_plot.py

Reads what `cargo run --profile ci --bin analytic_continuation` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "analytic_continuation"


def main() -> None:
    plt.rcParams["font.size"] = 13

    summary = read_table(EXAMPLE, "summary")
    size = int(summary["basis_size"][0])
    half = size // 2
    eta = summary["eta"][0]
    alpha = summary["alpha"][0]

    coefficients = read_table(EXAMPLE, "coefficients")
    tsvd = read_table(EXAMPLE, "tsvd")
    ridge = read_table(EXAMPLE, "ridge")
    discrete = read_table(EXAMPLE, "discrete")
    lorentz = read_table(EXAMPLE, "lorentz")
    kernel = read_table(EXAMPLE, "lorentz_kernel")

    # How fast the singular values fall: the size of the problem in one line.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(coefficients["l"], coefficients["s_ratio"], marker="+", ls="")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$s_l / s_0$")
    save(fig, f"{EXAMPLE}_singular_values")

    # The two models.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(tsvd["omega"], tsvd["semielliptic_exact"], lw=1.2, label="semi-elliptic")
    ax.plot(tsvd["omega"], tsvd["insulating_exact"], lw=1.2, label="insulating")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.legend()
    save(fig, f"{EXAMPLE}_models")

    # Their coefficients, before and after the noise.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, label in (("semielliptic", "semi-elliptic"), ("insulating", "insulating")):
        line = ax.semilogy(
            coefficients["l"],
            np.abs(coefficients[f"g_{name}"]),
            marker="+",
            ls="",
            label=label,
        )
        ax.semilogy(
            coefficients["l"],
            np.abs(coefficients[f"g_{name}_noisy"]),
            marker="x",
            ls="",
            color=line[0].get_color(),
            alpha=0.5,
            label=f"{label}, noisy",
        )
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|G_l|$")
    ax.legend(fontsize=10)
    save(fig, f"{EXAMPLE}_noisy_coefficients")

    # Truncated SVD: half the coefficients, then all of them.
    for name, label in (("semielliptic", "semi-elliptic"), ("insulating", "insulating")):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(tsvd["omega"], tsvd[f"{name}_exact"], lw=1.5, color="k", label="exact")
        ax.plot(tsvd["omega"], tsvd[f"{name}_half"], lw=1, label=rf"$L' = {half}$")
        ax.plot(tsvd["omega"], tsvd[f"{name}_full"], lw=1, label=rf"$L' = {size}$")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\rho(\omega)$")
        ax.legend()
        save(fig, f"{EXAMPLE}_tsvd_{name}")

    # Ridge regression, on the same axes as the exact answer.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, label in (("semielliptic", "semi-elliptic"), ("insulating", "insulating")):
        line = ax.plot(tsvd["omega"], tsvd[f"{name}_exact"], lw=1.5, label=label)
        ax.plot(
            ridge["omega"],
            ridge[f"{name}_ridge"],
            lw=1,
            ls="--",
            color=line[0].get_color(),
            label=f"{label}, ridge",
        )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.set_title(rf"$\alpha = {alpha:.2e}$", fontsize=11)
    ax.legend(fontsize=10)
    save(fig, f"{EXAMPLE}_ridge")

    # Why ρ_l is the wrong unknown: G_l decays, ρ_l does not.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(
        discrete["l"], np.abs(discrete["g_discrete"]), marker="+", ls="", label=r"$|G_l|$"
    )
    ax.semilogy(
        discrete["l"], np.abs(discrete["rho_discrete"]), marker="x", ls="", label=r"$|\rho_l|$"
    )
    ax.set_xlabel(r"$l$")
    ax.set_ylabel("four delta peaks")
    ax.legend()
    save(fig, f"{EXAMPLE}_discrete")

    # The real-axis basis function and the kernel it induces.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(lorentz["omega"], lorentz["f"], lw=1.2)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$f(\omega)$")
    ax.set_title(rf"$\eta = {eta:.4f}$", fontsize=11)
    save(fig, f"{EXAMPLE}_lorentz")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    columns = [name for name in kernel if name.startswith("k_")]
    matrix = np.array([kernel[name] for name in columns])  # (m, l)
    image = ax.imshow(
        np.log10(np.abs(matrix.T) + 1e-20),
        aspect="auto",
        origin="lower",
        extent=(-0.5, len(columns) - 0.5, -0.5, len(kernel["l"]) - 0.5),
    )
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$l$")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}|K_{lm}|$")
    save(fig, f"{EXAMPLE}_lorentz_kernel")


if __name__ == "__main__":
    main()
