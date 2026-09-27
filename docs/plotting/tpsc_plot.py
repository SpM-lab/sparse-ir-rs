"""Figures for the TPSC page.

    uv run --project . tpsc_plot.py

Reads what `cargo run --profile ci --bin tpsc` and `--bin tpsc_scan` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "tpsc"
SCAN = "tpsc_scan"
# Where Γ, X, M and Γ fall on the high-symmetry path of a 24 × 24 grid.
TICKS = (0, 12, 24, 36)
TICK_LABELS = (r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$")


def zone(table, column, nk_lin):
    """One column of the momentum table as an `nk × nk` array."""
    return table[column].reshape(nk_lin, nk_lin)


def main() -> None:
    plt.rcParams["font.size"] = 13

    summary = read_table(EXAMPLE, "summary")
    nk_lin = int(round(summary["nk_lin"][0]))
    momentum = read_table(EXAMPLE, "momentum")
    kx = zone(momentum, "kx", nk_lin)
    ky = zone(momentum, "ky", nk_lin)

    # The three quantities the notebook maps over the zone.
    panels = (
        ("g_re", r"$\mathrm{Re}\,G(k, \mathrm{i}\pi T)$"),
        ("sigma_im", r"$\mathrm{Im}\,\Sigma(k, \mathrm{i}\pi T)$"),
        ("chi_spin", r"$\chi_{\mathrm{sp}}(q, 0)$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (column, title) in zip(axes, panels):
        mesh = ax.pcolormesh(kx, ky, zone(momentum, column, nk_lin), shading="auto")
        ax.set_xlabel(r"$k_x/\pi$")
        ax.set_ylabel(r"$k_y/\pi$")
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax)
    save(fig, f"{EXAMPLE}_zone")

    # The same susceptibilities along Γ → X → M → Γ, where the enhancement of
    # the spin channel and the suppression of the charge one are easier to read
    # off than from a colour map.
    path = read_table(EXAMPLE, "path")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(path["distance"], path["chi_spin"], label=r"$\chi_{\mathrm{sp}}$")
    ax.plot(path["distance"], path["chi_0"], label=r"$\chi^0$", linestyle="--")
    ax.plot(path["distance"], path["chi_charge"], label=r"$\chi_{\mathrm{ch}}$")
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(path["distance"][0], path["distance"][-1])
    ax.set_ylabel("susceptibility")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_path")

    # The vertices against the bare interaction: Fig. 2 of Vilk (1997).
    scan_summary = read_table(SCAN, "summary")
    vertices = read_table(SCAN, "vertices")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(vertices["u"], vertices["u_ch"], label=r"$U_{\mathrm{ch}}$")
    ax.plot(vertices["u"], vertices["u_sp"], label=r"$U_{\mathrm{sp}}$")
    ax.plot(vertices["u"], vertices["u"], color="0.6", linewidth=1, label=r"$U$")
    ax.axhline(
        vertices["u_crit"][0],
        color="0.4",
        linestyle=":",
        linewidth=1,
    )
    ax.annotate(
        r"$U_{\mathrm{crit}} = 1/\max\chi^0$",
        (0.1, vertices["u_crit"][0]),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=11,
        color="0.3",
    )
    ax.set_xlabel(r"$U$")
    ax.set_xlim(0, vertices["u"][-1])
    ax.set_ylabel("effective interaction")
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, loc="upper left")
    save(fig, f"{SCAN}_vertices")

    # And the spin susceptibility along the path at three points of that scan,
    # at half filling, where the peak sits exactly on M.
    chi = read_table(SCAN, "chi_spin")
    probes = [name for name in chi if name.startswith("u_")]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(probes)))
    for name, colour in zip(probes, colours):
        u = scan_summary[f"probe_{name}"][0]
        ax.plot(chi["distance"], chi[name], color=colour, label=rf"$U = {u:.2f}$")
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(chi["distance"][0], chi["distance"][-1])
    ax.set_ylabel(r"$\chi_{\mathrm{sp}}(q, 0)$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    save(fig, f"{SCAN}_chi_spin")


if __name__ == "__main__":
    main()
