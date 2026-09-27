"""Figures for the FLEX page.

    uv run --project . flex_plot.py

Reads what `cargo run --profile ci --bin flex` and `--bin flex_scan` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "flex"
SCAN = "flex_scan"
TICK_LABELS = (r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$")


def ticks(distance):
    """Where Γ, X, M and Γ fall on a high-symmetry path of `nk_lin` steps."""
    last = distance[-1]
    return (0, last / 3, 2 * last / 3, last)


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

    # The self-consistent solution over the zone, and the gap function the
    # linearised Eliashberg equation returns from it.
    panels = (
        ("sigma_im", r"$\mathrm{Im}\,\Sigma(k, \mathrm{i}\pi T)$", "viridis"),
        ("chi_spin", r"$\chi_{\mathrm{sp}}(q, 0)$", "viridis"),
        ("delta_re", r"$\Delta(k, \mathrm{i}\pi T)$", "RdBu_r"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (column, title, cmap) in zip(axes, panels):
        values = zone(momentum, column, nk_lin)
        # The gap changes sign, so centre its colour scale on zero.
        limit = np.abs(values).max()
        bounds = {"vmin": -limit, "vmax": limit} if cmap == "RdBu_r" else {}
        mesh = ax.pcolormesh(kx, ky, values, shading="auto", cmap=cmap, **bounds)
        ax.set_xlabel(r"$k_x/\pi$")
        ax.set_ylabel(r"$k_y/\pi$")
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax)
    save(fig, f"{EXAMPLE}_zone")

    # The susceptibilities along Γ → X → M → Γ. The dressing of χ⁰ into χ_sp
    # is what makes the pairing interaction peak at the antiferromagnetic
    # wave vector.
    path = read_table(EXAMPLE, "path")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    ax.plot(path["distance"], path["chi_spin"], label=r"$\chi_{\mathrm{sp}}$")
    ax.plot(path["distance"], path["chi_0"], label=r"$\chi^0$", linestyle="--")
    ax.plot(path["distance"], path["chi_charge"], label=r"$\chi_{\mathrm{ch}}$")
    ax.set_xticks(ticks(path["distance"]))
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(path["distance"][0], path["distance"][-1])
    ax.set_ylabel("susceptibility")
    ax.legend(frameon=False)

    # And the frequency dependence at the antinode, where both the self-energy
    # and the gap are largest.
    self_energy = read_table(EXAMPLE, "self_energy")
    ax = axes[1]
    ax.plot(
        self_energy["nu"],
        self_energy["sigma_im"],
        ".-",
        color="C0",
        label=r"$\mathrm{Im}\,\Sigma$",
    )
    ax.axhline(0.0, color="0.6", linewidth=1)
    ax.set_xlabel(r"$\nu$")
    ax.set_xlim(-30, 30)
    ax.set_ylabel(r"$\mathrm{Im}\,\Sigma(k, \mathrm{i}\nu)$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title(r"at the antinode $(\pi, 0)$")
    # The gap is an eigenvector, so only its shape means anything; it lives on
    # its own axis for that reason.
    twin = ax.twinx()
    twin.plot(self_energy["nu"], self_energy["delta_re"], ".-", color="C3")
    twin.set_ylabel(r"$\Delta(k, \mathrm{i}\nu)$ (arbitrary scale)", color="C3")
    twin.tick_params(axis="y", labelcolor="C3")
    save(fig, f"{EXAMPLE}_path")

    # The figure the notebook ends on: cooling drives 1/χ_sp towards zero and
    # the leading eigenvalue towards one.
    temperature = read_table(SCAN, "temperature")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(
        temperature["temperature"],
        temperature["inverse_chi_spin_max"],
        "o-",
        color="C0",
        label=r"$1/\chi_{\mathrm{sp}}^{\max}$",
    )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$1/\chi_{\mathrm{sp}}^{\max}$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(0, None)
    twin = ax.twinx()
    twin.plot(
        temperature["temperature"],
        temperature["lambda_d"],
        "s--",
        color="C3",
        label=r"$\lambda_d$",
    )
    twin.axhline(1.0, color="0.6", linewidth=1, linestyle=":")
    twin.set_ylabel(r"$\lambda_d$", color="C3")
    twin.tick_params(axis="y", labelcolor="C3")
    twin.set_ylim(0, 1.05)
    save(fig, f"{SCAN}_lambda")

    # The spin susceptibility along the path at the coldest probe of the scan,
    # on the finer grid the sweep uses.
    scan_summary = read_table(SCAN, "summary")
    chi = read_table(SCAN, "chi_spin")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    probe = scan_summary["probe_temperature"][0]
    ax.plot(chi["distance"], chi["chi_spin"], label=rf"$T = {probe:g}$")
    ax.set_xticks(ticks(chi["distance"]))
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(chi["distance"][0], chi["distance"][-1])
    ax.set_ylabel(r"$\chi_{\mathrm{sp}}(q, 0)$")
    ax.legend(frameon=False)
    save(fig, f"{SCAN}_chi_spin")


if __name__ == "__main__":
    main()
