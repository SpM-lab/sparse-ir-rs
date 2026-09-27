"""Figures for the Eliashberg page.

    uv run --project . eliashberg_holstein_plot.py

Reads what `cargo run --profile ci --bin eliashberg_holstein` and
`--bin eliashberg_holstein_scan` wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "eliashberg_holstein"
SCAN = "eliashberg_holstein_scan"


def main() -> None:
    plt.rcParams["font.size"] = 13

    summary = read_table(EXAMPLE, "summary")
    omega0 = summary["omega0"][0]
    matsubara_f = read_table(EXAMPLE, "matsubara_f")
    matsubara_b = read_table(EXAMPLE, "matsubara_b")

    # The converged solution on the fermionic frequencies. The gap is even and
    # changes sign above the phonon frequency; the self-energy is odd. They
    # differ by two orders of magnitude, so each gets its own panel.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    nu = matsubara_f["nu"]
    panels = (
        ("delta_re", r"$\Delta(\mathrm{i}\nu)$"),
        ("sigma_im", r"$\mathrm{Im}\,\Sigma(\mathrm{i}\nu)$"),
        ("g_im", r"$\mathrm{Im}\,G(\mathrm{i}\nu)$"),
    )
    for ax, (column, label) in zip(axes, panels):
        ax.plot(nu, matsubara_f[column], "o-", ms=3)
        ax.axhline(0.0, color="0.7", lw=0.8)
        for sign in (-1, 1):
            ax.axvline(sign * omega0, color="C3", ls=":", lw=1.2)
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel(label)
        ax.set_xlim(-8 * omega0, 8 * omega0)
    axes[0].axvline(omega0, color="C3", ls=":", lw=1.2, label=r"$\omega_0$")
    axes[0].legend(frameon=False)
    save(fig, f"{EXAMPLE}_matsubara")

    # The phonon the electrons have dressed, against the bare propagator.
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    nu_b = matsubara_b["nu"]
    bare = 2 * omega0 / (-(nu_b**2) - omega0**2)
    ax.plot(nu_b, matsubara_b["d_re"], "o-", ms=3, label=r"$D(\mathrm{i}\nu)$")
    ax.plot(nu_b, bare, "--", label=r"$D_0(\mathrm{i}\nu)$")
    ax.set_xlabel(r"$\nu$")
    ax.set_xlim(-8 * omega0, 8 * omega0)
    ax.set_ylabel("phonon propagator")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_phonon")

    # The compactness claim, checked on the anomalous Green's function: its
    # coefficients fall with the singular values. Only the odd `l` carry
    # weight, because `F` is even in the frequency.
    basis = read_table(EXAMPLE, "basis")
    l = basis["l"]
    f_l = basis["f_l"]
    odd = l.astype(int) % 2 == 1
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.semilogy(l[odd], f_l[odd] / f_l[odd].max(), "o-", ms=4, label=r"$|F_l| / |F_1|$")
    ax.semilogy(l, basis["s_ratio"], "--", label=r"$s_l / s_0$")
    ax.set_xlabel(r"$l$")
    ax.set_ylim(1e-9, 5.0)
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_basis")

    # The jump the whole scan exists for.
    scan = read_table(SCAN, "specific_heat")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(scan["temperature"], scan["c_over_t"], "o-", ms=5)
    ax.set_xlabel(r"$T$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_ylabel(r"$C / T$")
    ax.set_ylim(bottom=0.0)
    save(fig, f"{SCAN}_specific_heat")


if __name__ == "__main__":
    main()
