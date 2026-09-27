"""Figures for the orbital magnetic susceptibility page.

    uv run --project . orbital_magnetic_susceptibility_plot.py

Reads what `cargo run --profile ci --bin orbital_magnetic_susceptibility`
wrote.
"""

from __future__ import annotations

import numpy as np

from common import plt, read_table, save

EXAMPLE = "orbital_magnetic_susceptibility"


def main() -> None:
    plt.rcParams["font.size"] = 13

    # The square lattice against its zero-temperature closed form. The two
    # part company at the van Hove filling, where the closed form diverges and
    # T = 0.1 does not, and at the band edge, where the Fermi function is what
    # rounds the step off.
    square = read_table(EXAMPLE, "square")
    analytic = read_table(EXAMPLE, "square_analytic")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(square["mu"], square["chi"], marker=".", label=r"IR ($T = 0.1$)")
    ax.plot(analytic["mu"], analytic["chi"], ls="--", label=r"closed form ($T = 0$)")
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title("square lattice")
    ax.legend(frameon=False)
    save(fig, f"{EXAMPLE}_square")

    # Graphene: diamagnetic at the Dirac point, paramagnetic on either side,
    # and flat zero outside the band.
    graphene = read_table(EXAMPLE, "graphene")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(graphene["mu"], graphene["chi"], marker=".", color="C2")
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title("graphene")
    save(fig, f"{EXAMPLE}_graphene")

    # What the basis is summing: the summand falls off as a power of nu, so a
    # naive sum would need a very long tail. Thirty sampling frequencies carry
    # the whole of it. The square lattice goes as nu^-4 and graphene as nu^-6;
    # graphene's velocity matrices are purely off-diagonal, which kills the
    # terms a general two-band model would have.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lattice, color, power in (("square", "C0", 4.0), ("graphene", "C2", 6.0)):
        table = read_table(EXAMPLE, f"{lattice}_matsubara")
        nu = np.abs(table["wn"])
        magnitude = np.hypot(table["chi_re"], table["chi_im"])
        ax.loglog(nu, magnitude, marker="x", ls="none", color=color, label=lattice)
        far = np.argmax(nu)
        guide = np.array([3.0, nu[far]])
        ax.loglog(
            guide,
            magnitude[far] * (guide / nu[far]) ** -power,
            ls="--",
            color=color,
            alpha=0.5,
            label=rf"$\propto \nu^{{-{power:.0f}}}$",
        )
    ax.set_xlabel(r"$|2n + 1|$")
    ax.set_ylabel(r"$|\chi(\mathrm{i}\nu)|$ at $\mu = -1$")
    ax.legend(frameon=False, ncols=2)
    save(fig, f"{EXAMPLE}_summand")


if __name__ == "__main__":
    main()
