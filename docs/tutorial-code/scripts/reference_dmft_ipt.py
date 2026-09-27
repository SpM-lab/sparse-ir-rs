"""Reference values for the `dmft_ipt` and `dmft_ipt_scan` examples.

This is the computation of `DMFT_IPT_py.ipynb` of sparse-ir-tutorial, with
the plotting removed and the numbers written out. Dynamical mean-field theory
on the Bethe lattice, with the impurity problem solved by iterated
perturbation theory: the self-energy is `U² 𝒢(τ)³`, so every iteration is one
round trip through the basis and the whole solver is four lines.

`write` produces the single `U = 5` calculation at the notebook's convergence
threshold. `write_scan` produces the renormalisation factor `Z(U)` from three
different starting points, converged far harder, which is slow enough to be a
separate example — and which disagrees with the single calculation about
whether `U = 5` is a metal.
"""

from __future__ import annotations

import numpy as np
import sparse_ir

from reference_common import fold_tau, provenance, write_table

EXAMPLE = "dmft_ipt"
SCAN = "dmft_ipt_scan"

# Half bandwidth; the hopping is t = D/2 = 1.
D = 2.0
WMAX = 2 * D
TEMPERATURE = 0.1 / D
BETA = 1.0 / TEMPERATURE
U = 5.0
EPS = 1e-15
MAXITER = 300
SFC_TOL = 1e-5
MIX = 0.25
# The same calculation carried on for a fixed number of iterations instead of
# stopped at a threshold, which is what shows the threshold up.
LONG_ITERATIONS = 5000
# The scan: 66 interaction strengths, each run for a fixed number of
# iterations rather than to a threshold on the change of `Σ`. Near the
# transition the loop spends hundreds of iterations next to the unstable fixed
# point that separates the metal from the insulator, and the change of `Σ`
# there is tiny while the distance still to go is not — so a threshold stops
# on the way past and reports a metal that is not one. `dmft_ipt` shows the
# same thing happening at `U = 5`.
U_MAX = 6.5
U_NUM = 66
SCAN_ITERATIONS = 5000
# The interaction strengths whose self-energy the scan writes out, as indices
# into `U_arr`: on either side of `U_c1 = 3.2` and of `U_c2`, which lies
# between 3.4 and 3.5.
SCAN_PROBES = (30, 32, 34, 35, 40)


def density_of_states(omega: np.ndarray) -> np.ndarray:
    """The semicircle of half bandwidth `D`, normalised to one."""
    return 2 * np.sqrt(D**2 - np.clip(omega, -D, D) ** 2) / (np.pi * D**2)


class Mesh:
    """The basis and its two sampling grids, with the round trip between."""

    def __init__(self, eps: float = EPS):
        self.basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=eps)
        self.tau = sparse_ir.TauSampling(self.basis)
        self.wn = sparse_ir.MatsubaraSampling(self.basis)
        self.iwn = 1j * self.wn.sampling_points * np.pi * TEMPERATURE
        # The index of the lowest positive Matsubara frequency, n = 1.
        self.iw0 = int(np.where(self.wn.sampling_points == 1)[0][0])

    def wn_to_tau(self, values: np.ndarray) -> np.ndarray:
        return self.tau.evaluate(self.wn.fit(values))

    def tau_to_wn(self, values: np.ndarray) -> np.ndarray:
        return self.wn.evaluate(self.tau.fit(values))

    def noninteracting(self) -> np.ndarray:
        """`G⁰(iν)` of the semicircle, through its spectral representation."""
        rho_l = self.basis.v.overlap(density_of_states)
        return self.wn.evaluate(-self.basis.s * rho_l)


def solve(
    mesh: Mesh, g_loc: np.ndarray, u: float, maxiter: int, sfc_tol: float
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """The DMFT loop. Returns the converged `G_loc`, `Σ`, and the residuals.

    Every iteration is the impurity solver `Σ(τ) = U² 𝒢(τ)³` — one fit and one
    evaluation each way — followed by the Dyson equation and the Bethe-lattice
    self-consistency `𝒢⁻¹ = iν − t² G_loc`.
    """
    t = D / 2
    sigma = np.zeros_like(g_loc)
    g_weiss = 1 / (mesh.iwn - t**2 * g_loc)
    residuals: list[float] = []

    for _ in range(maxiter):
        previous = sigma
        new = mesh.tau_to_wn(u**2 * mesh.wn_to_tau(g_weiss) ** 3)
        sigma = new * MIX + sigma * (1 - MIX)

        g_loc = 1 / (1 / g_weiss - sigma)
        g_weiss = 1 / (mesh.iwn - t**2 * g_loc)

        # At `U = 0` the self-energy stays identically zero, and the relative
        # change of nothing is not a number; that case is converged.
        scale = np.sum(abs(sigma))
        residuals.append(np.sum(abs(sigma - previous)) / scale if scale > 0 else 0.0)
        if residuals[-1] < sfc_tol:
            break
    return g_loc, sigma, residuals


def renormalisation(mesh: Mesh, sigma: np.ndarray) -> float:
    """`Z` from the slope of `Im Σ` between the two lowest frequencies.

    Negative slope means the self-energy turns upwards, which is the
    insulator; `Z` is zero there rather than negative.
    """
    slope = np.imag(sigma[mesh.iw0 + 1] - sigma[mesh.iw0])
    z = 1 / (1 - slope * BETA / (2 * np.pi))
    return max(float(z), 0.0)


def write() -> None:
    comment = provenance(EXAMPLE)
    mesh = Mesh()
    g0 = mesh.noninteracting()
    g_loc, sigma, residuals = solve(mesh, g0, U, MAXITER, SFC_TOL)
    # A threshold of zero is never met, so this one runs the full count.
    _, long_sigma, long_residuals = solve(mesh, g0, U, LONG_ITERATIONS, 0.0)

    write_table(
        EXAMPLE,
        "summary",
        {
            "d": np.array([D]),
            "wmax": np.array([WMAX]),
            "beta": np.array([BETA]),
            "u": np.array([U]),
            "eps": np.array([EPS]),
            "mix": np.array([MIX]),
            "sfc_tol": np.array([SFC_TOL]),
            "maxiter": np.array([MAXITER]),
            "basis_size": np.array([mesh.basis.size]),
            "n_tau": np.array([len(mesh.tau.sampling_points)]),
            "n_wn": np.array([len(mesh.wn.sampling_points)]),
            "iterations": np.array([len(residuals)]),
            "z": np.array([renormalisation(mesh, sigma)]),
            "long_iterations": np.array([len(long_residuals)]),
            "long_z": np.array([renormalisation(mesh, long_sigma)]),
        },
        comment,
    )

    n = mesh.wn.sampling_points.astype(float)
    write_table(EXAMPLE, "green", {"n": n, "g_re": g_loc.real, "g_im": g_loc.imag}, comment)
    write_table(
        EXAMPLE,
        "self_energy",
        {"n": n, "sigma_re": sigma.real, "sigma_im": sigma.imag},
        comment,
    )

    tau, sign, order = fold_tau(mesh.tau.sampling_points, BETA, "F")
    sigma_tau = sign * mesh.tau.evaluate(mesh.wn.fit(sigma))[order]
    write_table(
        EXAMPLE,
        "self_energy_tau",
        {"tau": tau, "sigma_re": sigma_tau.real, "sigma_im": sigma_tau.imag},
        comment,
    )

    for name, values in (("convergence", residuals), ("long_convergence", long_residuals)):
        write_table(
            EXAMPLE,
            name,
            {
                "iteration": np.arange(1, len(values) + 1, dtype=float),
                "residual": np.array(values),
            },
            comment,
        )


def write_scan() -> None:
    comment = provenance(SCAN)
    mesh = Mesh()
    g0 = mesh.noninteracting()
    u_arr = np.linspace(0.0, U_MAX, U_NUM)

    from_g0 = np.empty(U_NUM)
    metal = np.empty(U_NUM)
    insulator = np.empty(U_NUM)
    probes: dict[int, np.ndarray] = {}

    # Every calculation from the non-interacting Green's function.
    for index, u in enumerate(u_arr):
        _, sigma, _ = solve(mesh, g0, u, SCAN_ITERATIONS, 0.0)
        from_g0[index] = renormalisation(mesh, sigma)
        if index in SCAN_PROBES:
            probes[index] = sigma

    # And the hysteresis: walking up in U from the metal, and down from the
    # insulator, each calculation starting where the previous one stopped.
    g_metal, g_insulator = g0, g0
    for index in range(U_NUM):
        g_metal, sigma, _ = solve(mesh, g_metal, u_arr[index], SCAN_ITERATIONS, 0.0)
        metal[index] = renormalisation(mesh, sigma)

        other = U_NUM - 1 - index
        g_insulator, sigma, _ = solve(mesh, g_insulator, u_arr[other], SCAN_ITERATIONS, 0.0)
        insulator[other] = renormalisation(mesh, sigma)

    write_table(
        SCAN,
        "summary",
        {
            "d": np.array([D]),
            "beta": np.array([BETA]),
            "eps": np.array([EPS]),
            "u_max": np.array([U_MAX]),
            "u_num": np.array([U_NUM]),
            "iterations": np.array([SCAN_ITERATIONS]),
            "basis_size": np.array([mesh.basis.size]),
        },
        comment,
    )
    write_table(
        SCAN,
        "renormalisation",
        {"u": u_arr, "from_g0": from_g0, "metal": metal, "insulator": insulator},
        comment,
    )
    columns = {"n": mesh.wn.sampling_points.astype(float)}
    for index in SCAN_PROBES:
        columns[f"sigma_im_u{index}"] = probes[index].imag
    write_table(SCAN, "self_energy", columns, comment)


if __name__ == "__main__":
    write()
    write_scan()
