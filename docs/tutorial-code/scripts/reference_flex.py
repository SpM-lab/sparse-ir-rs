"""Reference values for the `flex` and `flex_scan` examples.

This is the computation of `FLEX_py.ipynb` of sparse-ir-tutorial (author:
Niklas Witt), with the plotting removed and the numbers written out. The
notebook solves the fluctuation-exchange approximation for the square-lattice
Hubbard model self-consistently, and then the linearised Eliashberg equation
for the d-wave pairing eigenvalue on top of the converged solution.

Three things differ from the notebook and are deliberate:

* the k grid is built with `indexing="ij"`, as in `reference_tpsc`, so that a
  flat index means the same point as in the Rust example. The dispersion is
  symmetric under `k_x <-> k_y`, so this transposes arrays without changing
  any value in them; the d-wave seed `cos k_x - cos k_y` does change sign,
  which the gap equation is blind to because it is linear and homogeneous.
* both loops run a fixed number of steps instead of stopping at a tolerance.
  The residual is written out instead, so the page can still say how far the
  solution has converged, but the answer no longer depends on exactly where a
  threshold happens to be crossed.
* the momentum transforms carry the notebook's factors. Every product taken
  in real space here has exactly two factors, for which the notebook's
  convention and the Rust one give identical answers.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import sparse_ir

from reference_common import provenance, write_table
from reference_tpsc import Mesh, high_symmetry_path

EXAMPLE = "flex"
SCAN_EXAMPLE = "flex_scan"

T_HOPPING = 1.0
TEMPERATURE = 0.1
BETA = 1 / TEMPERATURE
WMAX = 10.0
FILLING = 0.85
U = 4.0
NK_LIN = 24
EPS = 1e-10
MIX = 0.2
ITERATIONS = 30
GAP_ITERATIONS = 30
# The power method stops when lambda moves by less than this. It has to stop
# early: dropping the Hartree term from the singlet vertex is exact in the
# d-wave channel but leaves the operator with a spurious mode of larger
# magnitude, which takes over the iterate after some tens of steps. The
# tolerance is crossed three orders of magnitude before that.
GAP_TOL = 1e-4
RENORMALISATION_ITERATIONS = 50

# The transition line of Fig. 3(b) and Fig. 4 of Arita et al. (2000). The
# basis is built once at a fixed Lambda and re-evaluated at every temperature,
# so every step has the same basis size and the previous self-energy can be
# reused as the next starting point.
SCAN_TEMPERATURES = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025])
SCAN_NK_LIN = 64
SCAN_LAMBDA = 1e3
SCAN_EPS = 1e-8
# The temperature whose momentum dependence is written out as well.
SCAN_PROBE = 5

XTOL = 2e-12


class Solver:
    """The FLEX loop, in the notebook's order."""

    def __init__(self, mesh: Mesh, u: float, filling: float, sigma_init=None):
        self.mesh = mesh
        self.u_target = u
        self.u = u
        self.n = filling
        self.sigma = (
            np.zeros_like(mesh.iwn_f_) if sigma_init is None else sigma_init.copy()
        )
        self.renormalisation_steps = 0
        self.residual = np.nan

        self.mu = self.find_mu()
        self.gkio_calc(self.mu)
        self.grit_calc()
        self.ckio_calc()

    def solve(self) -> None:
        if np.amax(np.abs(self.ckio)) * self.u_target >= 1:
            self.renormalise()
        for _ in range(ITERATIONS):
            sigma_old = self.sigma
            self.loop()
            self.residual = np.sum(np.abs(self.sigma - sigma_old)) / np.sum(
                np.abs(self.sigma)
            )

    def loop(self) -> None:
        gkio_old = self.gkio
        self.v_calc()
        self.sigma_calc()
        self.mu = self.find_mu()
        self.gkio_calc(self.mu)
        self.gkio = MIX * self.gkio + (1 - MIX) * gkio_old
        self.grit_calc()
        self.ckio_calc()

    def renormalise(self) -> None:
        while self.u_target * np.amax(np.abs(self.ckio)) >= 1:
            self.renormalisation_steps += 1
            self.u = self.u_target / (
                np.amax(np.abs(self.ckio)) * self.u_target + 0.01
            )
            self.loop()
            self.u = self.u_target
            if self.renormalisation_steps == RENORMALISATION_ITERATIONS:
                break

    # --- Green's function and chemical potential ---------------------------
    def gkio_calc(self, mu: float) -> None:
        self.gkio = 1 / (self.mesh.iwn_f_ - (self.mesh.ek_ - mu) - self.sigma)

    def electron_density(self, mu: float) -> float:
        self.gkio_calc(mu)
        gio = np.sum(self.gkio, axis=1) / self.mesh.nk
        g_l = self.mesh.basis_set.smpl_wn_f.fit(gio)
        g_tau0 = self.mesh.basis_set.basis_f.u(0) @ g_l
        return 2 * (1 + np.real(g_tau0))

    def find_mu(self) -> float:
        lo, hi = 3 * np.amin(self.mesh.ek), 3 * np.amax(self.mesh.ek)
        return scipy.optimize.brentq(
            lambda mu: self.electron_density(mu) - self.n, lo, hi, xtol=XTOL
        )

    # --- susceptibilities and self-energy ----------------------------------
    def grit_calc(self) -> None:
        self.grit = self.mesh.wn_to_tau("F", self.mesh.k_to_r(self.gkio))

    def ckio_calc(self) -> None:
        # The sampling times are symmetric about beta/2, so reversing the rows
        # is G(beta - tau).
        ckio = self.grit * self.grit[::-1, :]
        self.ckio = self.mesh.tau_to_wn("B", self.mesh.r_to_k(ckio))

    def v_calc(self) -> None:
        self.chi_spin = self.ckio / (1 - self.u * self.ckio)
        self.chi_charge = self.ckio / (1 + self.u * self.ckio)
        v = self.u**2 * (
            1.5 * self.chi_spin + 0.5 * self.chi_charge - self.ckio
        )
        # The constant Hartree term ~U is left out: the basis cannot represent
        # it compactly, and in a single band it goes into the chemical
        # potential.
        self.v = self.mesh.wn_to_tau("B", self.mesh.k_to_r(v))

    def sigma_calc(self) -> None:
        self.sigma = self.mesh.tau_to_wn("F", self.mesh.r_to_k(self.v * self.grit))


class GapSolver:
    """The linearised Eliashberg equation by the power method."""

    def __init__(self, solver: Solver):
        mesh = solver.mesh
        self.mesh = mesh
        self.gkio = solver.gkio
        self.seed = np.cos(2 * np.pi * mesh.k1) - np.cos(2 * np.pi * mesh.k2)
        delta = np.tensordot(np.ones(len(mesh.iwn_f)), self.seed, axes=0)
        self.delta = delta / np.linalg.norm(delta)
        self.lam = 0.0
        self.iterations = 0

        # The singlet vertex: charge fluctuations enter with the opposite sign
        # to the self-energy, and the Hartree term drops out because a d-wave
        # gap sums to zero over the zone.
        v = solver.u**2 * (1.5 * solver.chi_spin - 0.5 * solver.chi_charge)
        self.v_singlet = mesh.wn_to_tau("B", mesh.k_to_r(v))

    def solve(self) -> None:
        for _ in range(GAP_ITERATIONS):
            lam_old = self.lam
            delta_old = self.delta
            self.fkio = -self.gkio * np.conj(self.gkio) * self.delta
            frit = self.mesh.wn_to_tau("F", self.mesh.k_to_r(self.fkio))
            delta = self.mesh.tau_to_wn(
                "F", self.mesh.r_to_k(self.v_singlet * frit)
            )
            self.lam = np.real(np.sum(np.conj(delta) * delta_old))
            self.delta = delta / np.linalg.norm(delta)
            self.iterations += 1
            if abs(self.lam - lam_old) < GAP_TOL:
                break


def _run(mesh: Mesh, sigma_init=None) -> tuple[Solver, GapSolver]:
    solver = Solver(mesh, U, FILLING)
    solver.solve()
    gap = GapSolver(solver)
    gap.solve()
    return solver, gap


def write() -> None:
    basis_set = sparse_ir.FiniteTempBasisSet(BETA, WMAX, eps=EPS)
    mesh = Mesh(basis_set, NK_LIN, NK_LIN)
    solver, gap = _run(mesh)

    comment = provenance(EXAMPLE)
    write_table(
        EXAMPLE,
        "summary",
        {
            "t": np.array([T_HOPPING]),
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "n": np.array([FILLING]),
            "u": np.array([U]),
            "nk_lin": np.array([float(NK_LIN)]),
            "eps": np.array([EPS]),
            "mix": np.array([MIX]),
            "iterations": np.array([float(ITERATIONS)]),
            "gap_max_iterations": np.array([float(GAP_ITERATIONS)]),
            "gap_tol": np.array([GAP_TOL]),
            "gap_iterations": np.array([float(gap.iterations)]),
            "basis_size_f": np.array([float(basis_set.basis_f.size)]),
            "basis_size_b": np.array([float(basis_set.basis_b.size)]),
            "n_tau": np.array([float(len(basis_set.smpl_tau_f.tau))]),
            "n_wn_f": np.array([float(len(basis_set.wn_f))]),
            "n_wn_b": np.array([float(len(basis_set.wn_b))]),
            "renormalisation_steps": np.array(
                [float(solver.renormalisation_steps)]
            ),
            "mu": np.array([solver.mu]),
            "residual": np.array([solver.residual]),
            "chi_spin_max": np.array([np.amax(solver.chi_spin.real)]),
            "lambda_d": np.array([gap.lam]),
        },
        comment,
    )

    row_f, row_b = mesh.iw0_f, mesh.iw0_b
    write_table(
        EXAMPLE,
        "momentum",
        {
            "kx": 2 * mesh.k1,
            "ky": 2 * mesh.k2,
            "ek": mesh.ek,
            "g_re": solver.gkio[row_f].real,
            "sigma_im": solver.sigma[row_f].imag,
            "chi_0": solver.ckio[row_b].real,
            "chi_spin": solver.chi_spin[row_b].real,
            "delta_re": gap.delta[row_f].real,
            "f_re": gap.fkio[row_f].real,
            "delta_seed": gap.seed,
        },
        comment,
    )

    path = high_symmetry_path(NK_LIN)
    write_table(
        EXAMPLE,
        "path",
        {
            "distance": np.arange(len(path), dtype=float),
            "chi_spin": solver.chi_spin[row_b][path].real,
            "chi_charge": solver.chi_charge[row_b][path].real,
            "chi_0": solver.ckio[row_b][path].real,
        },
        comment,
    )

    antinode = (NK_LIN // 2) * NK_LIN
    write_table(
        EXAMPLE,
        "self_energy",
        {
            "n": basis_set.wn_f.astype(float),
            "nu": basis_set.wn_f * np.pi / BETA,
            "sigma_im": solver.sigma[:, antinode].imag,
            "sigma_re": solver.sigma[:, antinode].real,
            "delta_re": gap.delta[:, antinode].real,
        },
        comment,
    )


def write_scan() -> None:
    beta_init = 1 / SCAN_TEMPERATURES[0]
    basis_set = sparse_ir.FiniteTempBasisSet(
        beta_init, SCAN_LAMBDA / beta_init, eps=SCAN_EPS
    )
    sve_result = basis_set.sve_result

    sigma_init = None
    lam, chi_max, mu, residual, steps, gap_its = [], [], [], [], [], []
    chi_path = None
    for index, temperature in enumerate(SCAN_TEMPERATURES):
        beta = 1 / temperature
        basis_set = sparse_ir.FiniteTempBasisSet(
            beta, SCAN_LAMBDA / beta, eps=SCAN_EPS, sve_result=sve_result
        )
        mesh = Mesh(basis_set, SCAN_NK_LIN, SCAN_NK_LIN)
        solver = Solver(mesh, U, FILLING, sigma_init=sigma_init)
        solver.solve()
        sigma_init = solver.sigma
        gap = GapSolver(solver)
        gap.solve()

        lam.append(gap.lam)
        chi_max.append(np.amax(solver.chi_spin.real))
        mu.append(solver.mu)
        residual.append(solver.residual)
        steps.append(float(solver.renormalisation_steps))
        gap_its.append(float(gap.iterations))
        if index == SCAN_PROBE:
            path = high_symmetry_path(SCAN_NK_LIN)
            chi_path = solver.chi_spin[mesh.iw0_b][path].real

    comment = provenance(SCAN_EXAMPLE)
    write_table(
        SCAN_EXAMPLE,
        "summary",
        {
            "t": np.array([T_HOPPING]),
            "n": np.array([FILLING]),
            "u": np.array([U]),
            "nk_lin": np.array([float(SCAN_NK_LIN)]),
            "lambda_ir": np.array([SCAN_LAMBDA]),
            "eps": np.array([SCAN_EPS]),
            "mix": np.array([MIX]),
            "iterations": np.array([float(ITERATIONS)]),
            "gap_max_iterations": np.array([float(GAP_ITERATIONS)]),
            "gap_tol": np.array([GAP_TOL]),
            "t_num": np.array([float(len(SCAN_TEMPERATURES))]),
            "basis_size_f": np.array([float(basis_set.basis_f.size)]),
            "n_tau": np.array([float(len(basis_set.smpl_tau_f.tau))]),
            "n_wn_f": np.array([float(len(basis_set.wn_f))]),
            "n_wn_b": np.array([float(len(basis_set.wn_b))]),
            "probe_temperature": np.array([SCAN_TEMPERATURES[SCAN_PROBE]]),
        },
        comment,
    )
    write_table(
        SCAN_EXAMPLE,
        "temperature",
        {
            "temperature": SCAN_TEMPERATURES,
            "beta": 1 / SCAN_TEMPERATURES,
            "mu": np.array(mu),
            "lambda_d": np.array(lam),
            "chi_spin_max": np.array(chi_max),
            "inverse_chi_spin_max": 1 / np.array(chi_max),
            "renormalisation_steps": np.array(steps),
            "residual": np.array(residual),
            "gap_iterations": np.array(gap_its),
        },
        comment,
    )
    write_table(
        SCAN_EXAMPLE,
        "chi_spin",
        {
            "distance": np.arange(len(chi_path), dtype=float),
            "chi_spin": chi_path,
        },
        comment,
    )
