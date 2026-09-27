"""Reference values for the `tpsc` and `tpsc_scan` examples.

This is the computation of `TPSC_py.ipynb` of sparse-ir-tutorial (author:
Niklas Witt), with the plotting removed and the numbers written out. The
notebook solves the two-particle self-consistent approximation for the
square-lattice Hubbard model: the spin and charge vertices are fixed by the
local sum rules rather than by summing diagrams, which makes each of them a
one-dimensional root search over a Matsubara-summed susceptibility.

Two conventions differ from the notebook and are harmless:

* the k grid is built with `indexing="ij"`, so that a flat index means the
  same point as in the Rust example. Since the dispersion is symmetric under
  `k_x <-> k_y`, this is a transpose of the notebook's array and changes no
  value in it.
* the momentum transforms carry the notebook's factors. Every product taken
  in real space here has exactly two factors, for which the notebook's
  convention and the Rust one give identical answers.

Nothing is compared on a sampling time, so no folding is needed: the outputs
are momentum-resolved quantities at a fixed Matsubara frequency, Matsubara
functions at a fixed momentum, and scalars.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import sparse_ir

from reference_common import provenance, write_table

EXAMPLE = "tpsc"
SCAN_EXAMPLE = "tpsc_scan"

T_HOPPING = 1.0
TEMPERATURE = 0.1
BETA = 1 / TEMPERATURE
WMAX = 10.0
FILLING = 0.85
U = 4.0
NK_LIN = 24
EPS = 1e-10

# The scan of Fig. 2 of Vilk and Tremblay (1997): half filling, well above the
# critical temperature so that `U_sp < U_crit` holds for every `U` on the grid.
SCAN_TEMPERATURE = 0.4
SCAN_FILLING = 1.0
SCAN_EPS = 1e-8
SCAN_U = np.linspace(1e-2, 5, 51)
# The interaction strengths whose momentum dependence is written out as well.
SCAN_PROBES = (0, 25, 50)

# The tolerance of every root search, matching SciPy's default `xtol` and the
# Rust example's constant.
XTOL = 2e-12


class Mesh:
    """k grid, sampling grids and the transforms between them."""

    def __init__(self, basis_set: sparse_ir.FiniteTempBasisSet, nk1: int, nk2: int):
        self.basis_set = basis_set
        self.nk1, self.nk2, self.nk = nk1, nk2, nk1 * nk2
        k1, k2 = np.meshgrid(
            np.arange(nk1) / nk1, np.arange(nk2) / nk2, indexing="ij"
        )
        self.k1, self.k2 = k1.reshape(self.nk), k2.reshape(self.nk)
        self.ek = -2 * T_HOPPING * (np.cos(2 * np.pi * k1) + np.cos(2 * np.pi * k2))
        self.ek = self.ek.reshape(self.nk)

        self.iw0_f = np.where(basis_set.wn_f == 1)[0][0]
        self.iw0_b = np.where(basis_set.wn_b == 0)[0][0]

        beta = basis_set.beta
        self.iwn_f = 1j * basis_set.wn_f * np.pi / beta
        self.iwn_f_ = np.tensordot(self.iwn_f, np.ones(self.nk), axes=0)
        self.ek_ = np.tensordot(np.ones(len(self.iwn_f)), self.ek, axes=0)

    def _sampling(self, statistics: str):
        smpl_tau = {"F": self.basis_set.smpl_tau_f, "B": self.basis_set.smpl_tau_b}[
            statistics
        ]
        smpl_wn = {"F": self.basis_set.smpl_wn_f, "B": self.basis_set.smpl_wn_b}[
            statistics
        ]
        return smpl_tau, smpl_wn

    def tau_to_wn(self, statistics: str, obj_tau: np.ndarray) -> np.ndarray:
        smpl_tau, smpl_wn = self._sampling(statistics)
        obj_l = smpl_tau.fit(obj_tau, axis=0)
        return smpl_wn.evaluate(obj_l, axis=0)

    def wn_to_tau(self, statistics: str, obj_wn: np.ndarray) -> np.ndarray:
        smpl_tau, smpl_wn = self._sampling(statistics)
        obj_l = smpl_wn.fit(obj_wn, axis=0)
        return smpl_tau.evaluate(obj_l, axis=0)

    def k_to_r(self, obj_k: np.ndarray) -> np.ndarray:
        obj_k = obj_k.reshape(-1, self.nk1, self.nk2)
        return np.fft.fftn(obj_k, axes=(1, 2)).reshape(-1, self.nk)

    def r_to_k(self, obj_r: np.ndarray) -> np.ndarray:
        obj_r = obj_r.reshape(-1, self.nk1, self.nk2)
        return (np.fft.ifftn(obj_r, axes=(1, 2)) / self.nk).reshape(-1, self.nk)


class Solver:
    """One TPSC calculation, in the notebook's order."""

    def __init__(self, mesh: Mesh, u: float, filling: float):
        self.mesh = mesh
        self.u = u
        self.n = filling

        self.sigma = np.zeros_like(mesh.iwn_f_)
        self.mu_0 = self.find_mu()
        self.gkio_calc(self.mu_0)
        self.ckio_calc()
        # 1/max(chi0) is where the RPA-like spin susceptibility diverges, so
        # no U_sp at or above it can satisfy the sum rule.
        self.u_crit = 1 / np.amax(self.ckio.real)

    def solve(self) -> None:
        self.spin_vertex_calc()
        self.docc = 0.25 * self.u_sp / self.u * self.n**2
        self.charge_vertex_calc()
        self.chi_spin = self.rpa(self.u_sp)
        self.chi_charge = self.rpa(-self.u_ch)
        self.sigma_calc()
        self.mu = self.find_mu()
        self.gkio_calc(self.mu)

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

    # --- susceptibilities ---------------------------------------------------
    def ckio_calc(self) -> None:
        grit = self.mesh.wn_to_tau("F", self.mesh.k_to_r(self.gkio))
        # The sampling times are symmetric about beta/2, so reversing the rows
        # is G(beta - tau).
        ckio = grit * grit[::-1, :]
        self.ckio = self.mesh.tau_to_wn("B", self.mesh.r_to_k(ckio))
        self.grit = grit

    def rpa(self, vertex: float) -> np.ndarray:
        return self.ckio / (1 - vertex * self.ckio)

    def chi_trace(self, vertex: float) -> float:
        chi = np.sum(self.rpa(vertex), axis=1) / self.mesh.nk
        chi_l = self.mesh.basis_set.smpl_wn_b.fit(chi)
        return np.real(self.mesh.basis_set.basis_b.u(0) @ chi_l)

    # --- vertices ------------------------------------------------------------
    def spin_vertex_calc(self) -> None:
        upper = np.floor(self.u_crit * 100) / 100

        def sum_rule(u_sp: float) -> float:
            return 2 * self.chi_trace(u_sp) - self.n + 0.5 * (u_sp / self.u) * self.n**2

        if sum_rule(upper) <= 0:
            raise SystemExit(
                f"{EXAMPLE}: U_sp would exceed U_crit = {self.u_crit} at U = {self.u}"
            )
        self.u_sp = scipy.optimize.brentq(sum_rule, 0.0, upper, xtol=XTOL)

    def charge_vertex_calc(self) -> None:
        def sum_rule(u_ch: float) -> float:
            return (
                2 * self.chi_trace(-u_ch) - self.n - 2 * self.docc + self.n**2
            )

        self.u_ch = scipy.optimize.brentq(sum_rule, 0.0, 100.0, xtol=XTOL)

    # --- self-energy ----------------------------------------------------------
    def sigma_calc(self) -> None:
        v = self.u / 4 * (3 * self.u_sp * self.chi_spin + self.u_ch * self.chi_charge)
        v = self.mesh.wn_to_tau("B", self.mesh.k_to_r(v))
        self.sigma = self.mesh.tau_to_wn("F", self.mesh.r_to_k(v * self.grit))


def high_symmetry_path(nk_lin: int) -> np.ndarray:
    """Gamma -> X -> M -> Gamma as flat indices of the `ij` grid."""
    half = nk_lin // 2
    path = [i * nk_lin for i in range(half + 1)]
    path += [half * nk_lin + j for j in range(1, half + 1)]
    path += [i * nk_lin + i for i in range(half - 1, -1, -1)]
    return np.array(path)


def write() -> None:
    basis_set = sparse_ir.FiniteTempBasisSet(BETA, WMAX, eps=EPS)
    mesh = Mesh(basis_set, NK_LIN, NK_LIN)
    solver = Solver(mesh, U, FILLING)
    solver.solve()

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
            "nk_lin": np.array([NK_LIN]),
            "eps": np.array([EPS]),
            "basis_size_f": np.array([basis_set.basis_f.size]),
            "basis_size_b": np.array([basis_set.basis_b.size]),
            "n_tau": np.array([basis_set.smpl_tau_f.tau.size]),
            "n_wn_f": np.array([basis_set.smpl_wn_f.wn.size]),
            "n_wn_b": np.array([basis_set.smpl_wn_b.wn.size]),
            "mu_0": np.array([solver.mu_0]),
            "mu": np.array([solver.mu]),
            "u_crit": np.array([solver.u_crit]),
            "u_sp": np.array([solver.u_sp]),
            "u_ch": np.array([solver.u_ch]),
            "docc": np.array([solver.docc]),
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
            "chi_charge": solver.chi_charge[row_b].real,
        },
        comment,
    )

    path = high_symmetry_path(NK_LIN)
    write_table(
        EXAMPLE,
        "path",
        {
            "distance": np.arange(path.size, dtype=float),
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
            "n": basis_set.smpl_wn_f.wn.astype(float),
            "nu": mesh.iwn_f.imag,
            "sigma_im": solver.sigma[:, antinode].imag,
            "sigma_re": solver.sigma[:, antinode].real,
        },
        comment,
    )


def write_scan() -> None:
    beta = 1 / SCAN_TEMPERATURE
    basis_set = sparse_ir.FiniteTempBasisSet(beta, WMAX, eps=SCAN_EPS)
    mesh = Mesh(basis_set, NK_LIN, NK_LIN)

    u_sp = np.empty(SCAN_U.size)
    u_ch = np.empty(SCAN_U.size)
    u_crit = np.empty(SCAN_U.size)
    docc = np.empty(SCAN_U.size)
    probes = {}
    for index, u in enumerate(SCAN_U):
        solver = Solver(mesh, u, SCAN_FILLING)
        solver.solve()
        u_sp[index] = solver.u_sp
        u_ch[index] = solver.u_ch
        u_crit[index] = solver.u_crit
        docc[index] = solver.docc
        if index in SCAN_PROBES:
            probes[index] = solver.chi_spin[mesh.iw0_b].real

    comment = provenance(SCAN_EXAMPLE)
    write_table(
        SCAN_EXAMPLE,
        "summary",
        {
            "t": np.array([T_HOPPING]),
            "beta": np.array([beta]),
            "wmax": np.array([WMAX]),
            "n": np.array([SCAN_FILLING]),
            "nk_lin": np.array([NK_LIN]),
            "eps": np.array([SCAN_EPS]),
            "u_num": np.array([SCAN_U.size]),
            "u_min": np.array([SCAN_U[0]]),
            "u_max": np.array([SCAN_U[-1]]),
            "basis_size_f": np.array([basis_set.basis_f.size]),
            **{f"probe_u_{index}": np.array([SCAN_U[index]]) for index in SCAN_PROBES},
        },
        comment,
    )
    write_table(
        SCAN_EXAMPLE,
        "vertices",
        {
            "u": SCAN_U,
            "u_sp": u_sp,
            "u_ch": u_ch,
            "u_crit": u_crit,
            "docc": docc,
        },
        comment,
    )
    path = high_symmetry_path(NK_LIN)
    write_table(
        SCAN_EXAMPLE,
        "chi_spin",
        {
            "distance": np.arange(path.size, dtype=float),
            # Named by grid index rather than by `U`, so that the column
            # names cannot depend on how a half-way value is rounded.
            **{f"u_{index}": probes[index][path] for index in SCAN_PROBES},
        },
        comment,
    )
