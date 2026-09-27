"""Reference values for the Eliashberg examples.

Ported from the Python notebook `eliashberg_holstein_py.ipynb` of
sparse-ir-tutorial, whose authors are Shintaro Hoshino and Hiroshi Shinaoka.

Three deliberate deviations from the notebook:

  * the noisy starting self-energy is read from the committed input file
    rather than redrawn, so that the Rust example starts from the same numbers
    without having to reproduce NumPy's generator;
  * the number of iterations each solve took is written out, because the two
    implementations only agree on the fixed point if they take the same walk
    towards it;
  * sampling times are folded onto `[−β/2, β/2]`, which is where the Rust
    implementation reports them.

Everything else — the Gauss-Legendre rule, the mixing, the convergence
threshold, the particle-hole symmetrisation, the clamp on `G(τ)` — follows the
notebook exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sparse_ir
from numpy.polynomial.legendre import leggauss

import reference_common

EXAMPLE = "eliashberg_holstein"
SCAN = "eliashberg_holstein_scan"

D = 0.5
U = 2.0
J = 0.03 * U
OMEGA0 = 0.15
MU = 0.0
EPS = 1e-7
WMAX = 10 * D
DEG_LEGGAUSS = 100
MIXING = 0.3

BETA = 500.0
LAMBDA0 = 0.125
MAX_ITERATIONS = 10000
ATOL = 1e-10

SCAN_LAMBDA0 = 0.175
SCAN_MAX_ITERATIONS = 100000
SCAN_ATOL = 1e-6
TEMPERATURES = np.linspace(0.009, 0.013, 10)
DT = 1e-5

INPUT_DIR = Path(__file__).resolve().parents[1] / "input" / EXAMPLE


def coupling(lambda0: float) -> float:
    """`g₀ = √(3 λ₀ ω₀ / 4)`."""
    return np.sqrt(3 * lambda0 * OMEGA0 / 4)


def density_of_states(omega: np.ndarray) -> np.ndarray:
    """Semicircle of half bandwidth `D`, normalised to one."""
    return np.sqrt(D**2 - omega**2) / (0.5 * D**2 * np.pi)


def read_noise(name: str) -> np.ndarray:
    """One column pair of the committed starting point."""
    rows = [
        line for line in (INPUT_DIR / f"{name}.csv").read_text().splitlines() if not line.startswith("#")
    ]
    values = np.array([[float(x) for x in line.split(",")] for line in rows[1:]])
    return values[:, 0] + 1j * values[:, 1]


def scale_quad(x, w, xmin, xmax):
    """Scales the weights and nodes of a quadrature onto `[xmin, xmax]`."""
    dx = xmax - xmin
    return (0.5 * dx) * (x + 1) + xmin, 0.5 * dx * w


class Eliashberg:
    def __init__(self, bset, omega_range, g):
        self.bset = bset
        self.beta = bset.beta
        self.g = g
        x, w = leggauss(DEG_LEGGAUSS)
        self.omega, weights = scale_quad(x, w, *omega_range)
        self.omega_coeff = density_of_states(self.omega) * weights
        self.iv_f = bset.wn_f * (1j * np.pi / self.beta)
        self.iv_b = bset.wn_b * (1j * np.pi / self.beta)
        self.d0_iv = 2 * OMEGA0 / (self.iv_b**2 - OMEGA0**2)

    def g_f_iv(self, sigma_iv, delta_iv):
        xi_iv = self.iv_f + MU - sigma_iv
        denominator = (xi_iv**2 - delta_iv**2)[:, None] - (self.omega**2)[None, :]
        numerator_g = xi_iv[:, None] + self.omega[None, :]
        numerator_f = delta_iv[:, None]
        g_iv = np.einsum("q,wq->w", self.omega_coeff, numerator_g / denominator, optimize=True)
        f_iv = np.einsum("q,wq->w", self.omega_coeff, numerator_f / denominator, optimize=True)
        return g_iv, f_iv

    def pi_tau(self, g_tau, f_tau):
        return -4 * (self.g**2) * (g_tau * g_tau[::-1] + f_tau**2)

    def d_iv(self, phi_iv):
        return 1 / (1 / self.d0_iv - phi_iv)

    def sigma_iv(self, g_tau, d_tau):
        return self.to_matsu(-4 * (self.g**2) * d_tau * g_tau, "F")

    def delta_iv(self, f_tau, d_tau):
        tl_delta_iv = self.to_matsu((4 * self.g**2) * d_tau * f_tau, "F")
        return tl_delta_iv + (U + 2 * J) * f_tau[0]

    def _smpl_tau(self, stat):
        return {"F": self.bset.smpl_tau_f, "B": self.bset.smpl_tau_b}[stat]

    def _smpl_wn(self, stat):
        return {"F": self.bset.smpl_wn_f, "B": self.bset.smpl_wn_b}[stat]

    def to_tau(self, obj_iv, stat):
        return self._smpl_tau(stat).evaluate(self._smpl_wn(stat).fit(obj_iv))

    def to_matsu(self, obj_tau, stat):
        return self._smpl_wn(stat).evaluate(self._smpl_tau(stat).fit(obj_tau))

    def internal_energy(self, sigma_iv, delta_iv, g_iv, d_iv):
        smpl_f = sparse_ir.TauSampling(self.bset.basis_f, [0.0])
        e1 = smpl_f.evaluate(self.bset.smpl_wn_f.fit(self.iv_f * g_iv - 1))
        e2 = smpl_f.evaluate(
            self.bset.smpl_wn_f.fit(
                g_iv * ((self.iv_f - sigma_iv) ** 2 - delta_iv**2) / (self.iv_f - sigma_iv) - 1
            )
        )
        f2 = self.bset.smpl_tau_b.evaluate(
            self.bset.smpl_wn_b.fit((OMEGA0**-2) * ((self.iv_b**2) * d_iv - 2 * OMEGA0))
        )
        return (3 * (e1 + e2 - OMEGA0 * f2))[0]


def solve(elsh, sigma_iv, delta_iv, max_iterations, atol):
    """The self-consistency loop, with particle-hole symmetry enforced."""
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        g_iv, f_iv = elsh.g_f_iv(sigma_iv, delta_iv)
        g_tau = elsh.to_tau(g_iv, "F")
        f_tau = elsh.to_tau(f_iv, "F")
        g_tau = 0.5 * (g_tau + g_tau[::-1])
        g_tau[g_tau > 0] = 0

        phi_tau = elsh.pi_tau(g_tau, f_tau)
        phi_iv = elsh.to_matsu(phi_tau, "B")
        phi_iv.imag = 0

        d_iv = elsh.d_iv(phi_iv)
        d_tau = elsh.to_tau(d_iv, "B")

        sigma_iv_new = elsh.sigma_iv(g_tau, d_tau)
        sigma_iv_prev = sigma_iv.copy()
        sigma_iv = (1 - MIXING) * sigma_iv + MIXING * sigma_iv_new

        delta_iv_new = elsh.delta_iv(f_tau, d_tau)
        delta_iv_prev = delta_iv.copy()
        delta_iv = (1 - MIXING) * delta_iv + MIXING * delta_iv_new
        delta_iv.imag = 0.0
        delta_iv = 0.5 * (delta_iv + delta_iv[::-1])

        diff = max(
            np.abs(sigma_iv_new - sigma_iv_prev).max(),
            np.abs(delta_iv_new - delta_iv_prev).max(),
        )
        if diff < atol:
            break

    energy = elsh.internal_energy(sigma_iv, delta_iv, g_iv, d_iv)
    return {
        "sigma_iv": sigma_iv,
        "delta_iv": delta_iv,
        "g_iv": g_iv,
        "f_iv": f_iv,
        "d_iv": d_iv,
        "d_tau": d_tau,
        "f_tau": f_tau,
        "g_tau": g_tau,
        "phi_iv": phi_iv,
        "energy": energy,
        "iterations": iterations,
    }


def write() -> None:
    comment = reference_common.provenance(EXAMPLE)
    bset = sparse_ir.FiniteTempBasisSet(BETA, WMAX, EPS)
    elsh = Eliashberg(bset, (-D, D), coupling(LAMBDA0))

    sigma_iv = read_noise("noise")
    delta_iv = np.full(bset.wn_f.size, 1.0, dtype=np.complex128)
    res = solve(elsh, sigma_iv, delta_iv, MAX_ITERATIONS, ATOL)

    summary = {
        "beta": BETA,
        "d": D,
        "u": U,
        "j": J,
        "omega0": OMEGA0,
        "lambda0": LAMBDA0,
        "g": coupling(LAMBDA0),
        "mu": MU,
        "eps": EPS,
        "wmax": WMAX,
        "deg_leggauss": DEG_LEGGAUSS,
        "mixing": MIXING,
        "max_iterations": MAX_ITERATIONS,
        "atol": ATOL,
        "iterations": res["iterations"],
        "basis_size_f": bset.basis_f.size,
        "basis_size_b": bset.basis_b.size,
        "n_tau_f": bset.tau.size,
        "n_wn_f": bset.wn_f.size,
        "n_wn_b": bset.wn_b.size,
        "gap": res["delta_iv"][bset.wn_f.size // 2].real,
        "energy": res["energy"].real,
    }
    reference_common.write_table(
        EXAMPLE, "summary", {k: np.array([v]) for k, v in summary.items()}, comment
    )

    reference_common.write_table(
        EXAMPLE,
        "matsubara_f",
        {
            "n": bset.wn_f.astype(float),
            "nu": elsh.iv_f.imag,
            "sigma_im": res["sigma_iv"].imag,
            "delta_re": res["delta_iv"].real,
            "g_im": res["g_iv"].imag,
            "f_re": res["f_iv"].real,
        },
        comment,
    )
    reference_common.write_table(
        EXAMPLE,
        "matsubara_b",
        {
            "n": bset.wn_b.astype(float),
            "nu": elsh.iv_b.imag,
            "d_re": res["d_iv"].real,
            "phi_re": res["phi_iv"].real,
        },
        comment,
    )

    tau, sign, order = reference_common.fold_tau(bset.smpl_tau_b.sampling_points, BETA, "B")
    reference_common.write_table(
        EXAMPLE,
        "tau_b",
        {"tau": tau, "d_tau": (sign * res["d_tau"][order]).real},
        comment,
    )

    # The compactness check of the notebook: the anomalous Green's function
    # decays with the singular values.
    f_l = bset.smpl_wn_f.fit(res["f_iv"])
    reference_common.write_table(
        EXAMPLE,
        "basis",
        {
            "l": np.arange(f_l.size, dtype=float),
            "f_l": np.abs(f_l),
            "s_ratio": bset.basis_f.s / bset.basis_f.s[0],
        },
        comment,
    )


def write_scan() -> None:
    comment = reference_common.provenance(SCAN)
    g = coupling(SCAN_LAMBDA0)
    lambda_ir = WMAX / TEMPERATURES.min()
    temperatures = np.unique(np.hstack((TEMPERATURES, TEMPERATURES + DT)))

    sve_result = None
    sigma_iv = read_noise("scan_noise")
    delta_iv = None
    energies, iterations, sizes = [], [], []
    for temperature in temperatures:
        beta = 1 / temperature
        if sve_result is None:
            bset = sparse_ir.FiniteTempBasisSet(beta, lambda_ir / beta, EPS)
            sve_result = bset.sve_result
        else:
            bset = sparse_ir.FiniteTempBasisSet(beta, lambda_ir / beta, EPS, sve_result=sve_result)
        if delta_iv is None:
            delta_iv = np.full(bset.wn_f.size, 1.0, dtype=np.complex128)
        elsh = Eliashberg(bset, (-D, D), g)
        res = solve(elsh, sigma_iv, delta_iv, SCAN_MAX_ITERATIONS, SCAN_ATOL)
        sigma_iv = res["sigma_iv"].copy()
        delta_iv = res["delta_iv"].copy()
        energies.append(res["energy"].real)
        iterations.append(res["iterations"])
        sizes.append(bset.wn_f.size)

    if len(set(sizes)) != 1:
        raise SystemExit(f"the basis changed size along the sweep: {sorted(set(sizes))}")

    energy = dict(zip(temperatures, energies))
    specific_heat = np.array([energy[t + DT] - energy[t] for t in TEMPERATURES]) / DT

    summary = {
        "d": D,
        "u": U,
        "j": J,
        "omega0": OMEGA0,
        "lambda0": SCAN_LAMBDA0,
        "g": g,
        "mu": MU,
        "eps": EPS,
        "lambda_ir": lambda_ir,
        "deg_leggauss": DEG_LEGGAUSS,
        "mixing": MIXING,
        "max_iterations": SCAN_MAX_ITERATIONS,
        "atol": SCAN_ATOL,
        "t_num": len(TEMPERATURES),
        "dt": DT,
        "basis_size_f": sizes[0],
        "n_wn_f": sizes[0],
    }
    reference_common.write_table(
        SCAN, "summary", {k: np.array([v]) for k, v in summary.items()}, comment
    )
    reference_common.write_table(
        SCAN,
        "temperature",
        {
            "temperature": temperatures,
            "beta": 1 / temperatures,
            "iterations": np.array(iterations, dtype=float),
            "energy": np.array(energies),
        },
        comment,
    )
    reference_common.write_table(
        SCAN,
        "specific_heat",
        {
            "temperature": TEMPERATURES,
            "specific_heat": specific_heat,
            "c_over_t": specific_heat / TEMPERATURES,
        },
        comment,
    )


if __name__ == "__main__":
    write()
    write_scan()
