"""Reference values for the `orbital_magnetic_susceptibility` example.

This is the computation of `orbital_magnetic_susceptibility_py.ipynb` of
sparse-ir-tutorial, with the plotting removed and the numbers written out.
The notebook evaluates the orbital magnetic susceptibility

    χ = T Σ_ν Σ_k Tr[γx G γy G γx G γy G
                      + ½ (γx G γy G + γy G γx G) γxy G]

of a tight-binding model as a function of the chemical potential, once for the
square lattice (one band, so every matrix is a number) and once for graphene
(two bands). The Matsubara sum is the whole point of the example: it is done
by fitting χ(iν) on the sampling frequencies and evaluating the result at
τ = 0, which is what `T Σ_ν` means.

The notebook loops over the k points one at a time; here they are summed in
chunks so that the reference does not take an hour to produce. The summands
are the same, only their order differs.

Nothing lives on a sampling time, so no τ folding is needed.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import sparse_ir

from reference_common import provenance, write_table

EXAMPLE = "orbital_magnetic_susceptibility"

T_HOPPING = 1.0
LATTICE = 1.0
TEMPERATURE = 0.1
BETA = 1.0 / TEMPERATURE
WMAX = 10.0
EPS = 1e-10
NK_LIN = 200
MU = np.linspace(-4.5, 4.5, 91)
PROBE_MU = -1.0
# k points are summed in blocks of this many; a block holds
# CHUNK × bands × |MU| × |ν| complex numbers.
CHUNK = 400


def k_points() -> np.ndarray:
    """The `NK_LIN²` points of the zone, in the notebook's order."""
    line = np.arange(NK_LIN) / NK_LIN
    k1, k2 = np.meshgrid(line, line, indexing="ij")
    return np.stack([k1.ravel(), k2.ravel()], axis=1)


def green(iw: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """`G(iν, k)` for every chemical potential, shaped `(..., |MU|, |ν|)`."""
    return 1.0 / (iw - (energies[..., None, None] - MU[:, None]))


def square(k: np.ndarray) -> tuple[np.ndarray, ...]:
    """`ε`, `γx`, `γy`, `γxy` of the square lattice at the given k points."""
    kx, ky = 2 * np.pi * k[:, 0], 2 * np.pi * k[:, 1]
    ek = -2 * T_HOPPING * (np.cos(kx) + np.cos(ky))
    gx = 2 * T_HOPPING * LATTICE * np.sin(kx)
    gy = 2 * T_HOPPING * LATTICE * np.sin(ky)
    return ek, gx, gy, np.zeros_like(ek)


SQRT3 = np.sqrt(3.0)


def graphene(k: np.ndarray) -> tuple[np.ndarray, ...]:
    """`H`, `γx`, `γy`, `γxy` of graphene: 2×2 matrices at each k point."""
    kx = 2 * np.pi * k[:, 0] / LATTICE
    ky = 2 * np.pi * (k[:, 0] + 2 * k[:, 1]) / (LATTICE * SQRT3)

    def offdiagonal(upper: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(upper), 2, 2), dtype=complex)
        matrix[:, 0, 1] = upper
        matrix[:, 1, 0] = np.conj(upper)
        return matrix

    phase = np.exp(-1j * ky * LATTICE / (2 * SQRT3))
    h = -T_HOPPING * (np.exp(1j * ky * LATTICE / SQRT3) + 2 * np.cos(kx * LATTICE / 2) * phase)
    hx = T_HOPPING * LATTICE * np.sin(kx / 2) * phase
    hy = -T_HOPPING * LATTICE * (1j / SQRT3) * (np.exp(1j * ky / SQRT3) - np.cos(kx / 2) * phase)
    hxy = -T_HOPPING * LATTICE**2 * 1j / (2 * SQRT3) * np.sin(kx / 2) * phase
    return offdiagonal(h), offdiagonal(hx), offdiagonal(hy), offdiagonal(hxy)


def chi_single_band(iw: np.ndarray) -> np.ndarray:
    """`χ(iν)` of the square lattice, summed over the zone."""
    points = k_points()
    total = np.zeros((len(MU), len(iw)), dtype=complex)
    for start in range(0, len(points), CHUNK):
        ek, gx, gy, gxy = square(points[start : start + CHUNK])
        g = green(iw, ek)
        weight_four = (gx**2 * gy**2)[:, None, None]
        weight_three = (gx * gy * gxy)[:, None, None]
        total += np.sum(weight_four * g**4 + weight_three * g**3, axis=0)
    return total / len(points)


def chi_two_band(iw: np.ndarray) -> np.ndarray:
    """`χ(iν)` of graphene, summed over the zone.

    In the band basis `G` is diagonal, so the trace is the sum over band
    indices of the velocity matrix elements times the four (or three) Green's
    functions of the bands they connect.
    """
    points = k_points()
    total = np.zeros((len(MU), len(iw)), dtype=complex)
    for start in range(0, len(points), CHUNK):
        hk, gx, gy, gxy = graphene(points[start : start + CHUNK])
        energies, vectors = np.linalg.eigh(hk)
        adjoint = np.conj(np.swapaxes(vectors, 1, 2))
        gx = adjoint @ gx @ vectors
        gy = adjoint @ gy @ vectors
        gxy = adjoint @ gxy @ vectors
        g = green(iw, np.swapaxes(energies, 0, 1))  # (band, k, |MU|, |ν|)
        g = np.swapaxes(g, 0, 1)  # (k, band, |MU|, |ν|)
        total += np.einsum(
            "kab, kbmn, kbc, kcmn, kcd, kdmn, kda, kamn->mn",
            gx, g, gy, g, gx, g, gy, g, optimize=True,
        )
        for first, second in ((gx, gy), (gy, gx)):
            total += 0.5 * np.einsum(
                "kab, kbmn, kbc, kcmn, kca, kamn->mn",
                first, g, second, g, gxy, g, optimize=True,
            )
    return total / len(points)


def matsubara_sum(basis, sampling, chi_iw: np.ndarray) -> np.ndarray:
    """`T Σ_ν χ(iν)`, which is the fit of `χ(iν)` evaluated at τ = 0."""
    return np.real(sampling.fit(chi_iw, axis=1) @ basis.u(0))


def analytic() -> tuple[np.ndarray, np.ndarray]:
    """The square lattice at `T = 0`, in complete elliptic integrals.

    `K(1)` is infinite, so the closed form diverges at the van Hove filling
    μ = 0; that one chemical potential is left out rather than written as an
    infinity nothing can be compared against.
    """
    mu = MU[MU != 0.0]
    m = 1 - mu**2 / 16
    chi = np.where(
        m >= 0,
        -(scipy.special.ellipe(m) - scipy.special.ellipk(m) / 2) * (2 / 3) / np.pi**2,
        0.0,
    )
    return mu, chi


def write() -> None:
    comment = provenance(EXAMPLE)

    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    sampling = sparse_ir.MatsubaraSampling(basis)
    wn = sampling.sampling_points
    iw = 1j * np.pi * wn / BETA

    write_table(
        EXAMPLE,
        "summary",
        {
            "t": np.array([T_HOPPING]),
            "a": np.array([LATTICE]),
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "eps": np.array([EPS]),
            "nk_lin": np.array([NK_LIN]),
            "n_mu": np.array([len(MU)]),
            "basis_size": np.array([basis.size]),
            "n_matsubara": np.array([len(wn)]),
        },
        comment,
    )

    # The chemical potential the sampled χ(iν) is written out at. Not μ = 0:
    # the square lattice is particle-hole symmetric there and χ(iν) comes out
    # real, which would hide a mistake in its imaginary part.
    probe = int(np.argmin(np.abs(MU - PROBE_MU)))
    for name, chi_iw in (
        ("square", chi_single_band(iw)),
        ("graphene", chi_two_band(iw)),
    ):
        write_table(
            EXAMPLE,
            f"{name}_matsubara",
            {
                "mu": np.full(len(wn), MU[probe]),
                "wn": wn.astype(float),
                "chi_re": chi_iw[probe].real,
                "chi_im": chi_iw[probe].imag,
            },
            comment,
        )
        write_table(
            EXAMPLE, name, {"mu": MU, "chi": matsubara_sum(basis, sampling, chi_iw)}, comment
        )

    mu, chi = analytic()
    write_table(EXAMPLE, "square_analytic", {"mu": mu, "chi": chi}, comment)


if __name__ == "__main__":
    write()
