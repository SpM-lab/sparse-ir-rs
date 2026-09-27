"""Reference values for the `liechtenstein` example.

This is the computation of `liechtenstein_py.ipynb` of sparse-ir-tutorial,
with the plotting removed and the numbers written out. The notebook evaluates
the exchange interactions of a square-lattice tight-binding model in the
Liechtenstein formulation, once on a naive Matsubara grid of increasing size
and once through the basis; the point of the example is that the second
answer is the limit of the first at a fraction of the cost.

Nothing lives on a sampling time here, so no folding is needed: everything is
either a Matsubara sum or a real-space array.
"""

from __future__ import annotations

import numpy as np
import sparse_ir

from reference_common import provenance, write_table

EXAMPLE = "liechtenstein"

T_HOPPING = 1.0
BETA = 50.0
NK_LIN = 36
B_EFF = 3.0
EPS = 1e-7
LAMBDA = 2 * max(8 * T_HOPPING, B_EFF) * BETA
# The chemical potentials J₀ is scanned over, and the sizes of the naive
# Matsubara grids it is compared against.
MU = np.linspace(-10, 10, 41)
NAIVE = (100, 200, 400, 800, 1600)


def dispersion() -> np.ndarray:
    k1, k2 = np.meshgrid(
        np.arange(NK_LIN) / NK_LIN, np.arange(NK_LIN) / NK_LIN, indexing="ij"
    )
    return -2 * T_HOPPING * (np.cos(2 * np.pi * k1) + np.cos(2 * np.pi * k2))


def j0(basis, sampling, ek: np.ndarray, nm: int | None) -> np.ndarray:
    """J₀ at every chemical potential in `MU`.

    With `nm` given, the Matsubara sum runs over `2 nm` frequencies and is
    truncated; with `nm = None` it is evaluated as `G⁺G⁻` at τ = 0 through the
    basis, which is the whole sum.
    """
    if nm is None:
        iw = 1j * np.pi * sampling.sampling_points / BETA
    else:
        iw = 1j * np.pi * (2 * np.arange(-nm, nm) + 1) / BETA

    # The occupation-number term, which has nothing to do with the frequency
    # grid and is therefore the same in both branches.
    n_up = 0.5 * (1 - np.tanh(0.5 * BETA * (ek[None] - MU[:, None, None] - B_EFF)))
    n_dn = 0.5 * (1 - np.tanh(0.5 * BETA * (ek[None] - MU[:, None, None] + B_EFF)))
    first = 0.5 * B_EFF * np.mean(n_up - n_dn, axis=(1, 2))

    shifted = ek[None, None] - MU[None, :, None, None]
    g_up = 1 / (iw[:, None, None, None] - (shifted - B_EFF))
    g_dn = 1 / (iw[:, None, None, None] - (shifted + B_EFF))
    second_w = B_EFF**2 * np.mean(g_up, axis=(2, 3)) * np.mean(g_dn, axis=(2, 3))

    if nm is None:
        second = basis.u(0) @ sampling.fit(second_w.reshape(len(iw), len(MU)))
    else:
        second = np.sum(second_w, axis=0) / BETA
    return np.real(first + second)


def jij(basis, sampling, ek: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """J_ij at one chemical potential, and the distance of each site."""
    iw = 1j * np.pi * sampling.sampling_points / BETA
    g_up = 1 / (iw[:, None, None] - (ek[None] - B_EFF - mu))
    g_dn = 1 / (iw[:, None, None] - (ek[None] + B_EFF - mu))
    # G_ij carries e^{−ikr}, G_ji carries e^{+ikr}; both are averages over the
    # zone, so both transforms bring a 1/N.
    gr_up = np.fft.fftn(g_up, axes=(1, 2)) / NK_LIN**2
    gr_dn = np.fft.ifftn(g_dn, axes=(1, 2))
    jij_w = -(B_EFF**2) * gr_up * gr_dn

    values = np.real(basis.u(0) @ sampling.fit(jij_w.reshape(len(iw), NK_LIN**2)))
    values[0] = 0.0  # the on-site term is not an exchange interaction
    distance = np.linalg.norm(
        np.array(np.meshgrid(np.arange(NK_LIN), np.arange(NK_LIN), indexing="ij")),
        axis=0,
    ).flatten()
    return values, distance


def write() -> None:
    comment = provenance(EXAMPLE)

    basis = sparse_ir.FiniteTempBasis("F", BETA, LAMBDA / BETA, eps=EPS)
    sampling = sparse_ir.MatsubaraSampling(basis)
    ek = dispersion()

    write_table(
        EXAMPLE,
        "summary",
        {
            "t": np.array([T_HOPPING]),
            "beta": np.array([BETA]),
            "wmax": np.array([LAMBDA / BETA]),
            "eps": np.array([EPS]),
            "lambda": np.array([LAMBDA]),
            "b_eff": np.array([B_EFF]),
            "nk_lin": np.array([NK_LIN]),
            "basis_size": np.array([basis.size]),
            "n_matsubara": np.array([len(sampling.sampling_points)]),
        },
        comment,
    )

    columns = {"mu": MU, "j0": j0(basis, sampling, ek, None)}
    for nm in NAIVE:
        columns[f"j0_nm{nm}"] = j0(basis, sampling, ek, nm)
    write_table(EXAMPLE, "j0", columns, comment)

    values, distance = jij(basis, sampling, ek, 0.0)
    write_table(
        EXAMPLE,
        "jij",
        {"distance": distance, "j_ij": values},
        comment,
    )

    # The sum rule of the notebook's last cell: J₀ from the two-site exchange
    # interactions must be the J₀ computed directly, at the same μ.
    write_table(
        EXAMPLE,
        "sum_rule",
        {
            "mu": np.array([0.0]),
            "j0_direct": np.array([columns["j0"][len(MU) // 2]]),
            "j0_from_jij": np.array([values.sum()]),
        },
        comment,
    )


if __name__ == "__main__":
    write()
