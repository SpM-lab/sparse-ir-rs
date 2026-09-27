"""Reference values for the `second_order_perturbation` example.

This is the computation of `second_order_perturbation_py.ipynb` of
sparse-ir-tutorial, with the plotting removed and the numbers written out.

The Python implementation reports its sampling times on `[0, β)`; the Rust one
reports them on `[−β/2, β/2]`. Everything that depends on τ is therefore
carried across with `fold_tau` before it is written, so that the reference file
and the example's output are indexed the same way.
"""

from __future__ import annotations

import numpy as np
import sparse_ir
from numpy.fft import fftn, ifftn

from reference_common import fold_tau, provenance, write_table

EXAMPLE = "second_order_perturbation"
LAMBDA = 1e5
BETA = 1e3
EPS = 1e-7
NK_LIN = 256
U = 2.0
WMAX = LAMBDA / BETA

# The last cell of the notebook evaluates Σ on frequencies of its own choosing,
# far outside the sampling set: every tenth fermionic frequency out to
# |n| = 20000.
FREQUENCIES = 2 * np.arange(-10000, 10000, 20) + 1


def write() -> None:
    comment = provenance(EXAMPLE)

    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, eps=EPS)
    smpl_tau = sparse_ir.TauSampling(basis)
    smpl_matsu = sparse_ir.MatsubaraSampling(basis)
    tau, sign, order = fold_tau(smpl_tau.sampling_points, BETA, "F")

    kps = (NK_LIN, NK_LIN)
    nk = int(np.prod(kps))
    kgrid = [2 * np.pi * np.arange(kp) / kp for kp in kps]
    k1, k2 = np.meshgrid(*kgrid, indexing="ij")
    ek = -2 * (np.cos(k1) + np.cos(k2))

    # G₀(iν, k) = 1/(iν − ε(k)), at half filling with μ = U/2 absorbed into ε.
    iv = 1j * np.pi * smpl_matsu.sampling_points / BETA
    gkf = 1.0 / (iv[:, None] - ek.ravel()[None, :])
    gkl = smpl_matsu.fit(gkf, axis=0)
    gkt = smpl_tau.evaluate(gkl)

    # G(τ, r), and the second-order self-energy on the sampling times. The
    # Python sampling times lie in [0, β), so `grt[::-1]` is G(β − τ, r).
    grt = fftn(gkt.reshape(len(tau), *kps), axes=(1, 2)).reshape(len(tau), nk) / nk
    srt = U * U * grt * grt * grt[::-1, :]

    srl = smpl_tau.fit(srt)
    skl = (ifftn(srl.reshape(basis.size, *kps), axes=(1, 2)) * nk).reshape(basis.size, nk)
    sigma_iv = smpl_matsu.evaluate(skl, axis=0)
    sigma_far = sparse_ir.MatsubaraSampling(basis, FREQUENCIES).evaluate(skl, axis=0)

    # Γ = (0, 0) is the first point of the grid, M = (π, π) the middle one.
    gamma = 0
    m_point = (NK_LIN // 2) * NK_LIN + NK_LIN // 2

    write_table(
        EXAMPLE,
        "summary",
        {
            "beta": np.array([BETA]),
            "wmax": np.array([WMAX]),
            "eps": np.array([EPS]),
            "lambda": np.array([LAMBDA]),
            "u": np.array([U]),
            "nk_lin": np.array([float(NK_LIN)]),
            "basis_size": np.array([float(basis.size)]),
            "n_tau": np.array([float(len(tau))]),
            "n_matsubara": np.array([float(smpl_matsu.sampling_points.size)]),
            "cond_tau": np.array([smpl_tau.cond]),
            "cond_matsubara": np.array([smpl_matsu.cond]),
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "green_matsubara",
        {
            "n": smpl_matsu.sampling_points.astype(float),
            "nu": iv.imag,
            "g_gamma_im": gkf[:, gamma].imag,
            "g_gamma_re": gkf[:, gamma].real,
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "green_coefficients",
        {
            "l": np.arange(basis.size, dtype=float),
            "s_l": basis.s,
            "g_gamma_abs": np.abs(gkl[:, gamma]),
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "green_tau",
        {
            "tau": tau,
            "g_gamma": (sign * gkt[order, gamma]).real,
            "g_m": (sign * gkt[order, m_point]).real,
            "g_origin": (sign * grt[order, 0]).real,
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "self_energy_tau",
        {
            "tau": tau,
            # Σ is a product of three fermionic G's, so it picks up the same
            # sign as one of them under τ → τ − β.
            "sigma_origin": (sign * srt[order, 0]).real,
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "self_energy_coefficients",
        {
            "l": np.arange(basis.size, dtype=float),
            "s_l": basis.s,
            "sigma_origin_abs": np.abs(srl[:, 0]),
            "sigma_gamma_abs": np.abs(skl[:, gamma]),
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "self_energy_matsubara",
        {
            "n": smpl_matsu.sampling_points.astype(float),
            "nu": iv.imag,
            "sigma_gamma_im": sigma_iv[:, gamma].imag,
            "sigma_gamma_re": sigma_iv[:, gamma].real,
        },
        comment,
    )

    write_table(
        EXAMPLE,
        "self_energy_far",
        {
            "n": FREQUENCIES.astype(float),
            "nu": FREQUENCIES * (np.pi / BETA),
            "sigma_gamma_im": sigma_far[:, gamma].imag,
            "sigma_gamma_re": sigma_far[:, gamma].real,
        },
        comment,
    )
