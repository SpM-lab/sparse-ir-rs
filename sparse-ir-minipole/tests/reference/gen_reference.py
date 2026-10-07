"""Generate reference outputs of Green-Phys/MiniPole for the parity tests.

Usage:
    MINIPOLE_REF=/path/to/MiniPole python gen_reference.py

MINIPOLE_REF is a checkout of https://github.com/Green-Phys/MiniPole at
commit 15e4a54 (MIT License, Copyright (c) 2024 lzphy). Requires numpy,
scipy and kneed. Writes one `<case>.txt` per case next to this script.

File format: one array per line, `name kind n v_0 ... v_{n-1}` with kind
`r` (real), `c` (complex, written as `re im` pairs) or `i` (integer).
Matrices are flattened in C order.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.environ["MINIPOLE_REF"])
from mini_pole import ESPRIT, MiniPole, MiniPoleDLR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def write(case, **arrays):
    with open(os.path.join(HERE, case + ".txt"), "w") as f:
        for name, a in arrays.items():
            a = np.asarray(a)
            flat = a.reshape(-1)
            if np.iscomplexobj(a):
                vals = " ".join(f"{v.real:.17e} {v.imag:.17e}" for v in flat)
                kind = "c"
            elif np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
                vals = " ".join(str(int(v)) for v in flat)
                kind = "i"
            else:
                vals = " ".join(f"{v:.17e}" for v in flat)
                kind = "r"
            f.write(f"{name} {kind} {flat.size} {vals}\n")


def args(kw):
    """Keyword arguments as `arg_<name>` arrays (strings as 0/1 codes)."""
    codes = {"abs": 0, "rel": 1, "z": 0, "w": 1}
    out = {}
    for k_, v in kw.items():
        if isinstance(v, str):
            v = codes[v]
        out["arg_" + k_] = np.array([v], dtype=float if isinstance(v, float) else int)
    return out


def g_poles(z, xs, amps):
    """sum_j A_j / (z - x_j); amps has shape (r,) or (r, n, n)."""
    z = np.asarray(z).reshape(-1, *([1] * (np.ndim(amps) - 1)))
    return sum(a / (z - x) for x, a in zip(xs, amps))


rng = np.random.default_rng(20260930)

# ---------------------------------------------------------------- ESPRIT
k = np.arange(60)
nodes = np.array([0.9 * np.exp(0.3j), 0.7, 0.5 * np.exp(-1.1j)])
amps = np.array([1.0, -0.5 + 0.2j, 0.3j])
h = (amps[None, :] * nodes[None, :] ** k[:, None]).sum(axis=1)
for name, kw in [
    ("esprit_abs", dict(err=1e-10, err_type="abs")),
    ("esprit_rel", dict(err=1e-8, err_type="rel")),
    ("esprit_fixed_m", dict(M=2)),
]:
    p = ESPRIT(h, **kw)
    write(name, h=h, S=p.S, M=np.array([p.M]), gamma=p.gamma,
          omega=p.omega, err_max=np.array([p.err_max]), **args(kw))

# Two channels with noise, and a real sequence.
h2 = np.stack([h, (0.2 * nodes[None, :] ** k[:, None]).sum(axis=1)], axis=1)
h2 = h2 + 1e-9 * (rng.standard_normal(h2.shape) + 1j * rng.standard_normal(h2.shape))
kw = dict(err=1e-7, Lfactor=0.5)
p = ESPRIT(h2, **kw)
write("esprit_matrix", h=h2, S=p.S, M=np.array([p.M]), gamma=p.gamma,
      omega=p.omega, err_max=np.array([p.err_max]), **args(kw))
hr = (np.array([1.0, 0.4])[None, :] * np.array([0.8, -0.6])[None, :] ** k[:, None]).sum(axis=1)
kw = dict(err=1e-12)
p = ESPRIT(hr, **kw)
write("esprit_real", h=hr.astype(complex), S=p.S, M=np.array([p.M]), gamma=p.gamma,
      omega=p.omega, err_max=np.array([p.err_max]), **args(kw))


# ---------------------------------------------------------- MiniPoleDLR
def dlr_like(beta, xs, amps, zeta, n_grid=60, wmax=None):
    """Residues on a fixed real grid fitted to G on Matsubara frequencies."""
    wmax = wmax or 1.2 * np.max(np.abs(xs))
    xl = wmax * np.sin(0.5 * np.pi * np.linspace(-1, 1, n_grid))
    n = np.arange(-200, 200)
    iw = 1j * (2 * n + zeta) * np.pi / beta
    A = 1.0 / (iw[:, None] - xl[None, :])
    G = g_poles(iw, xs, amps).reshape(iw.size, -1)
    al = np.linalg.lstsq(A, G, rcond=-1)[0]
    return al.reshape(n_grid, *np.shape(amps)[1:]), xl


def dlr_case(name, al, xl, beta, **kw):
    p = MiniPoleDLR(al, xl, beta, **kw)
    write(name, al=al.astype(complex), xl=xl, beta=np.array([beta]),
          h_k=p.h_k, S=p.p.S, M=np.array([p.p.M]),
          pole_location=p.pole_location, pole_weight=np.asarray(p.pole_weight).astype(complex),
          **args(kw))


xs = np.array([-1.42, 0.26, 1.16])
amps = np.array([0.3, 0.5, 0.2])
# Exact input: the true poles are grid points.
xl = np.sort(np.concatenate([xs, [-1.7, -0.8, 0.0, 0.6, 1.5]]))
al = np.array([dict(zip(xs, amps)).get(x, 0.0) for x in xl])
dlr_case("dlr_exact", al, xl, 50.0, n0=2, err=1e-10)
al, xl = dlr_like(50.0, xs, amps, 1)
dlr_case("dlr_fermion", al, xl, 50.0, n0=5, err=1e-8)
dlr_case("dlr_fermion_rel", al, xl, 50.0, n0=5, nmax=30.5, err=1e-7, err_type="rel")
dlr_case("dlr_fermion_fixed_m", al, xl, 50.0, n0=5, M=3)
al, xl = dlr_like(20.0, np.array([-2.13, 0.39, 1.74]), amps, 0)
dlr_case("dlr_boson", al, xl, 20.0, n0=2, err=1e-8)
res = np.array([
    [[0.5, 0.1 - 0.2j], [0.1 + 0.2j, 0.2]],
    [[0.3, 0.0], [0.0, 0.6]],
    [[0.2, -0.1 + 0.1j], [-0.1 - 0.1j, 0.2]],
])
al, xl = dlr_like(40.0, np.array([-0.6, 0.2, 0.75]), res, 1)
dlr_case("dlr_matrix", al, xl, 40.0, n0=4, err=1e-8)
xs_sym = np.array([-0.9, -0.3, 0.3, 0.9])
amps_sym = np.array([0.2, 0.3, 0.3, 0.2])
al, xl = dlr_like(40.0, xs_sym, amps_sym, 1)
al = 0.5 * (al + al[::-1])  # exact up-down symmetry on the symmetric grid
dlr_case("dlr_symmetric", al, xl, 40.0, n0=4, err=1e-8, symmetry=True)


# ------------------------------------------------------------- MiniPole
def mats_case(name, G_w, w, **kw):
    p = MiniPole(G_w, w, **kw)
    const = np.zeros(G_w.shape[1:], dtype=complex) + p.const
    write(name, G_w=G_w.astype(complex), w=w, n0=np.array([p.n0]),
          err_max=np.array([p.err_max]), h_k=p.h_k,
          pole_location=p.pole_location, pole_weight=p.pole_weight.astype(complex),
          const=const, **args(kw))


beta = 100.0
wf = (2 * np.arange(300) + 1) * np.pi / beta
wb = 2 * np.arange(300) * np.pi / beta
noise = lambda shape, eta: eta * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
G = g_poles(1j * wf, xs, amps)
mats_case("mats_fermion_n0", G, wf, n0=3, err=1e-10)
Gn = G + noise(G.shape, 1e-7)
mats_case("mats_fermion_auto", Gn, wf, err=1e-6)
mats_case("mats_fermion_plane_w", Gn, wf, n0=3, err=1e-6, plane="w")
mats_case("mats_fermion_fixed_m", Gn, wf, n0=3, err=1e-6, M=3)
mats_case("mats_fermion_const", G + 0.05, wf, n0=3, err=1e-10, compute_const=True)
Gb = g_poles(1j * wb, xs, amps)
mats_case("mats_boson", Gb, wb, n0=2, err=1e-10)
Gm = g_poles(1j * wf, np.array([-0.6, 0.2, 0.75]), res)
mats_case("mats_matrix", Gm, wf, n0=3, err=1e-10)
mats_case("mats_matrix_gsym", Gm.real + 0j, wf, n0=3, err=1e-10, G_symmetric=True)
Gs = g_poles(1j * wf, xs_sym, amps_sym)
mats_case("mats_symmetric", Gs, wf, n0=3, err=1e-10, symmetry=True)
mats_case("mats_matrix_symmetric", Gm, wf, n0=3, err=1e-10, symmetry=True)
print("done")
