"""
Tests for the C API independent DLR constructor and MiniPole entry points.
"""

import numpy as np
import pytest
from ctypes import c_int, c_double, byref, POINTER, c_int64

from pylibsparseir.core import _lib, COMPUTATION_SUCCESS, c_double_complex
from pylibsparseir.constants import (
    SPIR_STATISTICS_FERMIONIC,
    SPIR_STATISTICS_BOSONIC,
    SPIR_ORDER_ROW_MAJOR,
    SPIR_ORDER_COLUMN_MAJOR,
)

SPIR_NOT_SUPPORTED = -5
SPIR_INPUT_DIMENSION_MISMATCH = -3


def _dlr_independent(stat, beta, wmax, eps):
    status = c_int()
    dlr = _lib.spir_dlr_new_independent(stat, beta, wmax, eps, byref(status))
    assert status.value == COMPUTATION_SUCCESS
    return dlr


def _cptr(a):
    return a.ctypes.data_as(POINTER(c_double_complex))


def _green(beta, ns, xs, amps):
    """G[ch, i] = sum_j amps[ch, j] / (i nu_i - xs[j])."""
    z = 1j * np.pi * np.asarray(ns) / beta
    return (amps[:, None, :] / (z[None, :, None] - xs[None, None, :])).sum(axis=2)


@pytest.mark.parametrize("stat", [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC])
def test_dlr_independent_default_points(stat):
    dlr = _dlr_independent(stat, 50.0, 2.0, 1e-10)
    npoles = c_int()
    assert _lib.spir_dlr_get_npoles(dlr, byref(npoles)) == COMPUTATION_SUCCESS
    assert npoles.value > 0

    ntau = c_int()
    assert _lib.spir_basis_get_n_default_taus(dlr, byref(ntau)) == COMPUTATION_SUCCESS
    assert ntau.value == npoles.value
    taus = np.zeros(ntau.value)
    assert _lib.spir_basis_get_default_taus(
        dlr, taus.ctypes.data_as(POINTER(c_double))) == COMPUTATION_SUCCESS
    assert np.all(np.diff(taus) > 0)
    assert 0 <= taus[0] and taus[-1] <= 50.0

    nm = c_int()
    assert _lib.spir_basis_get_n_default_matsus(dlr, False, byref(nm)) == COMPUTATION_SUCCESS
    assert nm.value == npoles.value
    ns = np.zeros(nm.value, dtype=np.int64)
    assert _lib.spir_basis_get_default_matsus(
        dlr, False, ns.ctypes.data_as(POINTER(c_int64))) == COMPUTATION_SUCCESS
    zeta = 1 if stat == SPIR_STATISTICS_FERMIONIC else 0
    assert np.all(np.abs(ns) % 2 == zeta)
    _lib.spir_basis_release(dlr)


def test_dlr_independent_invalid():
    status = c_int()
    dlr = _lib.spir_dlr_new_independent(7, 10.0, 1.0, 1e-8, byref(status))
    assert not dlr
    assert status.value != COMPUTATION_SUCCESS
    dlr = _lib.spir_dlr_new_independent(SPIR_STATISTICS_FERMIONIC, -1.0, 1.0, 1e-8, byref(status))
    assert not dlr
    assert status.value != COMPUTATION_SUCCESS


def _pole_repr_contents(rep, nch):
    npoles = c_int()
    assert _lib.spir_pole_repr_get_npoles(rep, byref(npoles)) == COMPUTATION_SUCCESS
    poles = np.zeros(npoles.value, dtype=np.complex128)
    assert _lib.spir_pole_repr_get_poles(rep, _cptr(poles)) == COMPUTATION_SUCCESS
    res = np.zeros(npoles.value * nch, dtype=np.complex128)
    assert _lib.spir_pole_repr_get_residues(rep, _cptr(res)) == COMPUTATION_SUCCESS
    return poles, res


def test_minipole_from_matsubara_row_major():
    beta, wmax = 40.0, 1.0
    xs = np.array([-0.5, 0.3])
    amps = np.array([[0.4, 0.6], [0.1, 0.2]])
    ns = np.arange(-60, 60, dtype=np.int64) * 2 + 1
    g = np.ascontiguousarray(_green(beta, ns, xs, amps))  # row-major [2, nf]
    dims = (c_int * 2)(2, len(ns))
    status = c_int()
    rep = _lib.spir_minipole_from_matsubara(
        SPIR_STATISTICS_FERMIONIC, beta, wmax, 1e-12, len(ns),
        ns.ctypes.data_as(POINTER(c_int64)), SPIR_ORDER_ROW_MAJOR, 2, dims, 1,
        _cptr(g), 1e-8, 0, 0.0, 0.0, 0, byref(status))
    assert status.value == COMPUTATION_SUCCESS
    poles, res = _pole_repr_contents(rep, 2)
    np.testing.assert_allclose(poles, xs, atol=1e-6)
    np.testing.assert_allclose(res.reshape(2, 2), amps, atol=1e-6)  # row-major [2, npoles]

    resid = c_double()
    assert _lib.spir_pole_repr_get_dlr_fit_residual(rep, byref(resid)) == COMPUTATION_SUCCESS
    assert resid.value < 1e-8

    rep2 = _lib.spir_pole_repr_clone(rep)
    _lib.spir_pole_repr_release(rep)
    assert _lib.spir_pole_repr_is_assigned(rep2) == 1
    poles2, _ = _pole_repr_contents(rep2, 2)
    np.testing.assert_allclose(poles2, poles)
    _lib.spir_pole_repr_release(rep2)


def test_minipole_from_dlr_column_major():
    beta, wmax = 40.0, 1.0
    xs = np.array([-0.5, 0.3])
    amps = np.array([[0.4, 0.6], [0.1, 0.2]])
    dlr = _dlr_independent(SPIR_STATISTICS_FERMIONIC, beta, wmax, 1e-12)
    nm = c_int()
    _lib.spir_basis_get_n_default_matsus(dlr, False, byref(nm))
    ns = np.zeros(nm.value, dtype=np.int64)
    _lib.spir_basis_get_default_matsus(dlr, False, ns.ctypes.data_as(POINTER(c_int64)))

    status = c_int()
    smpl = _lib.spir_matsu_sampling_new(
        dlr, False, nm.value, ns.ctypes.data_as(POINTER(c_int64)), byref(status))
    assert status.value == COMPUTATION_SUCCESS
    g = np.asfortranarray(_green(beta, ns, xs, amps).T)  # column-major [nf, 2]
    coeffs = np.zeros((nm.value, 2), dtype=np.complex128, order="F")
    dims = (c_int * 2)(nm.value, 2)
    assert _lib.spir_sampling_fit_zz(
        smpl, None, SPIR_ORDER_COLUMN_MAJOR, 2, dims, 0, _cptr(g), _cptr(coeffs)
    ) == COMPUTATION_SUCCESS

    rep = _lib.spir_minipole_from_dlr(
        dlr, SPIR_ORDER_COLUMN_MAJOR, 2, dims, 0, _cptr(coeffs),
        1e-8, 0, 0.0, 0.0, 0, byref(status))
    assert status.value == COMPUTATION_SUCCESS
    poles, res = _pole_repr_contents(rep, 2)
    np.testing.assert_allclose(poles, xs, atol=1e-6)
    np.testing.assert_allclose(res.reshape(2, 2, order="F"), amps.T, atol=1e-6)
    resid = c_double()
    assert _lib.spir_pole_repr_get_dlr_fit_residual(rep, byref(resid)) == SPIR_NOT_SUPPORTED
    _lib.spir_pole_repr_release(rep)

    bad_dims = (c_int * 2)(nm.value + 1, 2)
    bad = np.zeros((nm.value + 1) * 2, dtype=np.complex128)
    rep = _lib.spir_minipole_from_dlr(
        dlr, SPIR_ORDER_COLUMN_MAJOR, 2, bad_dims, 0, _cptr(bad),
        1e-8, 0, 0.0, 0.0, 0, byref(status))
    assert not rep
    assert status.value == SPIR_INPUT_DIMENSION_MISMATCH

    _lib.spir_sampling_release(smpl)
    _lib.spir_basis_release(dlr)
