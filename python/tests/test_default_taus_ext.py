"""
Default tau sampling points for a requested count (sparse-ir-rs#339)
"""

from ctypes import POINTER, byref, c_double, c_int

import numpy as np

from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_STATISTICS_FERMIONIC
from pylibsparseir.core import (
    _lib,
    basis_get_default_tau_sampling_points_ext,
    basis_get_size,
    basis_new,
    logistic_kernel_new,
    sve_result_new,
)


def _fermionic_basis():
    kernel = logistic_kernel_new(10.0)
    sve = sve_result_new(kernel, 1e-6)
    return basis_new(SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, sve, -1)


def _raw_ext(basis, n_points):
    points = np.zeros(n_points, dtype=np.float64)
    n_returned = c_int()
    status = _lib.spir_basis_get_default_taus_ext(
        basis, n_points, points.ctypes.data_as(POINTER(c_double)), byref(n_returned))
    assert status == COMPUTATION_SUCCESS
    return points, n_returned.value


def test_returns_only_the_points_the_c_api_wrote():
    """A request larger than the available points must not pad with zeros."""
    basis = _fermionic_basis()
    n_request = basis_get_size(basis) + 50
    raw, n_returned = _raw_ext(basis, n_request)
    assert n_returned <= n_request

    points = basis_get_default_tau_sampling_points_ext(basis, n_request)
    assert len(points) == n_returned
    np.testing.assert_array_equal(points, raw[:n_returned])


def test_request_within_range_is_unchanged():
    basis = _fermionic_basis()
    n_request = basis_get_size(basis)
    raw, n_returned = _raw_ext(basis, n_request)
    points = basis_get_default_tau_sampling_points_ext(basis, n_request)
    np.testing.assert_array_equal(points, raw[:n_returned])
    assert len(points) == n_returned
