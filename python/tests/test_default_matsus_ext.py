"""
Default Matsubara sampling points for a given basis size (sparse-ir-rs#268)
"""

from ctypes import POINTER, byref, c_bool, c_int, c_int64

import numpy as np

from pylibsparseir.constants import SPIR_INVALID_ARGUMENT, SPIR_STATISTICS_BOSONIC
from pylibsparseir.core import (
    _lib,
    basis_get_default_matsus_ext,
    basis_get_n_default_matsus_ext,
    basis_get_size,
    basis_new,
    logistic_kernel_new,
    sve_result_new,
)


def _bosonic_basis():
    kernel = logistic_kernel_new(10.0)
    sve = sve_result_new(kernel, 1e-6)
    return basis_new(SPIR_STATISTICS_BOSONIC, 10.0, 1.0, 1e-6, kernel, sve, -1)


def test_points_for_basis_size_larger_than_basis():
    """All points are returned when the basis size exceeds the basis (augmented case)."""
    basis = _bosonic_basis()
    assert basis_get_size(basis) == 10

    full = basis_get_default_matsus_ext(basis, 12, False)
    np.testing.assert_array_equal(full, [-38, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 38])

    assert basis_get_n_default_matsus_ext(basis, 12, True) == 7
    positive = basis_get_default_matsus_ext(basis, 12, True)
    np.testing.assert_array_equal(positive, [0, 2, 4, 6, 8, 12, 38])


def test_undersized_buffer_is_rejected():
    """A buffer shorter than the point set is left untouched and the count is reported."""
    basis = _bosonic_basis()
    n_total = basis_get_n_default_matsus_ext(basis, 12, False)
    sentinel = np.iinfo(np.int64).min
    points = np.full(n_total, sentinel, dtype=np.int64)
    n_required = c_int(-1)
    status = _lib.spir_basis_get_default_matsus_ext(
        basis, c_bool(False), c_bool(False), c_int(12), c_int(n_total - 1),
        points.ctypes.data_as(POINTER(c_int64)), byref(n_required))
    assert status == SPIR_INVALID_ARGUMENT
    assert n_required.value == n_total
    assert np.all(points == sentinel)
