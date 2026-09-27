"""
Status codes of invalid input through the C API (one test per case)
"""

import math
from ctypes import POINTER, byref, c_double, c_int, c_int64

import numpy as np

from pylibsparseir.constants import (
    SPIR_COMPUTATION_SUCCESS,
    SPIR_INPUT_DIMENSION_MISMATCH,
    SPIR_INVALID_ARGUMENT,
    SPIR_NOT_SUPPORTED,
    SPIR_ORDER_ROW_MAJOR,
    SPIR_STATISTICS_FERMIONIC,
)
from pylibsparseir.core import _lib, c_double_complex

SPIR_TWORK_AUTO = -1


def _stand_in_kernel_matrix(nx, ny):
    """A stand-in kernel matrix of nx * ny finite entries, row-major."""
    return np.array(
        [0.1 * (1.0 + math.sin(k)) for k in range(nx * ny)], dtype=np.float64
    )


def _from_matrix(k, nx, ny, segments):
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    segs = np.asarray(segments, dtype=np.float64)
    sve = _lib.spir_sve_result_from_matrix(
        k.ctypes.data_as(POINTER(c_double)), None, nx, ny, SPIR_ORDER_ROW_MAJOR,
        segs.ctypes.data_as(POINTER(c_double)), len(segs) - 1,
        segs.ctypes.data_as(POINTER(c_double)), len(segs) - 1,
        2, 1e-8, byref(status))
    return status.value, sve


def _fermionic_basis(max_size):
    """A fermionic basis with beta = 10, omega_max = 1 and the given max_size."""
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    kernel = _lib.spir_logistic_kernel_new(10.0, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    basis = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, None, max_size,
        byref(status))
    _lib.spir_kernel_release(kernel)
    assert status.value == SPIR_COMPUTATION_SUCCESS
    return basis


def test_from_matrix_rejects_a_nan_entry():
    """S1: a kernel matrix with a NaN entry is an invalid argument."""
    k = _stand_in_kernel_matrix(4, 4)
    k[0] = math.nan
    status, sve = _from_matrix(k, 4, 4, [-1.0, 0.0, 1.0])
    assert status == SPIR_INVALID_ARGUMENT
    assert not sve


def test_from_matrix_needs_a_row_per_gauss_point():
    """S2: nx other than n_segments * n_gauss was -7 (larger) or a wrong SVE (smaller)."""
    for nx in (5, 3):
        status, sve = _from_matrix(
            _stand_in_kernel_matrix(nx, 4), nx, 4, [-1.0, 0.0, 1.0])
        assert status == SPIR_INVALID_ARGUMENT
        assert not sve


def test_the_size_and_the_accuracy_of_a_basis():
    """S3: max_size = 0 and epsilon = 1 are invalid; max_size = 1 gives one function."""
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    kernel = _lib.spir_logistic_kernel_new(10.0, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS

    empty = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, None, 0, byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not empty

    inaccurate = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1.0, kernel, None, -1, byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not inaccurate

    smallest = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, None, 1, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    size = c_int(-1)
    assert _lib.spir_basis_get_size(smallest, byref(size)) == SPIR_COMPUTATION_SUCCESS
    assert size.value == 1

    _lib.spir_basis_release(smallest)
    _lib.spir_kernel_release(kernel)


def test_the_accuracy_and_the_size_of_a_truncated_sve():
    """S4: an accuracy of 2 and a size of 0 are invalid arguments."""
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    kernel = _lib.spir_logistic_kernel_new(10.0, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    sve = _lib.spir_sve_result_new(kernel, 1e-6, -1, -1, SPIR_TWORK_AUTO, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS

    inaccurate = _lib.spir_sve_result_truncate(sve, 2.0, -1, byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not inaccurate

    empty = _lib.spir_sve_result_truncate(sve, 1e-6, 0, byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not empty

    _lib.spir_sve_result_release(sve)
    _lib.spir_kernel_release(kernel)


def test_a_basis_needs_an_sve_on_the_unit_square():
    """S5: an SVE on [-2, 2] x [-2, 2] cannot carry a basis."""
    status, sve = _from_matrix(
        _stand_in_kernel_matrix(4, 4), 4, 4, [-2.0, 0.0, 2.0])
    assert status == SPIR_COMPUTATION_SUCCESS

    kernel_status = c_int(SPIR_COMPUTATION_SUCCESS)
    kernel = _lib.spir_logistic_kernel_new(10.0, byref(kernel_status))
    assert kernel_status.value == SPIR_COMPUTATION_SUCCESS
    basis = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, sve, -1,
        byref(kernel_status))
    assert kernel_status.value == SPIR_INVALID_ARGUMENT
    assert not basis

    _lib.spir_kernel_release(kernel)
    _lib.spir_sve_result_release(sve)


def test_the_poles_of_a_dlr():
    """S6: a pole outside [-omega_max, omega_max] and a pole that is not a number."""
    basis = _fermionic_basis(1)
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    for poles in ([0.1, 2.0], [0.1, math.nan]):
        p = np.array(poles, dtype=np.float64)
        dlr = _lib.spir_dlr_new_with_poles(
            basis, 2, p.ctypes.data_as(POINTER(c_double)), byref(status))
        assert status.value == SPIR_INVALID_ARGUMENT
        assert not dlr
    _lib.spir_basis_release(basis)


def test_the_segments_of_a_gauss_legendre_rule():
    """S7: a boundary that is not a number is an invalid argument."""
    segments = np.array([0.0, math.nan], dtype=np.float64)
    x = np.zeros(4, dtype=np.float64)
    w = np.zeros(4, dtype=np.float64)
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    ret = _lib.spir_gauss_legendre_rule_piecewise_double(
        4, segments.ctypes.data_as(POINTER(c_double)), 1,
        x.ctypes.data_as(POINTER(c_double)), w.ctypes.data_as(POINTER(c_double)),
        byref(status))
    assert ret == SPIR_INVALID_ARGUMENT
    assert status.value == SPIR_INVALID_ARGUMENT


def test_a_sampling_matrix_with_a_nan_entry():
    """S8: neither a tau nor a Matsubara sampling matrix may hold a NaN."""
    status = c_int(SPIR_COMPUTATION_SUCCESS)

    taus = np.array([0.1, 0.2], dtype=np.float64)
    tau_matrix = np.array([1.0, math.nan, 1.0, 1.0], dtype=np.float64)
    tau_sampling = _lib.spir_tau_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR, SPIR_STATISTICS_FERMIONIC, 2, 2,
        taus.ctypes.data_as(POINTER(c_double)),
        tau_matrix.ctypes.data_as(POINTER(c_double)), byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not tau_sampling

    ns = np.array([1, 3], dtype=np.int64)
    matsu_matrix = np.array([1.0, math.nan, 1.0, 1.0], dtype=np.complex128)
    matsu_sampling = _lib.spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR, SPIR_STATISTICS_FERMIONIC, 2, False, 2,
        ns.ctypes.data_as(POINTER(c_int64)),
        matsu_matrix.ctypes.data_as(POINTER(c_double_complex)), byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not matsu_sampling


def test_default_matsus_need_a_definite_parity():
    """S9: a basis on an SVE from a matrix has no default Matsubara points."""
    status, sve = _from_matrix(
        _stand_in_kernel_matrix(4, 4), 4, 4, [-1.0, 0.0, 1.0])
    assert status == SPIR_COMPUTATION_SUCCESS

    basis_status = c_int(SPIR_COMPUTATION_SUCCESS)
    kernel = _lib.spir_logistic_kernel_new(10.0, byref(basis_status))
    assert basis_status.value == SPIR_COMPUTATION_SUCCESS
    basis = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 10.0, 1.0, 1e-6, kernel, sve, -1,
        byref(basis_status))
    assert basis_status.value == SPIR_COMPUTATION_SUCCESS

    n_points = c_int(-12345)
    assert _lib.spir_basis_get_n_default_matsus(
        basis, False, byref(n_points)) == SPIR_NOT_SUPPORTED
    assert n_points.value == -12345

    _lib.spir_basis_release(basis)
    _lib.spir_kernel_release(kernel)
    _lib.spir_sve_result_release(sve)


def test_a_dlr_transform_checks_the_target_extent_first():
    """S10: a wrong target extent is reported before the output is written."""
    basis = _fermionic_basis(1)
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    dlr = _lib.spir_dlr_new(basis, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS

    n_poles = c_int()
    assert _lib.spir_dlr_get_npoles(dlr, byref(n_poles)) == SPIR_COMPUTATION_SUCCESS
    assert n_poles.value > 0

    # One coefficient too many along the transformed axis
    input_dims = np.array([n_poles.value + 1], dtype=np.int32)
    values = np.ones(n_poles.value + 1, dtype=np.float64)
    sentinel = -12345.5
    out = np.full(n_poles.value + 1, sentinel, dtype=np.float64)
    assert _lib.spir_dlr2ir_dd(
        dlr, None, SPIR_ORDER_ROW_MAJOR, 1,
        input_dims.ctypes.data_as(POINTER(c_int)), 0,
        values.ctypes.data_as(POINTER(c_double)),
        out.ctypes.data_as(POINTER(c_double))) == SPIR_INPUT_DIMENSION_MISMATCH
    assert np.all(out == sentinel)

    _lib.spir_basis_release(dlr)
    _lib.spir_basis_release(basis)


def test_a_regularizer_undefined_at_half_of_omega_max():
    """S11: u of a basis with omega_max = 1 is not defined at omega_max / 2 = 5."""
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    k1 = _lib.spir_logistic_kernel_new(1.0, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    b1 = _lib.spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, 1.0, 1.0, 1e-6, k1, None, -1, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    u = _lib.spir_basis_get_u(b1, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS

    k10 = _lib.spir_logistic_kernel_new(10.0, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    sve = _lib.spir_sve_result_new(k10, 1e-6, -1, -1, SPIR_TWORK_AUTO, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS
    basis = _lib.spir_basis_new_from_sve_and_regularizer(
        SPIR_STATISTICS_FERMIONIC, 1.0, 10.0, 1e-6, 10.0, 0, 0.0, sve, u, -1,
        byref(status))
    assert status.value == SPIR_INVALID_ARGUMENT
    assert not basis

    _lib.spir_sve_result_release(sve)
    _lib.spir_kernel_release(k10)
    _lib.spir_funcs_release(u)
    _lib.spir_basis_release(b1)
    _lib.spir_kernel_release(k1)


def test_a_dlr_has_no_default_sampling_points():
    """S12: a DLR reports no default points, with success."""
    basis = _fermionic_basis(1)
    status = c_int(SPIR_COMPUTATION_SUCCESS)
    dlr = _lib.spir_dlr_new(basis, byref(status))
    assert status.value == SPIR_COMPUTATION_SUCCESS

    n_taus = c_int(-1)
    assert _lib.spir_basis_get_n_default_taus(
        dlr, byref(n_taus)) == SPIR_COMPUTATION_SUCCESS
    assert n_taus.value == 0

    n_matsus = c_int(-1)
    assert _lib.spir_basis_get_n_default_matsus(
        dlr, False, byref(n_matsus)) == SPIR_COMPUTATION_SUCCESS
    assert n_matsus.value == 0

    _lib.spir_basis_release(dlr)
    _lib.spir_basis_release(basis)
