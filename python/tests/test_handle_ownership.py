"""
Ownership of the opaque C handles created by pylibsparseir (issue #250).

Every handle a pylibsparseir function creates must be released exactly once,
with the matching ``spir_*_release``, when its owner becomes unreachable, when
``close()`` is called, or at interpreter exit, and must never be released while
it can still be used.

The C API makes every derived handle independent of its parent:
``spir_basis_new`` copies the kernel and deep-clones the SVE result, and
funcs, samplings and DLRs own their data (``Arc`` clones or freshly computed
matrices).  Parents may therefore be released before their dependents, which
``test_dependent_handles_keep_working_after_parents_are_released`` checks.
"""

import ctypes
import gc
import subprocess
import sys
import textwrap
from ctypes import POINTER, byref, c_double, c_int, c_void_p

import numpy as np
import pytest

import pylibsparseir.core as core
from pylibsparseir.core import _lib, c_double_complex
from pylibsparseir.constants import (
    COMPUTATION_SUCCESS,
    SPIR_ORDER_ROW_MAJOR,
    SPIR_STATISTICS_FERMIONIC,
)

BETA = 1.0
WMAX = 10.0
EPS = 1e-6

RELEASE_FUNCTIONS = (
    "spir_kernel_release",
    "spir_sve_result_release",
    "spir_basis_release",
    "spir_funcs_release",
    "spir_sampling_release",
    "spir_gemm_backend_release",
)


def _address(handle):
    """C address held by a handle (a raw ctypes pointer or an owning wrapper)."""
    return ctypes.cast(handle, c_void_p).value


@pytest.fixture
def releases(monkeypatch):
    """Record every successful ``spir_*_release`` call made through ``_lib``.

    Owners look their release function up on ``_lib`` when they are created,
    so handles created inside a test release through these forwarding
    recorders.  A call is recorded only after the C release returned, so a
    rejected call is not counted as a release.
    """
    calls = []
    for name in RELEASE_FUNCTIONS:
        original = getattr(_lib, name)

        def recorder(ptr, _name=name, _original=original):
            address = ctypes.cast(ptr, c_void_p).value
            _original(ptr)
            calls.append((_name, address))

        monkeypatch.setattr(_lib, name, recorder)
    return calls


def _fermionic_basis():
    kernel = core.logistic_kernel_new(BETA * WMAX)
    sve = core.sve_result_new(kernel, EPS)
    basis = core.basis_new(
        SPIR_STATISTICS_FERMIONIC, BETA, WMAX, EPS, kernel, sve, -1)
    return kernel, sve, basis


def _tau_sampling_with_matrix(basis):
    u = core.basis_get_u(basis)
    taus = core.basis_get_default_tau_sampling_points(basis)
    # Row-major (n_points, basis_size) matrix A[i, l] = u_l(tau_i).
    matrix = np.ascontiguousarray(
        [core.funcs_eval_single_float64(u, tau) for tau in taus],
        dtype=np.float64)
    return core.tau_sampling_new_with_matrix(basis, "F", taus, matrix)


def _matsubara_sampling_with_matrix(basis):
    uhat = core.basis_get_uhat(basis)
    matsus = core.basis_get_default_matsubara_sampling_points(basis, False)
    matrix = np.ascontiguousarray(
        [core.funcs_eval_single_complex128(uhat, n) for n in matsus],
        dtype=np.complex128)
    return core.matsubara_sampling_new_with_matrix(
        "F", core.basis_get_size(basis), False, matsus, matrix)


# Every pylibsparseir function that creates a handle -> (matching release,
# factory taking the parents (kernel, sve, basis)).
CONSTRUCTORS = {
    "logistic_kernel_new": (
        "spir_kernel_release",
        lambda k, s, b: core.logistic_kernel_new(BETA * WMAX)),
    "reg_bose_kernel_new": (
        "spir_kernel_release",
        lambda k, s, b: core.reg_bose_kernel_new(BETA * WMAX)),
    "sve_result_new": (
        "spir_sve_result_release",
        lambda k, s, b: core.sve_result_new(k, EPS)),
    "sve_result_truncate": (
        "spir_sve_result_release",
        lambda k, s, b: core.sve_result_truncate(s, 1e-3, 4)),
    "basis_new": (
        "spir_basis_release",
        lambda k, s, b: core.basis_new(
            SPIR_STATISTICS_FERMIONIC, BETA, WMAX, EPS, k, s, -1)),
    "basis_get_u": (
        "spir_funcs_release", lambda k, s, b: core.basis_get_u(b)),
    "basis_get_v": (
        "spir_funcs_release", lambda k, s, b: core.basis_get_v(b)),
    "basis_get_uhat": (
        "spir_funcs_release", lambda k, s, b: core.basis_get_uhat(b)),
    "tau_sampling_new": (
        "spir_sampling_release", lambda k, s, b: core.tau_sampling_new(b)),
    "tau_sampling_new_with_matrix": (
        "spir_sampling_release",
        lambda k, s, b: _tau_sampling_with_matrix(b)),
    "matsubara_sampling_new": (
        "spir_sampling_release",
        lambda k, s, b: core.matsubara_sampling_new(b)),
    "matsubara_sampling_new_with_matrix": (
        "spir_sampling_release",
        lambda k, s, b: _matsubara_sampling_with_matrix(b)),
}


@pytest.mark.parametrize("constructor", sorted(CONSTRUCTORS))
def test_handle_is_released_exactly_once_when_unreachable(constructor, releases):
    release_name, make = CONSTRUCTORS[constructor]
    parents = _fermionic_basis()
    gc.collect()

    handle = make(*parents)
    address = _address(handle)
    assert address

    del handle
    gc.collect()

    matching = [call for call in releases if call[1] == address]
    assert matching == [(release_name, address)], (
        f"{constructor}: expected exactly one {release_name}(0x{address:x}), "
        f"got {matching}")


def test_close_releases_once_and_is_idempotent(releases):
    with core.logistic_kernel_new(BETA * WMAX) as kernel:
        assert not kernel.closed
        assert kernel
        address = _address(kernel)
    assert kernel.closed
    assert not kernel

    kernel.close()  # a second close is a no-op
    del kernel
    gc.collect()
    assert releases.count(("spir_kernel_release", address)) == 1


def test_closed_handle_is_rejected_before_reaching_c(releases):
    kernel, sve, basis = _fermionic_basis()
    sve.close()
    basis.close()

    # A closed handle must never hand its (freed) address to C.
    with pytest.raises(ctypes.ArgumentError, match="already been released"):
        core.sve_result_get_size(sve)
    with pytest.raises(ctypes.ArgumentError, match="already been released"):
        core.basis_get_svals(basis)
    size = c_int()
    with pytest.raises(ctypes.ArgumentError, match="already been released"):
        _lib.spir_basis_get_size(basis, byref(size))

    # Handles that were not closed are unaffected.
    lam = c_double()
    assert _lib.spir_kernel_get_lambda(kernel, byref(lam)) == COMPUTATION_SUCCESS
    assert lam.value == BETA * WMAX


def test_raw_release_of_an_owned_handle_is_rejected(releases):
    kernel, sve, basis = _fermionic_basis()
    owned = [
        ("spir_kernel_release", kernel),
        ("spir_sve_result_release", sve),
        ("spir_basis_release", basis),
        ("spir_funcs_release", core.basis_get_v(basis)),
        ("spir_sampling_release", core.tau_sampling_new(basis)),
    ]
    for release_name, handle in owned:
        # Releasing through the raw C entry point would free the handle a
        # second time when its owner is finalized; it must be refused.
        with pytest.raises(ctypes.ArgumentError, match=r"close\(\)"):
            getattr(_lib, release_name)(handle)
        assert not handle.closed
    assert releases == []

    # The refused calls freed nothing: the handles still work ...
    np.testing.assert_array_equal(
        core.basis_get_svals(basis), core.basis_get_svals(basis))
    assert core.sve_result_get_size(sve) > 0

    # ... and each one is released exactly once by its owner.
    expected = sorted((name, _address(handle)) for name, handle in owned)
    del kernel, sve, basis, owned, handle
    gc.collect()
    assert sorted(call for call in releases if call in expected) == expected


def test_raw_handles_keep_raw_release_and_can_be_adopted(releases):
    status = c_int()
    raw = _lib.spir_logistic_kernel_new(c_double(BETA * WMAX), byref(status))
    assert status.value == COMPUTATION_SUCCESS and raw
    address = _address(raw)
    # Handles created directly through _lib are not owned by pylibsparseir;
    # their raw release keeps working.
    _lib.spir_kernel_release(raw)
    assert releases == [("spir_kernel_release", address)]
    releases.clear()

    raw = _lib.spir_logistic_kernel_new(c_double(BETA * WMAX), byref(status))
    assert status.value == COMPUTATION_SUCCESS and raw
    with pytest.raises(TypeError, match="spir_basis"):
        core.BasisHandle(raw)  # a kernel pointer is not a basis handle
    null = type(raw)()
    with pytest.raises(ValueError, match="NULL"):
        core.KernelHandle(null)
    with pytest.raises(ValueError, match="NULL"):
        core.KernelHandle(None)

    kernel = core.KernelHandle(raw)  # adopt: the owner now releases it
    with pytest.raises(TypeError, match="spir_kernel"):
        core.KernelHandle(kernel)  # an owner cannot be adopted twice
    address = _address(kernel)
    del kernel, raw
    gc.collect()
    assert releases == [("spir_kernel_release", address)]


def _eval_tau(sampling, coeffs, n_points):
    coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
    dims = np.array([coeffs.size], dtype=np.int32)
    out = np.zeros(n_points, dtype=np.float64)
    status = _lib.spir_sampling_eval_dd(
        sampling, core.get_default_blas_backend(), SPIR_ORDER_ROW_MAJOR, 1,
        dims.ctypes.data_as(POINTER(c_int)), 0,
        coeffs.ctypes.data_as(POINTER(c_double)),
        out.ctypes.data_as(POINTER(c_double)))
    assert status == COMPUTATION_SUCCESS
    return out


def _eval_matsubara(sampling, coeffs, n_points):
    coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)
    dims = np.array([coeffs.size], dtype=np.int32)
    out = np.zeros(n_points, dtype=np.complex128)
    status = _lib.spir_sampling_eval_dz(
        sampling, core.get_default_blas_backend(), SPIR_ORDER_ROW_MAJOR, 1,
        dims.ctypes.data_as(POINTER(c_int)), 0,
        coeffs.ctypes.data_as(POINTER(c_double)),
        out.ctypes.data_as(POINTER(c_double_complex)))
    assert status == COMPUTATION_SUCCESS
    return out


def test_dependent_handles_keep_working_after_parents_are_released(releases):
    kernel = core.logistic_kernel_new(BETA * WMAX)
    sve = core.sve_result_new(kernel, EPS)
    truncated = core.sve_result_truncate(sve, 1e-3, -1)
    basis = core.basis_new(
        SPIR_STATISTICS_FERMIONIC, BETA, WMAX, EPS, kernel, sve, -1)
    svals = core.basis_get_svals(basis)
    truncated_svals = core.sve_result_get_svals(truncated)
    assert np.all(svals > 0) and np.all(truncated_svals > 0)

    parents = {
        ("spir_kernel_release", _address(kernel)),
        ("spir_sve_result_release", _address(sve)),
    }
    del kernel, sve
    gc.collect()
    assert parents <= set(releases), "kernel and SVE result were not released"

    # The basis and the truncated SVE result were built from the released
    # kernel/SVE result and must be unaffected.
    np.testing.assert_array_equal(core.basis_get_svals(basis), svals)
    np.testing.assert_array_equal(
        core.sve_result_get_svals(truncated), truncated_svals)

    u = core.basis_get_u(basis)
    uhat = core.basis_get_uhat(basis)
    tau_sampling = core.tau_sampling_new(basis)
    matsubara_sampling = core.matsubara_sampling_new(basis)
    taus = core.basis_get_default_tau_sampling_points(basis)
    matsus = core.basis_get_default_matsubara_sampling_points(basis, False)
    coeffs = np.random.default_rng(250).standard_normal(core.basis_get_size(basis))

    def snapshot():
        u_at_taus = np.array(
            [core.funcs_eval_single_float64(u, tau) for tau in taus])
        uhat_at_matsus = np.array(
            [core.funcs_eval_single_complex128(uhat, n) for n in matsus])
        gtau = _eval_tau(tau_sampling, coeffs, taus.size)
        giw = _eval_matsubara(matsubara_sampling, coeffs, matsus.size)
        return u_at_taus, uhat_at_matsus, gtau, giw

    u_at_taus, uhat_at_matsus, gtau, giw = before = snapshot()
    # The samplings evaluate the same basis functions as u/uhat.  Both sides
    # form the same length-L dot products in double precision, differing only
    # in summation order (error ~ L * 1e-16 relative), so 1e-10 relative to
    # the norm is loose for rounding yet catches any wrong or garbage matrix.
    assert np.linalg.norm(gtau) > 0 and np.linalg.norm(giw.imag) > 0
    np.testing.assert_allclose(
        gtau, u_at_taus @ coeffs, rtol=0, atol=1e-10 * np.linalg.norm(gtau))
    np.testing.assert_allclose(
        giw, uhat_at_matsus @ coeffs, rtol=0, atol=1e-10 * np.linalg.norm(giw))

    basis_release = ("spir_basis_release", _address(basis))
    del basis
    gc.collect()
    assert basis_release in releases, "basis was not released"

    # Funcs and samplings derived from the released basis still evaluate to
    # exactly the same values.
    for got, expected in zip(snapshot(), before):
        np.testing.assert_array_equal(got, expected)


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=600)


def test_live_handles_are_released_once_at_interpreter_exit():
    result = _run_python("""
        import ctypes
        import pylibsparseir.core as core
        from pylibsparseir.core import _lib

        def recording(name):
            original = getattr(_lib, name)
            def release(ptr):
                address = ctypes.cast(ptr, ctypes.c_void_p).value
                original(ptr)
                print("released", name, address, flush=True)
            return release

        for name in ("spir_kernel_release", "spir_sve_result_release",
                     "spir_basis_release", "spir_funcs_release",
                     "spir_sampling_release"):
            setattr(_lib, name, recording(name))

        kernel = core.logistic_kernel_new(10.0)
        sve = core.sve_result_new(kernel, 1e-6)
        basis = core.basis_new(1, 1.0, 10.0, 1e-6, kernel, sve, -1)
        uhat = core.basis_get_uhat(basis)
        tau_sampling = core.tau_sampling_new(basis)

        class Node:
            pass

        # A handle reachable only from an uncollected reference cycle.
        node = Node()
        node.self = node
        node.sampling = core.matsubara_sampling_new(basis)

        for name, handle in [
                ("spir_kernel_release", kernel),
                ("spir_sve_result_release", sve),
                ("spir_basis_release", basis),
                ("spir_funcs_release", uhat),
                ("spir_sampling_release", tau_sampling),
                ("spir_sampling_release", node.sampling)]:
            print("created", name, ctypes.cast(handle, ctypes.c_void_p).value,
                  flush=True)
        del node, handle
    """)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert "Exception ignored" not in result.stderr, result.stderr

    def events(kind):
        return sorted(tuple(line.split()[1:]) for line in result.stdout.splitlines()
                      if line.startswith(kind + " "))

    created = events("created")
    assert len(created) == 6
    assert events("released") == created


def test_explicitly_released_default_backend_is_not_freed_again():
    result = _run_python("""
        import ctypes
        import numpy as np
        import pylibsparseir.core as core

        backend = core.get_default_blas_backend()
        core.release_blas_backend(backend)
        core.release_blas_backend(backend)  # a second release is a no-op
        assert backend.closed and not backend
        assert core.get_default_blas_backend() is backend

        # Using the released backend fails in Python instead of reading freed
        # memory in C.
        kernel = core.logistic_kernel_new(10.0)
        sve = core.sve_result_new(kernel, 1e-6)
        basis = core.basis_new(1, 1.0, 10.0, 1e-6, kernel, sve, -1)
        sampling = core.tau_sampling_new(basis)
        coeffs = np.ones(core.basis_get_size(basis))
        dims = np.array([coeffs.size], dtype=np.int32)
        out = np.zeros(len(core.basis_get_default_tau_sampling_points(basis)))
        try:
            core._lib.spir_sampling_eval_dd(
                sampling, backend, 0, 1,
                dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 0,
                coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        except ctypes.ArgumentError as err:
            assert "already been released" in str(err), err
        else:
            raise AssertionError("released backend reached C")
        print("done", flush=True)
    """)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert result.stdout.strip() == "done"
    assert "Traceback" not in result.stderr, result.stderr
    assert "Exception ignored" not in result.stderr, result.stderr
