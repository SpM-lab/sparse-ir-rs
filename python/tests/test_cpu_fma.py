"""
The import refuses a library that needs FMA on a CPU without it (sparse-ir-rs#332)
"""

import pytest

from pylibsparseir.core import _check_cpu_supports_build, _lib


class _Fn:
    """A stand-in for a ctypes function returning a fixed value"""

    def __init__(self, value):
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


class _Lib:
    def __init__(self, uses_fma, has_fma):
        self.spir_build_uses_fma = _Fn(uses_fma)
        self.spir_cpu_has_fma = _Fn(has_fma)


def test_the_loaded_library_runs_on_this_cpu():
    assert _lib.spir_build_uses_fma() in (0, 1)
    assert _lib.spir_cpu_has_fma() in (0, 1)
    _check_cpu_supports_build(_lib)


@pytest.mark.parametrize("uses_fma, has_fma", [(0, 0), (0, 1), (1, 1)])
def test_supported_combinations_pass(uses_fma, has_fma):
    _check_cpu_supports_build(_Lib(uses_fma, has_fma))


def test_fma_build_on_a_cpu_without_fma_raises():
    with pytest.raises(RuntimeError, match="fused multiply-add"):
        _check_cpu_supports_build(_Lib(1, 0))


def test_a_library_without_the_check_is_accepted():
    _check_cpu_supports_build(object())
