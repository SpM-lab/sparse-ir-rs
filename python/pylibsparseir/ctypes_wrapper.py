"""
Type definitions for the SparseIR C API.
"""

from ctypes import *
import numpy as np

__all__ = [
    "c_complex",
    "spir_kernel",
    "spir_funcs",
    "spir_basis",
    "spir_sampling",
    "spir_sve_result",
    "spir_gemm_backend",
    "c_int64",
    "COMPLEX_DTYPE",
    "DOUBLE_DTYPE",
    "INT64_DTYPE",
]

# Define complex type
c_complex = c_double * 2

# Opaque types
class _spir_kernel(Structure):
    _fields_ = []

class _spir_funcs(Structure):
    _fields_ = []

class _spir_basis(Structure):
    _fields_ = []

class _spir_sampling(Structure):
    _fields_ = []

class _spir_sve_result(Structure):
    _fields_ = []

class _spir_gemm_backend(Structure):
    _fields_ = []

class _spir_pole_repr(Structure):
    _fields_ = []

# Type aliases
spir_kernel = POINTER(_spir_kernel)
spir_funcs = POINTER(_spir_funcs)
spir_basis = POINTER(_spir_basis)
spir_sampling = POINTER(_spir_sampling)
spir_sve_result = POINTER(_spir_sve_result)
spir_gemm_backend = POINTER(_spir_gemm_backend)
spir_pole_repr = POINTER(_spir_pole_repr)

# Additional ctypes definitions
c_int64 = c_longlong

# NumPy dtype mappings
COMPLEX_DTYPE = np.dtype(np.complex128)
DOUBLE_DTYPE = np.dtype(np.float64)
INT64_DTYPE = np.dtype(np.int64)
