# example.py
# git checkout python-interface
# cd python
# uv sync
# uv run example.py
"""
uv run example.py
Warning: Expecting to get 62 sampling points for corresponding basis function, instead got 40. This may happen if not enough precision is left in the polynomial.
Warning: Expecting to get 62 sampling points for corresponding basis function, instead got 40. This may happen if not enough precision is left in the polynomial.
Warning: Expecting to get 62 sampling points for corresponding basis function, instead got 40. This may happen if not enough precision is left in the polynomial.

thread '<unnamed>' (5716820) panicked at /Users/terasaki/work/atelierarith/spm-lab/sparse-ir-rs/sparse-ir/src/dlr.rs:283:9:
The number of poles must be greater than or equal to the basis size
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
"""

from ctypes import c_int, byref

from pylibsparseir.core import (
    _lib,
    logistic_kernel_new, reg_bose_kernel_new,
    sve_result_new, basis_new,
    COMPUTATION_SUCCESS
)
from pylibsparseir.ctypes_wrapper import *
from pylibsparseir.constants import *
import numpy as np
from ctypes import POINTER
def _spir_basis_new(stat, beta, wmax, epsilon):
    """Helper function to create basis directly via C API (for testing)."""
    # Create kernel
    if stat == SPIR_STATISTICS_FERMIONIC:
        kernel = logistic_kernel_new(beta * wmax)
    else:
        kernel = reg_bose_kernel_new(beta * wmax)

    # Create SVE result
    sve = sve_result_new(kernel, epsilon)

    # Create basis
    max_size = -1
    basis = basis_new(stat, beta, wmax, epsilon,kernel, sve, max_size)

    return basis

beta = 10000.0  # Large beta for better conditioning
wmax = 1.0
epsilon = 1e-12

# statistics が SPIR_STATISTICS_FERMIONIC ならば動く
statistics = SPIR_STATISTICS_BOSONIC

# Create base IR basis
ir_basis = _spir_basis_new(statistics, beta, wmax, epsilon)
assert ir_basis is not None

# Get basis size
basis_size = c_int()
status = _lib.spir_basis_get_size(ir_basis, byref(basis_size))
assert status == COMPUTATION_SUCCESS
assert basis_size.value >= 0

# Get default poles
n_default_poles = c_int()
status = _lib.spir_basis_get_n_default_ws(ir_basis, byref(n_default_poles))
assert status == COMPUTATION_SUCCESS
assert n_default_poles.value >= 0

poles = np.zeros(n_default_poles.value, dtype=np.float64)
status = _lib.spir_basis_get_default_ws(ir_basis,
                                        poles.ctypes.data_as(POINTER(c_double)))
assert status == COMPUTATION_SUCCESS

dlr_status = c_int()
dlr_ptr = _lib.spir_dlr_new_with_poles(ir_basis, n_default_poles.value,
                                        poles.ctypes.data_as(POINTER(c_double)), byref(dlr_status))
assert dlr_status.value == COMPUTATION_SUCCESS
assert dlr_ptr is not None

