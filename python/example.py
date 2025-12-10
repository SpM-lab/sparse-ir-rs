# example.py
# git checkout python-interface
# cd python
# uv sync
# uv run example.py
"""
RUST_BACKTRACE=1 uv run example.py
Warning: Expecting to get 62 sampling points for corresponding basis function, instead got 40. This may happen if not enough precision is left in the polynomial.

thread '<unnamed>' (6049940) panicked at /Users/terasaki/work/atelierarith/spm-lab/sparse-ir-rs/sparse-ir/src/dlr.rs:283:9:
The number of poles must be greater than or equal to the basis size
stack backtrace:
   0: __rustc::rust_begin_unwind
   1: core::panicking::panic_fmt
   2: _spir_dlr_new
   3: _ffi_call_SYSV
   4: _ffi_call_int
   5: _ffi_call
   6: __call_function_pointer
   7: __ctypes_callproc
   8: _PyCFuncPtr_call
   9: __PyEval_EvalFrameDefault
  10: _PyEval_EvalCode
  11: _run_eval_code_obj
  12: _run_mod.llvm.9637556200777832325
  13: _pyrun_file
  14: __PyRun_SimpleFileObject
  15: __PyRun_AnyFileObject
  16: _pymain_run_file_obj
  17: _pymain_run_file
  18: _Py_RunMain
  19: _pymain_main
  20: _Py_BytesMain
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
Traceback (most recent call last):
  File "/Users/terasaki/work/atelierarith/spm-lab/sparse-ir-rs/python/example.py", line 58, in <module>
    assert dlr_status.value == COMPUTATION_SUCCESS
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
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

# Create base IR basis
statistics = SPIR_STATISTICS_BOSONIC
ir_basis = _spir_basis_new(statistics, beta, wmax, epsilon)
assert ir_basis is not None

# Create DLR using default poles
dlr_status = c_int()
dlr = _lib.spir_dlr_new(ir_basis, byref(dlr_status))
assert dlr_status.value == COMPUTATION_SUCCESS
assert dlr is not None
