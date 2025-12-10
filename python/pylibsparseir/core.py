"""
Core functionality for the SparseIR Python bindings.
"""

import os
import sys
import ctypes
from ctypes import c_int, c_double, c_int64, c_size_t, c_bool, POINTER, byref
from ctypes import CDLL
import numpy as np
import platform

# Enable only on Linux
import os
import sys
import ctypes
import platform

import os
import sys
import ctypes

from .ctypes_wrapper import spir_kernel, spir_sve_result, spir_basis, spir_funcs, spir_sampling
from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2, SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC


def _find_library():
    """Find the SparseIR shared library."""
    if sys.platform == "darwin":
        libname = "libsparseir.dylib"
    elif sys.platform == "win32":
        libname = "sparseir.dll"
    else:
        libname = "libsparseir.so"

    # Try to find the library in common locations
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "build"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "build"),
    ]

    for path in search_paths:
        libpath = os.path.join(path, libname)
        if os.path.exists(libpath):
            return libpath

    raise RuntimeError(f"Could not find {libname} in {search_paths}")


# Load the library
try:
    import scipy.linalg.cython_blas as blas
    # dgemm capsule
    # Get the PyCapsule objects for dgemm and zgemm
    capsule = blas.__pyx_capi__["dgemm"]
    capsule_z = blas.__pyx_capi__["zgemm"]

    # Get the name of the PyCapsule (optional, but safer to be explicit)
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    name = ctypes.pythonapi.PyCapsule_GetName(capsule)
    name_z = ctypes.pythonapi.PyCapsule_GetName(capsule_z)
    # Extract the pointer from the PyCapsule
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, name)
    ptr_z = ctypes.pythonapi.PyCapsule_GetPointer(capsule_z, name_z)

    _lib = ctypes.CDLL(_find_library())
    # Register both dgemm and zgemm at once using LP64 interface
    # Note: SciPy BLAS typically uses LP64 interface (32-bit integers)
    _lib.spir_register_dgemm_zgemm_lp64.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.spir_register_dgemm_zgemm_lp64.restype = None
    _lib.spir_register_dgemm_zgemm_lp64(ptr, ptr_z)
    if os.environ.get("SPARSEIR_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        print(f"[core.py] Registered SciPy BLAS dgemm @ {hex(ptr)}")
        print(f"[core.py] Registered SciPy BLAS zgemm @ {hex(ptr_z)}")
except Exception as e:
    raise RuntimeError(f"Failed to load SparseIR library: {e}")


class c_double_complex(ctypes.Structure):
    """complex is a c structure
    https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
    to use ctypes.Structure to pass structures (and, therefore, complex)
    See: https://stackoverflow.com/questions/13373291/complex-number-in-ctypes
    """
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @property
    def value(self):
        return self.real+1j*self.imag  # fields declared above

# Set up function prototypes using auto-generated bindings
try:
    from .ctypes_autogen import FUNCTIONS, c_double_complex as _autogen_c_double_complex
    # Use the generated c_double_complex if available, otherwise use the one defined below
    # (They should be identical, but we keep the local definition for backward compatibility)
except ImportError:
    # Fallback: if autogen file doesn't exist, use manual setup
    FUNCTIONS = {}
    print("WARNING: ctypes_autogen.py not found. Run tools/gen_ctypes.py to generate it.")


def _setup_prototypes():
    """Set up function prototypes from auto-generated bindings."""
    if not FUNCTIONS:
        # Fallback to manual setup if generation failed
        return
    
    # Import necessary types into local namespace for eval
    from ctypes import c_int, c_double, c_int64, c_size_t, c_bool, POINTER, c_char_p
    from .ctypes_wrapper import spir_kernel, spir_funcs, spir_basis, spir_sampling, spir_sve_result
    # Use the c_double_complex from this module (core.py), not from ctypes_autogen
    # This ensures type consistency
    
    # Type mapping for eval
    type_map = {
        'c_int': c_int, 'c_double': c_double, 'c_int64': c_int64,
        'c_size_t': c_size_t, 'c_bool': c_bool,
        'POINTER': POINTER, 'c_char_p': c_char_p,
        'spir_kernel': spir_kernel, 'spir_funcs': spir_funcs,
        'spir_basis': spir_basis, 'spir_sampling': spir_sampling,
        'spir_sve_result': spir_sve_result,
        'c_double_complex': c_double_complex,  # Use the one defined in this module
    }
    
    # Apply generated prototypes to the library
    for name, (restype_str, argtypes_list) in FUNCTIONS.items():
        if not hasattr(_lib, name):
            continue
        
        func = getattr(_lib, name)
        try:
            # Evaluate restype
            if restype_str == 'None':
                func.restype = None
            else:
                func.restype = eval(restype_str, globals(), type_map)
            
            # Evaluate argtypes
            evaluated_argtypes = []
            for argtype_str in argtypes_list:
                evaluated_argtypes.append(eval(argtype_str, globals(), type_map))
            func.argtypes = evaluated_argtypes
        except (NameError, AttributeError, SyntaxError) as e:
            # Skip functions that can't be evaluated (might be missing types)
            if os.environ.get("SPARSEIR_DEBUG", "").lower() in ("1", "true", "yes", "on"):
                print(f"WARNING: Could not set prototype for {name}: {e}")


_setup_prototypes()

# Python wrapper functions


def logistic_kernel_new(lambda_val):
    """Create a new logistic kernel."""
    status = c_int()
    kernel = _lib.spir_logistic_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create logistic kernel: {status.value}")
    return kernel


def reg_bose_kernel_new(lambda_val):
    """Create a new regularized bosonic kernel."""
    status = c_int()
    kernel = _lib.spir_reg_bose_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to create regularized bosonic kernel: {status.value}")
    return kernel


def sve_result_new(kernel, epsilon, cutoff=None, lmax=None, n_gauss=None, Twork=None):
    """Create a new SVE result.
    
    Note: cutoff parameter is deprecated and ignored (C-API doesn't have it).
    It's kept for backward compatibility but not passed to C-API.
    """
    # Validate epsilon
    if epsilon <= 0:
        raise RuntimeError(
            f"Failed to create SVE result: epsilon must be positive, got {epsilon}")

    # Note: cutoff parameter was removed from C-API, kept for backward compatibility
    if lmax is None:
        lmax = -1
    if n_gauss is None:
        n_gauss = -1
    if Twork is None:
        Twork = SPIR_TWORK_FLOAT64X2

    status = c_int()
    # C-API signature: spir_sve_result_new(kernel, epsilon, lmax, n_gauss, Twork, status)
    sve = _lib.spir_sve_result_new(
        kernel, c_double(epsilon), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create SVE result: {status.value}")
    return sve


def sve_result_get_size(sve):
    """Get the size of an SVE result."""
    size = c_int()
    status = _lib.spir_sve_result_get_size(sve, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get SVE result size: {status}")
    return size.value

def sve_result_truncate(sve, epsilon, max_size):
    """Truncate an SVE result."""
    status = c_int()
    sve = _lib.spir_sve_result_truncate(sve, epsilon, max_size, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to truncate SVE result: {status.value}")
    return sve

def sve_result_get_svals(sve):
    """Get the singular values from an SVE result."""
    size = sve_result_get_size(sve)
    svals = np.zeros(size, dtype=np.float64)
    status = _lib.spir_sve_result_get_svals(
        sve, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals

def basis_new(statistics, beta, omega_max, epsilon, kernel, sve, max_size):
    """Create a new basis."""
    status = c_int()
    basis = _lib.spir_basis_new(
        statistics, beta, omega_max, epsilon, kernel, sve, max_size, byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create basis: {status.value}")
    return basis


def basis_get_size(basis):
    """Get the size of a basis."""
    size = c_int()
    status = _lib.spir_basis_get_size(basis, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis size: {status}")
    return size.value


def basis_get_svals(basis):
    """Get the singular values of a basis."""
    size = basis_get_size(basis)
    svals = np.zeros(size, dtype=np.float64)
    status = _lib.spir_basis_get_svals(
        basis, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals


def basis_get_stats(basis):
    """Get the statistics type of a basis."""
    stats = c_int()
    status = _lib.spir_basis_get_stats(basis, byref(stats))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis statistics: {status}")
    return stats.value


def basis_get_u(basis):
    """Get the imaginary-time basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_u(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get u basis functions: {status.value}")
    return funcs


def basis_get_v(basis):
    """Get the real-frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_v(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get v basis functions: {status.value}")
    return funcs


def basis_get_uhat(basis):
    """Get the Matsubara frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_uhat(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to get uhat basis functions: {status.value}")
    return funcs


def funcs_get_size(funcs):
    """Get the size of a basis function set."""
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")
    return size.value

# TODO: Rename funcs_eval_single


def funcs_eval_single_float64(funcs, x):
    """Evaluate basis functions at a single point."""
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")

    # Prepare output array
    out = np.zeros(size.value, dtype=np.float64)

    # Evaluate
    status = _lib.spir_funcs_eval(
        funcs, c_double(x),
        out.ctypes.data_as(POINTER(c_double))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")

    return out

# TODO: Rename to funcs_eval_matsu_single


def funcs_eval_single_complex128(funcs, x):
    """Evaluate basis functions at a single point."""
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")

    # Prepare output array
    out = np.zeros(size.value, dtype=np.complex128)

    # Evaluate
    status = _lib.spir_funcs_eval_matsu(
        funcs, c_int64(x),
        out.ctypes.data_as(POINTER(c_double_complex))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")

    return out


def funcs_get_n_knots(funcs):
    """Get the number of knots of the underlying piecewise Legendre polynomial."""
    n_knots = c_int()
    status = _lib.spir_funcs_get_n_knots(funcs, byref(n_knots))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of knots: {status}")
    return n_knots.value


def funcs_get_knots(funcs):
    """Get the knots of the underlying piecewise Legendre polynomial."""
    n_knots = funcs_get_n_knots(funcs)
    knots = np.zeros(n_knots, dtype=np.float64)
    status = _lib.spir_funcs_get_knots(
        funcs, knots.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get knots: {status}")
    return knots


def basis_get_default_tau_sampling_points(basis):
    """Get default tau sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_taus(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to get number of default tau points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_taus(
        basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default tau points: {status}")

    return points


def basis_get_default_tau_sampling_points_ext(basis, n_points):
    """Get default tau sampling points for a basis."""
    points = np.zeros(n_points, dtype=np.float64)
    n_points_returned = c_int()
    status = _lib.spir_basis_get_default_taus_ext(
        basis, n_points, points.ctypes.data_as(POINTER(c_double)), byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default tau points: {status}")
    return points


def basis_get_default_omega_sampling_points(basis):
    """Get default omega (real frequency) sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_ws(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to get number of default omega points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_ws(
        basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default omega points: {status}")

    return points


def basis_get_default_matsubara_sampling_points(basis, positive_only=False):
    """Get default Matsubara sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_matsus(
        basis, c_int(1 if positive_only else 0), byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to get number of default Matsubara points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.int64)
    status = _lib.spir_basis_get_default_matsus(basis, c_int(
        1 if positive_only else 0), points.ctypes.data_as(POINTER(c_int64)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default Matsubara points: {status}")

    return points


def basis_get_n_default_matsus_ext(basis, n_points, positive_only):
    """Get the number of default Matsubara sampling points for a basis."""
    n_points_returned = c_int()
    status = _lib.spir_basis_get_n_default_matsus_ext(
        basis, c_int(1 if positive_only else 0), n_points, byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to get number of default Matsubara points: {status}")
    return n_points_returned.value


def basis_get_default_matsus_ext(basis, positive_only, points):
    n_points = len(points)
    n_points_returned = c_int()
    status = _lib.spir_basis_get_default_matsus_ext(basis, c_int(
        1 if positive_only else 0), c_int(0), n_points, points.ctypes.data_as(POINTER(c_int64)), byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default Matsubara points: {status}")
    return points


def tau_sampling_new(basis, sampling_points=None):
    """Create a new tau sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_tau_sampling_points(basis)

    sampling_points = np.asarray(sampling_points, dtype=np.float64)
    n_points = len(sampling_points)

    status = c_int()
    sampling = _lib.spir_tau_sampling_new(
        basis, n_points,
        sampling_points.ctypes.data_as(POINTER(c_double)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create tau sampling: {status.value}")

    return sampling


def _statistics_to_c(statistics):
    """Convert statistics to c type."""
    if statistics == "F":
        return SPIR_STATISTICS_FERMIONIC
    elif statistics == "B":
        return SPIR_STATISTICS_BOSONIC
    else:
        raise ValueError(f"Invalid statistics: {statistics}")


def tau_sampling_new_with_matrix(basis, statistics, sampling_points, matrix):
    """Create a new tau sampling object with a matrix."""
    status = c_int()
    sampling = _lib.spir_tau_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR,
        _statistics_to_c(statistics),
        basis.size,
        sampling_points.size,
        sampling_points.ctypes.data_as(POINTER(c_double)),
        matrix.ctypes.data_as(POINTER(c_double)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create tau sampling: {status.value}")

    return sampling


def matsubara_sampling_new(basis, positive_only=False, sampling_points=None):
    """Create a new Matsubara sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_matsubara_sampling_points(
            basis, positive_only)

    sampling_points = np.asarray(sampling_points, dtype=np.int64)
    n_points = len(sampling_points)

    status = c_int()
    sampling = _lib.spir_matsu_sampling_new(
        basis, c_bool(positive_only), n_points,
        sampling_points.ctypes.data_as(POINTER(c_int64)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to create Matsubara sampling: {status.value}")

    return sampling


def matsubara_sampling_new_with_matrix(statistics, basis_size, positive_only, sampling_points, matrix):
    """Create a new Matsubara sampling object with a matrix."""
    status = c_int()
    sampling = _lib.spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR,                           # order
        _statistics_to_c(statistics),                   # statistics
        c_int(basis_size),                              # basis_size
        c_bool(positive_only),                          # positive_only
        c_int(len(sampling_points)),                    # num_points
        sampling_points.ctypes.data_as(POINTER(c_int64)),  # points
        matrix.ctypes.data_as(POINTER(c_double_complex)),  # matrix
        byref(status)                                   # status
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(
            f"Failed to create Matsubara sampling: {status.value}")

    return sampling
