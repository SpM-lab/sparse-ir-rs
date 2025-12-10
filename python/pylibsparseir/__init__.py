"""
pylibsparseir - Python bindings for the libsparseir library

This package provides Python bindings for the libsparseir C++ library,
enabling efficient computation of sparse intermediate representation (IR) 
basis functions used in many-body physics calculations.
"""

from .core import *
from .constants import *

# Import version from package metadata
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("pylibsparseir")
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("pylibsparseir").version
    except Exception:
        __version__ = "unknown"
except PackageNotFoundError:
    # Package not installed, fallback
    __version__ = "unknown"

# Create version info tuple
try:
    __version_info__ = tuple(map(int, __version__.split(".")))
except (ValueError, AttributeError):
    __version_info__ = (0, 0, 0)
