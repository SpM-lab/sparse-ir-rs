"""Custom build hook for hatchling that builds the Rust library."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_lib_extension():
    """Get the library extension for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "dylib"
    elif system == "Windows":
        return "dll"
    else:
        return "so"


def get_lib_name():
    """Get the library name for the current platform."""
    ext = get_lib_extension()
    return f"libsparse_ir_capi.{ext}"


def build_rust_library(workspace_root: Path, verbose: bool = True):
    """Build the Rust library using cargo."""
    if verbose:
        print("Building sparse-ir-capi with cargo...", file=sys.stderr)

    # Run cargo build --release
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "sparse-ir-capi"],
        cwd=workspace_root,
        capture_output=not verbose,
        text=True,
    )

    if result.returncode != 0:
        if not verbose and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Cargo build failed with return code {result.returncode}")

    if verbose:
        print("Cargo build completed successfully.", file=sys.stderr)


def copy_library(workspace_root: Path, dest_dir: Path, verbose: bool = True):
    """Copy the built library to the destination directory."""
    lib_name = get_lib_name()
    src_path = workspace_root / "target" / "release" / lib_name
    dest_path = dest_dir / lib_name

    if not src_path.exists():
        raise RuntimeError(f"Built library not found at {src_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Copying {src_path} -> {dest_path}", file=sys.stderr)

    shutil.copy2(src_path, dest_path)

    return dest_path


def copy_header(workspace_root: Path, dest_dir: Path, verbose: bool = True):
    """Copy the header file to the destination directory."""
    src_path = workspace_root / "sparse-ir-capi" / "include" / "sparseir" / "sparseir.h"
    dest_path = dest_dir / "sparseir.h"

    if src_path.exists():
        if verbose:
            print(f"Copying {src_path} -> {dest_path}", file=sys.stderr)
        shutil.copy2(src_path, dest_path)
    else:
        if verbose:
            print(f"Header file not found at {src_path}, skipping.", file=sys.stderr)


class CustomBuildHook(BuildHookInterface):
    """Custom build hook that builds the Rust library during installation."""

    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        """Initialize the build hook - runs before the build."""
        # Only run for editable and wheel builds
        if self.target_name not in ("wheel", "editable"):
            return

        # Get paths
        python_dir = Path(self.root)
        workspace_root = python_dir.parent
        pylibsparseir_dir = python_dir / "pylibsparseir"

        print(f"Building Rust library (target: {self.target_name})...", file=sys.stderr)

        # Build the Rust library
        build_rust_library(workspace_root)

        # Copy the library to pylibsparseir
        copy_library(workspace_root, pylibsparseir_dir)

        # Copy the header file
        copy_header(workspace_root, pylibsparseir_dir)

        print("Rust library build and copy completed.", file=sys.stderr)

        # Include the shared library in the wheel
        lib_name = get_lib_name()
        if "shared_data" not in build_data:
            build_data["shared_data"] = {}

        # Force inclusion of the library file
        if "force_include" not in build_data:
            build_data["force_include"] = {}

        lib_path = str(pylibsparseir_dir / lib_name)
        build_data["force_include"][lib_path] = f"pylibsparseir/{lib_name}"

        # Also include header
        header_path = str(pylibsparseir_dir / "sparseir.h")
        if Path(header_path).exists():
            build_data["force_include"][header_path] = "pylibsparseir/sparseir.h"
