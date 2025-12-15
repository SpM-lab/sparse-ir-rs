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


def clean_old_libraries(pylibsparseir_dir: Path, verbose: bool = True):
    """Remove old shared libraries from pylibsparseir directory."""
    if not pylibsparseir_dir.exists():
        return

    for pattern in ["*.dylib", "*.so", "*.so.*", "*.dll"]:
        for old_lib in pylibsparseir_dir.glob(pattern):
            if verbose:
                print(f"Removing old library: {old_lib}", file=sys.stderr)
            old_lib.unlink()


def build_rust_library(workspace_root: Path, verbose: bool = True):
    """Build the Rust library using cargo."""
    if verbose:
        print("Building sparse-ir-capi with cargo...", file=sys.stderr)

    # Prepare environment with cargo in PATH
    env = os.environ.copy()
    # Try multiple ways to find cargo bin directory
    home = os.environ.get("HOME") or os.path.expanduser("~")
    cargo_bin = os.path.join(home, ".cargo", "bin")

    # Add cargo bin to PATH if it exists and is not already in PATH
    if os.path.exists(cargo_bin) and cargo_bin not in env.get("PATH", ""):
        env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"
        if verbose:
            print(f"Added {cargo_bin} to PATH", file=sys.stderr)
    elif verbose:
        print(f"Cargo bin path: {cargo_bin} (exists: {os.path.exists(cargo_bin)})", file=sys.stderr)
        print(f"Current PATH: {env.get('PATH', '')[:200]}...", file=sys.stderr)

    # Run cargo build --release
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "sparse-ir-capi"],
        cwd=workspace_root,
        env=env,
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

        # Clean up old libraries before building
        clean_old_libraries(pylibsparseir_dir)

        # Build the Rust library
        build_rust_library(workspace_root)

        # Copy the library to pylibsparseir
        copy_library(workspace_root, pylibsparseir_dir)

        # Copy the header file
        copy_header(workspace_root, pylibsparseir_dir)

        print("Rust library build and copy completed.", file=sys.stderr)

        # Include the shared library in the wheel
        lib_name = get_lib_name()

        # Mark as platform-specific wheel (not pure Python)
        # This is critical for cibuildwheel to recognize this as a platform wheel
        build_data["pure_python"] = False
        # Let hatchling infer the correct platform tag
        build_data["infer_tag"] = True

        # Register the shared library as an artifact (makes wheel platform-specific)
        if "artifacts" not in build_data:
            build_data["artifacts"] = []
        build_data["artifacts"].append(f"pylibsparseir/{lib_name}")

        # Force inclusion of the library file
        if "force_include" not in build_data:
            build_data["force_include"] = {}

        lib_path = str(pylibsparseir_dir / lib_name)
        build_data["force_include"][lib_path] = f"pylibsparseir/{lib_name}"

        # Also include header
        header_path = str(pylibsparseir_dir / "sparseir.h")
        if Path(header_path).exists():
            build_data["force_include"][header_path] = "pylibsparseir/sparseir.h"
