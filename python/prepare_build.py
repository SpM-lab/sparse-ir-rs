#!/usr/bin/env python3
"""
Prepare build by ensuring build environment is ready.
The Rust library will be built automatically during the CMake build process.
"""

import os
from pathlib import Path

def clean_old_libraries():
    """Remove old shared libraries from pylibsparseir directory."""
    script_dir = Path(__file__).parent
    pylibsparseir_dir = script_dir / "pylibsparseir"

    if pylibsparseir_dir.exists():
        # Remove old .dylib, .so, and .dll files
        for pattern in ["*.dylib", "*.so*", "*.dll"]:
            for old_lib in pylibsparseir_dir.glob(pattern):
                print(f"Removing old library: {old_lib}")
                old_lib.unlink()

def main():
    """Main function to prepare build files."""
    script_dir = Path(__file__).parent

    print("Preparing build environment...")

    # Clean up old shared libraries
    clean_old_libraries()

    print("Build preparation complete!")
    print("Note: The Rust library will be built automatically during the CMake build process.")

if __name__ == "__main__":
    main()
