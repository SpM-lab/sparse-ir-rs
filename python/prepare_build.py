#!/usr/bin/env python3
"""
Prepare build by copying necessary files from parent directory.
This avoids the need for symbolic links while ensuring all files are included in sdist.
"""

import os
import shutil
import sys
import glob
from pathlib import Path

def copy_directory_contents(src_dir, dst_dir, patterns=None):
    """Copy directory contents, optionally filtering by patterns."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        print(f"Warning: Source directory {src_path} does not exist")
        return

    dst_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.rglob("*"):
        if item.is_file():
            if patterns and not any(item.match(pattern) for pattern in patterns):
                continue

            relative_path = item.relative_to(src_path)
            dst_file = dst_path / relative_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_file)
            print(f"Copied: {item} -> {dst_file}")

def clean_old_libraries():
    """Remove old shared libraries from pylibsparseir directory."""
    script_dir = Path(__file__).parent
    pylibsparseir_dir = script_dir / "pylibsparseir"
    
    if pylibsparseir_dir.exists():
        # Remove old .dylib files
        for pattern in ["*.dylib", "*.so*"]:
            for old_lib in pylibsparseir_dir.glob(pattern):
                print(f"Removing old library: {old_lib}")
                old_lib.unlink()

def main():
    """Main function to prepare build files."""
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    print("Preparing build files...")
    
    # Clean up old shared libraries first
    clean_old_libraries()

    # Copy source files
    copy_directory_contents(
        parent_dir / "src",
        script_dir / "src",
        patterns=["*.cpp", "*.hpp", "*.h"]
    )

    # Copy include files
    copy_directory_contents(
        parent_dir / "include",
        script_dir / "include",
        patterns=["*.h", "*.hpp", "*.ipp"]
    )

    # Copy cmake files
    copy_directory_contents(
        parent_dir / "cmake",
        script_dir / "cmake"
    )

    print("Build preparation complete!")

if __name__ == "__main__":
    main()
