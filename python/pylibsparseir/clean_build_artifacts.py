#!/usr/bin/env python3
"""
Clean build artifacts and files copied by setup_build.py and included via MANIFEST.in
"""

import os
import shutil
import glob

def clean_build_artifacts():
    """Remove build artifacts and copied files."""
    current_dir = os.getcwd()

    # Directories and files to remove (from setup_build.py and build artifacts)
    items_to_remove = [
        'build',
        'dist',
        '*.egg-info',
        'include',
        'src',
        'fortran',
        'cmake',
    ]

    # Files in pylibsparseir to remove
    pylibsparseir_patterns = [
        'pylibsparseir/*.so',
        'pylibsparseir/*.dylib',
        'pylibsparseir/*.dll',
        'pylibsparseir/__pycache__'
    ]

    print(f"Cleaning build artifacts in: {current_dir}")

    # Remove main items
    for item in items_to_remove:
        if '*' in item:
            # Handle glob patterns
            for path in glob.glob(item):
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed directory: {path}")
                    else:
                        os.remove(path)
                        print(f"Removed file: {path}")
        else:
            # Handle direct paths
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"Removed file: {item}")

    # Remove pylibsparseir artifacts
    for pattern in pylibsparseir_patterns:
        for path in glob.glob(pattern):
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    print(f"Removed file: {path}")

    print("Clean completed!")

if __name__ == "__main__":
    clean_build_artifacts()
