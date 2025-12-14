#!/usr/bin/env python3
"""
Check version consistency across the workspace.

This script:
1. Reads the canonical version from [workspace.package] in Cargo.toml
2. Warns if Julia (build_tarballs.jl) or Python (pyproject.toml) versions don't match
"""

import re
import sys
from pathlib import Path


def extract_workspace_version(cargo_toml_path: Path) -> str | None:
    """Extract version from [workspace.package] section in Cargo.toml"""
    try:
        content = cargo_toml_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: {cargo_toml_path} not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading {cargo_toml_path}: {e}", file=sys.stderr)
        return None

    # Find [workspace.package] section
    workspace_package_match = re.search(
        r"\[workspace\.package\]\s*\n(.*?)(?=\n\[|\Z)", content, re.DOTALL
    )
    if not workspace_package_match:
        print(
            f"Error: [workspace.package] section not found in {cargo_toml_path}",
            file=sys.stderr,
        )
        return None

    section_content = workspace_package_match.group(1)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', section_content)
    if not version_match:
        print(
            f"Error: version not found in [workspace.package] section",
            file=sys.stderr,
        )
        return None

    return version_match.group(1)


def extract_julia_version(build_tarballs_path: Path) -> str | None:
    """Extract version from julia/build_tarballs.jl"""
    try:
        content = build_tarballs_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None  # Julia bindings are optional
    except Exception as e:
        print(f"Warning: Error reading {build_tarballs_path}: {e}", file=sys.stderr)
        return None

    # Match: version = v"0.7.2"
    version_match = re.search(r'version\s*=\s*v"([^"]+)"', content)
    if not version_match:
        print(
            f"Warning: version not found in {build_tarballs_path}",
            file=sys.stderr,
        )
        return None

    return version_match.group(1)


def extract_python_version(pyproject_path: Path) -> str | None:
    """Extract version from python/pyproject.toml"""
    try:
        content = pyproject_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None  # Python bindings are optional
    except Exception as e:
        print(f"Warning: Error reading {pyproject_path}: {e}", file=sys.stderr)
        return None

    # Match: version = "0.7.2" in [project] section
    project_match = re.search(
        r"\[project\]\s*\n(.*?)(?=\n\[|\Z)", content, re.DOTALL
    )
    if not project_match:
        print(
            f"Warning: [project] section not found in {pyproject_path}",
            file=sys.stderr,
        )
        return None

    section_content = project_match.group(1)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', section_content)
    if not version_match:
        print(
            f"Warning: version not found in [project] section of {pyproject_path}",
            file=sys.stderr,
        )
        return None

    return version_match.group(1)


def main() -> int:
    """Main function to check version consistency"""
    script_dir = Path(__file__).parent
    workspace_cargo_toml = script_dir / "Cargo.toml"
    julia_build_tarballs = script_dir / "julia" / "build_tarballs.jl"
    python_pyproject = script_dir / "python" / "pyproject.toml"

    # Extract canonical version from workspace
    workspace_version = extract_workspace_version(workspace_cargo_toml)
    if workspace_version is None:
        return 1

    print(f"Workspace version: {workspace_version}")

    errors = []
    warnings = []

    # Check Python version (error if mismatch)
    python_version = extract_python_version(python_pyproject)
    if python_version is not None:
        if python_version != workspace_version:
            errors.append(
                f"  Python (python/pyproject.toml): {python_version} != {workspace_version}"
            )
        else:
            print(f"  ✓ Python version matches: {python_version}")
    else:
        print(f"  - Python bindings not found (skipped)")

    # Check Julia version (warning only)
    julia_version = extract_julia_version(julia_build_tarballs)
    if julia_version is not None:
        if julia_version != workspace_version:
            warnings.append(
                f"  Julia (julia/build_tarballs.jl): {julia_version} != {workspace_version}"
            )
        else:
            print(f"  ✓ Julia version matches: {julia_version}")
    else:
        print(f"  - Julia bindings not found (skipped)")

    # Print warnings if any
    if warnings:
        print()
        print("⚠ Warnings (update after release):")
        for warning in warnings:
            print(warning)

    # Print errors and fail if any
    if errors:
        print()
        print("✗ Version mismatch errors:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print()
    print("✓ All version checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
