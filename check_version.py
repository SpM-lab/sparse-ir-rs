#!/usr/bin/env python3
"""
Check version consistency across the workspace.

This script:
1. Reads the canonical version from [workspace.package] in Cargo.toml
2. Fails if the Python (pyproject.toml) version doesn't match
3. Fails if a sparse-ir / sparse-ir-capi dependency snippet in a README
   (README.md or */README.md) pins a different version
4. Warns if the Julia (build_tarballs.jl) version doesn't match
"""

import re
import sys
from pathlib import Path

# Crates whose install snippets must advertise the workspace version.
README_CRATES = ("sparse-ir", "sparse-ir-capi")

# A Cargo dependency declaration of one of README_CRATES, in either form:
#   sparse-ir = "0.9.0"
#   sparse-ir = { version = "0.9.0", features = ["system-blas"] }
README_DEPENDENCY_PATTERN = re.compile(
    r"^[ \t]*(?P<crate>" + "|".join(map(re.escape, README_CRATES)) + r")"
    r'[ \t]*=[ \t]*(?:"(?P<plain>[^"]*)"|\{(?P<table>[^}]*)\})',
    re.MULTILINE,
)

# Only numeric versions are checked; placeholders such as "X.Y.Z" in the
# release instructions are skipped.
NUMERIC_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


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


def extract_readme_dependency_versions(
    readme_path: Path,
) -> list[tuple[int, str, str]]:
    """Extract (line, crate, version) of sparse-ir dependency snippets in a README"""
    try:
        content = readme_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Error reading {readme_path}: {e}", file=sys.stderr)
        return []

    snippets = []
    for match in README_DEPENDENCY_PATTERN.finditer(content):
        if match.group("plain") is not None:
            version = match.group("plain")
        else:
            version_match = re.search(
                r'\bversion\s*=\s*"([^"]*)"', match.group("table")
            )
            if not version_match:
                continue  # e.g. a path-only dependency
            version = version_match.group(1)

        if not NUMERIC_VERSION_PATTERN.match(version):
            continue  # placeholder such as "X.Y.Z"

        line = content.count("\n", 0, match.start("crate")) + 1
        snippets.append((line, match.group("crate"), version))

    return snippets


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

    # Check README install snippets (error if mismatch). The crate README is
    # packaged into the published crate and rendered on crates.io, so it must
    # be bumped in the same release PR as Cargo.toml, before publishing.
    readme_paths = [script_dir / "README.md", *sorted(script_dir.glob("*/README.md"))]
    readme_snippets = []
    for readme_path in readme_paths:
        if not readme_path.is_file():
            continue
        relative_path = readme_path.relative_to(script_dir).as_posix()
        for line, crate, version in extract_readme_dependency_versions(readme_path):
            readme_snippets.append((f"{relative_path}:{line}", crate, version))
    readme_mismatches = [
        f"  README ({location}): {crate} {version} != {workspace_version}"
        for location, crate, version in readme_snippets
        if version != workspace_version
    ]
    if not readme_snippets:
        print("  - README install snippets not found (skipped)")
    elif readme_mismatches:
        errors.extend(readme_mismatches)
    else:
        print(f"  ✓ README install snippets match: {len(readme_snippets)} snippet(s)")

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
