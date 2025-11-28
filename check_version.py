#!/usr/bin/env python3
"""
Check version consistency between workspace Cargo.toml and sparse-ir-capi/Cargo.toml

This script verifies that:
1. The workspace version in Cargo.toml matches
2. The pkg-config version in sparse-ir-capi/Cargo.toml matches the workspace version
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


def extract_pkg_config_version(capi_cargo_toml_path: Path) -> str | None:
    """Extract version from [package.metadata.capi.pkg_config] section"""
    try:
        content = capi_cargo_toml_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: {capi_cargo_toml_path} not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading {capi_cargo_toml_path}: {e}", file=sys.stderr)
        return None

    # Find [package.metadata.capi.pkg_config] section
    pkg_config_match = re.search(
        r"\[package\.metadata\.capi\.pkg_config\]\s*\n(.*?)(?=\n\[|\Z)",
        content,
        re.DOTALL,
    )
    if not pkg_config_match:
        print(
            f"Error: [package.metadata.capi.pkg_config] section not found in {capi_cargo_toml_path}",
            file=sys.stderr,
        )
        return None

    section_content = pkg_config_match.group(1)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', section_content)
    if not version_match:
        print(
            f"Error: version not found in [package.metadata.capi.pkg_config] section",
            file=sys.stderr,
        )
        return None

    return version_match.group(1)


def main() -> int:
    """Main function to check version consistency"""
    script_dir = Path(__file__).parent
    workspace_cargo_toml = script_dir / "Cargo.toml"
    capi_cargo_toml = script_dir / "sparse-ir-capi" / "Cargo.toml"

    # Extract versions
    workspace_version = extract_workspace_version(workspace_cargo_toml)
    pkg_config_version = extract_pkg_config_version(capi_cargo_toml)

    if workspace_version is None or pkg_config_version is None:
        return 1

    # Check consistency
    if workspace_version != pkg_config_version:
        print(
            f"Error: Version mismatch!",
            file=sys.stderr,
        )
        print(
            f"  Workspace version (Cargo.toml): {workspace_version}",
            file=sys.stderr,
        )
        print(
            f"  pkg-config version (sparse-ir-capi/Cargo.toml): {pkg_config_version}",
            file=sys.stderr,
        )
        return 1

    print(f"✓ Version consistency check passed: {workspace_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

