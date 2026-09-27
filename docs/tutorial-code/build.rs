//! Records which version of `sparse-ir` the examples were built against, so
//! that every CSV they write can say so on its first line.

use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let workspace_manifest = Path::new(&manifest_dir).join("../../Cargo.toml");
    println!("cargo:rerun-if-changed={}", workspace_manifest.display());

    let version = read_workspace_version(&workspace_manifest).unwrap_or_else(|| {
        panic!(
            "could not find the [workspace.package] version in {}",
            workspace_manifest.display()
        )
    });
    println!("cargo:rustc-env=SPARSE_IR_VERSION={version}");
}

/// The `version` of the `[workspace.package]` table, which is what
/// `sparse-ir/Cargo.toml` inherits with `version.workspace = true`.
fn read_workspace_version(manifest: &Path) -> Option<String> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let mut in_workspace_package = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_workspace_package = line == "[workspace.package]";
            continue;
        }
        if !in_workspace_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix("version") {
            let value = rest.trim_start().strip_prefix('=')?.trim();
            return Some(value.trim_matches('"').to_string());
        }
    }
    None
}
