use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // With `bundled-libm` the library defines the C math functions itself
    // (src/bundled_libm.rs). On ELF targets its calls to them would still go
    // to the system libm whenever that is loaded first, as in every Julia or
    // Python process, unless the library binds its own definitions.
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if env::var_os("CARGO_FEATURE_BUNDLED_LIBM").is_some()
        && !matches!(os.as_str(), "macos" | "ios" | "windows")
    {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-Bsymbolic");
    }

    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");

    // Track source files for rebuild detection
    let src_dir = PathBuf::from("src");
    for entry in fs::read_dir(&src_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    // Check if cbindgen is available
    let cbindgen_output = Command::new("cbindgen").arg("--version").output();

    if cbindgen_output.is_err() {
        println!("cargo:warning=cbindgen not found. Install with: cargo install cbindgen");
        println!("cargo:warning=Header files will not be automatically generated.");
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let include_dir = PathBuf::from(&crate_dir).join("include");
    let sparseir_subdir = include_dir.join("sparseir");
    let sparseir_header = sparseir_subdir.join("sparseir.h");

    // Create sparseir subdirectory if it doesn't exist
    fs::create_dir_all(&sparseir_subdir).expect("Failed to create include/sparseir directory");

    // Generate sparseir.h directly with C++ compatibility (adds extern "C" blocks)
    let output = Command::new("cbindgen")
        .arg("--config")
        .arg("cbindgen.toml")
        .arg("--cpp-compat")
        .arg("--output")
        .arg(&sparseir_header)
        .current_dir(&crate_dir)
        .output()
        .expect("Failed to run cbindgen");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("cbindgen failed:\n{}", stderr);
    }
}
