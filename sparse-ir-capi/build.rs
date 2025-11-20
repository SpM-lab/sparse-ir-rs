use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
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
    let cbindgen_output = Command::new("cbindgen")
        .arg("--version")
        .output();

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
    fs::create_dir_all(&sparseir_subdir)
        .expect("Failed to create include/sparseir directory");

    // Generate sparseir.h directly
    let output = Command::new("cbindgen")
        .arg("--config")
        .arg("cbindgen.toml")
        .arg("--output")
        .arg(&sparseir_header)
        .current_dir(&crate_dir)
        .output()
        .expect("Failed to run cbindgen");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("cbindgen failed:\n{}", stderr);
    }

    // Post-process to add header guard and fix header comment
    add_header_guard(&sparseir_header);
}

fn add_header_guard(header_path: &PathBuf) {
    let mut content = fs::read_to_string(header_path)
        .expect("Failed to read generated header");

    // Replace StatusCode with int
    content = content.replace("StatusCode", "int");
    content = content.replace("typedef int int;", ""); // Remove redundant typedef if generated

    // Remove Complex64 struct definition and its comment
    // First, remove the comment block before typedef struct Complex64
    if let Some(comment_start) = content.find("/**\n * Complex number type for C API") {
        if let Some(comment_end) = content[comment_start..].find("*/\n") {
            let comment_end_pos = comment_start + comment_end + "*/\n".len();
            content.replace_range(comment_start..comment_end_pos, "");
        }
    }
    
    // Then remove the typedef struct Complex64 { ... } definition
    if let Some(start) = content.find("typedef struct Complex64") {
        if let Some(end) = content[start..].find("} Complex64;") {
            let end_pos = start + end + "} Complex64;".len();
            // Remove the Complex64 struct definition
            content.replace_range(start..end_pos, "");
        }
    }
    
    // Replace all references to "struct Complex64" with "c_complex"
    content = content.replace("struct Complex64", "c_complex");

    // Remove #ifndef/#define/#endif guard if present
    if let Some(ifndef_pos) = content.find("#ifndef") {
        if let Some(define_pos) = content.find("#define") {
            // Remove #ifndef and #define lines
            let after_define = content[define_pos..].lines().skip(1).collect::<Vec<_>>();
            content = format!("{}\n{}", &content[..ifndef_pos], after_define.join("\n"));
        }
    }
    if let Some(endif_pos) = content.find("#endif") {
        // Remove #endif line and anything after it if it's the guard
        if content[endif_pos..].contains("SPARSEIR_H") {
            content = content[..endif_pos].trim_end().to_string();
        }
    }
    
    // Add header comment, #pragma once, and c_complex typedef
    let header_comment = r#"/**
 * @file sparseir.h
 * @brief C API for SparseIR library
 *
 * This header provides C-compatible interface for the SparseIR library.
 * Compatible with libsparseir C API.
 *
 * This header is automatically generated from Rust source code using cbindgen.
 * Do not edit manually - changes will be overwritten on next build.
 */

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_MSC_VER) || defined(__cplusplus)
// MSVC doesn't support C99 complex types by default
// For C++ compilation, use std::complex to avoid C99 extension warnings
#include <complex>
typedef std::complex<double> c_complex;
#else
#include <complex.h>
// Define a C-compatible type alias for the C99 complex number.
typedef double _Complex c_complex;
#endif

"#;

    // Find where the includes are and insert after them
    let includes_end = if let Some(pos) = content.find("#include <stdlib.h>") {
        // Find the end of this line
        if let Some(newline) = content[pos..].find('\n') {
            pos + newline + 1
        } else {
            pos + "#include <stdlib.h>".len()
        }
    } else {
        0
    };
    
    // Remove existing includes from content
    let content_without_includes = if includes_end > 0 {
        content[includes_end..].trim_start().to_string()
    } else {
        content
    };

    // Find where function declarations start (after typedefs and struct definitions)
    // Look for the first function declaration (starts with return type like "int ", "void ", "struct ")
    let mut func_start = 0;
    let lines: Vec<&str> = content_without_includes.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Check if this line starts a function declaration (not a typedef or struct definition)
        if (trimmed.starts_with("int ") || trimmed.starts_with("void ") || trimmed.starts_with("struct ") || trimmed.starts_with("spir_"))
            && !trimmed.starts_with("typedef")
            && !trimmed.starts_with("struct _")
            && trimmed.contains('(') {
            func_start = i;
            break;
        }
    }

    // Split content into before functions and functions
    let before_funcs = if func_start > 0 {
        lines[..func_start].join("\n")
    } else {
        content_without_includes.clone()
    };

    let funcs_and_after = if func_start > 0 {
        lines[func_start..].join("\n")
    } else {
        String::new()
    };

    // Add extern "C" block around function declarations
    let extern_c_start = if !funcs_and_after.is_empty() {
        "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
    } else {
        ""
    };

    let extern_c_end = if !funcs_and_after.is_empty() {
        "\n\n#ifdef __cplusplus\n}\n#endif\n"
    } else {
        ""
    };

    let new_content = format!("{}{}{}{}{}", header_comment, before_funcs, extern_c_start, funcs_and_after, extern_c_end);

    fs::write(header_path, new_content)
        .expect("Failed to write sparseir.h");
}
