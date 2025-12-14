SparseIR Rust Workspace
=======================

[![sparse-ir](https://img.shields.io/crates/v/sparse-ir.svg?label=sparse-ir)](https://crates.io/crates/sparse-ir)
[![sparse-ir-capi](https://img.shields.io/crates/v/sparse-ir-capi.svg?label=sparse-ir-capi)](https://crates.io/crates/sparse-ir-capi)
[![docs.rs sparse-ir](https://docs.rs/sparse-ir/badge.svg)](https://docs.rs/sparse-ir)
[![docs.rs sparse-ir-capi](https://docs.rs/sparse-ir-capi/badge.svg)](https://docs.rs/sparse-ir-capi)

High-performance Rust implementation of the sparse intermediate representation (IR) for quantum many-body physics.

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| **[Ecosystem Documentation](https://spm-lab.github.io/sparse-ir-doc/)** | Complete documentation for sparse-ir (theory, all languages) |
| **[Python/Julia Tutorials](https://spm-lab.github.io/sparse-ir-tutorial/)** | Interactive tutorials with Jupyter notebooks |
| **[Rust API: sparse-ir (docs.rs)](https://docs.rs/sparse-ir)** | Core Rust crate API documentation |
| **[Rust API: sparse-ir-capi (docs.rs)](https://docs.rs/sparse-ir-capi)** | C-API Rust crate API documentation |

## Ecosystem Overview

The sparse-ir ecosystem provides IR basis implementations for multiple languages. Most end users should start from the ecosystem docs / tutorials and use the full-featured Python/Julia libraries; this workspace focuses on Rust crates and low-level bindings.

### This Workspace

| Language | Component | Notes |
|----------|----------|-------|
| **Rust** | `sparse-ir` | High-performance core crate ([README](sparse-ir/README.md), [docs.rs](https://docs.rs/sparse-ir)) |
| **C/C++** | `sparse-ir-capi` | Rust crate providing C-compatible shared library ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi)) |
| **Fortran** | `fortran/` | Bindings via C-API ([README](fortran/README.md)) |
| **Python** | `pylibsparseir` | Thin wrapper for the C-API (not user-facing) ([README](python/README.md)) |

### Full-featured Libraries (External)

| Language | Library | Notes |
|----------|---------|-------|
| **Python** | [sparse-ir](https://github.com/SpM-lab/sparse-ir) | Recommended for most Python users |
| **Julia** | [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl) | Full-featured Julia implementation |

## Crates

- **`sparse-ir`** — Core Rust implementation of IR basis, DLR, and sampling ([README](sparse-ir/README.md), [docs.rs](https://docs.rs/sparse-ir))
- **`sparse-ir-capi`** — Rust crate providing a C-compatible API (shared library + C header) ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi))

## Sample Code & Tutorials

### Rust Examples

| Example | Description |
|---------|-------------|
| [`sparse-ir/examples/roundtrip.rs`](sparse-ir/examples/roundtrip.rs) | Complete DLR/IR/sampling cycle with round-trip tests |
| [`sparse-ir/tests/readme_sample.rs`](sparse-ir/tests/readme_sample.rs) | Basic usage examples |

```bash
cargo run --example roundtrip --release
```

### Python/Julia Tutorials (Online)

For comprehensive tutorials, see the **[online tutorial site](https://spm-lab.github.io/sparse-ir-tutorial/)** which covers:

- IR basis fundamentals
- Sparse sampling
- DLR (Discrete Lehmann Representation)
- Second-order perturbation, GW, FLEX, TPSC
- Eliashberg theory, DMFT, analytic continuation

### Python Test Scripts

Located in [`python/tutorials/`](python/tutorials/) — these scripts test `pylibsparseir` (the thin C-API wrapper) and can serve as usage examples for the low-level bindings.

### Fortran Sample

| Sample | Description |
|--------|-------------|
| [`fortran/sample/second_order_perturbation_fort.f90`](fortran/sample/second_order_perturbation_fort.f90) | Second-order perturbation theory |

### C/C++ Integration Tests

| Test | Description |
|------|-------------|
| [`cxx_tests/cinterface_core.cxx`](cxx_tests/cinterface_core.cxx) | Core C-API tests |
| [`cxx_tests/cinterface_integration.cxx`](cxx_tests/cinterface_integration.cxx) | DLR/IR round-trip tests |

## Project Structure

```
sparseir-rust/
├── sparse-ir/           # Core Rust library (crates.io: sparse-ir)
│   ├── src/             # Source code
│   ├── examples/        # Rust examples
│   └── tests/           # Integration tests
├── sparse-ir-capi/      # C-compatible API (shared library)
├── python/              # Python thin wrapper (pylibsparseir)
│   ├── pylibsparseir/   # ctypes bindings to C-API
│   └── tutorials/       # Test scripts
├── fortran/             # Fortran bindings via C-API
│   ├── src/             # Fortran modules
│   ├── sample/          # Sample programs
│   └── test/            # Test programs
├── cxx_tests/           # C/C++ integration tests
├── notebook/            # Technical notes (algorithms, design)
└── docs/                # Development documentation
```

## Build

From the workspace root:

```bash
cargo build            # build all crates in debug mode
cargo build --release  # optimized build
```

The default build uses the pure-Rust `faer` backend for matrix–matrix products.  
Faer is reasonably fast, but usually considerably slower than an optimized BLAS implementation.
To enable system BLAS (LP64) for the Rust `sparse-ir` crate at compile time, use:

```bash
cargo build -p sparse-ir --features system-blas
```

With `system-blas`, the default GEMM backend becomes BLAS at compile time. Regardless of the feature, arbitrary BLAS function pointers (LP64/ILP64) can be injected at runtime via the C API or the internal GEMM dispatcher.

## Test

### Rust Tests

```bash
cargo test --all-targets --release   # recommended for speed
```

### C++ Integration Tests

Tests C-API with different BLAS configurations (default, OpenBLAS LP64, OpenBLAS ILP64):

```bash
cd cxx_tests && ./run_with_rust_capi.sh
```

### Fortran Tests

```bash
cd fortran && ./test_with_rust_capi.sh
```

### Python Tests

```bash
cd python && uv sync && uv run pytest tests/ -v
```

### C API Benchmarks

```bash
cd capi_benchmark && ./run_with_rust_capi.sh
```

See [`.github/workflows/`](.github/workflows/) for CI configurations.

## Version Consistency Check

Check version consistency across the workspace:

```bash
python3 check_version.py
```

This script reads the canonical version from `[workspace.package]` in `Cargo.toml` and warns if Julia (`julia/build_tarballs.jl`) or Python (`python/pyproject.toml`) versions don't match.

## C API (sparse-ir-capi)

To build the C-compatible shared library and header:

```bash
cargo build -p sparse-ir-capi --release
```

See also: [README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi).

The generated library and the public header live under `sparse-ir-capi/` and can be linked from C, C++, Fortran, or other languages that can call C ABIs.

## Fortran bindings

For details on the Fortran interface and build instructions, see:

- [`fortran/README.md`](fortran/README.md)

## Related Projects

| Project | Language | Description |
|---------|----------|-------------|
| [**sparse-ir**](https://github.com/SpM-lab/sparse-ir) | Python | Full-featured Python library (recommended for Python users) |
| [**SparseIR.jl**](https://github.com/SpM-lab/SparseIR.jl) | Julia | Full-featured Julia library |
| [**libsparseir**](https://github.com/SpM-lab/libsparseir) | C++ | C++ implementation |

## References

- **sparse-ir: optimal compression and sparse sampling of many-body propagators**  
  Markus Wallerberger, Samuel Badr, Shintaro Hoshino, Fumiya Kakizawa, Takashi Koretsune, Yuki Nagai, Kosuke Nogaki, Takuya Nomoto, Hitoshi Mori, Junya Otsuki, Soshun Ozaki, Rihito Sakurai, Constanze Vogel, Niklas Witt, Kazuyoshi Yoshimi, Hiroshi Shinaoka  
  [arXiv:2206.11762](https://arxiv.org/abs/2206.11762) | [SoftwareX 21, 101266 (2023)](https://doi.org/10.1016/j.softx.2022.101266)

## License

This workspace is dual-licensed under the terms of the MIT license and the Apache License (Version 2.0).

- You may use the code in this repository under the terms of either license, at your option:
  - [MIT License](LICENSE)
  - [Apache License 2.0](LICENSE-APACHE)

Some components incorporate third-party code under Apache-2.0, such as the `col_piv_qr` module in the `sparse-ir` crate, which is based on nalgebra.  
See [`sparse-ir/README.md`](sparse-ir/README.md) and `LICENSE-APACHE` for details.
