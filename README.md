SparseIR Rust Workspace
=======================

[![sparse-ir](https://img.shields.io/crates/v/sparse-ir.svg?label=sparse-ir)](https://crates.io/crates/sparse-ir)
[![sparse-ir-capi](https://img.shields.io/crates/v/sparse-ir-capi.svg?label=sparse-ir-capi)](https://crates.io/crates/sparse-ir-capi)
[![docs.rs sparse-ir](https://docs.rs/sparse-ir/badge.svg)](https://docs.rs/sparse-ir)
[![docs.rs sparse-ir-capi](https://docs.rs/sparse-ir-capi/badge.svg)](https://docs.rs/sparse-ir-capi)

High-performance Rust implementation of the sparse intermediate representation (IR) for quantum many-body physics.

Most end users should start from the ecosystem documentation / tutorials and use the full-featured Python/Julia libraries; this workspace focuses on Rust crates and low-level bindings.

## Documentation

**Start here:** **[Ecosystem Documentation](https://spm-lab.github.io/sparse-ir-doc/)** (theory + usage across all languages)

| Resource | Description |
|----------|-------------|
| **[Python/Julia Tutorials](https://spm-lab.github.io/sparse-ir-tutorial/)** | Interactive tutorials with Jupyter notebooks |
| **[Rust API: sparse-ir (docs.rs)](https://docs.rs/sparse-ir)** | Core Rust crate API documentation |
| **[Rust API: sparse-ir-capi (docs.rs)](https://docs.rs/sparse-ir-capi)** | C-API Rust crate API documentation |

## Quick start (Rust)

- Use the Rust API docs: **[docs.rs/sparse-ir](https://docs.rs/sparse-ir)**
- Run the round-trip example (DLR/IR/sampling):

```bash
cargo run --example roundtrip --release
```

## What’s in this workspace

Rust users typically depend on the crates below. Users of other languages typically use the bindings (via the C API).

### Rust crates

- **`sparse-ir`** — Core Rust implementation of IR basis, DLR, and sampling ([README](sparse-ir/README.md), [docs.rs](https://docs.rs/sparse-ir))
- **`sparse-ir-capi`** — Rust crate providing a C-compatible API (shared library + C header) ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi))

### Bindings (other languages)

| Language | Component | Notes |
|----------|----------|-------|
| **C/C++** | `sparse-ir-capi` | C-compatible shared library ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi)) |
| **Fortran** | `fortran/` | Bindings via C-API ([README](fortran/README.md)) |
| **Python** | `pylibsparseir` | Thin wrapper for the C-API (not user-facing) ([README](python/README.md)) |

## Full-featured wrappers (external)

Note: the name **sparse-ir** is used both for the Rust crate (`sparse-ir`) and the full-featured Python project below.

| Project | Language | Notes |
|---------|----------|-------|
| [**sparse-ir (Python)**](https://github.com/SpM-lab/sparse-ir) | Python | Full-featured Python library (recommended for most users) |
| [**SparseIR.jl (Julia)**](https://github.com/SpM-lab/SparseIR.jl) | Julia | Full-featured Julia implementation |

## Examples

### Rust examples

| Example | Description |
|---------|-------------|
| [`sparse-ir/examples/roundtrip.rs`](sparse-ir/examples/roundtrip.rs) | Complete DLR/IR/sampling cycle with round-trip tests |
| [`sparse-ir/examples/basisgen.rs`](sparse-ir/examples/basisgen.rs) | Basis generation and sampling point validation tests |
| [`sparse-ir/tests/readme_examples.rs`](sparse-ir/tests/readme_examples.rs) | Basic usage examples |

### Fortran examples

| Example | Description |
|---------|-------------|
| [`fortran/examples/second_order_perturbation_fort.f90`](fortran/examples/second_order_perturbation_fort.f90) | Second-order perturbation theory |

### C/C++ integration tests

| Test | Description |
|------|-------------|
| [`cxx_tests/cinterface_core.cxx`](cxx_tests/cinterface_core.cxx) | Core C-API tests |
| [`cxx_tests/cinterface_integration.cxx`](cxx_tests/cinterface_integration.cxx) | DLR/IR round-trip tests |

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

---

## Development

### Project structure

```
sparseir-rust/
├── sparse-ir/           # Core Rust library (crates.io: sparse-ir)
│   ├── src/             # Source code
│   ├── examples/        # Rust examples
│   └── tests/           # Integration tests
├── sparse-ir-capi/      # C-compatible API (shared library)
├── python/              # Python thin wrapper (pylibsparseir)
│   ├── pylibsparseir/   # ctypes bindings to C-API
│   └── tests/           # Python tests
├── fortran/             # Fortran bindings via C-API
│   ├── src/             # Fortran modules
│   ├── examples/        # Example programs
│   └── test/            # Test programs
├── cxx_tests/           # C/C++ integration tests
├── capi_benchmark/      # C-API benchmarks
├── notebook/            # Technical notes (algorithms, design)
└── docs/                # Development documentation
```

### Build

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

### Test

#### Rust tests

```bash
cargo test --all-targets --release   # recommended for speed
```

#### C++ integration tests

Tests C-API with different BLAS configurations (default, OpenBLAS LP64, OpenBLAS ILP64):

```bash
cd cxx_tests && ./run_with_rust_capi.sh
```

#### Fortran tests

```bash
cd fortran && ./test_with_rust_capi.sh
```

#### Python tests

```bash
cd python && uv sync && uv run pytest tests/ -v
```

See [`.github/workflows/`](.github/workflows/) for CI configurations.

### Benchmarks

#### C API benchmarks

```bash
cd capi_benchmark && ./run_with_rust_capi.sh
```

### Version management

#### Version consistency check

Check version consistency across the workspace:

```bash
python3 check_version.py
```

This script reads the canonical version from `[workspace.package]` in `Cargo.toml` and warns if Julia (`julia/build_tarballs.jl`) or Python (`python/pyproject.toml`) versions don't match.

#### Releasing a new version

The release process is done in **two stages** because Julia bindings depend on the published crates.io version:

**Stage 1: Rust + Python version bump**

1. Update the version in `Cargo.toml`:
   ```toml
   [workspace.package]
   version = "0.8.0"  # Update this
   
   [workspace.dependencies]
   sparse-ir = { version = "0.8.0", path = "sparse-ir" }  # And this
   ```

2. Update the Python bindings version in `python/pyproject.toml`:
   ```toml
   [project]
   version = "0.8.0"  # Update this
   ```

3. Verify version consistency:
   ```bash
   python3 check_version.py
   ```

4. Commit and push:
   ```bash
   git add Cargo.toml python/pyproject.toml
   git commit -m "chore: bump version to 0.8.0"
   git push origin main  # or your release branch
   ```

5. Create a git tag and publish to crates.io:
   ```bash
   git tag v0.8.0
   git push origin v0.8.0
   cd sparse-ir && cargo publish
   cd ../sparse-ir-capi && cargo publish
   ```

**Stage 2: Julia version bump (after crates.io publication)**

After the new version is published to crates.io and available:

1. Update the version and commit hash in `julia/build_tarballs.jl`:
   ```julia
   version = v"0.8.0"  # Update version
   
   # Update the commit hash to match the tagged release
   sources = [
       GitSource("https://github.com/SpM-lab/sparse-ir-rs.git",
                 "abc123...")  # Update this hash
   ]
   ```
   
   To get the commit hash after tagging:
   ```bash
   git rev-parse v0.8.0
   ```

2. Verify version consistency:
   ```bash
   python3 check_version.py
   ```

3. Commit and push:
   ```bash
   git add julia/build_tarballs.jl
   git commit -m "chore: bump Julia bindings version to 0.8.0"
   git push origin main
   ```

4. Follow the Julia package release process (Yggdrasil PR, etc.)
