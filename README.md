SparseIR Rust Workspace
=======================

This repository is a Rust workspace providing:

- `sparse-ir` &mdash; core Rust implementation of the sparse intermediate representation (IR) and related numerical tools.
- `sparse-ir-capi` &mdash; C-compatible API built on top of `sparse-ir` (shared library + C header).
- `fortran` &mdash; Fortran bindings for the C API.

## Build

From the workspace root:

```bash
cargo build            # build all crates in debug mode
cargo build --release  # optimized build
```

The default build uses the pure-Rust `faer` backend for matrix–matrix products used in fitting and evaluation routines.  
Faer is reasonably fast, but usually considerably slower than an optimized BLAS implementation.
To enable system BLAS (LP64) for the Rust `sparse-ir` crate at compile time, use:

```bash
cargo build -p sparse-ir --features system-blas
```

Optional BLAS backends (e.g. Accelerate, OpenBLAS/CBLAS) can be injected at runtime via function pointers when using the C API or the internal GEMM dispatcher.

## Test

Run unit and integration tests:

```bash
cargo test --all-targets
```

Some doc tests that depend on external BLAS backends are marked as `ignore` and are not executed by default.

## Version Consistency Check

Check that versions in `Cargo.toml` and `sparse-ir-capi/Cargo.toml` are consistent:

```bash
python3 check_version.py
```

This script verifies that the workspace version matches the pkg-config version in the C API configuration.

## C API (sparse-ir-capi)

To build the C-compatible shared library and header:

```bash
cargo build -p sparse-ir-capi --release
```

The generated library and the public header live under `sparse-ir-capi/` and can be linked from C, C++, Fortran, or other languages that can call C ABIs.

## Fortran bindings

For details on the Fortran interface and build instructions, see:

- `fortran/README.md`

## License

This workspace is dual-licensed under the terms of the MIT license and the Apache License (Version 2.0).

- You may use the code in this repository under the terms of either license, at your option:
  - [MIT License](LICENSE)
  - [Apache License 2.0](LICENSE-APACHE)

Some components incorporate third-party code under Apache-2.0, such as the `col_piv_qr` module in the `sparse-ir` crate, which is based on nalgebra.  
See [`sparse-ir/README.md`](sparse-ir/README.md) and `LICENSE-APACHE` for details.


