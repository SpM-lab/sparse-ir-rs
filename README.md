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

The default build uses the pure-Rust `faer` backend for linear algebra.  
Optional BLAS backends (e.g. Accelerate, OpenBLAS/CBLAS) can be injected at runtime via function pointers when using the C API or the internal GEMM dispatcher.

## Test

Run unit and integration tests:

```bash
cargo test --all-targets
```

Some doc tests that depend on external BLAS backends are marked as `ignore` and are not executed by default.

## C API (sparse-ir-capi)

To build the C-compatible shared library and header:

```bash
cargo build -p sparse-ir-capi --release
```

The generated library and the public header live under `sparse-ir-capi/` and can be linked from C, C++, Fortran, or other languages that can call C ABIs.

## Fortran bindings

For details on the Fortran interface and build instructions, see:

- `fortran/README.md`


