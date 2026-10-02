# Installation

`sparse-ir` requires **Rust 1.96 or newer**. Add it from crates.io:

```toml
[dependencies]
sparse-ir = "0.11.0"
```

This book is written against `sparse-ir` 0.11. The `sparse-ir` crate re-exports
everything the book uses; it is built from four crates you can also depend on
directly: `sparse-ir-core` (statistics, sampling, fitting), `sparse-ir-basis`
(kernels, SVE and the IR basis), `sparse-ir-dlr` (the DLR) and
`sparse-ir-minipole` (ESPRIT and MiniPole). To follow the development branch
instead, use the Git dependency, pinned to a tested `rev` for reproducible
projects:

```toml
[dependencies]
sparse-ir = { git = "https://github.com/SpM-lab/sparse-ir-rs", branch = "main" }
```

`sparse-ir` is pure Rust and needs no system libraries: its linear algebra goes
through [faer](https://github.com/sarah-quinones/faer-rs) by default. If you
would rather route the matrix products through the LP64 BLAS already installed
on your machine, turn on the `system-blas` feature:

```toml
[dependencies]
sparse-ir = { version = "0.11.0", features = ["system-blas"] }
```

Both backends compute the same numbers; the feature only changes which
implementation multiplies the matrices.

The API reference is at
[docs.rs/sparse-ir](https://docs.rs/sparse-ir/) for the release and
[here](https://spm-lab.github.io/sparse-ir-rs/api/sparse_ir/index.html) for
`main`.
