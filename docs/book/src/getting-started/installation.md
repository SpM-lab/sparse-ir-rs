# Installation

This guide follows the repository's `main` branch and requires **Rust 1.96 or
newer**. Use the Git dependency to run its examples:

```toml
[dependencies]
sparse-ir = { git = "https://github.com/SpM-lab/sparse-ir-rs", branch = "main" }
```

For a crates.io release, consult its [released API documentation](https://docs.rs/sparse-ir/);
its API may differ from `main`. For reproducible projects, pin the Git dependency
to a tested `rev` rather than following `main`.

`sparse-ir` is pure Rust and needs no system libraries: its linear algebra goes
through [faer](https://github.com/sarah-quinones/faer-rs) by default. If you
would rather route the matrix products through the LP64 BLAS already installed
on your machine, turn on the `system-blas` feature:

```toml
[dependencies]
sparse-ir = { git = "https://github.com/SpM-lab/sparse-ir-rs", branch = "main", features = ["system-blas"] }
```

Both backends compute the same numbers; the feature only changes which
implementation multiplies the matrices.
