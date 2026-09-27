# Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
sparse-ir = "0.10"
```

`sparse-ir` is pure Rust and needs no system libraries: its linear algebra goes
through [faer](https://github.com/sarah-quinones/faer-rs) by default. If you
would rather route the matrix products through the LP64 BLAS already installed
on your machine, turn on the `system-blas` feature:

```toml
[dependencies]
sparse-ir = { version = "0.10", features = ["system-blas"] }
```

Both backends compute the same numbers; the feature only changes which
implementation multiplies the matrices.
