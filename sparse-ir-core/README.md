# sparse-ir-core

Core of [`sparse-ir`](https://crates.io/crates/sparse-ir): statistics and
Matsubara frequencies, the error type, GEMM dispatch, the least-squares
fitters, the `Basis` trait, and sparse sampling in imaginary time
(`TauSampling`) and Matsubara frequency (`MatsubaraSampling`) for any type
that implements `Basis`.

Most users should depend on [`sparse-ir`](https://crates.io/crates/sparse-ir),
which re-exports this crate together with the rest of the library under the
`sparse_ir::` paths used in the guide.

## Documentation

- [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/): tutorials and
  conventions, written against the 0.11 release.
- API reference: [docs.rs/sparse-ir-core](https://docs.rs/sparse-ir-core).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.
