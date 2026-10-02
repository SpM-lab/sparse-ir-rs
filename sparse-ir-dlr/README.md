# sparse-ir-dlr

The discrete Lehmann representation (DLR) of
[`sparse-ir`](https://crates.io/crates/sparse-ir).
`DiscreteLehmannRepresentation::new(beta, wmax, eps)` and `DlrBuilder` choose
the poles by an interpolative decomposition of the logistic kernel, without
building an IR basis. A DLR implements the `Basis` trait, so the sampling
types of `sparse-ir-core` work on it.

Most users should depend on [`sparse-ir`](https://crates.io/crates/sparse-ir),
which re-exports this crate together with the rest of the library under the
`sparse_ir::` paths used in the guide.

## Documentation

- [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/): tutorials and
  conventions, written against the 0.11 release.
- API reference: [docs.rs/sparse-ir-dlr](https://docs.rs/sparse-ir-dlr).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.
