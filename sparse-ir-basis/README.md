# sparse-ir-basis

The IR basis of [`sparse-ir`](https://crates.io/crates/sparse-ir): the
logistic and regularized Bose kernels, the Fourier transforms of the singular
functions, and `FiniteTempBasis`. It also builds a DLR from an IR basis
(`DlrFromIr`). The singular value expansion comes from
[`sparse-ir-sve`](https://crates.io/crates/sparse-ir-sve), which this crate
re-exports under its own paths.

Most users should depend on [`sparse-ir`](https://crates.io/crates/sparse-ir),
which re-exports this crate together with the rest of the library under the
`sparse_ir::` paths used in the guide.

## Documentation

- [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/): tutorials and
  conventions, written against the current release.
- API reference: [docs.rs/sparse-ir-basis](https://docs.rs/sparse-ir-basis).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.
