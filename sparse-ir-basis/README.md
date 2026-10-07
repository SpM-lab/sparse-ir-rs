# sparse-ir-basis

The IR basis of [`sparse-ir`](https://crates.io/crates/sparse-ir): the
logistic and regularized Bose kernels, the singular value expansion, piecewise
Legendre polynomials and their Fourier transforms, and `FiniteTempBasis`. It
also builds a DLR from an IR basis (`DlrFromIr`).

Most users should depend on [`sparse-ir`](https://crates.io/crates/sparse-ir),
which re-exports this crate together with the rest of the library under the
`sparse_ir::` paths used in the guide.

## Documentation

- [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/): tutorials and
  conventions, written against the 0.12 release.
- API reference: [docs.rs/sparse-ir-basis](https://docs.rs/sparse-ir-basis).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.

The `col_piv_qr` module is based on code from
[nalgebra](https://github.com/dimforge/nalgebra) (Apache License 2.0).
