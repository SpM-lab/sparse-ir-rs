# Crate and API selection

- **`sparse-ir`**: default dependency for Rust users. It re-exports the core
  types and the `basis`, `dlr`, `esprit`, and `minipole` APIs. See its
  [released API](https://docs.rs/sparse-ir/) and the [crate README](https://github.com/SpM-lab/sparse-ir-rs/blob/main/sparse-ir/README.md).
- **`sparse-ir-core`, `sparse-ir-dlr`, `sparse-ir-minipole`, `sparse-ir-basis`**:
  depend on one directly only when intentionally using that layer without the
  facade. The facade keeps their public paths stable for typical applications.
- **`sparse-ir-capi`**: C ABI, headers, and shared/static library. For C or
  Fortran consumers, follow the repository's binding documentation instead of
  treating this crate as the Rust facade.

The guide's Git dependency follows `main`; the versioned crates.io dependency
and docs.rs describe a published release. Pin a tested Git revision when a
reproducible unreleased build is required.
