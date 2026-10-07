# Numerical conventions and performance

- Statistics is represented by types such as `Fermionic` and `Bosonic`; it is
  not a runtime switch. `MatsubaraFreq` stores the *reduced* index `n`,
  with `iν = i n π/β` (`n` odd for fermions, even for bosons), not the textbook
  index `m` of `(2m + ζ)π/β`: `n = 2m + ζ`, so `FermionicFreq::new(0)` is an
  error and `FermionicFreq::new(1)` is `π/β`. The MiniPole DLR entry points use
  a contour `(2n+1)π/β`, even for bosons. See the [conventions chapter](https://spm-lab.github.io/sparse-ir-rs/getting-started/conventions.html).
- Imaginary-time sampling points are reported on the symmetric interval
  `[-beta/2, beta/2]`, even though the Green function's domain is `[0, beta)`.
  Apply the fermionic or bosonic folding relation when comparing another
  implementation.
- Public `Matrix<T>` values are rank-2 `TypedTensor`s. The crate's dense
  `Mat<T>` helper is column-major, with the first index varying fastest; do not
  infer a layout for an arbitrary external tensor from that helper. The C API
  documents its own column-major buffers.
- Constructing a `FiniteTempBasis` computes an SVE. Reuse the SVE result to
  create fermionic and bosonic bases when they share kernel and accuracy
  parameters; see the conventions chapter.
- Start with the pure-Rust backend. Enable `system-blas` only when you
  intentionally want the installed LP64 BLAS implementation.

The [transformation](https://spm-lab.github.io/sparse-ir-rs/tutorials/transformation.html)
and [DLR](https://spm-lab.github.io/sparse-ir-rs/tutorials/dlr.html) tutorials
show complete construction, fitting, and evaluation workflows.
