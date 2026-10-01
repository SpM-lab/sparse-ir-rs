---
name: sparse-ir-rust-usage
description: Use when choosing crates, learning APIs, or writing Rust code with sparse-ir.
---

# Using sparse-ir from Rust

Use the `sparse-ir` facade for ordinary Rust applications. It re-exports the
core, DLR, MiniPole, and IR-basis crates under the familiar `sparse_ir` paths.
Depend on `sparse-ir-capi` only when implementing or consuming the C ABI; it is
not the Rust facade.

Start with the [installation guide](https://spm-lab.github.io/sparse-ir-rs/getting-started/installation.html)
and use docs.rs for the API matching a published release. The guide follows
`main`, whose API may differ from crates.io. Read the relevant references below
before assuming array layout, frequency indexing, or accuracy semantics.

- [Crate and API selection](references/api-selection.md)
- [Numerical conventions and performance](references/numerics.md)
- [MiniPole inputs and accuracy](references/minipole.md)

Prefer the executable [Rust tutorials](https://spm-lab.github.io/sparse-ir-rs/)
for complete workflows. Reuse expensive basis/SVE objects when parameters
permit, propagate the crate's `Result` errors, and validate continuation results
at frequencies not used for fitting.
