# Repository Rules

These are sparse-ir-rs-specific rules. Apply them in addition to the shared
[SpM-lab agent rules](https://github.com/SpM-lab/spm-agent-rules).

The workspace publishes two Rust crates: the core `sparse-ir` crate and the
`sparse-ir-capi` C ABI crate. It also maintains thin Python and Fortran bindings
to that C ABI. Rules below protect the numerical contracts and the boundaries
between those layers.

Apply these rules to new and modified code. Existing violations are not
precedent, but they do not authorize unrelated cleanup. Remediation outside the
files and call paths required by the current task, or remediation that crosses
Rust API, C ABI, and language-binding ownership boundaries, requires an accepted
issue or explicit maintainer approval that defines its scope and verification.

## Workspace Layout

This is a Cargo workspace (`resolver = "2"`) with two published members,
defined in the top-level `Cargo.toml`:

- **`sparse-ir/`** — the core Rust crate (`sparse_ir`, lib crate types
  `cdylib` + `rlib`). Implements the sparse IR algorithms: kernels and SVE
  (`kernel.rs`, `kernelmatrix.rs`, `tsvd.rs`), piecewise Legendre polynomials
  and their Fourier transforms (`poly.rs`, `polyfourier.rs`), 1D/2D
  interpolation (`interpolation1d.rs`, `interpolation2d.rs`), Gauss
  quadrature (`gauss.rs`), the `FiniteTempBasis`/DLR basis types
  (`basis.rs`, `basis_trait.rs`, `dlr.rs`), tau/Matsubara sampling
  (`sampling.rs`, `matsubara_sampling.rs`, `taufuncs.rs`, `freq.rs`),
  special functions (`special_functions.rs`), the GEMM dispatch layer
  (`gemm.rs`, `col_piv_qr.rs`), and numeric/precision plumbing
  (`numeric.rs`, `working_buffer.rs`, `fpu_check.rs`). Test modules live
  next to their implementation as `*_tests.rs` files included via `#[cfg(test)]`
  from the corresponding `src/*.rs` module (e.g. `basis.rs` includes
  `basis_tests.rs`); additional integration tests live in `sparse-ir/tests/`.
  Supplemental, non-authoritative implementation notes for this crate are in
  [`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md).
- **`sparse-ir-capi/`** — the C ABI crate (`sparse_ir_capi`, lib crate types
  `cdylib` + `staticlib` + `rlib`). Depends on `sparse-ir` and translates it
  to a stable C interface: opaque handles and status codes (`types.rs`,
  `macros.rs`, `utils.rs`), and per-domain exported functions in
  `basis.rs`, `dlr.rs`, `funcs.rs`, `gemm.rs`, `kernel.rs`, `sampling.rs`,
  `sve.rs`. Integration and regression tests live in `sparse-ir-capi/tests/`
  (`integration_test.rs`, `test_capi_custom_sampling.rs`,
  `test_funcs_deriv.rs`); ad hoc debug scripts used while developing the ABI
  are under `sparse-ir-capi/examples/` (mostly `.jl` scripts, plus
  `test_julia.jl`).

Other top-level directories are thin bindings or consumers of the C API, not
part of the Cargo workspace:

- `python/` — the `pylibsparseir` ctypes wrapper package (see
  `python/tools/gen_ctypes.py` below).
- `julia/` — `SparseIR.jl`-adjacent build tooling (e.g.
  `julia/build_tarballs.jl`) used for downstream version bumps; the Julia
  wrapper package itself lives in the separate `SpM-lab/SparseIR.jl`
  repository.
- `fortran/` — Fortran bindings to the C API (`fortran/src/`, generated via
  scripts in `fortran/script/`).
- `cxx_tests/` — a standalone CMake project that links the built
  `sparse-ir-capi` library and header and exercises them from C++
  (`cinterface_core.cxx`, `cinterface_integration.cxx`).
- `capi_benchmark/` — benchmarking harness against the C API.
- `notebook/`, `docs/` — supporting notebooks and documentation.

## Building And Testing

All commands below were verified to run in this checkout (macOS, Rust 1.96,
Accelerate framework as the default BLAS backend) and mirror what CI runs in
`.github/workflows/`.

```bash
# Build/test the core crate (default backend: faer, pure Rust)
cargo build -p sparse-ir --release
cargo test -p sparse-ir --release

# Build/test the whole workspace
cargo build --all-targets --release --locked
cargo test --all-targets --release --locked

# Build/test sparse-ir with the system-BLAS (LP64) GEMM backend instead of faer
cargo build -p sparse-ir --features system-blas --all-targets --release --locked
cargo test -p sparse-ir --features system-blas --all-targets --release --locked

# Build/test the C API crate
cargo build -p sparse-ir-capi --release
cargo test -p sparse-ir-capi --release

# Run the DLR/IR/sampling round-trip example
cargo run --example roundtrip --release
```

Feature flags defined in `sparse-ir/Cargo.toml`:

- `system-blas` — switches the GEMM backend from the default pure-Rust
  `faer` implementation to system BLAS via `blas-sys` (LP64). Requires
  `libopenblas-dev`/`pkg-config` on Linux CI; macOS uses the Accelerate
  framework automatically (see the `build.rs` warning emitted during
  compilation).
- `shared-lib` — enables shared-library-oriented build settings.

`sparse-ir-capi/Cargo.toml` mirrors `system-blas`, forwarding it to
`sparse-ir/system-blas`, plus its own `capi` feature.

Git hooks (via `cargo-husky`, `dev-dependencies` in `sparse-ir/Cargo.toml`)
run `cargo fmt --all -- --check` on commit
(`.cargo-husky/hooks/pre-commit`); if formatting fails, run
`cargo fmt --all` before committing.

Version consistency across Rust/Python/Julia metadata and README install
snippets is checked with:

```bash
python3 check_version.py
```

This reads the canonical version from `[workspace.package]` in `Cargo.toml`.
See the "Version management" section of `README.md` and
`bump_version_downstream.md` for the release process itself; do not duplicate
that process here. See also "Release And Version Integrity" below for the
release-order rules themselves.

## C API Surface

- `sparse-ir-capi/src/` is the Rust source of the C ABI: exported `spir_*`
  functions, opaque handle types, and status codes.
- `sparse-ir-capi/include/sparseir/sparseir.h` is the canonical C header,
  generated from the Rust exports via `cbindgen` (config:
  `sparse-ir-capi/cbindgen.toml`; excludes internal types such as
  `BasisType`, `FuncsType`, `KernelType`, `SamplingType`, `SVEResultType`,
  `Arc_SVEResult`, and the `inner` field). CI's `header-sync` job in
  `.github/workflows/rust.yml` regenerates it with
  `cbindgen --config cbindgen.toml --cpp-compat --output /tmp/sparseir.h .`
  and fails the build on any diff; see "Generated Bindings And Cross-Language
  Synchronization" below for the do-not-hand-edit rule.
- `sparse-ir-capi/assets/sparse_ir_capi.h` is the `cargo-c` distribution copy
  of the same header (`[package.metadata.capi.header] generation = false`
  tells `cargo-c` to use this checked-in file rather than regenerate it).
  The same `header-sync` CI job diffs this file against the generated header
  too, so both copies must be updated together whenever the C API changes.
- `python/pylibsparseir/ctypes_autogen.py` is generated from the header by
  `python/tools/gen_ctypes.py` — regenerate it rather than hand-editing when
  the C API changes.
- `cxx_tests/` exercises the built C API from C++ against the same header;
  `cxx_tests/run_with_rust_capi.sh` builds `sparse-ir-capi` in release mode,
  installs it locally, and builds/runs the CMake test suite against it.

## CI Entry Points

Workflows in `.github/workflows/` (all triggered on push/PR to `main`):

- `rust.yml` — `rust-default` (faer backend) and `rust-system-blas` jobs run
  `cargo build`/`cargo test --all-targets --release --locked` for the
  workspace and for `sparse-ir --features system-blas` respectively; a
  `header-sync` job checks the two C headers described above stay in sync
  with cbindgen output.
- `rust_capi.yml` — builds and tests `sparse-ir-capi` in release mode
  (`cargo build -p sparse-ir-capi --release`, `cargo test -p sparse-ir-capi
  --release`).
- `cxx_tests.yml` — runs the C++ integration tests in `cxx_tests/` against
  the built C API, once with the default (faer) backend and once with
  OpenBLAS (LP64).
- `check_version.yml` — runs `python3 check_version.py`.
- `test_python.yml`, `test_fortran.yml` — test the Python and Fortran
  bindings against the built C API.
- `CI_PublishPyPI.yml`, `PublishPyPI.yml`, `publish_conda.yml`,
  `publish-libsparseir-reusable.yml`, `publish-libsparseir.yml`,
  `manual-release.yml` — publishing workflows; `manual-release.yml` is the
  manual Rust crates.io release flow referenced from `AGENTS.md` and
  `agent-skills/manual-rust-release/SKILL.md`. Do not bypass its
  expected-version and publication-order checks with ad hoc local
  `cargo publish` commands (see "Release And Version Integrity" below).
- `latest-dependencies.yml` — dependency-drift check.

## Before Editing

- Prefer `cargo build`/`cargo test` scoped to the crate you touched
  (`-p sparse-ir` or `-p sparse-ir-capi`) for fast local iteration; run the
  full workspace and `system-blas` variants before a PR that touches shared
  code (GEMM dispatch, numeric types, or the C API).
- A change to any exported `spir_*` function, struct, enum, or constant in
  `sparse-ir-capi/src/` requires regenerating and re-diffing both C headers
  (see "C API Surface" above and the rule in "Generated Bindings And
  Cross-Language Synchronization" below) in the same change, or the
  `header-sync` CI job will fail.
- `sparse-ir/*_tests.rs` files are included into their sibling `src/*.rs`
  module; when adding tests for a module that doesn't yet have one, follow
  the existing `mod foo; ... #[cfg(test)] mod foo_tests;`-style pattern
  rather than inlining a large test module.
- Numerical algorithm changes belong in `sparse-ir`; ABI translation,
  pointer/handle validation, and status-code mapping belong in
  `sparse-ir-capi`. See the shared
  [`ffi-boundary.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/ffi-boundary.md)
  and [`numerical-conventions.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/numerical-conventions.md)
  rules for what each layer must validate and document.

## Source Of Truth And Layering

- The core Rust implementation owns numerical algorithms, basis and sampling
  semantics, working-precision policy, and backend-neutral validation.
- `sparse-ir-capi` owns ABI translation: opaque handles, raw-pointer checks,
  memory-order conversion, status-code mapping, and panic containment. Do not
  move core numerical logic into the C API instead of adding an appropriate
  Rust abstraction.
- Python and Fortran code in this repository are thin bindings. Do not duplicate
  numerical algorithms there when the Rust core or C ABI owns the behavior.
- Public Rust docs, the checked-in C headers, generated binding declarations,
  and examples must describe the currently implemented surface. Historical
  notebooks and implementation notes are not public API specifications.
- Fix behavior at the layer that owns it. Avoid compatibility shims, duplicated
  validation, and downstream reach-through into private representations.

## Public Surface And Compatibility

- Planning helpers, intermediate matrices, scratch buffers, backend dispatch
  details, and test helpers should be private or crate-private unless
  downstream users are expected to rely on them.
- A public Rust item, exported C symbol, status code, opaque-handle lifecycle,
  struct layout, constant value, or calling convention is a compatibility
  contract. Review its Semantic Versioning impact before changing it.
- Do not expose low-level Rust internals through the C ABI merely to satisfy one
  wrapper. Add the narrowest language-neutral operation that preserves the
  core abstraction.
- When a public surface changes, update Rust documentation, C headers, generated
  declarations, wrappers, examples, and integration tests in the same change.

## Numerical Correctness And Precision

- Numerical behavior is part of the public contract. Changes to kernels,
  interpolation, Gauss rules, SVE, TSVD, DLR, basis construction, sampling, or
  fitters require tests that check values, reconstruction, or residuals rather
  than only shapes or successful execution.
- Cover representative values and difficult regimes: small and large
  `lambda`, requested `epsilon` near working-precision transitions, interval
  endpoints, roots and extrema, bosonic and fermionic statistics, positive-only
  and full Matsubara grids, real and complex data, and rank-deficient or
  ill-conditioned matrices where applicable.
- Treat `f64` and `Df64` as distinct working-precision contracts. Do not claim
  double-double accuracy for a path whose elementary functions, conversions,
  decomposition backend, or final storage reduce it to `f64` accuracy.
- Keep the automatic `TworkType` selection and epsilon clamping policy
  consistent between the core Rust API and C API constants. A threshold change
  is a numerical behavior change and requires focused boundary tests.
- Coordinate normalization and denormalization must be applied consistently in
  coefficient construction and evaluation. Polynomial and quadrature code must
  preserve the documented interval and endpoint conventions.
- Preserve symmetry, parity, statistics, Matsubara-frequency, singular-value
  ordering, and truncation semantics. Optimizations must not silently change
  these mathematical conventions.
- When parity with SparseIR.jl, Python sparse-ir, or libsparseir is intended,
  state which behavior is being matched and test it with reproducible reference
  data. Distinguish compatibility requirements from independent oracle checks.

## Numerical Test Tolerances

- Choose tolerances before evaluating a candidate change, based on an error
  model, conditioning, working precision, a higher-precision oracle, or a
  documented cross-language compatibility bound.
- Do not relax a tolerance merely until a failing result passes. A relaxation
  must record the observed residual, the previous and proposed bounds, the
  affected parameter range, and why the new bound still detects meaningful
  regressions.
- Prefer high-precision references such as `DBig`, analytic values, known
  identities, reconstruction residuals, or independently generated
  SparseIR.jl/Python reference data.
- Keep reference-data generation reproducible. Record the upstream project and
  version or commit, generator path, parameters, precision, and serialization
  convention near the fixture or generator.

## Public Boundary Validation And Errors

- Validate user-derived `lambda`, `epsilon`, statistics, rank, dimensions,
  target axes, sampling points, intervals, matrix shapes, and configuration
  values before allocation, indexing, backend calls, or no-op shortcuts.
- Public Rust library paths must not turn invalid user input into `panic!`,
  `unwrap`, `expect`, unchecked indexing, or debug-only assertions. Prefer
  crate-local typed errors with enough context for C and language bindings to
  report the failure.
- Panics are reserved for genuinely internal invariants whose proof is local.
  Do not copy an existing public panic into a new or modified API. Broad
  conversion of existing panic contracts requires an accepted issue or explicit
  maintainer approval.
- Preserve error categories across layers. Known invalid input must map to a
  specific `SPIR_INVALID_*` or dimension status, unsupported operations to
  `SPIR_NOT_SUPPORTED`, and unexpected internal failures to
  `SPIR_INTERNAL_ERROR`.
- Do not collapse an available typed Rust error into an internal error merely
  because the C ABI currently carries only a status code. Keep the mapping
  centralized and testable.

## C ABI Safety

- No Rust panic may unwind across an `extern "C"` boundary. Every exported
  entry point that can panic must use the repository's panic-containment pattern
  and return the documented failure status or null result.
- `catch_unwind` does not make an invalid non-null pointer safe. Regardless of
  caller obligations, Rust must check every verifiable precondition—nullability,
  lengths, dimensions, enum values, and arithmetic bounds—before dereferencing.
  Pointer provenance and lifetime remain caller obligations when they cannot be
  verified through the ABI.
- Before constructing a slice or tensor view from raw parts, validate pointer
  requirements and use checked arithmetic for dimension products, element
  counts, byte lengths, strides, and offsets.
- Check conversions among C integer types, `usize`, BLAS LP64 `i32`, and BLAS
  ILP64 `i64`. Reject values that are negative or out of range before casting.
- Validate output capacity, shape, and element type before writing. Unless an
  API explicitly documents partial output on failure, validation must complete
  before mutating caller-owned output buffers.
- Do not accept overlapping input and output buffers unless the operation's
  aliasing contract explicitly permits it and the implementation is tested for
  that overlap.
- Unsafe helper functions must state their complete caller obligations in a
  `# Safety` section; the call site must establish those obligations locally.

## Opaque Handle Ownership

- Every C opaque handle has one documented constructor/clone/release ownership
  model. A successful constructor returns an owned handle, clone returns a new
  independently releasable handle, and release consumes exactly one handle.
- Releasing a null pointer may be a no-op when documented. Double release,
  dereference after release, or passing an arbitrary non-null pointer remains a
  caller contract violation and is not recoverable with `catch_unwind`.
- Keep Rust implementation details behind opaque handles. Do not expose inner
  enum layouts, `Arc` internals, Rust vtables, or allocator-specific ownership
  through the ABI.
- Sharing an `Arc` proves shared lifetime, not mutable access. Do not derive
  mutable aliases from cloneable handles without a separate exclusivity
  contract.
- Constructors with an out-status parameter must set it deterministically on
  every success and failure path. A null return and status value must not
  contradict each other.

## Memory Order And Dimension Semantics

- The C API and binding layers explicitly support both row-major and
  column-major buffers. Do not introduce hidden row-major conversion into the
  core Rust numerical API. Each ABI function must document the logical shape,
  memory order, and meaning of every target or batch dimension.
- Keep row-major/column-major translation in shared C API helpers. Do not add
  operation-local dimension reversal or axis remapping when the common
  conversion path can express it.
- A memory-order conversion must preserve the logical tensor. Add tests that
  feed the same non-square, rank-greater-than-two value in both orders and
  compare logical results.
- Do not hide layout copies in core numerical APIs. When an ABI or backend
  requires packing or permutation, keep the copy at an explicit boundary and
  document its cost.
- Test singleton, zero-length where supported, non-square, batched, and
  target-dimension edge cases. Shape products and axis remapping must be checked
  before creating mdarray views.

## BLAS And GEMM Backend Contract

- The pure-Rust/faer backend, compile-time `system-blas` backend, and runtime
  injected BLAS function pointers must implement the same mathematical and
  memory-order contract.
- LP64 and ILP64 are distinct ABI choices. Keep function-pointer types,
  dimensions, leading dimensions, and tests separate; never infer one ABI from
  pointer presence alone.
- Validate all matrix dimensions and leading dimensions before an FFI call.
  Overflow or an unsupported dimension must return an error rather than narrow
  silently.
- Backend fallback must be intentional. If a fallback allocates, packs, changes
  threading, or uses a slower implementation, make that boundary visible in the
  implementation and cover it with correctness and performance tests.
- Do not select a backend independently inside inner numerical helpers. Route
  GEMM through the workspace's established dispatcher or explicit backend
  handle so one operation does not mix incompatible providers.
- A local manual matrix product is allowed only for a bounded reference path or
  when the owning backend cannot express the required generic scalar type.
  Document that exception and cover it with focused correctness tests.
- Benchmarks comparing providers must use release builds, pin relevant thread
  counts, separate setup from execution, and report the BLAS implementation and
  LP64/ILP64 mode.

## Unsafe Code And Working Buffers

- Keep `unsafe` localized to FFI, backend adapters, raw buffer management, and
  view construction that cannot be expressed safely. Numerical orchestration
  and public validation should remain safe Rust.
- Every new or modified unsafe block requires a nearby `// SAFETY:` comment
  explaining pointer validity, alignment, initialized range, bounds, lifetime,
  and aliasing facts relevant to that block. Auditing untouched legacy unsafe
  blocks requires a scoped issue or explicit maintainer approval; do not add
  perfunctory comments without re-verifying the invariant.
- `WorkingBuffer` and similar reusable scratch storage must maintain checked
  byte capacity and alignment before producing typed slices. A typed mutable
  slice must never outlive or alias another mutable view of the buffer.
- Distinguish uninitialized full-overwrite storage from initialized
  read-before-write storage. Do not fix an initialization bug by unconditionally
  zero-filling a shared hot-path buffer.
- Unsafe `Send` or `Sync` implementations require a concrete ownership and
  aliasing argument plus tests that exercise movement or sharing across threads
  when that behavior is supported.

## Generated Bindings And Cross-Language Synchronization

- `sparse-ir-capi/include/sparseir/sparseir.h` is the cbindgen-generated C API
  header. Regenerate it from the Rust exports; do not hand-edit it as the source
  of truth. `sparse-ir-capi/assets/sparse_ir_capi.h` is the checked-in
  `cargo-c` distribution copy and must match the generated header. A header
  synchronization check should fail when the two drift.
- `python/pylibsparseir/ctypes_autogen.py` is generated by
  `python/tools/gen_ctypes.py`. Regenerate it; do not hand-edit generated
  declarations to conceal drift.
- Files under `fortran/src/` marked as generated must be changed through their
  scripts under `fortran/script/`. Keep generated declarations, public exports,
  and implementation include files synchronized.
- A C ABI change is incomplete until the relevant C/C++ integration tests,
  Python tests, and Fortran tests compile and exercise the changed contract.
- Keep complex-number layout, boolean representation, enum/status constants,
  pointer ownership, and row/column-major conventions identical across Rust,
  headers, ctypes, C++, and Fortran declarations.
- Generated-file checks should be reproducible and fail CI when regeneration
  would produce a diff.

## Release And Version Integrity

- Keep `[workspace.package].version` and
  `[workspace.dependencies].sparse-ir.version` in `Cargo.toml` synchronized.
- Before a Rust release or release pull request, align
  `python/pyproject.toml` `[project].version` and the `sparse-ir/README.md`
  install snippets with the workspace version and run
  `python3 check_version.py`, which fails on either mismatch.
- Publish `sparse-ir` before `sparse-ir-capi`, and push the `vX.Y.Z` tag only
  after both intended crates are successfully published to crates.io.
- Update downstream Julia version metadata only after the corresponding crates
  are available from crates.io. Follow `bump_version_downstream.md` for that
  separate stage.
- Use `.github/workflows/manual-release.yml` for the manual Rust release flow.
  Do not bypass its expected-version and publication-order checks with ad hoc
  local publication commands.

## Performance And Allocation Discipline

- Reuse working buffers, precompute interpolation coefficients, and carry
  prepared shape/stride metadata into inner kernels. Prefer borrowed mdarray
  views and explicit output buffers at fitter and ABI boundaries.
- Cell and segment lookup should retain logarithmic or better lookup behavior
  where the grid permits it. Do not replace binary search with an unbounded
  linear scan without measured justification.
- Performance changes require representative sizes, parameter regimes, memory
  orders, working precisions, and providers. A single favorable matrix size is
  not sufficient evidence.
- A performance optimization must preserve the numerical and ABI contracts in
  this file. Do not trade away validation, precision, or deterministic ownership
  for an unmeasured speedup.

## Tests And Documentation

- Tests follow implementation ownership: core numerical behavior belongs in
  `sparse-ir`, ABI behavior in `sparse-ir-capi`, and wrapper marshaling in the
  corresponding language binding.
- Public `Result` APIs should document concrete error conditions. Exported C
  functions should document nullability, ownership, input/output dimensions,
  memory order, status results, and panic containment.
- When a public API or supported capability changes, audit the workspace
  README, crate READMEs, rustdoc, C headers, Python and Fortran documentation,
  examples, and ecosystem links for stale claims.
- Keep tests deterministic. Record random seeds, reference-generation inputs,
  provider configuration, and thread counts when they affect results.
- Changes to a Rust, C ABI, Python, or Fortran boundary must run focused checks
  for that boundary before review; hosted CI owns the full cross-language,
  feature, and provider matrix.

## Provenance And Scientific Credit

- When implementing or translating code while reading SparseIR.jl, Python
  sparse-ir, libsparseir, nalgebra, or another project, record the project,
  source file or symbol, revision when practical, and whether the code was
  ported, derived, convention-matched, or only validated against it.
- Preserve copyright notices and give scientific credit to the original
  authors and publications when code, algorithms, or reference data are
  ported, derived, or convention-matched from another project.
- Changing a citation, attribution, or reference to an upstream project, and
  any upstream-facing issue or pull-request work (filing, commenting, or
  proposing changes against SparseIR.jl, Python sparse-ir, libsparseir, or
  another upstream project), requires explicit user approval before acting.
- Reference fixtures must identify their generator and upstream version. Do not
  commit unexplained numerical tables whose provenance cannot be reproduced.
