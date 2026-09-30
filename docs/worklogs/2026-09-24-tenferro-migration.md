# tenferro migration (issue #259, phase 1)

Branch: `work/tenferro-migration` (not to be merged into `main` without human review).

## Goal

Move the dense tensor data model and construction-time linear algebra from
`mdarray` / `mdarray-linalg-faer` / `nalgebra` to tenferro 0.7.1
(`tenferro-tensor::TypedTensor`, `tenferro-linalg`, `tenferro-cpu`), CPU only,
no AD. Hard requirement: no speed regression in basis generation or in
sampling fit/evaluate. The C ABI stays backward compatible (Python and Julia
wrap it).

## Baseline (main @ branch point, Apple M5 Max, rustc 1.96.0, `RAYON_NUM_THREADS=1`)

`cargo run --release --example bench_core` (median of 7 heavy / 51 light runs):

| case | ms |
| --- | ---: |
| SVE Λ=1e1 ε=1e-6 f64 | 2.27 |
| SVE Λ=1e3 ε=1e-6 f64 | 23.3 |
| SVE Λ=1e5 ε=1e-6 f64 | 109 |
| SVE Λ=1e3 ε=1e-10 Df64 | 1056 |
| SVE Λ=1e5 ε=1e-10 Df64 | 5368 |
| MatsubaraSampling::new Λ=1e5 L=95 | 356 |
| TauSampling::new Λ=1e5 L=95 | 1.01 |
| tau fit Λ=1e5 extra=1 / 100 / 10000 (dim 0) | 0.0018 / 0.069 / 6.63 |
| matsu fit Λ=1e5 extra=1 / 100 / 10000 (dim 0) | 0.0074 / 0.317 / 30.8 |

Full output: see the commit message of the benchmark commit / rerun the example.

## tenferro microbenchmark vs faer 0.23 (same machine)

- GEMM for large extra dimension (n = 100, 10000): parity (0.85x–1.05x) for f64 and c64,
  single- and multi-threaded.
- Thin SVD: 1.04x–1.14x slower single-threaded; 0.56x–0.68x (faster) multi-threaded
  for 95x95 and larger.
- Per-call overhead: single-thread session entry 0.35 µs, matmul inside a session
  1.8–3.4 µs versus 0.8 µs for direct faer (95x95 · 95x1).
  **Multi-threaded `CpuBackend::new()`: entering `with_backend_session` costs ~29 µs.**
  A single-vector fit (0.5–7 µs today) would regress 5–50x if every fit opened a session.

## Design decision

- Data model: `TypedTensor<T>` (column-major) replaces `mdarray` tensors in the Rust API.
- Construction-time factorizations (SVD for fitters, QR, eig for later ESPRIT): tenferro-linalg.
- Fit/evaluate hot path: GEMM is called directly on `TypedTensor` host slices through
  sparse-ir's existing dispatcher (faer or injected BLAS), not through a tenferro session,
  because of the session-entry overhead above. Revisit when tenferro offers a cheap
  caller-thread session.
- Df64 SVE path (xprec + in-house TSVD / column-pivoted QR) stays; tenferro has no
  mainline extended-precision scalar.

## Feedback for tenferro (collected during migration)

Premise: sparse-ir owns its extended-precision linear algebra (Df64 SVD/QR stay
in-house). tenferro only has to provide the tensor type. The main request is
therefore to fold `HostTensor<T>` into `TypedTensor`, not to add Df64 kernels.

### A. Unify `HostTensor<T>` into `TypedTensor<T, R>` (main request)

Filed as tensor4all/tenferro-rs#1903 (2026-09-25), with a reproducer.

Today (origin/main, after #1800) there are two parallel host containers:

- `TypedTensor<T, R>`: static rank, views, integrated with tenferro-cpu and
  tenferro-linalg, but `T: TensorScalar` is sealed to the preset set.
- `HostTensor<T>` (tenferro-tensor-core): unconstrained `T`, `Clone`, used by
  external scalar sets (`ext/df64-proof`), but dynamic rank only, no mutable
  views, and view ops limited to reshape/transpose/slice.

Generic numerical code over `f64` and `Df64` cannot share one tensor type, so
sparse-ir keeps an in-house column-major matrix for generic `T`. Wish: one
`TypedTensor<T, R>` whose host storage accepts any `T: Copy` (for example
`T: Copy + 'static`). Backend kernels, dtype dispatch and linalg stay gated by
`TensorScalar`, so the preset path is unchanged. The concrete requirements for
the unified type:

1. Static rank (`Rank<N>`) and `DynRank`, with the same view API (strided,
   mutable, reshape/transpose/slice) for every `T`.
2. `get`/`get_mut` without `One + Zero`. Reading an element should need no
   arithmetic bounds.
3. Infallible `Index<[usize; N]>` / `IndexMut` on host-owned compact tensors
   and on column-major views. Today `get2` and `host_col_major_view()` return
   `Result`, which fills dense index loops with `?`/`unwrap` noise.
4. `Clone`. `HostTensor` already derives it, `TypedTensor` does not. Structs
   holding a tensor cannot `#[derive(Clone)]`, which forces `Arc` or a
   hand-written impl.
5. A compaction primitive, `view.to_col_major()` (or a `duplicate()` that
   compacts). `duplicate()`/`as_slice()` currently reject strided views, so
   sparse-ir carries its own gather (`fitters::common::gather_col_major`).

### B. Small-call overhead (independent of A)

Filed as tensor4all/tenferro-rs#1904 (2026-09-25), with a reproducer. The re-measurement
against faer 0.24 shows that single-threaded thin SVD is *faster* in tenferro for 95x95 and
larger (0.3–0.44x of faer's time), so item 9 was dropped from the issue.

6. Session entry on a multi-threaded `CpuBackend` costs ~29 µs (~0.35 µs
   single-threaded). Libraries called frequently through FFI (C/Python/Julia)
   cannot hold a closure-scoped session across calls. Wish: a cheap
   caller-thread/inline session, or a storable session handle.
7. Per-op overhead inside a session is 1–2.5 µs above direct faer for a tiny
   GEMM (95x95·95x1).
8. Construction/view overhead is large for small tensors. `from_vec_col_major`
   and `zeros` each take ~210 ns (vs ~30 ns for the raw allocation), and
   `as_view` takes 65–85 ns. For a 52-element tau evaluation this dominates
   (0.3 µs → 0.5 µs end to end). Wish: a cheap constructor for
   already-validated compact column-major buffers.
9. Single-threaded f64 thin SVD is 4–14% slower than faer 0.23 directly
   (52x52–400x200). Minor.

### C. Minor

10. tenferro-linalg entry points are `DynRank`-only, so `Rank<2>` matrices are
    converted in and out. Minor, because sparse-ir only uses linalg for f64
    SVD/eig.
11. MSRV 1.96 and a pinned faer 0.24 in tenferro-cpu force the same toolchain
    and faer on downstream crates.

Dropped: a published Df64 scalar crate and Df64 SVD/QR in tenferro. They are
not needed, because sparse-ir keeps its own.

## Status (2026-09-24)

- `sparse-ir`: all public array types are `TypedTensor<T>` (`DynRank`) / `Matrix<T>`
  (`TypedTensor<T, Rank<2>>`); mdarray, mdarray-linalg and mdarray-linalg-faer removed
  from the workspace. Fitter transforms return `sparse_ir::Result`. Tests: 252 passed.
- `sparse-ir-capi`: ported without ABI changes. Row-major C buffers are handled as
  column-major tensors with reversed dims (zero-copy views for the inplace sampling
  entry points; one flat copy for DLR conversion, as before). Library errors map to
  status codes (`Unsupported` → `SPIR_NOT_SUPPORTED`, `ShapeMismatch` →
  `SPIR_INPUT_DIMENSION_MISMATCH`, ...). Condition numbers use
  `sparse_ir::fitters::singular_values`.
- Df64 TSVD/QR remain in-house (`tsvd.rs`); tenferro-linalg is not extended.
- Small matrix-vector products (`pre == 1`, `post <= 4`, `m*n <= 128^2`, default
  backend) bypass faer's GEMM dispatch (375 ns for a 52×52 matvec) in
  `apply_along_axis`.
- Tests: sparse-ir 252 passed; sparse-ir-capi 53+18+2+3 (59 with `system-blas`);
  C++ `cxx_tests` via `run_with_rust_capi.sh` all passed.

## Benchmark after migration (same machine, `RAYON_NUM_THREADS=1`)

| case | main (ms) | tenferro (ms) |
| --- | ---: | ---: |
| SVE Λ=1e5 ε=1e-6 f64 | 109 | 112 |
| SVE Λ=1e5 ε=1e-10 Df64 | 5368 | 5258 |
| basis Λ=1e5 L=95 (incl. SVE) | 5305 | 5324 |
| MatsubaraSampling::new Λ=1e5 | 356 | 375 |
| tau eval Λ=1e3 extra=1 dim=0 | 0.0003 | 0.0005 |
| tau fit Λ=1e5 extra=1 / 100 / 10000 (dim 0) | 0.0018 / 0.069 / 6.63 | 0.0023 / 0.070 / 6.57 |
| tau fit Λ=1e5 extra=10000 dim=1 | 7.95 | 6.46 |
| matsu fit Λ=1e5 extra=1 / 100 / 10000 (dim 0) | 0.0074 / 0.317 / 30.8 | 0.0090 / 0.296 / 29.2 |
| ir2dlr Λ=1e5 extra=10000 dim=1 | 8.06 | 6.44 |

Basis generation (SVE) is at parity within run-to-run noise (±3%). Fits and
evaluations with non-trivial batch sizes are equal or faster (up to 20–25% for
`dim=1`). Single-vector calls are 0.2–0.5 µs slower in absolute terms because of
tensor construction overhead (feedback item 12).

## Rebuilt on 0.10.0 (2026-09-30)

The branch was rebuilt on `origin/main` 8ef34ad (0.10.0) instead of 0579962;
the old head is kept as `archive/tenferro-migration-0579962`. Each commit of
the old branch is re-applied in order.

- Array migration: `sparse_ir::Error` of main (typed errors, #296/#300)
  replaces the branch's error type; it gains `Gemm` and `Tensor` variants.
  The fitters keep main's contract (whole-shape checks of `out`, nothing
  written on an error, SVD failures naming the matrix, condition numbers).
  Samplings keep main's point order and argument checks.
- Tests: every test of main is kept. The in-file fitter and gemm tests whose
  API is gone are covered by the rewritten tests; the ones not covered were
  carried over (`fitters/contract_tests.rs`), and the doc examples restored.
- Local results after the migration commit (Linux x86_64): sparse-ir lib 383
  passed, integration tests 19, doctests 18; sparse-ir-capi 79 + 37 + 11 + 2
  + 3 + 43 + 1; Rust tutorial `scripts/check.sh --run` passes (examples match
  the committed reference values); C headers unchanged.
- All 11 commits of the old branch are re-applied, one commit each; the
  per-milestone sections below carry a "Rebuild on 0.10.0" note where the
  port differs from the old branch.
- Rust API changes against 0.10.0 (breaking, allowed by #259):
  `DiscreteLehmannRepresentation::new(&basis)` is `from_ir(&basis)`,
  `with_poles` is `from_ir_with_poles`, and `new(beta, wmax, eps)` /
  `DlrBuilder` build the independent DLR; `ir_basis_size()` returns
  `Option<usize>` (`None` for an independent DLR, whose `from_ir_nd` /
  `to_ir_nd` are `NotSupported`); sampling `from_matrix` takes `&Matrix`;
  arrays are `TypedTensor` / `Matrix`.
- Behavior change decided by the maintainer (2026-09-30): the default τ and
  Matsubara points of a DLR, in Rust and through
  `spir_basis_get_{n_,}default_{taus,matsus}`, are its interpolation nodes,
  one per pole (0.10.0 reported none). The status tests of 0.10.0 (S12 in
  C++, Fortran, Python) and `test_dlr_basis_methods_report_errors` were
  changed to this contract, and the C docs say so. The `_ext` variants still
  report 0 points for a DLR.
- Kept from main over the old branch: DLR pole weights are the kernel
  regularizers (#286), Matsubara samplings keep the given point order
  (#291), and every error is a `sparse_ir::Error`.
- Final local results on the last code commit (Linux x86_64, 4 build jobs):
  sparse-ir lib 407 passed (1 ignored), integration tests 19, doctests 19;
  sparse-ir-capi lib 81 plus integration tests; system-blas layer green;
  C++ 2/2 suites (26 cases in cinterface_core); Fortran 13/13; Python 102
  passed; headers match cbindgen 0.29.2. The Rust tutorial check passes.


## Milestone B: independent DLR (2026-09-25)

- `dlr_id.rs`: Kaye–Chen–Parcollet construction (composite Chebyshev τ/ω
  candidate grids, 24-point panels, dyadic refinement; column-pivoted
  Gram–Schmidt with re-orthogonalization for poles; row selection for τ and
  Matsubara nodes). Matsubara candidates: all `|n| <= 2·128+ζ`, then 32 per
  octave up to `~8Λ`.
- `DiscreteLehmannRepresentation::new(beta, wmax, eps)` / `DlrBuilder` is now
  the default and needs no IR basis. The old constructors were renamed to
  `from_ir` / `from_ir_with_poles` and still attach an `IrDlrTransform`.
  `IrDlrTransform::new(basis, dlr)` connects any compatible pair. It rescales by
  the ratio of pole weights, so a logistic DLR also works with a
  `RegularizedBoseKernel` basis.
- The DLR now implements `default_tau_sampling_points` and
  `default_matsubara_sampling_points` (full and positive-only), so
  `TauSampling::new(&dlr)` and `MatsubaraSampling::new(&dlr)` interpolate on
  the ID nodes. The nodes are computed lazily and cached.
- Ranks vs IR size, all within a few: Λ=1e3/ε=1e-10: 51 vs 52;
  Λ=1e5/ε=1e-10: 92 vs 95.
- Bench (`RAYON_NUM_THREADS=1`, ε=1e-10), independent DLR including τ and
  Matsubara nodes: Λ=1e3 10.7 ms, Λ=1e5 65 ms. Building the IR basis at the
  same ε (Df64 SVE) takes 1065 ms and 5281 ms.
- Rebuild on 0.10.0: the pole weights of an IR-derived DLR are the kernel
  regularizers, as on main (#286); construction errors are `sparse_ir::Error`
  (`DlrError` is gone). `ir_basis_size()` is `None` for an independent DLR, and
  `from_ir_nd` / `to_ir_nd` are `NotSupported` for it.

## Milestone C: ESPRIT module (2026-09-25)

- `sparse_ir::esprit`: block-Hankel ESPRIT with nodes shared across channels
  (input `[N, ...]`, trailing axes flattened), `ModelOrder::Fixed(r)` or
  `ModelOrder::Tolerance(rtol)` (σ_i > rtol·σ_0, optional `max_order`), default
  pencil `L ≈ N d/(d+1)`. Diagnostics: all Hankel singular values, order,
  pencil, σ_r/σ_0, max and relative residual. `EspritResult::evaluate`
  handles non-integer positions.
- Linear algebra: Hankel SVD and the r×r general eigenproblem go through
  tenferro-linalg (`svd`, `eig`). Least squares reuse the crate pinv factors.
- Rebuild on 0.10.0: argument errors are `Error::InvalidParameter` (named
  `samples`, `pencil`, `order`, `tolerance`) and failed decompositions
  `Error::DecompositionFailed`.

## Milestones D/E: MiniPole (2026-09-25)

- `sparse_ir::minipole::{minipole_from_dlr, minipole_from_matsubara}`, following
  Zhang & Gull (PRB 110, 035154): the Joukowski map sends a segment
  `[iν_a, iν_b]` of the imaginary axis onto the unit circle, the moments
  `h_k = Σ_l Ã_l ξ̃_l^k` are exact residue sums over the DLR poles, and ESPRIT
  (tolerance) gives the nodes. The DLR replaces the paper's first Prony step.
- Rejected designs, for the record:
  - Joukowski ellipse around `[-ωmax, ωmax]` (Laurent coefficients of `G`)
    and Möbius half-plane moments. Both put the contour near the real axis.
    There, a DLR fitted on Matsubara frequencies is not pinned. Example:
    β=50, ε=1e-12. The error is 0.6 at ν=0, 9e-4 at 1+0.3i and 4e-5 at z=2.2,
    but 1e-9 at ν=0.5 on the imaginary axis. The result was 15–18 spurious
    poles for a 3-pole spectrum.
- Segment defaults:
  - `ν_a = max(2.5 ln(1/tol)/β, π/β)` and `ν_b = ν_a + 10 ωmax`.
  - M is chosen from the slowest node modulus.
  - A scan on noisy data (η=1e-7) showed that longer segments and larger ν_a
    are both clearly better.
- Matsubara input is fitted with a truncated-SVD DLR fit, cutoff
  `max(dlr_accuracy, tol/100)`. An unregularized fit follows the noise, and
  σ0 of the moments rose from 0.5 to 127.
- Nodes whose pole is closer to the segment than to the real axis
  (`|Im ξ| >= ν_a/2`) are dropped, and the amplitudes are refitted.
- Accuracy: from an exact DLR, poles match to about 1e-7 (fermionic, bosonic,
  2x2 matrix). From noisy Matsubara data (η=1e-7), poles match to 2e-3.
- Rebuild on 0.10.0: shape errors are `Error::ShapeMismatch` of the input,
  option errors `Error::InvalidParameter` (named `tolerance`,
  `freq_min, freq_max`, `n_moments`), no frequencies `Error::EmptyInput`; DLR
  construction errors pass through unchanged.

## Milestone F: additive C API and wrappers (2026-09-25)

- New C entry points. All are additive, and no existing signature changed.
  - `spir_dlr_new_independent(statistics, beta, omega_max, epsilon, status)`
    builds the ID-based DLR without an IR basis. The IR route
    (`spir_dlr_new`, `spir_dlr_new_with_poles`) is unchanged.
  - `spir_pole_repr` opaque type: `_release`, `_clone` and `_is_assigned` are
    written out by hand, because cbindgen does not expand
    `impl_opaque_type_common!`.
  - `spir_minipole_from_dlr` and `spir_minipole_from_matsubara` follow the
    usual order/ndim/dims/target_dim convention. Pass `<= 0` for the
    optional parameters to use their defaults.
  - Getters: `spir_pole_repr_get_{npoles,poles,residues,dlr_fit_residual}`.
    Residues keep the caller layout, with the pole axis at `target_dim`.
- Behavior change: `spir_basis_get_{n_,}default_taus` and
  `spir_basis_get_{n_,}default_matsus` on a DLR handle now return the
  DLR's own nodes. The node count equals npoles. Before this change they
  returned 0 points. The `_ext` variants (size_requested and mitigate) still
  return 0 points for a DLR.
- Wrappers and generated bindings:
  - Python ctypes (`ctypes_autogen.py`), header and assets copy were
    regenerated.
  - Fortran `generate_c_binding.py` now resolves typedefs through their
    canonical type, maps `int64_t` to `c_int64_t`, and passes the clang
    resource dir and the macOS SDK to libclang. Without these, `bool` and
    `StatusCode` were misparsed.
  - Regenerating the Fortran bindings also fixed two existing binding bugs.
    `c_spir_basis_get_n_default_matsus_ext` was missing its `mitigate`
    argument, and the `n` argument of `spir_funcs_eval_matsu` was bound as
    `c_int` instead of `c_int64_t`.
- Tests:
  - Rust capi unit tests.
  - `python/tests/c_api/minipole_tests.py`: 5 tests; the full suite (69) passes.
  - C++ `cinterface_core.cxx`: the test case for spir_dlr_new_independent
    and MiniPole passes.
  - Fortran `test/test_minipole.f90`: 12/12 pass.
- Deferred (out of scope for this branch):
  - A C API for ESPRIT.
  - An evaluate function for `spir_pole_repr`.
  - DLR support in the `_ext` default-point variants.
  - High-level Julia and Python wrappers. Only the C API and ctypes are
    in place.
- Rebuild on 0.10.0: the C entry points report errors through main's
  `status_from`, and `spir_minipole_*` validate `input_dims` with
  `validate_dims` (a zero or negative extent is `SPIR_INVALID_DIMENSION`, as
  for every array of the C API). The status tests of 0.10.0 (S12 in C++,
  Fortran and Python) that asserted 0 default points for a DLR now assert one
  node per pole, and the C docs say so. The Fortran bindings keep main's
  argument names; main had already fixed the two binding bugs above.

## Final audit (2026-09-25, branch `work/tenferro-migration`)

- `cargo fmt --all --check`: clean.
- `cargo test --workspace --release`: all green (sparse-ir: 276 passed,
  5 ignored; sparse-ir-capi: 55 lib tests plus the integration tests).
- Clippy: the remaining warnings all predate this branch, for example
  `doc list item overindented` and `is_multiple_of` in `basis.rs`. The new
  modules (`minipole`, `esprit`, independent DLR, the capi `minipole.rs`)
  add no warnings.
- Wrapper suites: Python 69 passed; C++ `ctest` 2/2; Fortran 12/12.
- `bench_core` final run (`RAYON_NUM_THREADS=1`) matches the post-migration
  table. Selected times:
  - basis Λ=1e5 (incl. SVE): 5272 ms.
  - MatsubaraSampling::new: 372 ms.
  - tau fit extra=10000 dim=0 / dim=1: 6.53 / 6.42 ms.
  - matsu fit extra=10000 dim=0: 29.0 ms.
  - ir2dlr extra=10000 dim=1: 6.42 ms.
  - New: `DiscreteLehmannRepresentation::new` (independent, incl. nodes)
    at Λ=1e5: 65.5 ms. That is about 80x cheaper than going through the IR
    basis (5272 + 1.2 ms).
- Not merged to main. Julia and Python compatibility is kept because the
  C ABI changes are additive only.

## Batched fit/evaluate: Rust port of `test_timing` and a tenferro-einsum engine (2026-09-25)

`sparse-ir/examples/bench_batch.rs` is a port of `fortran/test/test_timing.f90`.

- Setup: Λ=1e6, ε=1e-8, β=100, L=96, ntau=96.
- Each run is `fit_matsubara → evaluate_tau → fit_tau → evaluate_matsubara` over `[lsize, npts]`, with pre=lsize.
- The calls go through `InplaceFitter::*_to`, as the C API does.
- Engines:
  - `--blas` injects `dgemm_`/`zgemm_` as the Fortran wrapper does.
  - `--engine=einsum|plan|plan1s` runs the same pinv two-step contractions as tenferro-einsum calls.
- With `--blas` the numbers match the Fortran benchmark on the branch (see below).
- Every timed loop has one untimed warm-up call, so the lazy pinv SVD is not measured.

Per-vector seconds, `RAYON_NUM_THREADS=1`, num=185640, Apple M-series.

"pos/real" is the positive-only pattern with real IR coefficients (zd/dd/dd/dz). "full/cplx" is full Matsubara with all four transforms zz. Both are fermionic; the bosonic numbers are the same.

| engine | pos/real l=1 | l=10 | l=120 | full/cplx l=1 | l=10 | l=120 |
|---|---|---|---|---|---|---|
| inhouse, faer (default) | 6.18e-6 | 2.81e-6 | 2.05e-6 | 1.83e-5 | 8.58e-6 | 6.42e-6 |
| inhouse, injected Accelerate BLAS | 3.44e-6 | 1.07e-6 | 3.82e-7 | 5.17e-5 | 6.69e-6 | 1.82e-6 |
| einsum string, faer | 5.10e-5 | 7.07e-6 | 2.77e-6 | 5.29e-5 | 1.19e-5 | 7.37e-6 |
| einsum prepared plan, faer | 3.34e-5 | 5.24e-6 | 2.57e-6 | 3.50e-5 | 1.01e-5 | 7.21e-6 |
| plan, one session per loop, faer | 3.30e-5 | 5.21e-6 | 2.59e-6 | 3.47e-5 | 1.02e-5 | 7.27e-6 |
| einsum string, Accelerate | 5.63e-5 | 6.68e-6 | 1.23e-6 | 1.00e-4 | 1.30e-5 | 2.81e-6 |
| plan, Accelerate | 3.59e-5 | 4.48e-6 | 1.02e-6 | 7.98e-5 | 1.09e-5 | 2.63e-6 |

The Accelerate build of tenferro needs `--features tenferro-linalg/blas-accelerate`. `tenferro-cpu/blas-accelerate` alone breaks the SVD at run time (tenferro-rs#1905).

Findings:

- **Einsum is not ready to replace the in-house fitters.**
  - Each prepared contraction costs about 4 µs fixed; string parsing adds ~3 µs (tenferro-rs#1906).
  - At lsize=1 a cycle is 6 contractions, so it is 5–10× slower than in-house.
  - With faer, even lsize=120 is 10–25% slower than in-house faer.
  - With Accelerate, lsize=120 is still 2.6× (real) and 1.4× (complex) slower than the injected-BLAS in-house path.
- Session entry with `with_threads(1)` costs nothing: plan and plan1s give the same numbers.
- Staying in-house means the default faer backend is about 5× slower than Accelerate for batched real transforms.
  - Accelerate zgemm is slower than faer at lsize=1, for the full complex pattern.
  - A possible follow-up: route small calls to faer even when a BLAS backend is injected.

## Crate split (2026-09-30)

The library is split into four crates plus the `sparse-ir` facade, which
re-exports every module and item under its previous path (`sparse_ir::basis`,
`sparse_ir::dlr`, `sparse_ir::FiniteTempBasis`, ...), so `sparse-ir-capi`,
the examples, the integration tests and the Rust tutorial compile unchanged.

```text
sparse-ir-core      error, traits, freq, taufuncs, gemm, matrix, fpu_check,
                    fitters, basis_trait (Basis), sampling, matsubara_sampling
sparse-ir-dlr       dlr, dlr_id                          -> core
sparse-ir-minipole  esprit, minipole                     -> dlr, core
sparse-ir-basis     numeric, gauss, col_piv_qr, tsvd, interpolation1d, kernel,
                    kernelmatrix, poly, polyfourier, special_functions, sve,
                    basis, ir_dlr                        -> dlr, core
sparse-ir           re-exports only; examples and integration tests
sparse-ir-capi      -> sparse-ir (C ABI unchanged)
```

- Stage 1 (single crate, commit "Decouple the DLR from the IR basis"):
  `Basis` lost `Kernel`/`kernel()`; the DLR no longer holds a kernel;
  `ir_dlr.rs` holds `IrBasis` (a basis with a kernel), `DlrFromIr`
  (`from_ir`, `from_ir_with_poles`) and `IrBasis::dlr_transform`.
- Stage 2 moves the files with `git mv`. Each crate imports the modules of
  the crates it depends on at its root (`use sparse_ir_core::{error, ...}`),
  so the moved code keeps its `crate::error::...` paths. `debug_warn!` and
  the internal `mat!` are `#[macro_export]` in the core.
- Internals of the core that the other crates need (`RealMatrixFitter`,
  `PinvFactors`, `compute_pinv*`, `require_*`, `freq::is_zero`,
  `sampling::mat_from_matrix`, ...) are `#[doc(hidden)] pub`.
- The unit tests that need an IR basis (τ/Matsubara sampling, the `Basis`
  trait, the IR-derived and independent DLR) moved to `sparse-ir-basis`,
  which tests the samplings and the DLR on an IR basis; the others stay
  with their crate. Counts are unchanged: library unit tests 407
  (basis 306, core 82, minipole 10, dlr 9), doctests 19 (core 15, dlr 4),
  integration tests 19.
- Doc examples keep their `sparse_ir::` paths: each library crate
  dev-depends on the facade (a dev-dependency cycle, which cargo allows and
  strips on publish).
- `system-blas` and `build.rs` live in `sparse-ir-core`; `sparse-ir`
  forwards the feature. The unused `special` and `statrs` dependencies are
  gone; `proptest` was unused and is no longer a dev-dependency.
- Release: `manual-release.yml` publishes core, dlr, minipole, basis,
  sparse-ir and capi in that order, waiting for each on crates.io. CI, the
  docs workflow, the local test gate, README and REPOSITORY_RULES use the new
  crate names.
- Dependency weight (`cargo tree -e normal`): sparse-ir-core 96 crates,
  sparse-ir-dlr 97, sparse-ir-minipole 98, sparse-ir-basis 109. A user of the
  DLR or MiniPole alone no longer pulls xprec, simba or nalgebra; tenferro
  (and faer) stay in the core by design.
