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

1. Session entry on a multi-threaded `CpuBackend` costs ~29 µs (~0.35 µs single-threaded).
   Libraries whose calls are small and frequent (sparse-ir fit on a single vector,
   called from C/Python/Julia) cannot hold a closure-scoped session across FFI calls.
   Wish: a cheap caller-thread/inline session, or a storable session handle.
2. Per-op overhead inside a session is 1–2.5 µs above direct faer for tiny GEMM (95x95·95x1).
3. No mainline extended-precision (double-double) scalar; `ext/df64-proof` is unpublished.
   sparse-ir needs Df64 SVD/QR for the SVE at ε < 1e-8.
4. Single-threaded thin SVD is 4–14% slower than calling faer 0.23 directly for 52x52–400x200.
5. `TypedTensor<T, Rank<2>>` is a natural matrix type, but element access is fallible and verbose:
   no `Index<[usize; 2]>`, `get2(i, j)` returns `Result<&T>`, and `host_col_major_view()`
   also returns `Result`. Numerical code with dense index loops gets `?`/`unwrap` noise.
   Wish: an infallible `Index`/`IndexMut` on host-owned compact tensors (and on `ColMajorView`).
6. Owned `TypedTensor<T>` requires `T: TensorScalar` (sealed). Extended-precision matrices
   (Df64 in the SVE) cannot use the same container, so sparse-ir keeps a small in-house
   column-major matrix for generic `T`. Wish: host-only storage for arbitrary `T: Copy`.
7. `TypedTensor` is not `Clone`; every copy is an explicit `duplicate()` (fallible for views).
   Structs holding a tensor cannot `#[derive(Clone)]`, which pushes the tensor behind `Arc`
   or forces hand-written `Clone` impls.
8. `TypedTensorView::duplicate()` and `as_slice()` reject non-contiguous (strided) views.
   There is no "gather a strided host view into a compact column-major buffer" primitive,
   so sparse-ir carries its own strided gather (`fitters::common::gather_col_major`).
   Wish: `view.to_col_major()` (or `duplicate()` that compacts).
9. `TypedTensor::get` / `get_mut` require `T: One + Zero` in addition to `TensorScalar`,
   which leaks into every generic helper that only wants to read an element.
10. tenferro-linalg entry points are `DynRank`-only; a `TypedTensor<T, Rank<2>>` matrix has
    to be converted before calling SVD/QR, and results come back as `DynRank`.
11. MSRV 1.96 and a pinned faer 0.24 in tenferro-cpu force the same toolchain/faer on
    downstream crates (sparse-ir previously used faer 0.23 directly).
12. Construction/view overhead is large for small tensors: `from_vec_col_major` costs
    ~210 ns and `zeros` ~210 ns (vs ~30 ns for the raw allocations), and `as_view`
    65–85 ns. For a 52-element tau evaluation this is the dominant cost
    (0.3 µs → 0.5 µs end to end). A cheap constructor for already-validated compact
    column-major buffers would remove most of it.

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

## Milestone C: ESPRIT module (2026-09-25)

- `sparse_ir::esprit`: block-Hankel ESPRIT with nodes shared across channels
  (input `[N, ...]`, trailing axes flattened), `ModelOrder::Fixed(r)` or
  `ModelOrder::Tolerance(rtol)` (σ_i > rtol·σ_0, optional `max_order`), default
  pencil `L ≈ N d/(d+1)`. Diagnostics: all Hankel singular values, order,
  pencil, σ_r/σ_0, max and relative residual. `EspritResult::evaluate`
  handles non-integer positions.
- Linear algebra: Hankel SVD and the r×r general eigenproblem go through
  tenferro-linalg (`svd`, `eig`). Least squares reuse the crate pinv factors.

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
