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
