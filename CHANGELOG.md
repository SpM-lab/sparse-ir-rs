# Changelog

User-visible changes of sparse-ir-rs (the Rust crates and the C API
`libsparseir`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The `parallel` feature (opt-in, off by default): the SVE decomposes its
  blocks on the current rayon pool (`RAYON_NUM_THREADS` or
  `ThreadPool::install` bound it). The results do not depend on the number
  of threads. Without it, as before, the library runs on the calling
  thread; the C API never enables it and runs every call on the calling
  thread.
- `SVEStrategy::blocks` (provided method) and `KernelBlock`: the weighted
  kernel blocks of an SVE strategy, evaluated on demand.

### Changed

- The double-double SVE of `compute_sve` (ε below about 1e-8) computes the
  SVDs of its even and odd blocks by adaptive cross approximation (ACA),
  evaluating the kernel lazily: 7–11× faster on one thread (logistic
  kernel, Λ = 10⁴ to 10⁶, ε = 10⁻¹⁵). Against the dense column-pivoted
  QR, the singular values agree to the working precision relative to s₀
  and to 1e-15 relative over those a basis uses; their number can differ
  by one at the noise floor (a value near 2·ε_work·s₀, about 5e-32·s₀,
  e.g. for Λ = 3·10⁵ and 10⁶), so the length of the `SVEResult` can
  differ, but not the bases. The f64 SVE and `compute_sve_general` are
  unchanged (bitwise those of the previous release).
- Requires xprec 0.2.3.

### Changed in the C API

- `spir_sve_result_from_matrix` and
  `spir_sve_result_from_matrix_centrosymmetric` normalize the double-double
  pairs `(K_high, K_low)` passed in, and return `SPIR_INVALID_ARGUMENT` if
  their sum overflows. Normalized pairs, such as those of a correctly
  rounded double-double value, give the same result as before.

### Fixed

- The fitters return an error, not zeros or NaN, if the SVD of the sampling
  matrix has non-finite singular values.
