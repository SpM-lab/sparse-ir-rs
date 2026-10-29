# Changelog

User-visible changes of sparse-ir-rs (the Rust crates and the C API
`libsparseir`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.14.0] - 2026-10-29

### Added

- The `parallel` feature (opt-in, off by default): the SVE decomposes its
  blocks on the current rayon pool (`RAYON_NUM_THREADS` or
  `ThreadPool::install` bound it). The results do not depend on the number
  of threads. Without it, as before, the SVE runs on the calling thread;
  the C API never enables it and keeps the parallel regions of sparse-ir on
  the calling thread. The SVDs of the fitters and of MiniPole (tenferro's
  `CpuBackend`) are unchanged: as before, they use a thread pool of their
  own, sized by `RAYON_NUM_THREADS` or all cores (#344).
- `SVEStrategy::blocks` (provided method) and `KernelBlock`: the weighted
  kernel blocks of an SVE strategy, evaluated on demand.
- `TworkType::auto_for(epsilon)`: the working precision that
  `TworkType::Auto` selects, or `Error::InvalidParameter` for an epsilon
  outside (0, 1) (#345).
- C API: `spir_build_uses_fma()` and `spir_cpu_has_fma()`. A binding that
  finds `spir_build_uses_fma() && !spir_cpu_has_fma()` can raise a clear
  error at import instead of the process dying with SIGILL on the first FMA
  instruction. `pylibsparseir` does so (#332).

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
- `spir_choose_working_type` returns `SPIR_INVALID_ARGUMENT` for an epsilon
  that no SVE accepts (NaN, zero, negative, infinite, or 1 or more) instead
  of a precision; it used to return `SPIR_TWORK_FLOAT64X2` for NaN, 0 and
  negative values. Valid epsilons give the same precision as before (#345).

### Fixed

- The fitters return an error, not zeros or NaN, if the SVD of the sampling
  matrix has non-finite singular values.
- C API: `spir_sve_result_from_matrix` and
  `spir_sve_result_from_matrix_centrosymmetric` return
  `SPIR_INVALID_ARGUMENT` for an `order` other than the two
  `SPIR_ORDER_*` constants; they read the matrix as row-major (#325).
- C API: `spir_gauss_legendre_rule_piecewise_double` and `_ddouble` return
  `SPIR_INVALID_DIMENSION`, before reading `segments`, when
  `n_segments + 1` or `n * n_segments` overflows, and write no output on
  any error (#292).
- C API: the documentation of `spir_basis_get_default_matsus` states the
  buffer size it needs, the count of `spir_basis_get_n_default_matsus`,
  which can exceed the basis size (#347).
- The Fortran C bindings and the Python ctypes declarations are checked
  against the header in CI (#294).
