# Changelog

User-visible changes of sparse-ir-rs (the Rust crates and the C API
`libsparseir`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- `LogisticKernelWindowed`: the logistic kernel on an asymmetric frequency
  window `[omega_min, omega_max]` containing 0, the first built-in kernel
  that is not centrosymmetric. Its SVE is taken over the full domain, and
  `FiniteTempBasis` works on the kernel's domain (`omega_min()`; the DLR
  rejects poles outside the window). Not yet available through the C API.
- Default Matsubara sampling points for bases whose functions have no
  definite parity (`LogisticKernelWindowed`, `compute_sve_general`,
  `spir_sve_result_from_matrix`): a set symmetric in ±n selected in pairs
  by an interpolative decomposition (pivoted QR of the rows, as Kaye, Chen
  and Parcollet do for the DLR); with `mitigate`, about 10% more
  frequencies that minimize the noise variance of the fit. They were
  `NotSupported` (#183).
- `MatsubaraPointMethod` (`SignChanges`, `PairSelection`) with
  `FiniteTempBasis::matsubara_sampling_points` and
  `matsubara_sampling_points_from_uhat`, to choose the method of the
  Matsubara points explicitly, e.g. the pair selection for a basis of a
  centrosymmetric kernel.
- `BasisKernel`: the kernels `compute_sve`, `FiniteTempBasis` and `IrBasis`
  accept. It is implemented for every `CentrosymmKernel` (plus
  `KernelProperties + Clone + 'static`), so every kernel accepted before
  still is, and for `LogisticKernelWindowed`.
- `KernelProperties::xmin()` and `ymin()`, defaulting to the symmetric
  domain.
- `compute_sve_for_basis` and `BASIS_CUTOFF_FACTOR`: an SVE that resolves
  only the singular values a basis for ε uses, down to
  `max(2·ε_work, 1e-10·ε)`; in double-double 1.5–2.3× faster than
  `compute_sve`, with about 20% fewer singular values. Opt-in: the default
  SVE is unchanged.

### Changed

- The double-double SVE of `compute_sve` (ε below about 1e-8) computes the
  SVDs of its even and odd blocks by adaptive cross approximation (ACA),
  evaluating the kernel lazily: 7–11× faster on one thread (logistic
  kernel, Λ = 10⁴ to 10⁶, ε = 10⁻¹⁵). Against the dense column-pivoted
  QR, the singular values agree to the working precision relative to s₀
  and to 1e-15 relative over those a basis uses; their number can differ
  by one at the noise floor (a value near 2·ε_work·s₀, about 5e-32·s₀,
  e.g. for Λ = 3·10⁵ and 10⁶), so the length of the `SVEResult` can
  differ, but not the bases. The f64 SVE is unchanged (bitwise that of
  the previous release).
- `compute_sve_general` computes the double-double SVE of a kernel that is
  not centrosymmetric by ACA as well (for a centrosymmetric kernel it
  keeps the dense column-pivoted QR, whose single block holds nearly
  degenerate pairs of singular values).
- `FiniteTempBasis::from_sve_result` accepts an SVE without parity for a
  `LogisticKernel` (it was `NotSupported`).
- The default Matsubara points are computed once per basis.
- The default sampling points of bases of centrosymmetric kernels are
  unchanged (Matsubara exactly, tau to rounding).
- Real-coefficient fits of `MatsubaraSampling` (`fit_real`, `fit_nd_real`,
  `spir_sampling_fit_zd`) solve the real least-squares problem; for points
  symmetric in ±n, such as the default ones, the result is the same up to
  rounding.
- Requires xprec 0.2.3.

### Changed in the C API

- `spir_basis_get_n_default_matsus`, `spir_basis_get_default_matsus` (and
  their `_ext` variants) and `spir_uhat_get_default_matsus` now return
  `SPIR_COMPUTATION_SUCCESS` with the pair-selected points for bases whose
  functions have no definite parity (from `spir_sve_result_from_matrix`),
  instead of `SPIR_NOT_SUPPORTED`.
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

- The asymptotic Matsubara model of functions without definite parity
  (wrong values at |n| ≥ n_asymp).
- `CentrosymmSVE::new` (and so `compute_sve`) rejects a centrosymmetric
  kernel whose domain is not symmetric, instead of returning the SVE of
  another kernel.
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
