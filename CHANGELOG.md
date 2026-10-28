# Changelog

User-visible changes of sparse-ir-rs (the Rust crates and the C API
`libsparseir`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The `sparse-ir-sve` crate: the SVE of integral kernels K(x, y), with the
  ACA-accelerated truncated SVD, and the numerics it is built on (Gauss
  quadrature, piecewise Legendre polynomials, `Df64`, the dense containers,
  the error type, the FPU guard, the rayon pool), without the physics of the
  IR basis, for use with other kernels. `sparse-ir-basis` re-exports its
  modules under the previous paths (`sparse_ir::sve`, `sparse_ir::gauss`,
  `sparse_ir::poly`, ...). Its `parallel` feature is off by default and
  turned on by that of `sparse-ir-core`. The machinery of the SVE (its
  strategies, the kernel matrices, the truncated SVD, the FPU guard, the
  thread pool) is public only under `sparse_ir_sve::unstable`, outside the
  semver guarantees of that crate; the `sparse_ir::` paths are unchanged.
- `compute_sve_dense`: the SVE by the dense column-pivoted QR of the fully
  formed blocks (what `compute_sve` does in f64), in f64 and double-double,
  to check the ACA path on a new kernel.
- `SegmentHints`: SVE hints given by the segment boundaries on the domain
  of a kernel (`new`), or by a number of segments of equal length
  (`uniform`), for any kernel, so that a kernel defined outside the crates
  needs no hints type of its own.
- `sve_residual`: max |K − Σ s_l u_l v_l| / s₀ of an SVE at points next to
  the ends and at the midpoint of every segment, which shows whether the
  segments of the hints resolve the kernel (`compute_sve` does not check
  it). Limited from below by the rounding of the evaluation in f64.
- `compute_sve_with` and `SVEOptions`: the SVE with options set by name
  (accuracy, cutoff, maximum number of singular values, working precision,
  full-domain expansion of a centrosymmetric kernel). `compute_sve` is
  unchanged. `compute_sve_general`, `compute_sve_for_basis` and
  `BASIS_CUTOFF_FACTOR` are deprecated in favor of
  `SVEOptions::full_domain`, `SVEOptions::cutoff_from_epsilon` and
  `ACCURACY_CUTOFF_FACTOR`.
- `SVEResult::truncate`: the expansion truncated at a relative threshold
  (its accuracy) and a maximum size, as an `SVEResult`. `SVEResult::part`,
  which returned the parts as a tuple, is deprecated.

### Changed

- The kernel traits are split into the mathematics of the SVE, defined in
  `sparse-ir-sve`, and the physics of the basis. `AbstractKernel` (values),
  `SVEKernel` (domain and SVE hints, taken out of `KernelProperties`),
  `ReducedKernel` (`compute_reduced`, taken out of `CentrosymmKernel`) and
  `ExpandableKernel` (the kernels `compute_sve` expands, by their symmetry,
  taken out of `BasisKernel`) come
  from `sparse-ir-sve`; `KernelProperties` (`ypower`, `conv_radius`,
  `regularizer`), `CentrosymmKernel` (`lambda`) and `BasisKernel`
  (`kernel_lambda`) stay in `sparse-ir-basis`, each extending the trait it
  was split from. Code that bounds on these traits is unchanged; a kernel
  defined outside the crates implements `SVEKernel` next to
  `KernelProperties`, `ReducedKernel` next to `CentrosymmKernel`, and, if it
  is not centrosymmetric, `ExpandableKernel` next to `BasisKernel`. Calling
  `xmax()` or `sve_hints()` on a concrete kernel needs `SVEKernel` in scope.
- `SVEHints::segments_x` and `segments_y` return the segment boundaries of
  the full domain `[xmin, xmax]` for every kernel, the centrosymmetric ones
  included (`LogisticSVEHints` and `RegularizedBoseSVEHints` returned those
  of `[0, 1]`); the SVE of a centrosymmetric kernel takes their half itself
  (`sparse_ir::sve::half_domain_segments`). `AbstractKernel::is_centrosymmetric`
  is deprecated: the SVE uses the symmetry of the kernels that implement
  `ReducedKernel`. `spir_kernel_get_sve_hints_segments_x` and `_y` of the
  C API are unchanged.
- The functions of the SVE, the quadrature and the polynomials return
  `sparse_ir_sve::Error`, which converts into `sparse_ir::Error` (`?`,
  `From`).

## [0.15.0] - 2026-10-31

### Added

- Fortran: `check_cpu_fma` and `fma_combination_is_supported`. `init_ir`
  calls the check before its first numerical call, so a library built with
  FMA stops with a message naming the requirement and the
  `RUSTFLAGS="-C target-feature=-fma"` build, instead of dying with SIGILL on
  a CPU without FMA (#353).

### Fixed

- C API: `spir_sve_result_from_matrix_centrosymmetric` ran its SVD on the
  matrix narrowed to `f64`, so the double-double low parts `K_even_low` /
  `K_odd_low` never reached the decomposition: the function delivered f64
  accuracy while its documentation promised DDouble, and the two entry
  points disagreed on precision for the same input. The decomposition now
  runs in `Df64`, as the non-centrosymmetric `spir_sve_result_from_matrix`
  already did; with f64 inputs the result is unchanged (#293).

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
- `LogisticKernelWindowed`: the logistic kernel on an asymmetric frequency
  window `[omega_min, omega_max]`, which need not contain 0 (e.g.
  `[0, omega_max]`), the first built-in kernel that is not
  centrosymmetric. Its SVE is taken over the full domain, and
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
  `spir_sampling_fit_zd`) solve the real least-squares problem. For points
  symmetric in ±n, such as the default ones, the result is unchanged (up
  to rounding) for any data; for other points see Fixed.
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

- Real-coefficient fits of `MatsubaraSampling` (`fit_real`, `fit_nd_real`,
  `spir_sampling_fit_zd`) at points not symmetric in ±n returned the real
  part of the complex least-squares solution, which with fewer points than
  functions does not fit its own input: at the positive halves of the
  default points, ‖A x − y‖/‖y‖ was 0.50 for data that real coefficients
  reproduce; it is now at rounding level.
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
