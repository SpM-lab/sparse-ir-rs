# Changelog

User-visible changes of sparse-ir-rs (the Rust crates and the C API
`libsparseir`). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `LogisticKernelWindowed`: the logistic kernel on an asymmetric frequency
  window `[omega_min, omega_max]` containing 0, the first built-in kernel
  that is not centrosymmetric. Its SVE is taken over the full domain, and
  `FiniteTempBasis` works on the kernel's domain (`omega_min()`; the DLR
  rejects poles outside the window). Not yet available through the C API.
- Default Matsubara sampling points for bases whose functions have no
  definite parity (`LogisticKernelWindowed`, `compute_sve_general`,
  `spir_sve_result_from_matrix`): a set symmetric in ±n selected by pairs
  (bias + variance criterion of Box and Draper). They were `NotSupported`
  (#183).
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
- The `parallel` feature (on by default in the `sparse-ir` crate only): the
  SVE and the batched Matsubara evaluations run on the current rayon pool
  (`RAYON_NUM_THREADS` or `ThreadPool::install` bound it). The results do
  not depend on the number of threads. The C API is built without it and
  runs every call on the calling thread.

### Changed

- The double-double SVE (ε below about 1e-8) computes the SVDs of its
  blocks by adaptive cross approximation (ACA), evaluating the kernel
  lazily: 11–16× faster on one thread, with the same number of singular
  values as the dense column-pivoted QR in every case tested (logistic,
  regularized Bose and windowed kernels, Λ = 10 to 1e5; logistic and
  windowed at 1e6), the values agreeing to
  the working precision relative to s₀ and to 1e-15 relative over those a
  basis uses. The f64 SVE is unchanged (bitwise that of the previous
  release).
- The default sampling points of bases of centrosymmetric kernels are
  unchanged, bitwise (Matsubara and tau).
- Real-coefficient fits of `MatsubaraSampling` (`fit_real`, `fit_nd_real`,
  `spir_sampling_fit_zd`) solve the real least-squares problem; for points
  symmetric in ±n, such as the default ones, the result is the same up to
  rounding.
- The pseudo-inverse of the fitters drops singular values below
  `max(n_rows, n_cols)·ε·s₀` (the minimum-norm least-squares solution), so
  rank-deficient sampling matrices give finite coefficients.
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
  their sum overflows.

### Fixed

- The asymptotic Matsubara model of functions without definite parity
  (wrong values at |n| ≥ n_asymp).
- The fitters return an error, not zeros, if the SVD of the sampling matrix
  has non-finite singular values.
- `CentrosymmSVE::new` (and so `compute_sve`) rejects a centrosymmetric
  kernel whose domain is not symmetric, instead of returning the SVE of
  another kernel.
