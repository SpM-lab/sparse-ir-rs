# MiniPole inputs and accuracy

Use the [MiniPole contour tutorial](https://spm-lab.github.io/sparse-ir-rs/tutorials/minipole.html)
for a runnable DLR-to-pole example and contour interpretation.

- `mini_pole_dlr_from` accepts DLR coefficients, not bare residues; it applies
  the DLR pole weights internally. `mini_pole_dlr` accepts actual residues and
  real pole locations.
- `nmax: None` uses the numerical value of `beta` as the contour cutoff. The
  tutorial's `n0=5`, `nmax=50` is specific to its analytic example, not a
  general recommendation.
- ESPRIT's tolerance selects a model order; it is not a bound on reconstruction
  error. Compare contours, inspect pole stability, and test at held-out
  frequencies, especially near zero.
- Direct Matsubara input requires a uniform, finite, non-negative,
  strictly-increasing frequency grid. Matrices use column-major channel layout.
