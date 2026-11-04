# sparse-ir-sve

The singular value expansion (SVE) of integral kernels K(x, y), computed in
extended precision with an ACA-accelerated truncated SVD. It is the lowest
layer of [`sparse-ir`](https://crates.io/crates/sparse-ir), but does not depend
on statistics or Matsubara frequencies, so it can expand other kernels as well.

- `compute_sve` expands a kernel that implements `AbstractKernel` (its values
  K(x, y)), `SVEKernel` (its domain and the quadrature segments of its SVE)
  and either `ReducedKernel` (a centrosymmetric kernel, expanded in even and
  odd blocks) or `ExpandableKernel` (any other kernel, expanded over its
  full domain).
- `SegmentHints` discretizes a kernel on given segments (or segments of equal
  length) of its domain, so that a kernel needs no hints type of its own.
- `sve_residual` measures how well an SVE reproduces its kernel, which shows
  whether the segments of the hints resolve the kernel.
- In `f64` the SVE takes the SVDs of the discretized kernel by a dense
  column-pivoted QR, in double-double by adaptive cross approximation, which
  reads only a few of its rows and columns; `compute_sve_dense` takes the
  dense path in both, to check the SVE of a new kernel.
- The `parallel` feature (off by default) runs the SVE on the rayon thread
  pool.
- The singular functions are piecewise Legendre polynomials
  (`PiecewiseLegendrePolyVector`).
- The crate is mathematics only and defines no concrete kernel;
  `sparse-ir-basis` defines the kernels of the IR basis and builds the basis
  on top of this crate.
- The truncated SVD (`tsvd`), Gauss–Legendre quadrature (`gauss`) and
  double-double arithmetic (`Df64`) are available on their own.

The crate documentation has an example that expands a kernel defined outside
this crate.

## Documentation

- API reference: [docs.rs/sparse-ir-sve](https://docs.rs/sparse-ir-sve).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.

The `col_piv_qr` module is based on code from
[nalgebra](https://github.com/dimforge/nalgebra) (Apache License 2.0).
