# sparse-ir

[![Crates.io](https://img.shields.io/crates/v/sparse-ir.svg)](https://crates.io/crates/sparse-ir)
[![Documentation](https://docs.rs/sparse-ir/badge.svg)](https://docs.rs/sparse-ir)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

A high-performance Rust implementation of the SparseIR (Sparse Intermediate Representation) library, providing analytical continuation and sparse representation functionality for quantum many-body physics calculations.

## Features

- Intermediate Representation (IR) basis for fermionic and bosonic statistics
- Discrete Lehmann Representation (DLR) basis for fermionic and bosonic statistics
- Sparse Sampling in imaginary time and Matsubara frequencies

## Installation

### As a Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
sparse-ir = "0.1.0"
```

## Usage

### Basic Example

```rust
use sparse_ir::*;

// Create a finite temperature basis
let beta = 10.0;
let lambda = 10.0; // beta * omega_max
let kernel = LogisticKernel::new(lambda);
let basis = FermionicBasis::new(kernel, beta, None, None);

// Generate sampling points
let sampling = TauSampling::new(&basis);

// Use the basis for calculations
let tau_points = sampling.sampling_points();
println!("Generated {} sampling points", tau_points.len());
```

### SVE Example

```rust
use sparse_ir::*;

// Create a kernel for analytical continuation
let kernel = LogisticKernel::new(1.0);

// Compute SVE
let sve_result = compute_sve(kernel, 1e-12, None, Some(100), TworkType::Auto);

println!("SVE computed with {} singular values", sve_result.s.len());
```

## API Documentation

The complete API documentation is available at [docs.rs/sparse-ir](https://docs.rs/sparse-ir).

## Performance

This implementation is optimized for high performance:

- **Pure Rust**: No external C/C++ dependencies for core functionality
- **SIMD Optimized**: Uses Faer for matrix-matrix products (evaluate and fit routines). Optionally, system BLAS can be used for these operations for better performance.

## Dependencies

- **Linear Algebra**: [mdarray](https://crates.io/crates/mdarray) + [Faer](https://crates.io/crates/faer)
- **Extended Precision**: [xprec-rs](https://github.com/tuwien-cms/xprec-rs)
- **Special Functions**: [special](https://crates.io/crates/special)

## License

This crate is dual-licensed under the terms of the MIT license and the Apache License (Version 2.0).

- You may use this crate under the terms of either license, at your option:
  - [MIT License](../LICENSE)
  - [Apache License 2.0](../LICENSE-APACHE)

### Third-Party Licenses

The `col_piv_qr` module is based on code from the [nalgebra](https://github.com/dimforge/nalgebra) library, which is licensed under the Apache License 2.0:

- **nalgebra**: Apache License 2.0
  - Original source: `nalgebra/src/linalg/col_piv_qr.rs`
  - Copyright 2020 Sébastien Crozet
  - See [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) for details

Modifications and additions to the nalgebra code (including early termination support) are available under the same dual license as this crate (MIT OR Apache-2.0).

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## References

- **sparse-ir: optimal compression and sparse sampling of many-body propagators**  
  Markus Wallerberger, Samuel Badr, Shintaro Hoshino, Fumiya Kakizawa, Takashi Koretsune, Yuki Nagai, Kosuke Nogaki, Takuya Nomoto, Hitoshi Mori, Junya Otsuki, Soshun Ozaki, Rihito Sakurai, Constanze Vogel, Niklas Witt, Kazuyoshi Yoshimi, Hiroshi Shinaoka  
  [arXiv:2206.11762](https://arxiv.org/abs/2206.11762) | [SoftwareX 21, 101266 (2023)](https://doi.org/10.1016/j.softx.2022.101266)
- Python wrapper: [sparse-ir](https://github.com/SpM-lab/sparse-ir)
- Julia wrapper: [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
