# sparseir-rust

[![Crates.io](https://img.shields.io/crates/v/sparseir-rust.svg)](https://crates.io/crates/sparseir-rust)
[![Documentation](https://docs.rs/sparseir-rust/badge.svg)](https://docs.rs/sparseir-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Rust implementation of the SparseIR (Sparse Intermediate Representation) library, providing analytical continuation and sparse representation functionality for quantum many-body physics calculations.

## Features

- **Finite Temperature Basis**: Bosonic and fermionic basis representations
- **Singular Value Expansion (SVE)**: Efficient kernel decomposition
- **Discrete Lehmann Representation (DLR)**: Sparse representation of Green's functions
- **Piecewise Legendre Polynomials**: High-precision interpolation
- **Sparse Sampling**: Efficient sampling in imaginary time and Matsubara frequencies
- **High-Performance Linear Algebra**: Built on Faer for pure Rust performance

## Installation

### As a Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
sparseir-rust = "0.1.0"
```

### As a Shared Library

The library can be built as a shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows) for use with other languages:

```bash
# Build shared library
cargo build --release

# The shared library will be available at:
# target/release/libsparseir_rust.so (Linux)
# target/release/libsparseir_rust.dylib (macOS)
# target/release/sparseir_rust.dll (Windows)
```

## Usage

### Basic Example

```rust
use sparseir_rust::*;

// Create a finite temperature basis
let basis = FiniteTempBasis::new(10.0, 100, Statistics::Fermionic);

// Generate sampling points
let sampling = TauSampling::new(&basis);

// Use the basis for calculations
let tau_points = sampling.tau_points();
println!("Generated {} sampling points", tau_points.len());
```

### SVE Example

```rust
use sparseir_rust::*;

// Create a kernel for analytical continuation
let kernel = LogisticKernel::new(1.0, 0.1);

// Compute SVE
let sve_result = compute_sve(&kernel, 100, 1e-12);

println!("SVE computed with {} singular values", sve_result.singular_values.len());
```

## API Documentation

The complete API documentation is available at [docs.rs/sparseir-rust](https://docs.rs/sparseir-rust).

## Performance

This implementation is optimized for high performance:

- **Pure Rust**: No external C/C++ dependencies for core functionality
- **SIMD Optimized**: Uses Faer for vectorized linear algebra
- **Memory Efficient**: Sparse representations minimize memory usage
- **Parallel Processing**: Rayon-based parallelization where beneficial

## Dependencies

- **Linear Algebra**: [mdarray](https://crates.io/crates/mdarray) + [Faer](https://crates.io/crates/faer)
- **Extended Precision**: [xprec-rs](https://github.com/tuwien-cms/xprec-rs)
- **Special Functions**: [special](https://crates.io/crates/special)
- **Parallel Processing**: [rayon](https://crates.io/crates/rayon)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

The `col_piv_qr` module is based on code from the [nalgebra](https://github.com/dimforge/nalgebra) library, which is licensed under the Apache License 2.0:

- **nalgebra**: Apache License 2.0
  - Original source: `nalgebra/src/linalg/col_piv_qr.rs`
  - Copyright 2020 Sébastien Crozet
  - See [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) for details

Modifications and additions to the nalgebra code (including early termination support) are licensed under the MIT License as part of this project.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## References

- SparseIR: [arXiv:2007.09002](https://arxiv.org/abs/2007.09002)
- Original C++ implementation: [libsparseir](https://github.com/SpM-lab/libsparseir)
- Julia implementation: [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
