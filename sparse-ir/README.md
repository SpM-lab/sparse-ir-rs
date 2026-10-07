# sparse-ir

[![Crates.io](https://img.shields.io/crates/v/sparse-ir.svg)](https://crates.io/crates/sparse-ir)
[![Documentation](https://docs.rs/sparse-ir/badge.svg)](https://docs.rs/sparse-ir)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

IR basis, DLR, ESPRIT/MiniPole and sparse sampling for imaginary-time Green's
functions, in pure Rust.

## Features

All of these are in the 0.12 release on crates.io:

- Intermediate representation (IR) basis for fermionic and bosonic statistics
- Discrete Lehmann representation (DLR), built independently of an IR basis
  (`DiscreteLehmannRepresentation::new`, `DlrBuilder`) or from one (`from_ir`)
- ESPRIT and MiniPole pole reconstruction (`sparse_ir::esprit`,
  `sparse_ir::minipole`)
- Sparse sampling in imaginary time and Matsubara frequencies

## Documentation

The **[Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/)** has
tutorials with figures and runnable examples, written against the 0.12
release; start with its
[installation](https://spm-lab.github.io/sparse-ir-rs/getting-started/installation.html)
and [conventions](https://spm-lab.github.io/sparse-ir-rs/getting-started/conventions.html)
pages. The API reference for the release is on
[docs.rs/sparse-ir](https://docs.rs/sparse-ir); the guide also hosts the
[API reference built from `main`](https://spm-lab.github.io/sparse-ir-rs/api/sparse_ir/index.html).
For AI-assisted Rust use, load the [SparseIR Rust usage
skill](https://github.com/SpM-lab/sparse-ir-rs/blob/main/agent-skills/sparse-ir-rust-usage/SKILL.md)
when choosing crates, imports, or numerical conventions.

## Installation

`sparse-ir` requires Rust 1.96 or newer. Add it to your `Cargo.toml`:

```toml
[dependencies]
sparse-ir = "0.12.0"
```

### Optional: system BLAS

By default, `sparse-ir` uses [faer](https://crates.io/crates/faer) (pure Rust)
for the matrix products in `fit` and `evaluate`. To route them through the
LP64 BLAS installed on your system (e.g. OpenBLAS, Intel MKL), enable the
`system-blas` feature:

```toml
[dependencies]
sparse-ir = { version = "0.12.0", features = ["system-blas"] }
```

Both backends compute the same numbers; the feature only changes which
implementation multiplies the matrices.

## Usage

Matsubara frequencies are indexed by the reduced integer `n`, with
iν_n = i n π/β (`n` odd for fermions, even for bosons). Imaginary-time
sampling points lie in [−β/2, β/2]. See the guide's
[conventions](https://spm-lab.github.io/sparse-ir-rs/getting-started/conventions.html)
page for details.

### Quick start: fit and evaluate with the IR basis

Build an IR basis, sample a Green's function G(τ) on the basis's τ points,
fit its IR coefficients g_l, and evaluate them on the Matsubara sampling
points. The model is a single pole at ω = 0.5, whose exact values are known
on both axes.

```rust
use sparse_ir::{
    FermionicBasis, LogisticKernel, MatsubaraSampling, TauSampling, fermionic_single_pole,
    giwn_single_pole,
};

fn main() -> Result<(), sparse_ir::Error> {
    let (beta, wmax, omega) = (10.0, 1.0, 0.5);

    // IR basis for Λ = β ω_max, truncated at s_l / s_0 < 1e-10
    let kernel = LogisticKernel::new(beta * wmax)?;
    let basis = FermionicBasis::new(kernel, beta, Some(1e-10), None)?;

    // G(τ) on the τ sampling points → IR coefficients g_l
    let tau_sampling = TauSampling::new(&basis)?;
    let g_tau = tau_sampling
        .sampling_points()
        .iter()
        .map(|&tau| fermionic_single_pole(tau, omega, beta))
        .collect::<Result<Vec<_>, _>>()?;
    let g_l = tau_sampling.fit(&g_tau)?;

    // g_l → G(iν_n) on the Matsubara sampling points
    let matsubara_sampling = MatsubaraSampling::new(&basis)?;
    let g_iw = matsubara_sampling.evaluate_real(&g_l)?;
    for (freq, value) in matsubara_sampling.sampling_points().iter().zip(&g_iw) {
        let exact = giwn_single_pole(freq, omega, beta)?;
        assert!((value - exact).norm() < 1e-8);
    }
    Ok(())
}
```

### DLR without an IR basis

The DLR represents G as a sum of poles ω_p with coefficients c_p.
`DiscreteLehmannRepresentation::new(beta, wmax, eps)` chooses the poles
directly, without building an IR basis, and works with the same sampling
types.

```rust
use sparse_ir::{
    DiscreteLehmannRepresentation, Fermionic, MatsubaraSampling, TauSampling,
    fermionic_single_pole, giwn_single_pole,
};

fn main() -> Result<(), sparse_ir::Error> {
    let (beta, wmax, omega) = (10.0, 1.0, 0.5);

    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(beta, wmax, 1e-10)?;
    println!("{} poles", dlr.poles().len());

    // G(τ) on the DLR's τ nodes → pole coefficients c_p
    let tau_sampling = TauSampling::new(&dlr)?;
    let g_tau = tau_sampling
        .sampling_points()
        .iter()
        .map(|&tau| fermionic_single_pole(tau, omega, beta))
        .collect::<Result<Vec<_>, _>>()?;
    let c_p = tau_sampling.fit(&g_tau)?;

    // c_p → G(iν_n) on the DLR's Matsubara nodes
    let matsubara_sampling = MatsubaraSampling::new(&dlr)?;
    let g_iw = matsubara_sampling.evaluate_real(&c_p)?;
    for (freq, value) in matsubara_sampling.sampling_points().iter().zip(&g_iw) {
        let exact = giwn_single_pole(freq, omega, beta)?;
        assert!((value - exact).norm() < 1e-8);
    }
    Ok(())
}
```

More examples, including IR ↔ DLR transforms, MiniPole and applied
calculations, are in the [user guide](https://spm-lab.github.io/sparse-ir-rs/).

## Crate layout

`sparse-ir` re-exports the crates it is made of, under their usual paths; you
can also depend on them directly:

- [`sparse-ir-core`](https://crates.io/crates/sparse-ir-core): statistics,
  errors, GEMM, fitters, the `Basis` trait and sparse sampling;
- [`sparse-ir-basis`](https://crates.io/crates/sparse-ir-basis): kernels, the
  singular value expansion and the IR basis `FiniteTempBasis`;
- [`sparse-ir-dlr`](https://crates.io/crates/sparse-ir-dlr): the discrete
  Lehmann representation;
- [`sparse-ir-minipole`](https://crates.io/crates/sparse-ir-minipole): ESPRIT
  and minimal pole representations.

## Dependencies

- **Tensors and linear algebra**: [tenferro-rs](https://github.com/tensor4all/tenferro-rs) (CPU) + [faer](https://crates.io/crates/faer)
- **Extended precision**: [xprec-rs](https://github.com/tuwien-cms/xprec-rs)

## License

This crate is dual-licensed under the terms of the MIT license and the Apache License (Version 2.0).

- You may use this crate under the terms of either license, at your option:
  - [MIT License](https://github.com/SpM-lab/sparse-ir-rs/blob/main/LICENSE)
  - [Apache License 2.0](https://github.com/SpM-lab/sparse-ir-rs/blob/main/LICENSE-APACHE)

### Third-party licenses

The `col_piv_qr` module of `sparse-ir-basis` is based on code from the [nalgebra](https://github.com/dimforge/nalgebra) library, which is licensed under the Apache License 2.0:

- **nalgebra**: Apache License 2.0
  - Original source: `nalgebra/src/linalg/col_piv_qr.rs`
  - Copyright 2020 Sébastien Crozet
  - See [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) for details

Modifications and additions to the nalgebra code (including early termination support) are available under the same dual license as this crate (MIT OR Apache-2.0).

The ESPRIT and MiniPole code of `sparse-ir-minipole` is a port of
[MiniPole](https://github.com/Green-Phys/MiniPole) (MIT License); see
`sparse-ir-minipole/LICENSE-THIRD-PARTY`.

## Contributing

Contributions are welcome! Please see the [development guide](https://github.com/SpM-lab/sparse-ir-rs#development) for details.

## References

- **sparse-ir: optimal compression and sparse sampling of many-body propagators**  
  Markus Wallerberger, Samuel Badr, Shintaro Hoshino, Fumiya Kakizawa, Takashi Koretsune, Yuki Nagai, Kosuke Nogaki, Takuya Nomoto, Hitoshi Mori, Junya Otsuki, Soshun Ozaki, Rihito Sakurai, Constanze Vogel, Niklas Witt, Kazuyoshi Yoshimi, Hiroshi Shinaoka  
  [arXiv:2206.11762](https://arxiv.org/abs/2206.11762) | [SoftwareX 21, 101266 (2023)](https://doi.org/10.1016/j.softx.2022.101266)
- Python wrapper: [sparse-ir](https://github.com/SpM-lab/sparse-ir)
- Julia wrapper: [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
