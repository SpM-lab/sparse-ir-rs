# sparse-ir-minipole

ESPRIT and MiniPole for [`sparse-ir`](https://crates.io/crates/sparse-ir):
compact pole representations of a Green's function from DLR coefficients or
from Matsubara data (L. Zhang and E. Gull, Phys. Rev. B 110, 035154 (2024);
L. Zhang, Y. Yu and E. Gull, Phys. Rev. B 110, 235131 (2024)).

Most users should depend on [`sparse-ir`](https://crates.io/crates/sparse-ir),
which re-exports this crate together with the rest of the library under the
`sparse_ir::` paths used in the guide.

## Documentation

- [Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/): tutorials and
  conventions, written against the 0.11 release.
- API reference: [docs.rs/sparse-ir-minipole](https://docs.rs/sparse-ir-minipole).
- Source: [SpM-lab/sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs).

## License

Dual-licensed under the MIT license and the Apache License (Version 2.0), at
your option.

The ESPRIT and MiniPole code is a port of
[Green-Phys/MiniPole](https://github.com/Green-Phys/MiniPole) (MIT License);
see `LICENSE-THIRD-PARTY`.
