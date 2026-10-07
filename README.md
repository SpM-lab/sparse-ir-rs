SparseIR Rust Workspace
=======================

[![sparse-ir](https://img.shields.io/crates/v/sparse-ir.svg?label=sparse-ir)](https://crates.io/crates/sparse-ir)
[![sparse-ir-capi](https://img.shields.io/crates/v/sparse-ir-capi.svg?label=sparse-ir-capi)](https://crates.io/crates/sparse-ir-capi)
[![docs.rs sparse-ir](https://docs.rs/sparse-ir/badge.svg)](https://docs.rs/sparse-ir)
[![docs.rs sparse-ir-capi](https://docs.rs/sparse-ir-capi/badge.svg)](https://docs.rs/sparse-ir-capi)

Rust implementation of the intermediate representation (IR), the discrete Lehmann representation (DLR, built with or without an IR basis), ESPRIT/MiniPole, and sparse sampling for imaginary-time Green's functions. All of these are in the 0.12 release on crates.io.

Most end users should start from the ecosystem documentation / tutorials and use the full-featured Python/Julia libraries; this workspace focuses on Rust crates and low-level bindings.

## Documentation

**Start here for Rust:** **[Rust User Guide](https://spm-lab.github.io/sparse-ir-rs/)** — browser-readable tutorials with figures, runnable examples, and conventions.

| Resource | Description |
|----------|-------------|
| **[Rust Tutorials](https://spm-lab.github.io/sparse-ir-rs/)** | Sparse sampling, IR/DLR transformations, MiniPole, and applied examples |
| **[MiniPole](https://spm-lab.github.io/sparse-ir-rs/tutorials/minipole.html)** | A few complex poles from Matsubara data (ESPRIT); entry points and contour choice |
| **[Current Rust API](https://spm-lab.github.io/sparse-ir-rs/api/sparse_ir/index.html)** | API documentation built from `main` |
| **[Ecosystem Documentation](https://spm-lab.github.io/sparse-ir-doc/)** | Theory and usage across languages |
| **[IR, DLR and MiniPole: history and comparison](https://spm-lab.github.io/sparse-ir-doc/src/history_comparison.html)** | How the three representations are related and when to use which |
| **[Python/Julia Tutorials](https://spm-lab.github.io/sparse-ir-tutorial-v2/)** | Interactive tutorials with Jupyter notebooks |
| **[Rust API: sparse-ir (docs.rs)](https://docs.rs/sparse-ir)** | Core Rust crate API documentation |
| **[Rust API: sparse-ir-capi (docs.rs)](https://docs.rs/sparse-ir-capi)** | C-API Rust crate API documentation |

The Rust guide is written against the 0.12 release on crates.io, which docs.rs
documents. The guide's "Current Rust API" link is built from `main` and may run
ahead of the release.

### Tutorial sources and local browsing

- [Book contents](docs/book/src/SUMMARY.md): Markdown chapters and figures in `docs/book/`.
- [Runnable Rust examples](docs/tutorial-code/src/bin/): the source of the included code and CSV data.
- [Plotting scripts](docs/plotting/): render figures from those CSVs.

With [mdBook 0.5.2](https://rust-lang.github.io/mdBook/) installed, browse locally:

```bash
mdbook serve docs/book --open
```

To reproduce the MiniPole figures from the repository root:

```bash
cargo run --manifest-path docs/tutorial-code/Cargo.toml --profile ci --locked --bin minipole
uv run --project docs/plotting python docs/plotting/minipole_plot.py
```

## Quick start (Rust)

- Add the crate from crates.io (requires **Rust 1.96 or newer**):

  ```toml
  [dependencies]
  sparse-ir = "0.12.0"
  ```

  The [installation guide](https://spm-lab.github.io/sparse-ir-rs/getting-started/installation.html) covers the optional `system-blas` feature and the Git dependency on `main`; the [`sparse-ir` README](sparse-ir/README.md) has fit/evaluate and DLR examples.
- From a checkout, run the round-trip example (DLR/IR/sampling):

```bash
cargo run --example roundtrip --release
```

## What’s in this workspace

Rust users typically depend on the crates below. Users of other languages typically use the bindings (via the C API).

### Rust crates

- **`sparse-ir`** — Rust implementation of the IR basis, DLR, MiniPole and sampling ([README](sparse-ir/README.md), [docs.rs](https://docs.rs/sparse-ir)). It re-exports the crates it is made of; all of them are published on crates.io and can also be used on their own:
  - [`sparse-ir-core`](sparse-ir-core/README.md) — statistics, errors, GEMM, fitters, the `Basis` trait and sparse sampling
  - [`sparse-ir-dlr`](sparse-ir-dlr/README.md) — the discrete Lehmann representation (no IR basis needed)
  - [`sparse-ir-minipole`](sparse-ir-minipole/README.md) — ESPRIT and minimal pole representations
  - [`sparse-ir-basis`](sparse-ir-basis/README.md) — kernels, the singular value expansion and the IR basis
- **`sparse-ir-capi`** — Rust crate providing a C-compatible API (shared library + C header) ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi))

### Bindings (other languages)

| Language | Component | Notes |
|----------|----------|-------|
| **C/C++** | `sparse-ir-capi` | C-compatible shared library ([README](sparse-ir-capi/README.md), [docs.rs](https://docs.rs/sparse-ir-capi)) |
| **Fortran** | `fortran/` | Bindings via C-API ([README](fortran/README.md)) |
| **Python** | `pylibsparseir` | Thin wrapper for the C-API (not user-facing) ([README](python/README.md)) |

## Full-featured wrappers (external)

Note: the name **sparse-ir** is used both for the Rust crate (`sparse-ir`) and the full-featured Python project below.

| Project | Language | Notes |
|---------|----------|-------|
| [**sparse-ir (Python)**](https://github.com/SpM-lab/sparse-ir) | Python | Full-featured Python library (recommended for most users) |
| [**SparseIR.jl (Julia)**](https://github.com/SpM-lab/SparseIR.jl) | Julia | Full-featured Julia implementation |

## Examples

### Rust examples

| Example | Description |
|---------|-------------|
| [`sparse-ir/examples/roundtrip.rs`](sparse-ir/examples/roundtrip.rs) | Complete DLR/IR/sampling cycle with round-trip tests |
| [`sparse-ir/README.md`](sparse-ir/README.md) | IR fit/evaluate and DLR quick starts, compiled as doctests |

### Fortran examples

| Example | Description |
|---------|-------------|
| [`fortran/examples/second_order_perturbation_fort.f90`](fortran/examples/second_order_perturbation_fort.f90) | Second-order perturbation theory |

### C/C++ integration tests

| Test | Description |
|------|-------------|
| [`cxx_tests/cinterface_core.cxx`](cxx_tests/cinterface_core.cxx) | Core C-API tests |
| [`cxx_tests/cinterface_integration.cxx`](cxx_tests/cinterface_integration.cxx) | DLR/IR round-trip tests |

## References

- **sparse-ir: optimal compression and sparse sampling of many-body propagators**  
  Markus Wallerberger, Samuel Badr, Shintaro Hoshino, Fumiya Kakizawa, Takashi Koretsune, Yuki Nagai, Kosuke Nogaki, Takuya Nomoto, Hitoshi Mori, Junya Otsuki, Soshun Ozaki, Rihito Sakurai, Constanze Vogel, Niklas Witt, Kazuyoshi Yoshimi, Hiroshi Shinaoka  
  [arXiv:2206.11762](https://arxiv.org/abs/2206.11762) | [SoftwareX 21, 101266 (2023)](https://doi.org/10.1016/j.softx.2022.101266)

## License

This workspace is dual-licensed under the terms of the MIT license and the Apache License (Version 2.0).

- You may use the code in this repository under the terms of either license, at your option:
  - [MIT License](LICENSE)
  - [Apache License 2.0](LICENSE-APACHE)

Some components incorporate third-party code:

- the `col_piv_qr` module in the `sparse-ir-basis` crate is based on nalgebra (Apache-2.0); see [`sparse-ir/README.md`](sparse-ir/README.md) and `LICENSE-APACHE`;
- the ESPRIT and MiniPole code in the `sparse-ir-minipole` crate is a port of [MiniPole](https://github.com/Green-Phys/MiniPole) (MIT); see [`sparse-ir-minipole/LICENSE-THIRD-PARTY`](sparse-ir-minipole/LICENSE-THIRD-PARTY).

---

## Development

### Project structure

```
sparse-ir-rs/
├── sparse-ir/           # Rust library facade (crates.io: sparse-ir): re-exports
│   ├── examples/        # Rust examples
│   └── tests/           # Integration tests
├── sparse-ir-core/      # Statistics, errors, GEMM, fitters, Basis trait, sampling
├── sparse-ir-dlr/       # Discrete Lehmann representation
├── sparse-ir-minipole/  # ESPRIT and minimal pole representations
├── sparse-ir-basis/     # Kernels, SVE, IR basis
├── sparse-ir-capi/      # C-compatible API (shared library)
├── python/              # Python thin wrapper (pylibsparseir)
│   ├── pylibsparseir/   # ctypes bindings to C-API
│   └── tests/           # Python tests
├── fortran/             # Fortran bindings via C-API
│   ├── src/             # Fortran modules
│   ├── examples/        # Example programs
│   └── test/            # Test programs
├── cxx_tests/           # C/C++ integration tests
├── capi_benchmark/      # C-API benchmarks
├── notebook/            # Technical notes (algorithms, design)
├── agent-skills/        # Repo-local agent skills (Rust usage, releases)
├── bump_version_downstream.md  # Release checklist for downstream wrappers
└── docs/
    ├── book/            # Browser-readable Rust guide (mdBook), including figures
    ├── tutorial-code/   # Executable examples, CSV outputs and numerical checks
    ├── plotting/        # CSV-to-figure scripts
    └── dev/             # Internal plans and development records (not user documentation)
```

### Build

From the workspace root:

```bash
cargo build            # build all crates in debug mode
cargo build --release  # optimized build
```

Builds are portable by default: they target the baseline CPU of each platform, so the published wheels and libraries run on any machine. For a build tuned to the local CPU, opt in through the environment (never in `.cargo/config.toml`, which also applies to release builds):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The default build uses the pure-Rust `faer` backend for matrix–matrix products.  
Faer is reasonably fast, but usually considerably slower than an optimized BLAS implementation.

To enable system BLAS (LP64) for the Rust `sparse-ir` crate at compile time, use:

```bash
cargo build -p sparse-ir --features system-blas   # forwards to sparse-ir-core/system-blas
```

With `system-blas`, the default GEMM backend becomes BLAS at compile time. Regardless of the feature, arbitrary BLAS function pointers (LP64/ILP64) can be injected at runtime via the C API or the internal GEMM dispatcher.

### Test

#### Rust tests

```bash
cargo test --all-targets --release   # recommended for speed
cargo test --workspace --exclude sparse-ir-capi --doc --release   # doctests (not included in --all-targets)
```

#### C++ integration tests

Tests C-API with different BLAS configurations (default, OpenBLAS LP64, OpenBLAS ILP64):

```bash
cd cxx_tests && ./run_with_rust_capi.sh
```

#### Fortran tests

```bash
cd fortran && ./test_with_rust_capi.sh
```

#### Python tests

```bash
cd python && uv sync --locked && uv run pytest tests/ -v
```

See [`.github/workflows/`](.github/workflows/) for CI configurations.

### Benchmarks

#### C API benchmarks

```bash
cd capi_benchmark && ./run_with_rust_capi.sh
```

### Version management

Agent-facing guidance for release work lives in [`AGENTS.md`](AGENTS.md). Repo-local skills for version suggestion and release execution live under [`agent-skills/`](agent-skills/).

#### Version consistency check

Check version consistency across the workspace:

```bash
python3 check_version.py
```

This script reads the canonical version from `[workspace.package]` in `Cargo.toml` and

- fails if the Python bindings version (`python/pyproject.toml`) doesn't match;
- fails if a `sparse-ir` / `sparse-ir-capi` dependency snippet in a README (`README.md` or `*/README.md`, e.g. the install snippets in [`sparse-ir/README.md`](sparse-ir/README.md)) pins a different version (placeholders such as `X.Y.Z` are ignored);
- warns if the Julia version (`julia/build_tarballs.jl`) doesn't match, because it is updated only after the crates are published.

#### Releasing a new version

The release process is done in **two stages** because Julia bindings depend on the published crates.io version. In the commands below, replace `X.Y.Z` with the version being released.

**Stage 1: Rust + Python version bump**

1. Update the version in `Cargo.toml`:
   ```toml
   [workspace.package]
   version = "X.Y.Z"  # Update this
   
   [workspace.dependencies]
   sparse-ir = { version = "X.Y.Z", path = "sparse-ir" }
   sparse-ir-core = { version = "X.Y.Z", path = "sparse-ir-core" }
   sparse-ir-dlr = { version = "X.Y.Z", path = "sparse-ir-dlr" }
   sparse-ir-minipole = { version = "X.Y.Z", path = "sparse-ir-minipole" }
   sparse-ir-basis = { version = "X.Y.Z", path = "sparse-ir-basis" }
   ```

2. Update the Python bindings version in `python/pyproject.toml`:
   ```toml
   [project]
   version = "X.Y.Z"  # Update this
   ```

3. Update the install snippets in `sparse-ir/README.md` (`sparse-ir = "X.Y.Z"` and
   `sparse-ir = { version = "X.Y.Z", features = ["system-blas"] }`) and the quick start of the
   root `README.md`. The crate README is packaged with the crate and rendered on crates.io, so
   `check_version.py` fails until they match.

4. Verify version consistency and test publishing (dry run):
   ```bash
   python3 check_version.py
   cargo publish -p sparse-ir-core --dry-run
   ```

   The library crates depend on each other (`sparse-ir-core` <- `sparse-ir-dlr` <-
   `sparse-ir-minipole`, `sparse-ir-basis` <- `sparse-ir` <- `sparse-ir-capi`), so the registry
   verification of each one only succeeds after the version it depends on is visible on crates.io;
   only `sparse-ir-core` can be dry-run first. The manual release workflow publishes them in that
   order and waits for each one.

5. Create a PR for the version bump:
   ```bash
   git checkout -b release/vX.Y.Z
   git add Cargo.toml python/pyproject.toml README.md sparse-ir/README.md
   git commit -m "chore: bump version to X.Y.Z"
   git push origin release/vX.Y.Z
   ```
   Then create a PR on GitHub and get it reviewed.

6. After the PR is merged, start the manual Rust release workflow:
   ```bash
   gh workflow run manual-release.yml \
     -f release_ref=main \
     -f expected_version=X.Y.Z \
     -f confirm_publish=true
   ```

7. Watch the workflow:
   ```bash
   RUN_ID=$(gh run list --workflow manual-release.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN_ID"
   ```

   The workflow publishes `sparse-ir-core`, `sparse-ir-dlr`, `sparse-ir-minipole`,
   `sparse-ir-basis`, `sparse-ir`, and `sparse-ir-capi` in dependency order,
   waiting for registry visibility before proceeding. Only then does it push `vX.Y.Z`.

8. The manual Rust release workflow updates the `libsparseir` Yggdrasil branch from the new release tag. After that workflow succeeds, dispatch the standalone PyPI workflow from the release tag. The upload job is defined directly in this workflow because PyPI Trusted Publishing does not support reusable workflow jobs:
   ```bash
   RUN_ID=$(gh run list --workflow manual-release.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN_ID"
   gh workflow run PublishPyPI.yml --ref vX.Y.Z
   PYPI_RUN_ID=$(gh run list --workflow PublishPyPI.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$PYPI_RUN_ID"
   curl -fsSL "https://pypi.org/pypi/pylibsparseir/X.Y.Z/json" >/dev/null
   ```

   If the Python publish needs to be retried after the tag already exists, rerun the same workflow from the release tag:
   ```bash
   gh workflow run PublishPyPI.yml --ref vX.Y.Z
   ```

   If the Yggdrasil update leg needs to be retried independently after the tag already exists, rerun `.github/workflows/publish-libsparseir.yml` manually:
   ```bash
   gh workflow run publish-libsparseir.yml
   ```

   Requirements:
   - GitHub Actions secret `CRATES_IO_TOKEN` must be configured.
   - The workflow file is `.github/workflows/manual-release.yml`.
   - Julia and BinaryBuilder follow-up work still happen after crates.io publication.

**Stage 2: Julia version bump (after crates.io publication)**

After the new version is published to crates.io and available:

1. Update `julia/build_tarballs.jl` using the update script:
   ```bash
   julia julia/update_build_tarballs.jl vX.Y.Z
   ```
   
   This script automatically:
   - Updates the version number
   - Fetches and updates the commit hash for the tag
   - Updates the comment with the tag reference

2. Verify version consistency:
   ```bash
   python3 check_version.py
   ```

3. Create a PR for the Julia version bump:
   ```bash
   git checkout -b update-julia-vX.Y.Z
   git add julia/build_tarballs.jl
   git commit -m "chore: bump Julia bindings version to X.Y.Z"
   git push origin update-julia-vX.Y.Z
   ```
   Then create a PR on GitHub and get it reviewed.

4. After the PR is merged, follow the Julia package release process (Yggdrasil PR, etc.)
