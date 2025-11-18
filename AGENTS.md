# Repository Guidelines

## Project Structure & Module Organization
The Rust crate lives in `sparseir-rust/`. Core modules under `src/` cover basis generation, kernels, interpolation, sampling, and numeric traits, with matching `*_tests.rs` companions kept alongside the implementation for targeted unit coverage. Shared SVE utilities sit in `src/sve/`. Integration-style regression checks live in `tests/` (e.g., `tests/basis_comparison.rs`). `build.rs` handles optional BLAS discovery, preferring Accelerate on macOS or falling back to `pkg-config` for OpenBLAS/CBLAS on other platforms. `benches/` is prepared for Criterion harnesses when performance baselines are added.

## Build, Test, and Development Commands
- `cargo build` / `cargo build --release` compile the crate; add `--features shared-lib` to emit the C-compatible shared library.
- `cargo test --all-targets` runs unit and integration suites; expect BLAS warnings if no system library is available, though core tests fall back to the pure Rust stack.
- `cargo doc --no-deps --open` validates the public API documentation.
- `cargo fmt` and `cargo clippy --all-targets --all-features -D warnings` keep formatting and linting aligned with the repository rules.

## Coding Style & Naming Conventions
Follow `CODING_RULES.md`: prefer generic functions with explicit trait bounds, add comments when precision trade-offs arise (especially around `TwoFloat`), and normalize Legendre coordinates to `[-1, 1]`. Use Rust 2024 defaults (4-space indent, `snake_case` for items, `CamelCase` for types). Test helpers belong in `src/test_utils.rs`, and new numeric backends should implement the `CustomNumeric` trait.

## Testing Guidelines
Name precision tests after the component under scrutiny (e.g., `kernel_interpolation_tests`). Configure tolerances from observed error bounds (`TwoFloat` checks currently assert around `1e-11`). Use `cargo test --package sparseir-rust -- --nocapture` during development for detailed diagnostics, but strip debug logging before committing. Ensure high-precision comparisons pull in `dashu` reference values where practical.

## Commit & Pull Request Guidelines
Write commits that describe the final outcome (“Add centrosymmetric kernel SVE hints”), not the intermediate investigation. Reference related issues and include reproduction snippets for bug fixes. Pull requests should summarize motivation, outline verification steps (build/test status, numeric baselines), and highlight any BLAS configuration requirements so reviewers can reproduce results.

**Before committing, always verify that build artifacts and other unnecessary files are not included:**
- Check `git status` to see what files are staged
- Ensure build directories (e.g., `target/`, `_build/`, `_install/`, `build/`) are not included
- Verify that backup files (e.g., `*.backup.*`, `*_backup.*`, `*.old`) are not included
- Make sure test binaries and temporary files are excluded
- Use `git status --short` to quickly review staged changes before committing
