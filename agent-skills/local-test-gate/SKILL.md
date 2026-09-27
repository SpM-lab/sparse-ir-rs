---
name: local-test-gate
description: Use when running the local test layers of sparse-ir-rs — cargo, the C++ Catch2 suite, Fortran, Python, and the cbindgen header check — either one layer while iterating or all of them as a pre-push gate
---

# Local Test Gate

## Overview

sparse-ir-rs has five test layers. While iterating, run only the layer that can
see your change; run the whole gate before pushing a branch that touches
`sparse-ir-capi` or the C header.

All commands are run from the workspace root.

## Layers

| Layer | Command | Sees |
|---|---|---|
| Rust | `cargo test --all-targets --release --locked` | core + capi unit and integration tests |
| Rust docs | `cargo test -p sparse-ir --doc --release --locked` | doctests |
| system BLAS | `cargo test -p sparse-ir --features system-blas --all-targets --release --locked` | the alternative BLAS backend |
| header sync | see below | `include/sparseir/sparseir.h` vs cbindgen and vs `assets/sparse_ir_capi.h` |
| C++ | `cxx_tests/run_with_rust_capi.sh` | the C API through Catch2 |
| Fortran | `fortran/test_with_rust_capi.sh --compiler=gfortran` | the C API through the Fortran bindings |
| Python | `cd python && uv sync --locked -q && uv run pytest tests/ -q` | the C API through `pylibsparseir` (ctypes) |

Header sync (cbindgen 0.29.2, the version CI pins):

```bash
cd sparse-ir-capi
cbindgen --config cbindgen.toml --cpp-compat --output /tmp/sparseir.h .
diff include/sparseir/sparseir.h /tmp/sparseir.h
diff include/sparseir/sparseir.h assets/sparse_ir_capi.h
```

A status-code or doc change must leave a comment-only diff here. Regenerate and
commit both headers whenever the diff is non-empty.

## Iterating

Both wrapper scripts take an incremental mode; use it while iterating and a
clean run to reproduce CI.

```bash
cxx_tests/run_with_rust_capi.sh --no-clean      # incremental (default: clean)
fortran/test_with_rust_capi.sh --compiler=gfortran   # incremental (--clean to wipe)
```

The two scripts default opposite ways: the C++ one wipes `target/`, `_build/`
and `_install/` unless `--no-clean` is given, the Fortran one keeps its build
directory unless `--clean` is given.

## Notes

- `cargo fmt --all` before committing; the repo installs a pre-commit hook via
  `.cargo-husky`. Do not bypass it.
- The C++ script removes the whole `target/` directory in clean mode, so a
  `cargo build` that follows it is a full rebuild. On macOS that first rebuild
  also emits spurious `(arm64) ... unable to open object file` linker warnings;
  see `build-warning-baseline`.
- After moving a worktree, delete `python/.venv` before the Python layer — its
  shebangs hold absolute paths.
- `python3 check_version.py` checks that the versions in `Cargo.toml`,
  `python/pyproject.toml` and the Julia build recipe agree.
