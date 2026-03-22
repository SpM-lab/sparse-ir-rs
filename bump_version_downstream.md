# Downstream Version Bumps

This note covers the downstream version bumps most likely to matter after a new `sparse-ir-rs` release:

- the repo-local Python wrapper in `python/`
- the repo-local Julia BinaryBuilder recipe in `julia/`
- the external Julia wrapper in `SpM-lab/SparseIR.jl`

It does not cover the external full-featured Python wrapper in `SpM-lab/sparse-ir`.

## Release Order

1. Bump the Rust workspace version in `Cargo.toml`.
2. Bump the Python wrapper version in `python/pyproject.toml`.
3. Run `python3 check_version.py`.
4. Merge the release PR.
5. Run `.github/workflows/manual-rust-release.yml` to publish `sparse-ir` and `sparse-ir-capi`.
6. Let the release workflow push tag `vX.Y.Z`.
7. Let `.github/workflows/PublishPyPI.yml` publish `pylibsparseir` from that tag.
8. Confirm that `pylibsparseir X.Y.Z` is actually available on PyPI before bumping downstream Python consumers that resolve from package indexes.
9. After crates.io publication and tag creation, bump `julia/build_tarballs.jl` if the BinaryBuilder recipe in this repo still needs to follow the release.
10. After crates.io publication, bump `SpM-lab/SparseIR.jl` to the new backend version if that wrapper should consume the release.
11. After `pylibsparseir X.Y.Z` is available, bump `SpM-lab/sparse-ir` to depend on it.

The critical dependency is that the Julia update script requires an existing `vX.Y.Z` tag, while the Python wrapper version must already match the workspace version before the Rust release.

## Python Wrapper

### Source of truth

- Manual version field: `python/pyproject.toml` `[project].version`
- Canonical version to match: `Cargo.toml` `[workspace.package].version`

`python3 check_version.py` treats a Python mismatch as an error, so the Python wrapper must be bumped in the same release PR as the Rust workspace version.

### What to edit

Update:

```toml
[project]
version = "X.Y.Z"
```

in `python/pyproject.toml`.

Do not manually edit these for version bumps:

- `python/pylibsparseir/__init__.py`
  It reads `__version__` from installed package metadata.
- `python/conda-recipe/meta.yaml`
  It derives the package version from `Cargo.toml`.

### Verification

Run:

```bash
python3 check_version.py
```

Recommended local checks before or during the release PR:

```bash
cd python && uv build
cd python && uv sync && uv run pytest tests/ -v
```

### Publish Path

- `.github/workflows/CI_PublishPyPI.yml` builds wheels on `main`, pull requests, and manual dispatch, then publishes to TestPyPI.
- `.github/workflows/PublishPyPI.yml` builds wheels on `v*` tags and publishes `pylibsparseir` to PyPI.

Normal release flow is:

1. bump `python/pyproject.toml` in the release PR
2. merge the PR
3. run `.github/workflows/manual-rust-release.yml`
4. let the pushed `vX.Y.Z` tag trigger `.github/workflows/PublishPyPI.yml`

If downstream projects resolve `pylibsparseir` from PyPI, wait until the package is visible there before updating those repositories.

## Julia Recipe

### Source of truth

- Manual release file: `julia/build_tarballs.jl`

`check_version.py` treats a Julia mismatch as a warning, not an error, because Julia is updated after the Rust crates are published.

In this repository, the Julia-side release input is still the BinaryBuilder recipe in `julia/build_tarballs.jl`. The file uses:

```julia
build_tarballs(...; julia_compat="1.10", compilers=[:c, :rust])
```

So the repo-local bump is still "update the BinaryBuilder recipe". This is separate from the external `SparseIR.jl` wrapper, which now builds through `RustToolChain`.

### Preconditions

Before bumping Julia:

- `sparse-ir` and `sparse-ir-capi` for `X.Y.Z` must already be published on crates.io
- tag `vX.Y.Z` must already exist

This ordering matters because `julia/update_build_tarballs.jl` runs `git rev-parse <tag>` and embeds the tagged commit hash into `julia/build_tarballs.jl`.

### What to run

Use the helper script:

```bash
julia julia/update_build_tarballs.jl vX.Y.Z
```

This updates:

- `version = v"X.Y.Z"`
- the `GitSource(..., "<commit>")` hash
- the `# sparse-ir-rs vX.Y.Z` comment

### Verification

Run:

```bash
python3 check_version.py
```

After the Julia bump, the warning about `julia/build_tarballs.jl` should disappear.

### Git Flow

Typical branch flow:

```bash
git checkout -b update-julia-vX.Y.Z
git add julia/build_tarballs.jl
git commit -m "chore: bump Julia bindings version to X.Y.Z"
git push origin update-julia-vX.Y.Z
```

Then open a PR and merge it after review.

### After Merge

The repository-side bump only updates the BinaryBuilder recipe. The actual Julia release still needs the downstream Julia packaging work that consumes this recipe.

## External Julia Wrapper: `SpM-lab/SparseIR.jl`

`SparseIR.jl` `main` currently builds the Rust backend during `Pkg.build("SparseIR")` and is already `RustToolChain`-based.

Relevant files on `SpM-lab/SparseIR.jl` `main`:

- `deps/build.jl`
  Uses `using RustToolChain: cargo`
- `deps/Project.toml`
  Declares the `RustToolChain` dependency
- `Project.toml`
  Pins the Rust backend version in `[tool.sparseir] rust_backend_version = "X.Y.Z"`

### What to bump

In `SpM-lab/SparseIR.jl`, update:

```toml
[tool.sparseir]
rust_backend_version = "X.Y.Z"
```

This should happen only after `sparse-ir-capi` `X.Y.Z` is available on crates.io, because `SparseIR.jl` falls back to a pinned crates.io backend when no local checkout is selected.

### Build behavior

`SparseIR.jl` `main` documents this build source priority:

1. `SPARSEIR_RUST_BACKEND_DIR` if set
2. `../sparse-ir-rs` if that sibling checkout exists
3. pinned `sparse-ir-capi` `X.Y.Z` from crates.io otherwise

So the version bump matters for the default, no-local-checkout install path.

### Verification

In the `SparseIR.jl` repository, rebuild after the bump:

```bash
julia --project=. -e 'using Pkg; Pkg.build("SparseIR")'
```

If needed, point it at a local backend checkout during coordinated development:

```bash
export SPARSEIR_RUST_BACKEND_DIR=/path/to/sparse-ir-rs
julia --project=. -e 'using Pkg; Pkg.build("SparseIR")'
```

## External Python Wrapper: `SpM-lab/sparse-ir`

`SpM-lab/sparse-ir` depends on the published `pylibsparseir` package, so its resolver-based CI will not pass until `pylibsparseir X.Y.Z` is available on PyPI.

### What to bump

In `SpM-lab/sparse-ir`, update:

- package version in `pyproject.toml`
- `pylibsparseir>=X.Y.Z,<0.9.0` in `pyproject.toml`
- `spm-lab::pylibsparseir >=X.Y.Z,<0.9.0` in `.conda/meta.yaml`

### Verification

Consistency check:

```bash
python3 check_libsparseir_version_consistency.py
```

After `pylibsparseir X.Y.Z` is published:

```bash
uv run pytest tests/ -q
```

Before publication, resolver-based `uv run` is expected to fail because the required `pylibsparseir` version is not on the index yet. In that case, verify against a local backend build instead:

```bash
python3 -m venv /tmp/sparse-ir-verify
source /tmp/sparse-ir-verify/bin/activate
python -m pip install -U pip setuptools wheel pytest numpy scipy
python -m pip install /path/to/sparse-ir-rs/python
python -m pip install -e /path/to/sparse-ir --no-deps
pytest tests/ -q
```
