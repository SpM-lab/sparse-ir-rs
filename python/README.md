# Python bindings for sparse-ir-capi

This is a low-level binding for the [sparse-ir-capi](https://github.com/SpM-lab/sparse-ir-rs) Rust library.

## Requirements

- Python >= 3.10
- Rust toolchain (for building the Rust library)
- [uv](https://docs.astral.sh/uv/) (for building and managing Python dependencies)
- numpy >= 1.26.4
- scipy

### BLAS Support

This package automatically uses SciPy's BLAS backend for optimal performance. No additional BLAS installation is required - SciPy will provide the necessary BLAS functionality.

## Build

### Install Dependencies and Build

```bash
# Build the package (Rust library will be built automatically)
cd python
uv build
```

This will:
- Automatically build the Rust sparse-ir-capi library with Cargo (through the hatchling build hook in `hatch_build.py`)
- Copy the built library and header file to the Python package
- Create both source distribution (sdist) and wheel packages

### Development Build

For development:

```bash
# Install in development mode (will auto-prepare if needed)
uv sync --locked
```

**Note for CI/CD**: The Rust library is built automatically during the Python package build. No separate build step is needed:

```bash
# In CI/CD scripts
cd python
uv build
```

See `.github/workflows/` in the repository for the workflows that build and test this package.

### BLAS Configuration

The package automatically uses SciPy's BLAS backend, which provides optimized BLAS operations without requiring separate BLAS installation. The build system is configured to use SciPy's BLAS functions directly.

### Clean Build Artifacts

To remove build artifacts and files copied from the parent directory:

```bash
make clean
```

This will remove:
- Build directories and caches: `build/`, `dist/`, `*.egg-info`, `__pycache__/`, `.pytest_cache/`, `.venv/`
- Files copied into the package: `pylibsparseir/*.so`, `pylibsparseir/*.dylib`, `pylibsparseir/*.dll`, `pylibsparseir/*.h`

### Build Process Overview

The build process works as follows:

1. **Build hook**: `pyproject.toml` uses the hatchling backend, which runs the
   custom build hook in `hatch_build.py`.

2. **Rust Library Build**: the hook runs `cargo build --release -p sparse-ir-capi`
   from the repository root:
   - Compiles the Rust library to a shared library (`.so`, `.dylib`, or `.dll`)
   - Removes old shared libraries from `pylibsparseir/`, then copies the new
     library and the header `sparse-ir-capi/include/sparseir/sparseir.h` there

3. **Python Package Building**: `uv build` or `uv sync --locked`:
   - Packages everything into distributable wheels and source distributions

4. **Installation**: The built package includes the compiled shared library and Python bindings

### Conda Build

This package can also be built and distributed via conda-forge. The conda recipe is located in `conda-recipe/` and supports multiple platforms and Python versions.

**Building conda packages locally:**

```bash
# Install conda-build
conda install conda-build

# Build the conda package
cd python
conda build conda-recipe

# Build for specific platforms
conda build conda-recipe --platform linux-64
conda build conda-recipe --platform osx-64
conda build conda-recipe --platform osx-arm64
```

**Supported platforms:**
- Linux x86_64
- macOS Intel (x86_64)
- macOS Apple Silicon (ARM64)

**Supported Python versions:**
- Python 3.11, 3.12, 3.13, 3.14 (the conda recipe; the PyPI package itself
  supports Python 3.10 and newer, see `pyproject.toml`)

**Supported NumPy versions:**
- NumPy 2.1, 2.2, 2.3

The conda build automatically:
- Uses SciPy's BLAS backend for optimal performance
- Cleans up old shared libraries before building
- Builds platform-specific packages with proper dependencies

## Handle Ownership

Every function in `pylibsparseir.core` that creates a C object returns an
owning handle instead of a raw ctypes pointer: `KernelHandle`,
`SVEResultHandle`, `BasisHandle`, `FuncsHandle`, `SamplingHandle`, or
`GemmBackendHandle` (the default BLAS backend returned by
`get_default_blas_backend()`).

- A handle is released exactly once with the matching `spir_*_release`: when
  it becomes unreachable, when `close()` is called (also on leaving a
  `with handle:` block), or at interpreter exit. Closing it again does nothing.
- An open handle can be passed directly to any `_lib.spir_*` function and is
  truthy. A released handle is falsy, and passing it to C raises
  `ctypes.ArgumentError` instead of handing C a freed pointer.
- Do not release an owned handle through `_lib.spir_*_release`. Those entry
  points refuse owned handles with `ctypes.ArgumentError`, because the owner
  would free them a second time. Handles created directly through `_lib`
  remain the caller's to release, or can be handed over to an owner, e.g.
  `FuncsHandle(_lib.spir_funcs_clone(funcs))`.
- The DLR and MiniPole entry points have no wrapper in `pylibsparseir.core`;
  call them through `_lib` (`spir_dlr_new_independent`, `spir_dlr_new`,
  `spir_minipole_from_dlr`, `spir_minipole_from_matsubara`). A DLR is a
  `spir_basis` and can be handed to `BasisHandle`. A `spir_pole_repr` has no
  owning handle class, so release it yourself with
  `_lib.spir_pole_repr_release`.
- C handles never depend on the handles they were created from
  (`spir_basis_new` copies the kernel and the SVE result; funcs and samplings
  own their data), so handles can be released in any order.

## Performance Notes

### BLAS Support

This package automatically uses SciPy's optimized BLAS backend for improved linear algebra performance:

- **Automatic BLAS**: Uses SciPy's BLAS functions for optimal performance
- **No additional setup**: SciPy provides all necessary BLAS functionality

The build system automatically configures BLAS support through SciPy. You can verify BLAS support with [debug output](#debug-output) enabled:

```bash
export SPARSEIR_DEBUG=1
python -c "import pylibsparseir"
```

This will show:
```
[core.py] Created SciPy BLAS backend handle
[core.py] Registered SciPy BLAS dgemm @ 0x...
[core.py] Registered SciPy BLAS zgemm @ 0x...
```

### Debug Output

The `SPARSEIR_DEBUG` environment variable enables debug output only when it is
set to `1`, `true`, `yes` or `on`, in any letter case. Any other value,
including `0`, `false` or an empty string, disables it, as does leaving it
unset. The same rule applies to both layers:

- pylibsparseir prints its messages to stdout. It checks the variable when it
  is imported.
- The Rust library prints `[SPARSEIR DEBUG]`, `[SPARSEIR DEBUG ERROR]` and
  `[SPARSEIR WARN]` lines to stderr. It checks the variable each time it
  would print one.

### Troubleshooting

**Build fails with missing Cargo:**
```bash
# Make sure Rust toolchain is installed
# Install from https://rustup.rs/
# Then retry:
cd python
uv build
```

**Clean rebuild:**
```bash
# Remove all build artifacts
make clean
cd ../sparse-ir-capi
cargo clean
cd ../python
uv build
```
