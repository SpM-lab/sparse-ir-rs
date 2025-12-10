# Python bindings for sparse-ir-capi

This is a low-level binding for the [sparse-ir-capi](https://github.com/SpM-lab/sparse-ir-rs) Rust library.

## Requirements

- Python >= 3.10
- Rust toolchain (for building the Rust library)
- numpy >= 1.26.4
- scipy

### BLAS Support

This package automatically uses SciPy's BLAS backend for optimal performance. No additional BLAS installation is required - SciPy will provide the necessary BLAS functionality.

## Build

### Install Dependencies and Build

```bash
# First, build the Rust sparse-ir-capi library
cd ../sparse-ir-capi
cargo build --release

# Then build the Python package
cd ../python
uv build
```

This will:
- Build the Rust sparse-ir-capi library using Cargo
- Copy the built library and header files
- Create both source distribution (sdist) and wheel packages

### Development Build

For development:

```bash
# Install in development mode (will auto-prepare if needed)
uv sync
```

**Note for CI/CD**: In CI environments, you must build the Rust library first:

```bash
# In CI/CD scripts
cd sparse-ir-capi
cargo build --release
cd ../python
uv build
```

See `.github-workflows-example.yml` for a complete GitHub Actions example.

### BLAS Configuration

The package automatically uses SciPy's BLAS backend, which provides optimized BLAS operations without requiring separate BLAS installation. The build system is configured to use SciPy's BLAS functions directly.

### Clean Build Artifacts

To remove build artifacts and files copied from the parent directory:

```bash
uv run clean
```

This will remove:
- Build directories: `build/`, `dist/`, `*.egg-info`
- Compiled libraries: `pylibsparseir/*.so`, `pylibsparseir/*.dylib`, `pylibsparseir/*.dll`
- Cache directories: `pylibsparseir/__pycache__`

### Build Process Overview

The build process works as follows:

1. **Rust Library Build**: Build the `sparse-ir-capi` Rust crate:
   - Compile the Rust library to a shared library (`.so`, `.dylib`, or `.dll`)
   - Generate C header file (`sparseir.h`) using cbindgen

2. **Python Package Building**: `uv build` or `uv sync`:
   - Copy the built Rust library and header files to the Python package
   - Package everything into distributable wheels and source distributions

3. **Installation**: The built package includes the compiled shared library and Python bindings

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
- Python 3.11, 3.12, 3.13

**Supported NumPy versions:**
- NumPy 2.1, 2.2, 2.3

The conda build automatically:
- Uses SciPy's BLAS backend for optimal performance
- Cleans up old shared libraries before building
- Builds platform-specific packages with proper dependencies

## Performance Notes

### BLAS Support

This package automatically uses SciPy's optimized BLAS backend for improved linear algebra performance:

- **Automatic BLAS**: Uses SciPy's BLAS functions for optimal performance
- **No additional setup**: SciPy provides all necessary BLAS functionality

The build system automatically configures BLAS support through SciPy. You can verify BLAS support by checking the build output for messages like:

```bash
export SPARSEIR_DEBUG=1
python -c "import pylibsparseir"
```

This will show:
```
BLAS support enabled
Registered SciPy BLAS dgemm @ 0x...
```

### Troubleshooting

**Build fails with missing library:**
```bash
# Make sure to build the Rust library first
cd ../sparse-ir-capi
cargo build --release
cd ../python
uv build
```

**Clean rebuild:**
```bash
# Remove all build artifacts
uv run clean
cd ../sparse-ir-capi
cargo clean
cargo build --release
cd ../python
uv build
```