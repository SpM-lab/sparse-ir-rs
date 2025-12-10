# Python bindings for libsparseir

This is a low-level binding for the [libsparseir](https://github.com/SpM-lab/libsparseir) library.

## Requirements

- Python >= 3.10
- CMake (for building the C++ library)
- C++11 compatible compiler
- numpy >= 1.26.4
- scipy

### BLAS Support

This package automatically uses SciPy's BLAS backend for optimal performance. No additional BLAS installation is required - SciPy will provide the necessary BLAS functionality.

## Build

### Install Dependencies and Build

```bash
# First, prepare the build by copying necessary files from parent directory
python3 prepare_build.py

# Then build the package
uv build
```

This will:
- Copy source files (`src/`, `include/`, `cmake/`) from the parent libsparseir directory
- Build the C++ libsparseir library using CMake with automatic BLAS support via SciPy
- Create both source distribution (sdist) and wheel packages

### Development Build

For development:

```bash
# Install in development mode (will auto-prepare if needed)
uv sync
```

**Note for CI/CD**: In CI environments, you must run `prepare_build.py` before building:

```bash
# In CI/CD scripts
cd python
python3 prepare_build.py
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
- Copied source files: `include/`, `src/`, `cmake/` (copied by `prepare_build.py`)
- Compiled libraries: `pylibsparseir/*.so`, `pylibsparseir/*.dylib`, `pylibsparseir/*.dll`
- Cache directories: `pylibsparseir/__pycache__`

### Build Process Overview

The build process works as follows:

1. **File Preparation**: `prepare_build.py` copies necessary files from the parent libsparseir directory:
   - Source files (`../src/` → `src/`)
   - Header files (`../include/` → `include/`)
   - CMake configuration (`../cmake/` → `cmake/`)

2. **Package Building**: `uv build` or `uv sync` uses scikit-build-core to:
   - Configure CMake with automatic BLAS support via SciPy
   - Compile the C++ library with dynamic BLAS symbol lookup (for SciPy compatibility)
   - Package everything into distributable wheels and source distributions

3. **Installation**: The built package includes the compiled shared library and Python bindings

**Why File Copying?**: The `prepare_build.py` script copies files from the parent directory instead of using symbolic links to ensure:
- Cross-platform compatibility (Windows doesn't handle symlinks well)
- Proper inclusion in source distributions (sdist)
- Clean separation between the main C++ library and Python bindings

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

**Build fails with missing source files:**
```bash
# Make sure to run prepare_build.py first
python3 prepare_build.py
uv build
```

**Clean rebuild:**
```bash
# Remove all copied files and build artifacts
uv run clean
python3 prepare_build.py
uv build
```