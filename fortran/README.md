# Fortran API documentation

Fortran bindings for the [sparse-ir C-API](../sparse-ir-capi/).

## Directory Structure

```
fortran/
├── src/                    # Source files and generated .inc files
│   ├── sparse_ir_c.f90        # Main Fortran module (C bindings)
│   ├── sparse_ir_extension.f90    # Extended Fortran module
│   └── *.inc               # Auto-generated include files
├── script/                 # Code generation scripts (for developers)
│   └── generate_*.py       # Scripts to generate .inc files
├── test/                   # Test programs
├── examples/               # Example programs
├── CMakeLists.txt          # CMake build configuration
└── test_with_rust_capi.sh # Main test script
```

## Prerequisites

- Fortran compiler
  - **Fortran 2003 or later** (required for `iso_c_binding` and `procedure` statements)
- CMake (3.15 or later)
- Rust toolchain (for building the C-API)
- BLAS/LAPACK library with **LP64 interface (32-bit integers)**
  - macOS: Accelerate framework (automatic)
  - Linux: OpenBLAS, MKL, or any LP64-compatible BLAS library
  - The wrapper accepts any BLAS library that provides `dgemm` and `zgemm` with 32-bit integer arguments

## Building and Testing

### Quick Start: Run All Tests

The easiest way to build and test the Fortran wrapper is to use the main test script:

```sh
cd fortran
./test_with_rust_capi.sh
```

This script will:
1. Configure CMake (which automatically detects and builds the Rust C-API library with cargo)
2. Build the Fortran bindings (CMake automatically builds the Rust C-API as a dependency)
3. Run all tests

**Note:** Code generation is not required for using the Fortran wrapper. The generated `.inc` files are already included in the repository.

### Clean Build

To start from scratch, use the `--clean` option:

```sh
./test_with_rust_capi.sh --clean
```

This will remove the `_build/` directory and Rust `target/` directory before building.

### Running Example Programs

To build and run the example program:

```sh
cd examples
./build_and_run.sh
```

This script will:
1. Build the Rust C-API library if needed
2. Build the Fortran example program
3. Run the example program

The example program demonstrates:
- Creating an IR basis
- Computing Green's functions
- Transforming between different representations
- Second-order perturbation theory calculations

## Manual Build Process

### Option 1: Automatic Build with CMake (Recommended)

CMake can automatically build the Rust C-API library with cargo. This is the simplest approach:

```sh
cd fortran
mkdir -p _build
cd _build
cmake .. -DSPARSEIR_BUILD_RUST_CAPI=ON -DSPARSEIR_BUILD_TESTING=ON
cmake --build .
ctest --output-on-failure
```

To install to a custom location:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DSPARSEIR_BUILD_RUST_CAPI=ON
cmake --build .
cmake --install .
```

This will install:
- Fortran library: `${CMAKE_INSTALL_PREFIX}/lib/libsparseir_fortran.*`
- Fortran modules: `${CMAKE_INSTALL_PREFIX}/include/sparseir/*.mod`
- Rust C-API library: `${CMAKE_INSTALL_PREFIX}/lib/libsparse_ir_capi.*`
- Rust C-API header: `${CMAKE_INSTALL_PREFIX}/include/sparseir/sparseir.h`

### Option 2: Manual Build (Advanced)

If you prefer to build the Rust C-API manually:

#### Step 1: Build Rust C-API

```sh
cd ../sparseir-rust
cargo build --release -p sparse-ir-capi
```

#### Step 2: Install C-API

```sh
cd fortran
mkdir -p _install/lib _install/include/sparseir
cp ../target/release/libsparse_ir_capi.* _install/lib/
cp ../sparse-ir-capi/include/sparseir/sparseir.h _install/include/sparseir/
```

#### Step 3: Build Fortran Bindings

```sh
mkdir -p _build
cd _build
cmake .. -DSPARSEIR_BUILD_RUST_CAPI=OFF -DSPARSEIR_CAPI_PREFIX=../_install
cmake --build .
```

#### Step 4: Run Tests

```sh
# Set library path
export DYLD_LIBRARY_PATH=../_install/lib:.:$DYLD_LIBRARY_PATH  # macOS
# or
export LD_LIBRARY_PATH=../_install/lib:.:$LD_LIBRARY_PATH      # Linux

# Run tests
ctest --output-on-failure
```

## Troubleshooting

### Library Not Found Errors

If you get library not found errors at runtime:

**macOS:**
```sh
export DYLD_LIBRARY_PATH=/path/to/fortran/_install/lib:$DYLD_LIBRARY_PATH
```

**Linux:**
```sh
export LD_LIBRARY_PATH=/path/to/fortran/_install/lib:$LD_LIBRARY_PATH
```

### BLAS/LAPACK Errors

Make sure BLAS/LAPACK libraries are available with LP64 interface (32-bit integers):

**macOS:** Accelerate framework is used automatically

**Linux:** Install OpenBLAS or another LP64-compatible BLAS:
```sh
sudo apt-get install libopenblas-dev  # Debian/Ubuntu
```

The wrapper works with any BLAS library that provides `dgemm` and `zgemm` functions with 32-bit integer arguments (LP64 interface).

### Code Generation Errors

If you encounter code generation errors (only relevant for developers modifying the C-API):
1. Ensure `libclang` is installed and accessible
2. Check that the C-API header path is correct
3. Verify Python environment has `libclang` package installed

## API Usage

See the `examples/` directory for example usage of the Fortran API.

The main modules are:
- `sparse_ir_c` - Core C-API bindings
- `sparse_ir_extension` - Extended Fortran-friendly interface

Key types:
- `IR` - IR basis object containing sampling points and basis functions

Key functions:
- `init_ir` - Initialize IR basis
- `evaluate_tau` - Evaluate functions on tau sampling points
- `evaluate_matsubara` - Evaluate functions on Matsubara frequencies
- `fit_tau` - Fit data on tau sampling points
- `fit_matsubara` - Fit data on Matsubara frequencies
- `ir2dlr` - Convert IR coefficients to DLR coefficients
- `dlr2ir` - Convert DLR coefficients to IR coefficients
- `finalize_ir` - Clean up IR basis object

---

## For Developers: Code Generation

The following sections are only relevant if you need to modify the C-API bindings or regenerate the implementation files.

### Prerequisites for Code Generation

- Python 3.10 or later
- `uv` package manager (recommended) or `pip`
- `libclang` library (for parsing C headers)

**Setting up the environment with `uv` (recommended):**

The project includes a `pyproject.toml` file that defines all required dependencies. To set up the environment:

```sh
cd fortran

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates virtual environment and installs libclang)
uv sync
```

This will automatically:
- Create a virtual environment
- Install `libclang` Python package
- Set up the correct Python version

**Installing libclang system library:**

The `libclang` Python package requires the system `libclang` library to be installed:

On macOS:
```sh
brew install llvm
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib:$DYLD_LIBRARY_PATH
```

On Linux:
```sh
sudo apt-get install libclang-dev
```

**Alternative: Using pip (not recommended):**

If you prefer to use `pip` instead of `uv`:

```sh
pip install libclang
```

However, using `uv` is recommended as it ensures consistent dependency versions and automatic virtual environment management.

### Updating C-API Bindings

If the C-API header (`sparseir.h`) changes, you need to regenerate the Fortran bindings:

```sh
cd fortran

# Set library path for libclang (macOS only)
# On Linux, this is usually not needed if libclang-dev is installed
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib:$DYLD_LIBRARY_PATH  # macOS only

# Generate bindings using uv (automatically uses the virtual environment)
uv run script/generate_c_binding.py ../sparse-ir-capi/include/sparseir/sparseir.h
```

**Note:** If you're using `uv`, the `uv run` command automatically activates the virtual environment and uses the dependencies defined in `pyproject.toml`. If you're using `pip`, make sure to activate your virtual environment first.

This will update:
- `src/_cbinding.inc` - C function bindings
- `src/_cbinding_public.inc` - Public declarations

### Generating Implementation Files

The implementation files (`*_impl.inc`) are generated by scripts in the `script/` directory:

```sh
cd fortran

# Generate evaluate_tau implementation
uv run script/generate_evaluate_tau.py

# Generate evaluate_matsubara implementation
uv run script/generate_evaluate_matsubara.py

# Generate fit_tau implementation
uv run script/generate_fit_tau.py

# Generate fit_matsubara implementation
uv run script/generate_fit_matsubara.py

# Generate ir2dlr implementation
uv run script/generate_ir2dlr.py

# Generate dlr2ir implementation
uv run script/generate_dlr2ir.py
```

**Note:** All scripts can be run with `uv run`, which automatically uses the virtual environment and dependencies defined in `pyproject.toml`.

All generated files are written to the `src/` directory.

**Important:** After regenerating files, make sure to test the build:
```sh
./test_with_rust_capi.sh
```
