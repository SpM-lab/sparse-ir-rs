#!/bin/bash

# Build sparseir-capi Rust library and Fortran bindings using CMake, then run tests
# CMake will automatically build the Rust C API with cargo and install everything
# Directory structure:
#   fortran/_build/            - Build directory for Fortran bindings and Rust C API
#   fortran/_build/_rust_capi_install/ - Temporary install directory for Rust C API (during build)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Move to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${SCRIPT_DIR}"

WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/_build"

echo -e "${GREEN}=== Testing Fortran with Rust sparseir-capi ===${NC}"

# Step 0: Clean build (optional - remove _build directory)
if [ "${1:-}" = "--clean" ]; then
    echo -e "${YELLOW}Step 0: Cleaning build directories...${NC}"
    cd "${WORKSPACE_ROOT}"
    if [ -d "target" ]; then
        rm -rf target
        echo -e "${GREEN}Removed target directory${NC}"
    fi
    cd "${SCRIPT_DIR}"
    if [ -d "_build" ]; then
        rm -rf _build
        echo -e "${GREEN}Removed _build directory${NC}"
    fi
fi

# Step 1: Configure and build with CMake (CMake will automatically build Rust C API)
echo -e "${YELLOW}Step 1: Configuring CMake...${NC}"
cd "${SCRIPT_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPARSEIR_BUILD_TESTING=ON \
    -DSPARSEIR_BUILD_RUST_CAPI=ON

# Step 2: Build (CMake will automatically build Rust C API with cargo)
echo -e "${YELLOW}Step 2: Building (CMake will automatically build Rust C API with cargo)...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Step 3: Run tests
echo -e "${YELLOW}Step 3: Running Fortran tests...${NC}"

# Determine library extension and set library path for runtime
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
    RUST_CAPI_INSTALL_DIR="${BUILD_DIR}/_rust_capi_install"
    export DYLD_LIBRARY_PATH="${RUST_CAPI_INSTALL_DIR}/lib:${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"
    # Update install_name for Fortran library to find Rust C API library
    if [ -f "${BUILD_DIR}/libsparseir_fortran.dylib" ]; then
        LIB_NAME="libsparse_ir_capi.dylib"
        install_name_tool -change "@rpath/${LIB_NAME}" "${RUST_CAPI_INSTALL_DIR}/lib/${LIB_NAME}" \
            "${BUILD_DIR}/libsparseir_fortran.dylib" 2>/dev/null || true
        install_name_tool -id "${BUILD_DIR}/libsparseir_fortran.dylib" \
            "${BUILD_DIR}/libsparseir_fortran.dylib" 2>/dev/null || true
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
    RUST_CAPI_INSTALL_DIR="${BUILD_DIR}/_rust_capi_install"
    export LD_LIBRARY_PATH="${RUST_CAPI_INSTALL_DIR}/lib:${BUILD_DIR}:${LD_LIBRARY_PATH:-}"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

