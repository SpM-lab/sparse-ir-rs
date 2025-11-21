#!/bin/bash

# Build sparseir-capi Rust library, install it to _install, then build and run Fortran tests
# Directory structure:
#   fortran/_install/          - Install directory for Rust C API library
#   fortran/_build/            - Build directory for Fortran bindings
#   fortran/_build/test/       - Build directory for Fortran tests

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
INSTALL_DIR="${SCRIPT_DIR}/_install"
BUILD_DIR="${SCRIPT_DIR}/_build"

echo -e "${GREEN}=== Testing Fortran with Rust sparseir-capi ===${NC}"

# Step 0: Clean build (optional - remove _build and _install directories)
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
    if [ -d "_install" ]; then
        rm -rf _install
        echo -e "${GREEN}Removed _install directory${NC}"
    fi
fi

# Step 1: Build Rust C API library
echo -e "${YELLOW}Step 1: Building sparseir-capi...${NC}"

# Ensure we're in the workspace root directory
cd "${WORKSPACE_ROOT}"
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Cargo.toml not found at ${WORKSPACE_ROOT}${NC}"
    exit 1
fi

# Check if xprec-rs exists and has Cargo.toml, initialize submodule if needed
if [ ! -f "xprec-rs/Cargo.toml" ]; then
    echo -e "${YELLOW}xprec-rs submodule not initialized, initializing...${NC}"
    if [ -f ".gitmodules" ] && grep -q "xprec-rs" .gitmodules; then
        git submodule update --init --force --recursive xprec-rs || {
            echo -e "${RED}Error: Failed to initialize xprec-rs submodule${NC}"
            exit 1
        }
        # Verify Cargo.toml exists after initialization
        if [ ! -f "xprec-rs/Cargo.toml" ]; then
            echo -e "${RED}Error: xprec-rs/Cargo.toml still not found after submodule initialization${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: xprec-rs/Cargo.toml not found and not a git submodule${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Building from workspace root: ${WORKSPACE_ROOT}${NC}"
cargo build --release -p sparse-ir-capi

# Determine library extension based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

LIB_NAME="libsparse_ir_capi.${LIB_EXT}"
LIB_SOURCE="${WORKSPACE_ROOT}/target/release/${LIB_NAME}"

if [[ ! -f "${LIB_SOURCE}" ]]; then
    echo -e "${RED}Error: Library not found at ${LIB_SOURCE}${NC}"
    exit 1
fi

# Step 2: Install library and header to _install
echo -e "${YELLOW}Step 2: Installing to ${INSTALL_DIR}...${NC}"
cd "${SCRIPT_DIR}"
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/include/sparseir"

cp "${LIB_SOURCE}" "${INSTALL_DIR}/lib/"
# Copy sparseir directory structure
cp "${WORKSPACE_ROOT}/sparse-ir-capi/include/sparseir/sparseir.h" "${INSTALL_DIR}/include/sparseir/"

echo -e "${GREEN}Installed:${NC}"
echo -e "  Library: ${INSTALL_DIR}/lib/${LIB_NAME}"
echo -e "  Header:  ${INSTALL_DIR}/include/sparseir/sparseir.h"

# Step 3: Build Fortran bindings
echo -e "${YELLOW}Step 3: Building Fortran bindings...${NC}"
cd "${SCRIPT_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo -e "${YELLOW}Configuring CMake...${NC}"
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPARSEIR_BUILD_TESTING=ON

echo -e "${YELLOW}Building Fortran bindings...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Step 4: Run tests
echo -e "${YELLOW}Step 4: Running Fortran tests...${NC}"

# Set library path for runtime
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${INSTALL_DIR}/lib:${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"
    # Update install_name for Fortran library to find Rust C API library
    if [ -f "${BUILD_DIR}/libsparseir_fortran.dylib" ]; then
        install_name_tool -change "@rpath/${LIB_NAME}" "${INSTALL_DIR}/lib/${LIB_NAME}" \
            "${BUILD_DIR}/libsparseir_fortran.dylib" 2>/dev/null || true
        install_name_tool -id "${BUILD_DIR}/libsparseir_fortran.dylib" \
            "${BUILD_DIR}/libsparseir_fortran.dylib" 2>/dev/null || true
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${BUILD_DIR}:${LD_LIBRARY_PATH:-}"
fi

ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

