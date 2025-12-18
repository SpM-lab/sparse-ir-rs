#!/bin/bash

# Build sparseir-capi Rust library and Fortran bindings using CMake with specified Fortran compiler, then run tests
# This script tests with a specified Fortran compiler (gfortran, ifx, or ifort) without modifying other compiler settings
# Directory structure:
#   fortran/_build_<compiler>/ - Build directory for specified Fortran compiler
#
# Options:
#   --clean              Clean build directories before building
#   --compiler=<name>    Use specified compiler: gfortran (default), ifx, or ifort

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

# Parse arguments
COMPILER="gfortran"
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --compiler=*)
            COMPILER="${arg#*=}"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# Validate compiler
if [ "$COMPILER" != "gfortran" ] && [ "$COMPILER" != "ifx" ] && [ "$COMPILER" != "ifort" ]; then
    echo -e "${RED}Error: Compiler must be 'gfortran', 'ifx', or 'ifort'${NC}"
    exit 1
fi

# Check if compiler is available
if ! command -v "$COMPILER" &> /dev/null; then
    echo -e "${RED}Error: $COMPILER not found in PATH${NC}"
    if [ "$COMPILER" = "ifx" ] || [ "$COMPILER" = "ifort" ]; then
        echo -e "${YELLOW}Please ensure Intel oneAPI is properly installed and sourced${NC}"
    fi
    exit 1
fi

# Determine build directory name
if [ "$COMPILER" = "gfortran" ]; then
    BUILD_DIR="${SCRIPT_DIR}/_build_gfortran"
else
    BUILD_DIR="${SCRIPT_DIR}/_build_${COMPILER}"
fi

echo -e "${GREEN}=== Testing Fortran with $COMPILER and Rust sparseir-capi ===${NC}"

# Step 0: Clean build (optional)
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Step 0: Cleaning build directories...${NC}"
    cd "${WORKSPACE_ROOT}"
    if [ -d "target" ]; then
        rm -rf target
        echo -e "${GREEN}Removed target directory${NC}"
    fi
    cd "${SCRIPT_DIR}"
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}Removed $BUILD_DIR directory${NC}"
    fi
fi

# Step 1: Configure and build with CMake (CMake will automatically build Rust C API)
echo -e "${YELLOW}Step 1: Configuring CMake with $COMPILER...${NC}"
cd "${SCRIPT_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with specified Fortran compiler
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPARSEIR_BUILD_TESTING=ON \
    -DSPARSEIR_BUILD_RUST_CAPI=ON \
    -DCMAKE_Fortran_COMPILER="$COMPILER" \
    -DCMAKE_Fortran_FLAGS="-O3"

# Step 2: Build (CMake will automatically build Rust C API with cargo)
echo -e "${YELLOW}Step 2: Building with $COMPILER (CMake will automatically build Rust C API with cargo)...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Step 3: Run tests
echo -e "${YELLOW}Step 3: Running Fortran tests with $COMPILER...${NC}"

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

echo -e "${GREEN}=== All tests with $COMPILER completed successfully ===${NC}"

