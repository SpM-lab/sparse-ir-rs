#!/bin/bash

# Build sparseir-capi Rust library, install it to _install, then build and run C++ tests
# Directory structure:
#   cxx_tests/_install/          - Install directory for Rust C API library
#   cxx_tests/_build/            - Build directory for C++ tests

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

echo -e "${GREEN}=== Testing C++ CAPI tests with Rust sparseir-capi ===${NC}"

# Step 0: Clean build (remove target, _build, and _install directories)
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
# Also clean FetchContent cache if it exists (in case of partial downloads)
if [ -d "_build/_deps" ]; then
    rm -rf _build/_deps
    echo -e "${GREEN}Removed FetchContent cache${NC}"
fi

# Step 1: Build Rust C API library
echo -e "${YELLOW}Step 1: Building sparseir-capi...${NC}"

# Build release version with shared library
cargo build --release -p sparseir-capi

# Determine library extension based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

LIB_NAME="libsparseir_capi.${LIB_EXT}"
LIB_SOURCE="${WORKSPACE_ROOT}/target/release/${LIB_NAME}"

if [[ ! -f "${LIB_SOURCE}" ]]; then
    echo -e "${RED}Error: Library not found at ${LIB_SOURCE}${NC}"
    exit 1
fi

# Step 2: Install library and header to _install
echo -e "${YELLOW}Step 2: Installing to ${INSTALL_DIR}...${NC}"
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/include/sparseir"

cp "${LIB_SOURCE}" "${INSTALL_DIR}/lib/"
# Copy sparseir directory structure
cp "${WORKSPACE_ROOT}/sparseir-capi/include/sparseir/sparseir.h" "${INSTALL_DIR}/include/sparseir/"

echo -e "${GREEN}Installed:${NC}"
echo -e "  Library: ${INSTALL_DIR}/lib/${LIB_NAME}"
echo -e "  Header:  ${INSTALL_DIR}/include/sparseir/sparseir.h"

# Step 3: Build C++ tests
echo -e "${YELLOW}Step 3: Building C++ tests...${NC}"
cd "${SCRIPT_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Temporarily disable Git URL rewriting to ensure HTTPS is used for GitLab and GitHub
# This prevents Git from rewriting https:// URLs to git@ URLs:
GIT_CONFIG_GLOBAL_OLD_GITLAB=$(git config --global --get url.git@gitlab.com:.insteadOf 2>/dev/null || echo "")
GIT_CONFIG_GLOBAL_OLD_GITHUB=$(git config --global --get url.git@github.com:.insteadOf 2>/dev/null || echo "")
GIT_CONFIG_RESTORE=false

if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITLAB}" ]; then
    git config --global --unset url.git@gitlab.com:.insteadOf
    GIT_CONFIG_RESTORE=true
fi
if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITHUB}" ]; then
    git config --global --unset url.git@github.com:.insteadOf
    GIT_CONFIG_RESTORE=true
fi

if [ "${GIT_CONFIG_RESTORE}" = "true" ]; then
    # Set up trap to restore config on exit (including errors)
    trap 'if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITLAB}" ]; then git config --global url.git@gitlab.com:.insteadOf https://gitlab.com/; fi; if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITHUB}" ]; then git config --global url.git@github.com:.insteadOf https://github.com/; fi' EXIT
fi

echo -e "${YELLOW}Configuring CMake...${NC}"
# Enable verbose output for CMake configuration to see what's happening
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=ON

# Restore Git config if it was changed (trap will also handle this on exit)
if [ "${GIT_CONFIG_RESTORE}" = "true" ]; then
    if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITLAB}" ]; then
        git config --global url.git@gitlab.com:.insteadOf https://gitlab.com/
    fi
    if [ -n "${GIT_CONFIG_GLOBAL_OLD_GITHUB}" ]; then
        git config --global url.git@github.com:.insteadOf https://github.com/
    fi
    trap - EXIT  # Remove trap since we've restored manually
fi

echo -e "${YELLOW}Building C++ tests...${NC}"
# Use verbose output to see progress
# Note: This may take a while if FetchContent is downloading dependencies
cmake --build . --verbose -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) 2>&1 | tee build.log || {
    echo -e "${RED}Build failed. Last 50 lines of build.log:${NC}"
    tail -50 build.log
    exit 1
}

# Step 4: Run tests
echo -e "${YELLOW}Step 4: Running C++ tests...${NC}"

# Set library path for runtime
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${INSTALL_DIR}/lib:${DYLD_LIBRARY_PATH:-}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

