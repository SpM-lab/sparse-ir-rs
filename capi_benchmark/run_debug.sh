#!/bin/bash
set -e

# Script to build and run debug_benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/_install"

# Step 1: Build Rust C API library
echo "Step 1: Building sparseir-capi..."
cd "${WORKSPACE_ROOT}"
cargo build --release -p sparse-ir-capi

# Determine library extension based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
else
    echo "Error: Unsupported OS: $OSTYPE"
    exit 1
fi

LIB_NAME="libsparse_ir_capi.${LIB_EXT}"
LIB_SOURCE="${WORKSPACE_ROOT}/target/release/${LIB_NAME}"

if [[ ! -f "${LIB_SOURCE}" ]]; then
    echo "Error: Library not found at ${LIB_SOURCE}"
    exit 1
fi

# Step 2: Install library and header to _install
echo "Step 2: Installing to ${INSTALL_DIR}..."
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/include/sparseir"

cp "${LIB_SOURCE}" "${INSTALL_DIR}/lib/"
cp "${WORKSPACE_ROOT}/sparse-ir-capi/include/sparseir/sparseir.h" "${INSTALL_DIR}/include/sparseir/"

echo "Installed:"
echo "  Library: ${INSTALL_DIR}/lib/${LIB_NAME}"
echo "  Header:  ${INSTALL_DIR}/include/sparseir/sparseir.h"

# Step 3: Build debug_benchmark
echo "Step 3: Building debug_benchmark..."
cd "${SCRIPT_DIR}"
if [ ! -d "_build" ]; then
    mkdir -p _build
fi

cd _build
cmake ..
make debug_benchmark

# Run debug_benchmark with default or provided arguments
# Usage: ./run_debug.sh [beta] [epsilon] [extra_size] [nrun] [positive_only]
# Default: beta=1e+5, epsilon=1e-10, extra_size=1000, nrun=10000, positive_only=0

BETA="${1:-1e+5}"
EPSILON="${2:-1e-10}"
EXTRA_SIZE="${3:-1000}"
NRUN="${4:-10000}"
POSITIVE_ONLY="${5:-0}"

echo "=========================================="
echo "Running debug_benchmark"
echo "=========================================="
echo "beta: $BETA"
echo "epsilon: $EPSILON"
echo "extra_size: $EXTRA_SIZE"
echo "nrun: $NRUN"
echo "positive_only: $POSITIVE_ONLY"
echo "=========================================="
echo ""

./debug_benchmark "$BETA" "$EPSILON" "$EXTRA_SIZE" "$NRUN" "$POSITIVE_ONLY"
