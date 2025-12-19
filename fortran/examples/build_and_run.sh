#!/bin/bash
# Build and run Fortran example using Rust C-API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$FORTRAN_DIR")"

# Build Rust C-API if needed
if [ ! -f "$FORTRAN_DIR/_install/lib/libsparse_ir_capi.dylib" ] && [ ! -f "$FORTRAN_DIR/_install/lib/libsparse_ir_capi.so" ]; then
    echo "Building Rust C-API..."
    cd "$ROOT_DIR"
    cargo build --release -p sparse-ir-capi
    
    # Create install directory
    mkdir -p "$FORTRAN_DIR/_install/lib"
    mkdir -p "$FORTRAN_DIR/_install/include/sparseir"
    
    # Copy library and header
    if [ -f "target/release/libsparse_ir_capi.dylib" ]; then
        cp target/release/libsparse_ir_capi.dylib "$FORTRAN_DIR/_install/lib/"
    elif [ -f "target/release/libsparse_ir_capi.so" ]; then
        cp target/release/libsparse_ir_capi.so "$FORTRAN_DIR/_install/lib/"
    fi
    
    if [ -f "sparse-ir-capi/include/sparseir/sparseir.h" ]; then
        cp sparse-ir-capi/include/sparseir/sparseir.h "$FORTRAN_DIR/_install/include/sparseir/"
    fi
fi

# Build example
cd "$SCRIPT_DIR"
rm -rf build
cmake -B build -DCMAKE_PREFIX_PATH="$FORTRAN_DIR/_install"
cmake --build build

# Set library path and run
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$FORTRAN_DIR/_install/lib:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$FORTRAN_DIR/_install/lib:$LD_LIBRARY_PATH"
fi

./build/second_order_perturbation_fort
