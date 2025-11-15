# C API Benchmarks

Benchmark programs for the SparseIR Rust C API.

## Build and run

After building and installing the Rust C API library, you can build the benchmarks as follows:

```bash
# Build the Rust C API library first (creates _install directory)
cd ../sparseir-capi
cargo build --release --features shared-lib
# Copy installation files to capi_benchmark/_install
mkdir -p ../capi_benchmark/_install
cp -r target/release/include ../capi_benchmark/_install/
cp -r target/release/lib ../capi_benchmark/_install/

# Build and run benchmarks
cd ../capi_benchmark
cmake -S . -B ./_build
cmake --build _build --target benchmark
```

Or use the helper script (similar to `cxx_tests/run_with_rust_capi.sh`):

```bash
./run_with_rust_capi.sh
```

