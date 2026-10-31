#!/usr/bin/env bash
# Standard run of the batched fit/evaluate benchmark (examples/bench_batch.rs).
#
# Sweeps the six Fortran test_timing patterns over lsize = 1, 10, 120, 1000,
# 10000 for each engine, single-threaded, and writes one file per engine to
# OUT_DIR (default: target/bench_batch/<timestamp>).
#
# Usage: sparse-ir/examples/run_bench_batch.sh [--quick] [--no-einsum] [--accelerate]
#   --quick       smoke test (num/10, one repetition)
#   --no-einsum   only the in-house engine (faer and, if available, injected BLAS)
#   --accelerate  also run the einsum engine with tenferro built against
#                 Accelerate (macOS; separate target dir, needs
#                 tenferro-linalg/blas-accelerate, see tenferro-rs#1905)
set -euo pipefail

cd "$(dirname "$0")/../.."
quick=()
einsum=1
accelerate=0
for a in "$@"; do
    case "$a" in
        --quick) quick=(--quick) ;;
        --no-einsum) einsum=0 ;;
        --accelerate) accelerate=1 ;;
        *) echo "unknown option $a" >&2; exit 1 ;;
    esac
done

export RAYON_NUM_THREADS=1
out="${OUT_DIR:-target/bench_batch/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"

cargo build --release -p sparse-ir --example bench_batch
bin=target/release/examples/bench_batch

run() {
    local name=$1; shift
    echo "== $name" >&2
    "$@" ${quick[@]+"${quick[@]}"} | tee "$out/$name.txt"
}

run inhouse_faer "$bin"
run inhouse_blas "$bin" --blas
if [[ $einsum == 1 ]]; then
    run einsum_string_faer "$bin" --engine=einsum
    run einsum_plan_faer "$bin" --engine=plan
    run einsum_plan1s_faer "$bin" --engine=plan1s
fi
if [[ $accelerate == 1 ]]; then
    accel_target=target/bench_batch-accelerate
    CARGO_TARGET_DIR=$accel_target cargo build --release -p sparse-ir \
        --example bench_batch --features tenferro-linalg/blas-accelerate
    run einsum_plan_accelerate "$accel_target/release/examples/bench_batch" --engine=plan
fi

{
    echo "# $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | grep -q . && echo '(dirty)')"
    echo "# $(uname -sm); $(rustc --version)"
    if [[ $(uname) == Darwin ]]; then sysctl -n machdep.cpu.brand_string; fi
} > "$out/env.txt"
echo "results in $out" >&2
