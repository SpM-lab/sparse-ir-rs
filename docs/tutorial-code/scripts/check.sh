#!/usr/bin/env bash
# Everything CI runs for the tutorial, in the order that fails fastest.
#
#   scripts/check.sh            format, lints, unit tests, book tests
#   scripts/check.sh --run      also run every example and verify its numbers
#   scripts/check.sh --run --scans   including the parameter scans, which are
#                               minutes rather than seconds
#
# Run it from anywhere; it works on its own directory.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TUTORIAL_DIR"

RUN_EXAMPLES=0
RUN_SCANS=0
PROFILE=ci
for arg in "$@"; do
    case "$arg" in
        --run) RUN_EXAMPLES=1 ;;
        --scans) RUN_SCANS=1 ;;
        --release) PROFILE=release ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n=== %s ===\n' "$1"; }

step "cargo fmt"
cargo fmt --all -- --check

step "cargo clippy"
cargo clippy --all-targets -- -D warnings

step "cargo test"
cargo test

if [[ $RUN_EXAMPLES -eq 1 ]]; then
    # `tutorial_binaries` runs every example, `verification` checks the
    # numbers they wrote against the committed reference values. Both are
    # skipped without SPARSEIR_TUTORIAL_RUN, so `cargo test` above stayed fast.
    step "examples and verification (--profile $PROFILE)"
    SPARSEIR_TUTORIAL_RUN=1 SPARSEIR_TUTORIAL_SCANS="$RUN_SCANS" \
        cargo test --profile "$PROFILE" \
        --test tutorial_binaries --test verification -- --nocapture --test-threads=1
fi

step "mdbook test"
SPARSEIR_TUTORIAL_PROFILE="$PROFILE" "$TUTORIAL_DIR/scripts/test-mdbook.sh"

printf '\nall checks passed\n'
