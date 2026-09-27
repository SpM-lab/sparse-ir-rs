#!/usr/bin/env bash
# Everything CI runs for the tutorial, in the order that fails fastest.
#
#   scripts/check.sh            format, lints, unit tests, book tests
#   scripts/check.sh --run      also run every example and verify its numbers
#
# Run it from anywhere; it works on its own directory.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TUTORIAL_DIR/../.." && pwd)"
cd "$TUTORIAL_DIR"

RUN_EXAMPLES=0
PROFILE=ci
for arg in "$@"; do
    case "$arg" in
        --run) RUN_EXAMPLES=1 ;;
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
    step "examples (--profile $PROFILE)"
    for manifest_bin in src/bin/*.rs; do
        [[ -e "$manifest_bin" ]] || continue
        name="$(basename "$manifest_bin" .rs)"
        printf -- '--- %s\n' "$name"
        cargo run --profile "$PROFILE" --bin "$name"
    done

    step "verification"
    SPARSEIR_TUTORIAL_VERIFY=1 cargo test --test verification -- --nocapture
fi

if [[ -d "$REPO_ROOT/docs/book-tests" ]]; then
    step "mdbook test"
    "$REPO_ROOT/docs/tutorial-code/scripts/test-mdbook.sh"
else
    step "mdbook test"
    echo "skipped: docs/book-tests does not exist yet"
fi

printf '\nall checks passed\n'
