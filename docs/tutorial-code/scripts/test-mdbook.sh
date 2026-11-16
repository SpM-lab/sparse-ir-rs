#!/usr/bin/env bash
# Compiles and runs the `rust` code blocks of the book.
#
# `mdbook test` calls rustdoc directly, and rustdoc has no idea how Cargo
# resolved `sparse-ir` to a path dependency. So: build the tutorial crate once,
# ask Cargo (with -vv) what `--extern` and `-L` flags it passed to rustc, and
# put a rustdoc on PATH that passes the same ones. Borrowed from tensor4all-rs,
# whose book has the same problem.
#
# The `-L` flags come from Cargo too, rather than from a fixed
# target/<profile>/deps: since Rust 1.100 the build directory keeps each
# crate in its own target/<profile>/build/<crate>/<hash>/out, and there is no
# deps directory to point at.
set -euo pipefail

TUTORIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOK_DIR="$(cd "$TUTORIAL_DIR/../book" && pwd)"
cd "$TUTORIAL_DIR"

PROFILE="${SPARSEIR_TUTORIAL_PROFILE:-ci}"

probe_log="$(mktemp)"
wrapper_dir="$(mktemp -d)"
# SPARSEIR_KEEP_WRAPPER keeps the generated rustdoc wrapper for inspection.
if [[ -n "${SPARSEIR_KEEP_WRAPPER:-}" ]]; then
    trap 'rm -f "$probe_log"' EXIT
    echo "rustdoc wrapper: $wrapper_dir/rustdoc" >&2
else
    trap 'rm -f "$probe_log"; rm -rf "$wrapper_dir"' EXIT
fi

# A fresh -Cmetadata forces cargo to run rustc rather than report a cache hit,
# which is the only way to see the command line we are after.
cargo rustc --profile "$PROFILE" --lib -vv -- \
    -Cmetadata="mdbook_probe_$(date +%s)_$$" >"$probe_log" 2>&1

rustc_line="$(grep -- '--crate-name sparse_ir_tutorial' "$probe_log" | tail -n 1 || true)"
if [[ -z "$rustc_line" ]]; then
    echo "could not find the rustc command for the tutorial crate" >&2
    tail -n 50 "$probe_log" >&2 || true
    exit 1
fi

extern_args="$(printf '%s\n' "$rustc_line" | grep -oE -- '--extern [^ ]+' | sed 's/^--extern //')"
if [[ -z "$extern_args" ]]; then
    echo "could not extract --extern flags from:" >&2
    printf '%s\n' "$rustc_line" >&2
    exit 1
fi

# Where rustdoc finds the dependencies of those crates (`dependency=...`).
library_args="$(printf '%s\n' "$rustc_line" | grep -oE -- '-L [^ ]+' | sed 's/^-L //')"
if [[ -z "$library_args" ]]; then
    echo "could not extract -L flags from:" >&2
    printf '%s\n' "$rustc_line" >&2
    exit 1
fi

real_rustdoc="$(rustup which rustdoc 2>/dev/null || command -v rustdoc)"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'exec %q ' "$real_rustdoc"
    while IFS= read -r extern_arg; do
        [[ -n "$extern_arg" ]] || continue
        crate_name="${extern_arg%%=*}"
        crate_path="${extern_arg#*=}"
        # A doctest links, so it needs the rlib, not just the metadata cargo
        # was content with for a `check`-like step.
        if [[ "$crate_path" == *.rmeta && -f "${crate_path%.rmeta}.rlib" ]]; then
            crate_path="${crate_path%.rmeta}.rlib"
        fi
        printf '%q %q ' --extern "${crate_name}=${crate_path}"
    done <<< "$extern_args"
    while IFS= read -r library_arg; do
        [[ -n "$library_arg" ]] || continue
        printf '%q %q ' -L "$library_arg"
    done <<< "$library_args"
    echo '"$@"'
} > "$wrapper_dir/rustdoc"
chmod +x "$wrapper_dir/rustdoc"

PATH="$wrapper_dir:$PATH" mdbook test "$BOOK_DIR" "$@"
