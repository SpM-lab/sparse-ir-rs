---
name: build-warning-baseline
description: Use when a change must not introduce new compiler or rustdoc warnings — captures a baseline from a reference commit and compares it against the branch in a machine-independent way
---

# Build Warning Baseline

## Overview

Refactors in this repository are held to "no new warnings", which means a set
comparison against a reference commit, not a raw diff of build logs. Build logs
differ between machines for reasons that have nothing to do with the change, so
normalize before comparing.

## Capture

Build the reference commit and the branch the same way, in separate worktrees so
neither disturbs the other:

```bash
git worktree add ../wt-base <reference-commit>
for d in ../wt-base .; do
  (cd "$d" && cargo clean && cargo build --workspace --all-targets 2>&1) \
    | grep -E '^(warning|error)' | sort | uniq -c
done
```

Capture rustdoc warnings separately with `cargo doc --workspace --no-deps`.

## Normalize before comparing

```bash
norm() { sed -E 's/^ +//; s/ +/ /' | grep -v 'BLAS' | LC_ALL=C sort; }
diff <(norm < base-warn.txt) <(norm < branch-warn.txt)
```

Each step removes a known machine difference:

- `sed` — BSD and GNU `uniq -c` pad the count column differently.
- `grep -v BLAS` — the backend line differs by platform (macOS prints
  `Using macOS Accelerate framework`, Linux `Found system BLAS: openblas`).
- `LC_ALL=C sort` — locale changes the sort order.

An empty diff is the pass condition. Any line only on the branch side is a new
warning and must be fixed or explained.

## macOS caveat

`cxx_tests/run_with_rust_capi.sh` deletes `target/` in its default clean mode.
A `cargo build` run straight afterwards emits dozens of

```
warning: (arm64) .../deps/*.rcgu.o unable to open object file
```

These are a stale-incremental artifact, not a code warning: they survive a
second build but disappear after `cargo clean`. Always `cargo clean` before
capturing warnings on macOS.
