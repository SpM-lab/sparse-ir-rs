---
name: semantic-version-suggestion
description: Use when deciding the next sparse-ir-rs release version or checking whether a change set should be patch, minor, or major under Semantic Versioning
---

# Semantic Version Suggestion

## Overview

Pick the next version from evidence, not intuition. Use the workspace version, crates.io state, the latest release tag, and the actual diff since that tag.

## Checklist

1. Read the current workspace version from `Cargo.toml`.
2. Check crates.io for published versions of `sparse-ir` and `sparse-ir-capi`.
3. Find the latest `vX.Y.Z` git tag and diff from that tag to `HEAD`.
4. Classify the change set:
   - Patch: bug fixes, internal refactors, thread-safety fixes, tests, docs, CI-only changes.
   - Minor: backward-compatible API or C-API additions, new features, new optional behavior.
   - Major: breaking Rust API, C ABI, serialized format, feature removal, raised minimum requirements that break users.
5. Suggest the exact next version and name every file that must be bumped:
   - `Cargo.toml`
   - `python/pyproject.toml`
   - later, after publish, `julia/build_tarballs.jl`

## Repository-Specific Rules

- If the workspace version is already published on crates.io and only patch-class changes were added afterward, suggest the next patch release.
- If the workspace version is unpublished but already matches the intended SemVer class, keep it.
- Do not bump Julia before crates.io publication.

## Commands

```bash
curl -s https://crates.io/api/v1/crates/sparse-ir | jq '.crate.max_version'
curl -s https://crates.io/api/v1/crates/sparse-ir-capi | jq '.crate.max_version'
git tag --sort=-creatordate | head
git diff --stat <latest-tag>..HEAD
git diff <latest-tag>..HEAD -- sparse-ir/src sparse-ir-capi/src
```
