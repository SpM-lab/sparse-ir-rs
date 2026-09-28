---
name: manual-rust-release
description: Use when releasing sparse-ir-rs through the manual GitHub Actions workflow that publishes crates first and pushes the release tag only after publish succeeds
---

# Manual Rust Release

## Overview

This repository uses a manual GitHub Actions workflow for the Rust release gate. The workflow publishes `sparse-ir` first, waits until that version is visible on crates.io, publishes `sparse-ir-capi`, and only then pushes `vX.Y.Z`.

## Preconditions

- The version bump PR is merged.
- `Cargo.toml` and `python/pyproject.toml` already contain the intended version.
- `python3 check_version.py` passes.
- `CRATES_IO_TOKEN` is configured as a GitHub Actions secret.
- The target version is not already published and the tag does not already exist.

## Start The Workflow

Replace `X.Y.Z` with the version being released.

```bash
gh workflow run manual-release.yml \
  -f release_ref=main \
  -f expected_version=X.Y.Z \
  -f confirm_publish=true
```

Use a different `release_ref` only when releasing from a specific branch or commit.

## Watch The Run

```bash
RUN_ID=$(gh run list --workflow manual-release.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID"
```

## Expected Outcome

- The workflow publishes `sparse-ir`.
- It waits until `sparse-ir` at that exact version is visible on crates.io.
- It publishes `sparse-ir-capi`.
- It pushes `vX.Y.Z` to `origin`.
- Job `publish-libsparseir` pushes branch `libsparseir-vX.Y.Z` to the
  `SpM-lab/Yggdrasil` fork. It does not open the upstream Yggdrasil PR.

The tag is pushed with `GITHUB_TOKEN`, so it does **not** start the tag
workflows. Dispatch both against the tag, or `pylibsparseir` stays on the
previous version on that channel with no error:

```bash
gh workflow run PublishPyPI.yml   --ref vX.Y.Z
gh workflow run publish_conda.yml --ref vX.Y.Z
```

## Next Steps

This skill covers one step. For the Yggdrasil PR, `libsparseir_jll`, the
downstream wrappers and the final verification of every channel, continue
with the shared `sparse-ir-release` skill:
`https://github.com/SpM-lab/spm-agent-rules/blob/main/skills/sparse-ir-release/SKILL.md`
(offline: `../spm-agent-rules/skills/sparse-ir-release/SKILL.md`).
