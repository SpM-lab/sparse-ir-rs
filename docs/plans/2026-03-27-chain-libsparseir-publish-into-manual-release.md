# Chain libsparseir Publish Into Manual Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the Yggdrasil update workflow for `libsparseir` as part of the same manual release flow that publishes Rust crates and Python wheels.

**Architecture:** Extract the current `publish-libsparseir.yml` logic into a reusable workflow that accepts an optional release tag. Keep a thin standalone entry workflow for manual reruns, and call the reusable workflow from `manual-release.yml` after the release tag exists.

**Tech Stack:** GitHub Actions, reusable workflows, GitHub App token auth, Bash, Julia

---

### Task 1: Extract reusable Yggdrasil publish workflow

**Files:**
- Create: `.github/workflows/publish-libsparseir-reusable.yml`
- Modify: `.github/workflows/publish-libsparseir.yml`

**Step 1: Define `workflow_call` input**

- Add an optional `release_tag` string input.

**Step 2: Preserve the existing update logic**

- Keep GitHub App token creation, both checkouts, Julia update, copy, commit, and push behavior.
- When `release_tag` is empty, resolve the latest `v*` tag as before.

**Step 3: Convert the standalone workflow into a wrapper**

- Keep `workflow_dispatch`.
- Delegate to the reusable workflow instead of duplicating the implementation.

### Task 2: Chain Yggdrasil publish from manual release

**Files:**
- Modify: `.github/workflows/manual-release.yml`

**Step 1: Add a downstream reusable workflow job**

- Call the new reusable workflow after the release tag has been created.
- Pass `needs.publish-and-tag.outputs.tag`.
- Inherit secrets so the GitHub App credentials are available.

### Task 3: Update documentation

**Files:**
- Modify: `README.md`

**Step 1: Document the additional downstream leg**

- Explain that the manual release workflow now also kicks off the `libsparseir` Yggdrasil branch update after publishing crates and Python wheels.
- Keep the manual retry path for `.github/workflows/publish-libsparseir.yml`.

### Task 4: Verify before completion

**Files:**
- Verify: `.github/workflows/publish-libsparseir-reusable.yml`
- Verify: `.github/workflows/publish-libsparseir.yml`
- Verify: `.github/workflows/manual-release.yml`
- Verify: `README.md`

**Step 1: Parse YAML and lint workflows**

Run: `ruby -e 'require "yaml"; %w[.github/workflows/publish-libsparseir-reusable.yml .github/workflows/publish-libsparseir.yml .github/workflows/manual-release.yml].each { |f| YAML.load_file(f); puts "#{f}: ok" }' && actionlint .github/workflows/*.yml`
Expected: all files parse and `actionlint` exits 0
