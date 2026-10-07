# Python Publish After Rust Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the manual Rust release workflow publish `pylibsparseir` in the same GitHub Actions run after crates.io publication and tag creation, while preserving a standalone PyPI publish entrypoint for retries.

**Architecture:** Move the wheel build and PyPI publish jobs into a reusable workflow that accepts a `release_ref`. Keep `PublishPyPI.yml` as a thin wrapper for `push` on release tags and `workflow_dispatch`, and call the reusable workflow from `manual-release.yml` with the freshly created tag.

**Tech Stack:** GitHub Actions, reusable workflows, cibuildwheel, Trusted Publishing, Markdown

---

### Task 1: Extract the reusable PyPI workflow

**Files:**
- Create: `.github/workflows/publish-pypi-reusable.yml`

**Step 1: Define the callable interface**

- Add `on.workflow_call` with a required `release_ref` string input.

**Step 2: Move the wheel build logic**

- Copy the existing cibuildwheel matrix and artifact upload jobs into the reusable workflow.
- Check out the requested ref before building.

**Step 3: Keep PyPI publishing inside the reusable workflow**

- Preserve the `pypi` environment and Trusted Publishing step.

### Task 2: Convert the standalone entrypoint into a wrapper

**Files:**
- Modify: `.github/workflows/PublishPyPI.yml`

**Step 1: Keep human- and tag-facing triggers**

- Retain `push` on `v*` and `workflow_dispatch`.

**Step 2: Delegate to the reusable workflow**

- Replace the duplicated build/publish jobs with a single job-level `uses` call.
- Pass `github.ref_name` as the `release_ref`.

### Task 3: Chain Python publication from manual Rust release

**Files:**
- Modify: `.github/workflows/manual-release.yml`

**Step 1: Export the release tag as a job output**

- Promote the preflight tag output to a job output so downstream jobs can reference it.

**Step 2: Add a reusable-workflow job**

- After `publish-and-tag` succeeds, call the reusable PyPI workflow with the new tag.
- Grant only the permissions needed for the publish job.

### Task 4: Update release documentation

**Files:**
- Modify: `README.md`

**Step 1: Replace the manual Python publish step**

- Document that the manual Rust release workflow now runs the Python publish automatically in the same overall release workflow.

**Step 2: Keep the recovery path**

- Note that `PublishPyPI.yml` still exists for manual reruns from a release tag if the Python leg needs to be retried independently.

### Task 5: Verify before completion

**Files:**
- Verify: `.github/workflows/publish-pypi-reusable.yml`
- Verify: `.github/workflows/PublishPyPI.yml`
- Verify: `.github/workflows/manual-release.yml`
- Verify: `README.md`

**Step 1: Parse the workflow YAML files**

Run: `ruby -e 'require "yaml"; %w[.github/workflows/publish-pypi-reusable.yml .github/workflows/PublishPyPI.yml .github/workflows/manual-release.yml].each { |f| YAML.load_file(f); puts "#{f}: ok" }'`
Expected: each file prints `ok`

**Step 2: Inspect the diff**

Run: `git diff --stat`
Expected: workflow files, README, and plan docs only
