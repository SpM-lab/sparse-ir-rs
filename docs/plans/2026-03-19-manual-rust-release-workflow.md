# Manual Rust Release Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a manual GitHub Actions workflow that publishes the Rust crates to crates.io and only pushes the `vX.Y.Z` tag after publish succeeds, while documenting the process for Codex and Claude.

**Architecture:** Keep the first-stage release automation GitHub-native and intentionally narrow. The workflow owns preflight validation, `cargo publish`, crates.io visibility checks, and tag push; project guidance lives in `AGENTS.md`, `CLAUDE.md`, repo-local skill documents, and the root `README.md`.

**Tech Stack:** GitHub Actions, Bash, Python 3, Cargo, crates.io API, Markdown

---

### Task 1: Add the manual release workflow

**Files:**
- Create: `.github/workflows/manual-release.yml`

**Step 1: Define manual inputs and permissions**

- Add `workflow_dispatch` inputs for `release_ref`, `expected_version`, and `confirm_publish`.
- Grant `contents: write` so the workflow can push the tag after publish succeeds.

**Step 2: Add preflight validation**

- Check out the requested ref with full tag history.
- Run `python3 check_version.py`.
- Read the canonical version from the workspace `Cargo.toml`.
- Fail if `expected_version` does not match, the crates already exist on crates.io, or the git tag already exists.

**Step 3: Publish crates in order**

- Publish `sparse-ir` first.
- Poll crates.io until the just-published version becomes visible.
- Publish `sparse-ir-capi` second.

**Step 4: Tag only after publish succeeds**

- Create `vX.Y.Z` from the checked-out commit.
- Push the tag to `origin`.

### Task 2: Add repo-local agent guidance

**Files:**
- Create: `AGENTS.md`
- Create: `CLAUDE.md`
- Create: `agent-skills/semantic-version-suggestion/SKILL.md`
- Create: `agent-skills/manual-rust-release/SKILL.md`

**Step 1: Document repository release invariants**

- State which version files must stay in sync.
- State that Julia version updates happen only after crates.io publication.
- State that tags are pushed only after successful publish.

**Step 2: Add the SemVer suggestion skill**

- Teach agents how to inspect the current version, published crates, latest tag, and diff since the last release.
- Encode the patch/minor/major decision rules used in this repository.

**Step 3: Add the manual release workflow skill**

- Teach agents how to run the new workflow via `gh`.
- Include the required inputs, secrets, and expected ordering.

**Step 4: Keep Claude bootstrap minimal**

- Make `CLAUDE.md` contain only `@AGENTS.md`.

### Task 3: Update the main release documentation

**Files:**
- Modify: `README.md`

**Step 1: Replace the manual tag-and-publish steps**

- Keep the version bump PR flow.
- Replace manual `git tag` plus `cargo publish` commands with the new `workflow_dispatch` flow.

**Step 2: Document agent-facing skill entry points**

- Point readers to `AGENTS.md` and the repo-local skills.
- Mention the exact workflow filename and `gh workflow run` example.

### Task 4: Verify before completion

**Files:**
- Verify: `.github/workflows/manual-release.yml`
- Verify: `AGENTS.md`
- Verify: `CLAUDE.md`
- Verify: `agent-skills/semantic-version-suggestion/SKILL.md`
- Verify: `agent-skills/manual-rust-release/SKILL.md`
- Verify: `README.md`

**Step 1: Validate YAML structure**

Run: `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/manual-release.yml"); puts "ok"'`
Expected: `ok`

**Step 2: Exercise the preflight logic locally**

Run the version extraction and crates.io existence checks against the current repository state.
Expected: current published version is detected and would block a duplicate release.

**Step 3: Inspect the diff**

Run: `git diff --stat`
Expected: only the workflow, docs, and skill files change.
