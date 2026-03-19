# Agent Instructions

Before changing release automation or version metadata, read the version-management section in [README.md](README.md).

Use these repo-local skills when the task matches:

- `agent-skills/semantic-version-suggestion/SKILL.md`
  Use when deciding the next sparse-ir-rs version under Semantic Versioning.
- `agent-skills/manual-rust-release/SKILL.md`
  Use when preparing or triggering the manual GitHub Actions workflow that publishes crates and pushes the release tag.

Release invariants for this repository:

- Keep `[workspace.package].version` and `[workspace.dependencies].sparse-ir.version` in [`Cargo.toml`](Cargo.toml) in sync.
- Keep [`python/pyproject.toml`](python/pyproject.toml) `[project].version` aligned with the workspace version before a Rust release.
- Run `python3 check_version.py` before any release or release PR.
- Update Julia version metadata only after the crates are published to crates.io.
- Push `vX.Y.Z` tags only after successful crates.io publication.
- The manual Rust release workflow file is `.github/workflows/manual-rust-release.yml`.
