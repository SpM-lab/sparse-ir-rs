# Agent Instructions

Before acting, read the latest shared SpM-lab agent rules from the
[`spm-agent-rules`](https://github.com/SpM-lab/spm-agent-rules)
repository. Start from:

- `https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/index.md`

If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout:

- `../spm-agent-rules/rules/index.md`

Load only the rule files relevant to the task. For this repository, that
typically resolves to:

- `common.md` — for any implementation work.
- `ffi-boundary.md` — whenever a change touches `sparse-ir-capi`, an
  `extern "C"` function, a raw pointer, a dtype conversion, memory order, or a
  status code.
- `numerical-conventions.md` — whenever a change touches basis construction,
  sampling, DLR, statistics (fermionic/bosonic), `tau`/Matsubara domains, or
  real/complex handling.
- `testing.md` — for any new test or change to existing test coverage.
- `rust.md` — the sparse-ir-rs- and libsparseir-specific rules.

Provenance and scientific-credit requirements when code is written while
referencing third-party code are defined in the "Provenance And Scientific
Credit" section of `REPOSITORY_RULES.md` (see below); no shared-rules
counterpart exists yet.

Then read [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md), which contains the
durable sparse-ir-rs-specific contracts. Repository-local rules override shared
rules when they are more specific. Existing code that predates a rule is not
precedent; remediation outside the current task requires the scope and approval
defined in `REPOSITORY_RULES.md`.

[`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md) contains supplemental
crate implementation notes. Read it only after the root rules, and do not treat
it as an independent policy source. `REPOSITORY_RULES.md`, current source,
public documentation, and generated binding contracts are authoritative.

Before changing release automation or version metadata, read the
version-management section in [README.md](README.md). For downstream wrapper
version bumps, also read
[`bump_version_downstream.md`](bump_version_downstream.md).

Use these repo-local skills when the task matches:

- `agent-skills/semantic-version-suggestion/SKILL.md`
  Use when deciding the next sparse-ir-rs version under Semantic Versioning.
- `agent-skills/manual-rust-release/SKILL.md`
  Use when preparing or triggering the manual GitHub Actions workflow that publishes crates and pushes the release tag.
- `agent-skills/local-test-gate/SKILL.md`
  Use when running the local test layers — cargo, C++, Fortran, Python, header sync — one layer while iterating or all of them before pushing.
- `agent-skills/build-warning-baseline/SKILL.md`
  Use when a change must not introduce new compiler or rustdoc warnings.

Release invariants for this repository:

- Follow the [Release And Version Integrity](REPOSITORY_RULES.md#release-and-version-integrity)
  rules in `REPOSITORY_RULES.md`.
