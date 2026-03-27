# Python Publish After Rust Release Design

**Context:** `manual-rust-release.yml` publishes Rust crates and pushes the `vX.Y.Z` tag, but `pylibsparseir` publication still depends on a separate `PublishPyPI.yml` run. In practice, the tag push is not a reliable trigger boundary for the Python publish step.

**Decision:** Extract the PyPI build-and-publish logic into a reusable workflow with `workflow_call`, keep a thin standalone `PublishPyPI.yml` entrypoint for manual or tag-based invocation, and call the reusable workflow directly from `manual-rust-release.yml` after the tag is pushed.

**Why this design:** This preserves a manual recovery path, keeps wheel-building logic in one place, and expresses the intended dependency in GitHub Actions itself instead of relying on event chaining across workflows.
