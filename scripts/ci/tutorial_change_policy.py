"""Decides whether the tutorial jobs have to run for a given set of changes.

    python3 scripts/ci/tutorial_change_policy.py <file> [<file> ...]

Writes `required=true|false` and a one-line `reason` to `$GITHUB_OUTPUT` (and
to stdout, so the log says why). It always exits 0: "not required" is an
answer, not a failure, and the rollup job downstream has to be able to report
success either way.

The tutorial is a separate cargo workspace that builds `sparse-ir` from a path
dependency, so it has to run whenever the library, the tutorial itself, or the
machinery around either one changes. Everything else — the C API, the
wrappers, the other workflows — cannot move its numbers.
"""

from __future__ import annotations

import os
import sys

# A change under any of these makes the tutorial jobs required.
WATCHED = (
    "sparse-ir/",
    "docs/tutorial-code/",
    "docs/book/",
    "docs/plotting/",
    "scripts/ci/tutorial_change_policy.py",
    ".github/workflows/tutorial.yml",
    "Cargo.toml",
    "Cargo.lock",
)


def decide(paths: list[str]) -> tuple[bool, str]:
    if not paths:
        # No list of changed files — a manual run, a scheduled run, or a push
        # whose diff could not be computed. Run everything.
        return True, "no list of changed files was given, so nothing can be ruled out"
    hits = sorted({prefix for prefix in WATCHED for path in paths if path.startswith(prefix)})
    if hits:
        return True, f"changed: {', '.join(hits)}"
    return False, f"none of the {len(paths)} changed files are under {', '.join(WATCHED)}"


def main(argv: list[str]) -> int:
    required, reason = decide(argv[1:])
    print(f"required={str(required).lower()}")
    print(f"reason: {reason}")
    if not required:
        print(f"::notice::tutorial not required: {reason}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"required={str(required).lower()}\n")
            handle.write(f"reason={reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
