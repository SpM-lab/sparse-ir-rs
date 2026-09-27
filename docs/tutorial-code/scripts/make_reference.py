"""Writes the reference values for every tutorial example.

    uv run --project . make_reference.py [example ...]

with no arguments, every example. Run it only when a reference is genuinely
meant to move — the committed files are what keeps the tutorial's numbers from
drifting.
"""

from __future__ import annotations

import sys

import reference_dlr
import reference_sparse_sampling_demo
import reference_transformation

EXAMPLES = {
    "dlr": reference_dlr.write,
    "sparse_sampling_demo": reference_sparse_sampling_demo.write,
    "transformation": reference_transformation.write,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or sorted(EXAMPLES)
    unknown = [name for name in wanted if name not in EXAMPLES]
    if unknown:
        print(f"unknown example(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(sorted(EXAMPLES))}", file=sys.stderr)
        return 2
    for name in wanted:
        EXAMPLES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
