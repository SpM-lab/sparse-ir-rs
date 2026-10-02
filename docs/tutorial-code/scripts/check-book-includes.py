#!/usr/bin/env python3
"""Fails if a `{{#include file:anchor}}` in the book points at nothing.

mdBook 0.5 renders an include whose anchor is missing as an empty code block,
without a warning, so a renamed or deleted anchor would silently empty a page.
This checks every include under docs/book/src: the file must exist and, when an
anchor is named, contain both `ANCHOR: name` and `ANCHOR_END: name`.

    python3 docs/tutorial-code/scripts/check-book-includes.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BOOK_SRC = Path(__file__).resolve().parents[2] / "book" / "src"
INCLUDE = re.compile(r"\{\{#include\s+([^}\s]+)\s*\}\}")


def main() -> int:
    errors = []
    count = 0
    for page in sorted(BOOK_SRC.rglob("*.md")):
        for target in INCLUDE.findall(page.read_text(encoding="utf-8")):
            count += 1
            path, _, anchor = target.partition(":")
            source = (page.parent / path).resolve()
            where = f"{page.relative_to(BOOK_SRC)}: {target}"
            if not source.is_file():
                errors.append(f"{where}: no such file")
                continue
            # A numeric suffix is a line range, not an anchor.
            if not anchor or re.fullmatch(r"[0-9:]*", anchor):
                continue
            text = source.read_text(encoding="utf-8")
            for marker in (f"ANCHOR: {anchor}", f"ANCHOR_END: {anchor}"):
                if not re.search(re.escape(marker) + r"\b", text):
                    errors.append(f"{where}: missing `{marker}`")
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"all {count} book includes resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
