#!/usr/bin/env python3
"""Fail closed when public crate or docs discovery metadata is incomplete."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://spm-lab.github.io/sparse-ir-rs/"
CRATES = (
    "sparse-ir-core",
    "sparse-ir-dlr",
    "sparse-ir-minipole",
    "sparse-ir-basis",
    "sparse-ir",
    "sparse-ir-capi",
)


def fail(message: str) -> None:
    raise SystemExit(message)


workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]
for crate in CRATES:
    manifest_path = ROOT / crate / "Cargo.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    package = manifest["package"]
    values = {}
    for key in ("version", "rust-version", "keywords", "categories"):
        value = package.get(key)
        if value == {"workspace": True}:
            value = workspace.get(key)
        values[key] = value
        if key in ("keywords", "categories"):
            if not isinstance(value, list) or not value:
                fail(f"{crate}: missing package.{key}")
        elif not value:
            fail(f"{crate}: missing package.{key}")
    if values["version"] != workspace["version"]:
        fail(f"{crate}: version differs from the workspace")
    if values["rust-version"] != workspace["rust-version"]:
        fail(f"{crate}: rust-version differs from the workspace")
    docs_rs = package.get("metadata", {}).get("docs", {}).get("rs", {})
    if not docs_rs.get("targets") or docs_rs.get("no-default-features") is not True:
        fail(f"{crate}: configure package.metadata.docs.rs targets and no-default-features")

book_src = ROOT / "docs/book/src"
summary = (book_src / "SUMMARY.md").read_text()
llms_path = book_src / "llms.txt"
llms = llms_path.read_text()
llms_urls = set(re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", llms))
page_urls: set[str] = set()
for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", summary):
    if "://" in target or target.startswith("#"):
        continue
    source_path = (book_src / target.split("#", 1)[0]).resolve()
    if not source_path.is_relative_to(book_src.resolve()) or not source_path.is_file():
        fail(f"SUMMARY.md: missing or invalid page {target}")
    output_path = Path(target.split("#", 1)[0]).with_suffix(".html")
    page_urls.add(BASE + output_path.as_posix())
missing_pages = page_urls - llms_urls
if missing_pages:
    fail("llms.txt omits SUMMARY pages: " + ", ".join(sorted(missing_pages)))

skill = ROOT / "agent-skills/sparse-ir-rust-usage/SKILL.md"
references = skill.parent / "references"
if not skill.is_file() or not any(references.glob("*.md")):
    fail("missing Rust usage skill or references")
skill_url = "https://github.com/SpM-lab/sparse-ir-rs/blob/main/agent-skills/sparse-ir-rust-usage/SKILL.md"
if skill_url not in llms_urls or skill_url not in (ROOT / "sparse-ir/README.md").read_text():
    fail("link the Rust usage skill from llms.txt and sparse-ir/README.md")
if '#![doc = include_str!("../README.md")]' not in (ROOT / "sparse-ir/src/lib.rs").read_text():
    fail("compile README Rust examples through the sparse-ir crate docs")

api_urls = {
    BASE + f"api/{crate.replace('-', '_')}/index.html"
    for crate in ("sparse-ir", "sparse-ir-core", "sparse-ir-dlr", "sparse-ir-minipole", "sparse-ir-basis")
}
if api_urls - llms_urls:
    fail("llms.txt omits generated API entry points: " + ", ".join(sorted(api_urls - llms_urls)))
expected_external = {skill_url, "https://docs.rs/sparse-ir-capi/"}
unexpected = llms_urls - page_urls - api_urls - expected_external
if unexpected or expected_external - llms_urls:
    fail("llms.txt has unexpected or missing links: " + ", ".join(sorted(unexpected | (expected_external - llms_urls))))

parser = argparse.ArgumentParser()
parser.add_argument("--check-built", action="store_true")
args = parser.parse_args()
if args.check_built:
    output = ROOT / "docs/book/book"
    if not (output / "llms.txt").is_file():
        fail("mdBook output is missing llms.txt")
    missing = {
        (output / url.removeprefix(BASE))
        for url in page_urls | api_urls
        if not (output / url.removeprefix(BASE)).is_file()
    }
    if missing:
        fail("built site is missing: " + ", ".join(str(path.relative_to(output)) for path in sorted(missing)))

print(f"Release metadata OK for {len(CRATES)} crates; llms.txt indexes all {len(page_urls)} guide pages, APIs, and the usage skill.")
