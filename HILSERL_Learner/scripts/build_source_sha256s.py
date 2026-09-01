#!/usr/bin/env python3
"""Regenerate or verify the Learner migration checksum inventory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_SHA256SUMS"
INCLUDE_ROOTS = (
    ROOT / "artifacts",
    ROOT / "docs",
    ROOT / "requirements",
    ROOT / "scripts",
    ROOT / "src",
)
# Learner deployment instructions live in the repository-level
# docs/Learner部署手册.md.  Keep this inventory scoped to the standalone
# Learner bundle roots above.
INCLUDE_FILES = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory() -> str:
    files = list(INCLUDE_FILES)
    for directory in INCLUDE_ROOTS:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    selected = sorted(
        {
            path.resolve()
            for path in files
            if "__pycache__" not in path.parts and path.suffix != ".pyc"
        },
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    return "".join(f"{_sha256(path)}  {path}\n" for path in selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = inventory()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != expected:
            raise SystemExit("SOURCE_SHA256SUMS: FAIL — regenerate after source changes")
        print(f"SOURCE_SHA256SUMS: PASS files={len(expected.splitlines())}")
        return
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"SOURCE_SHA256SUMS: WRITTEN files={len(expected.splitlines())}")


if __name__ == "__main__":
    main()
