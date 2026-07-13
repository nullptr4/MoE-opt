#!/usr/bin/env python3
"""Run dependency-free repository governance checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".editorconfig",
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
)
MAINTAINED_PREFIXES = ("benchmarks/", "config/", "docs/", "scripts/", ".github/")
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yml", ".yaml", ".toml"}
CONFLICT_MARKERS = ("<<<<<<< ", "=======", ">>>>>>> ")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
    )
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def check_required(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_syntax(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if not relative.startswith(MAINTAINED_PREFIXES):
            continue
        try:
            if path.suffix == ".py":
                compile(path.read_bytes(), relative, "exec", dont_inherit=True)
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".sh":
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise ValueError(result.stderr.strip())
        except (OSError, SyntaxError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{relative}: {exc}")


def check_conflict_markers(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix not in TEXT_SUFFIXES or not (
            relative.startswith(MAINTAINED_PREFIXES) or "/" not in relative
        ):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeError as exc:
            errors.append(f"{relative}: cannot decode as UTF-8: {exc}")
            continue
        for line_number, line in enumerate(lines, 1):
            if line.startswith(CONFLICT_MARKERS):
                errors.append(f"{relative}:{line_number}: unresolved merge marker")


def main() -> int:
    errors: list[str] = []
    files = tracked_files()
    check_required(errors)
    check_syntax(files, errors)
    check_conflict_markers(files, errors)
    if errors:
        print("repository checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"repository checks passed ({len(files)} tracked files inspected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
