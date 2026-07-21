#!/usr/bin/env python3
"""Verify that the standalone OJ submission is synchronized with formal SOTA."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


MANIFEST = Path("benchmarks/tilelang-moe/sota_submission_sync.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sync(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read SOTA/submission sync manifest: {exc}"]

    if manifest.get("schema_version") != "1.0":
        errors.append("unsupported SOTA/submission sync manifest schema")

    for section_name in ("formal_sota", "submission"):
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"missing manifest section {section_name!r}")
            continue
        relative = section.get("source_path")
        expected = section.get("source_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append(f"invalid source identity in section {section_name!r}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"source path escapes repository: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing synchronized source: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"{section_name} SHA mismatch for {relative}: expected {expected}, got {actual}; "
                "a SOTA/source change must synchronize submission.py and the manifest"
            )

    submission = manifest.get("submission", {})
    submission_path = root / submission.get(
        "source_path", "benchmarks/tilelang-moe/submission.py"
    )
    try:
        submission_source = submission_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot inspect submission synchronization markers: {exc}")
        return errors

    snippets = manifest.get("required_submission_snippets")
    if not isinstance(snippets, dict) or not snippets:
        errors.append("required_submission_snippets must be a non-empty object")
    else:
        for snippet, minimum_count in snippets.items():
            if not isinstance(snippet, str) or not isinstance(minimum_count, int):
                errors.append("invalid required submission snippet entry")
                continue
            actual_count = submission_source.count(snippet)
            if actual_count < minimum_count:
                errors.append(
                    f"submission is missing synchronized mechanism {snippet!r}: "
                    f"expected at least {minimum_count}, found {actual_count}"
                )

    mechanisms = manifest.get("synchronized_mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        errors.append("synchronized_mechanisms must document at least one adapted mechanism")

    return errors


def main() -> int:
    if len(sys.argv) > 2:
        print("usage: check_moe_sota_submission_sync.py [repository-root]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parents[1]
    errors = check_sync(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    print(
        "PASS SOTA/submission sync: "
        f"{manifest['formal_sota']['id']} -> {manifest['submission']['abi']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
