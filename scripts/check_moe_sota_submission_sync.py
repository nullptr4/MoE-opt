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

    remote = manifest.get("remote_contract")
    if not isinstance(remote, dict):
        errors.append("missing remote_contract binding")
    else:
        required_remote = {
            "source_path", "source_sha256", "contract_fingerprint",
            "submission_abi_fingerprint", "status", "known", "derived", "unknown",
        }
        if set(remote) != required_remote:
            errors.append("remote_contract binding fields differ from the v1 sync policy")
        else:
            remote_path = (root / remote["source_path"]).resolve()
            if not remote_path.is_relative_to(root.resolve()) or not remote_path.is_file():
                errors.append("remote contract mirror is missing or escapes the repository")
            elif sha256(remote_path) != remote["source_sha256"]:
                errors.append("remote contract mirror SHA differs from the sync manifest")
            else:
                sys.path.insert(0, str(remote_path.parent))
                try:
                    from remote_contract_tools import (  # type: ignore
                        contract_fingerprint,
                        load_contract,
                        submission_abi_fingerprint,
                    )

                    contract = load_contract(remote_path)
                    if contract_fingerprint(contract) != remote["contract_fingerprint"]:
                        errors.append("remote contract canonical fingerprint differs")
                    if submission_abi_fingerprint(contract) != remote["submission_abi_fingerprint"]:
                        errors.append("remote submission ABI fingerprint differs")
                    if remote["status"] != "LOCAL_PROXY_ONLY":
                        errors.append("unresolved remote contract must stay LOCAL_PROXY_ONLY")
                except (ImportError, ValueError) as exc:
                    errors.append(f"cannot validate remote contract binding: {exc}")

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
        f"{manifest['formal_sota']['id']} -> {manifest['submission']['abi']} "
        f"[{manifest['submission']['sync_status']}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
