#!/usr/bin/env python3
"""Archive mcProfiler output with a reproducible manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-id", default=os.environ.get("MOE_HOST_ID", "c500-unknown"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workload-json", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root or root / "data" / "profiler"
    destination = output_root / args.host_id / args.run_id
    raw_destination = destination / "raw"
    raw_destination.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    if not source.exists():
        parser.error(f"profiler source does not exist: {source}")

    if source.is_dir():
        shutil.copytree(source, raw_destination / source.name, dirs_exist_ok=True)
    else:
        shutil.copy2(source, raw_destination / source.name)

    files = sorted(path for path in raw_destination.rglob("*") if path.is_file())
    metadata = {
        "schema_version": 1,
        "host_id": args.host_id,
        "run_id": args.run_id,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "source_path": str(source),
        "workload": json.loads(args.workload_json.read_text()) if args.workload_json else None,
        "config": json.loads(args.config_json.read_text()) if args.config_json else None,
        "files": [{"path": str(path.relative_to(destination)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
