#!/usr/bin/env python3
"""Collect a stable hardware/software fingerprint for shared experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


def run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def git_value(root: Path, *args: str) -> str | None:
    return run_command(["git", "-C", str(root), *args])


def probe_python_packages() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_memory_gb"] = round(device.total_memory / (1024**3), 2)
            result["gpu_compute_capability"] = [device.major, device.minor]
    except Exception as exc:  # pragma: no cover - depends on the host image
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import tilelang

        result["tilelang"] = tilelang.__version__
    except Exception as exc:  # pragma: no cover - depends on the host image
        result["tilelang_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect(root: Path, host_id: str) -> dict[str, Any]:
    env_keys = (
        "MACA_PATH",
        "TILELANG_HOME",
        "TILELANG_CACHE_DIR",
        "TORCHINDUCTOR_MAX_AUTOTUNE",
        "TORCHINDUCTOR_AUTOHEURISTIC_USE",
    )
    return {
        "schema_version": 1,
        "host_id": host_id,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_branch": git_value(root, "branch", "--show-current"),
        "packages": probe_python_packages(),
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "commands": {
            "mx_smi_version": run_command(["mx-smi", "--version"]),
            "mx_smi": run_command(["mx-smi"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-id", default=os.environ.get("MOE_HOST_ID", "c500-unknown"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not HOST_ID_RE.fullmatch(args.host_id):
        parser.error("--host-id must contain only lowercase letters, digits, '.', '_' or '-'")

    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "data" / "hosts" / args.host_id / "host.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collect(root, args.host_id), ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
