#!/usr/bin/env python3
"""Archive E0-E5 generated MACA host/device kernels for reproducible profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS = tuple(f"E{index}" for index in range(6))
FC1_MARKER = 'extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel('
FC2_MARKER = 'extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel_1('


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stage_function_hashes(device_source: str) -> dict[str, str]:
    fc1_start = device_source.index(FC1_MARKER)
    fc2_start = device_source.index(FC2_MARKER)
    return {
        "fc1": sha256_bytes(device_source[fc1_start:fc2_start].encode()),
        "fc2": sha256_bytes(device_source[fc2_start:].encode()),
    }


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_generated_file(cache_root: Path, experiment: str, filename: str) -> Path:
    candidates = sorted(
        (cache_root / f"{experiment}-large").glob(
            f"*_maca_*/kernels/*/{filename}"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one MACA {filename} for {experiment}, found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-id", default="c500-32g")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/metax-race/.cache/moe-stage1-prof"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "data" / "profiler" / args.host_id / "stage1-generated-code"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for experiment in EXPERIMENTS:
        destination = output / experiment
        destination.mkdir(parents=True, exist_ok=True)
        device_source = find_generated_file(args.cache_root, experiment, "device_kernel.cu")
        host_source = find_generated_file(args.cache_root, experiment, "host_kernel.cu")
        device_destination = destination / "device_kernel.cu"
        host_destination = destination / "host_kernel.cu"
        shutil.copy2(device_source, device_destination)
        shutil.copy2(host_source, host_destination)
        records.append(
            {
                "experiment": experiment,
                "schedule": f"config/moe_stage1_{experiment.lower()}_schedule.json"
                if experiment != "E0"
                else "config/moe_baseline_schedule.json",
                "source_cache": str(device_source.parent),
                "device_kernel": {
                    "path": str(device_destination.relative_to(output)),
                    "sha256": sha256(device_destination),
                    "bytes": device_destination.stat().st_size,
                    "stage_function_sha256": stage_function_hashes(
                        device_destination.read_text()
                    ),
                },
                "host_kernel": {
                    "path": str(host_destination.relative_to(output)),
                    "sha256": sha256(host_destination),
                    "bytes": host_destination.stat().st_size,
                },
            }
        )

    metadata = {
        "schema_version": 1,
        "record_type": "moe-stage1-generated-code-archive",
        "host_id": args.host_id,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "cache_root": str(args.cache_root.resolve()),
        "compiler_replay": (
            "mxcc -x maca -device-obj -O3 -lineinfo --offload-arch=xcore1000 "
            "-std=c++17 -I${TILELANG_HOME}/src -I${TILELANG_HOME}/src/tl_templates "
            "-D__FAST_HALF_CVT__ -w -resource-usage -o /dev/null device_kernel.cu"
        ),
        "experiments": records,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
