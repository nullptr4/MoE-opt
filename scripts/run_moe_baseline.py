#!/usr/bin/env python3
"""Run and summarize the fixed MoE baseline in independent processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collect_host_profile import collect as collect_host_profile


BASELINE_TAG = "v1-fullrow-single-buffer-row8-row16"
BASELINE_SCHEDULE = {
    "block_token": 128,
    "block_dhidden": 128,
    "block_dexpert": 128,
    "threads": 256,
    "num_stages": 1,
    "num_stages_down": 1,
    "s1_bn": 128,
    "s1_bk": 128,
    "s1_stages": 1,
    "s2_bn": 128,
    "s2_bk": 128,
    "s2_stages": 1,
    "swizzle_panel": 8,
    "swizzle_order": "row",
    "swizzle_panel_down": 16,
    "swizzle_order_down": "row",
    "gemm_policy": "full_row",
    "gemm_policy_down": "full_row",
    "single_weight_buffer": True,
    "min_blocks_per_sm": None,
}
SOURCE_SNAPSHOT_PATHS = (
    "benchmarks/tilelang-moe/custom_fusedmoe.py",
    "benchmarks/tilelang-moe/fusedmoe_benchmark.py",
    "benchmarks/tilelang-moe/moe_test_configs.json",
    "benchmarks/tilelang-moe/moe_test_config_tools.py",
    "benchmarks/tilelang-moe/remote_contract.json",
    "benchmarks/tilelang-moe/remote_contract_tools.py",
    "benchmarks/tilelang-moe/moe_schedule.py",
    "benchmarks/tilelang-moe/submission.py",
    "benchmarks/tilelang-moe/test_moe_submission.py",
    "benchmarks/tilelang-moe/tune_moe.py",
    "config/moe_baseline_schedule.json",
    "scripts/collect_host_profile.py",
    "scripts/run_moe_baseline.py",
)


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_samples(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    samples = [float(value) for value in values]
    median = statistics.median(samples)
    return {
        "samples": len(samples),
        "latency_samples_ms": samples,
        "median_latency_ms": median,
        "mad_latency_ms": statistics.median(abs(value - median) for value in samples),
        "p95_latency_ms": percentile(samples, 95),
        "min_latency_ms": min(samples),
        "max_latency_ms": max(samples),
    }


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def aggregate_process_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    functional: dict[str, list[dict[str, Any]]] = defaultdict(list)
    performance: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if record.get("schedule") != BASELINE_SCHEDULE:
            raise ValueError("process record does not use the fixed baseline schedule")
        for result in record.get("results", []):
            config = result.get("config")
            if not isinstance(config, dict):
                raise ValueError("process result is missing its workload config")
            key = _config_key(config)
            if result.get("test_type") == "functional":
                functional[key].append(result)
            elif result.get("test_type") == "performance":
                performance[key].append(result)

    functional_summary = []
    for key, results in sorted(functional.items()):
        passed = sum(result.get("correct") is True for result in results)
        functional_summary.append(
            {
                "config": json.loads(key),
                "passed_processes": passed,
                "processes": len(results),
                "correct": passed == len(records) and len(results) == len(records),
                "errors": [result["error"] for result in results if result.get("error")],
            }
        )

    performance_summary = []
    for key, results in sorted(performance.items()):
        summary = summarize_samples([result["latency_ms"] for result in results])
        summary["config"] = json.loads(key)
        summary["complete"] = len(results) == len(records)
        performance_summary.append(summary)

    return {
        "functional": functional_summary,
        "performance": performance_summary,
        "all_functional_passed": bool(functional_summary)
        and all(item["correct"] for item in functional_summary),
        "all_performance_complete": bool(performance_summary)
        and all(item["complete"] for item in performance_summary),
    }


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + (result.stderr or ""))
    return result.returncode


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(root: Path) -> list[dict[str, str]]:
    snapshot = []
    for relative_path in SOURCE_SNAPSHOT_PATHS:
        path = root / relative_path
        snapshot.append(
            {
                "path": relative_path,
                "sha256": _sha256(path),
                "content": path.read_text(),
            }
        )
    return snapshot


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _baseline_command(python: str, moe_dir: Path, result_json: Path, warmup: int, iteration: int) -> list[str]:
    return [
        python,
        str(moe_dir / "tune_moe.py"),
        "--evaluation-target",
        "local-proxy-guard",
        "--block-token",
        "128",
        "--block-dhidden",
        "128",
        "--block-dexpert",
        "128",
        "--threads",
        "256",
        "--num-stages",
        "1",
        "--num-stages-down",
        "1",
        "--swizzle-panel",
        "8",
        "--swizzle-order",
        "row",
        "--swizzle-panel-down",
        "16",
        "--swizzle-order-down",
        "row",
        "--gemm-policy",
        "full_row",
        "--gemm-policy-down",
        "full_row",
        "--single-weight-buffer",
        "--mode",
        "all",
        "--warmup",
        str(warmup),
        "--iteration",
        str(iteration),
        "--result-json",
        str(result_json),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-target",
        choices=("local-proxy-guard",),
        required=True,
        help="explicit opt-in to the historical formal 11-tensor baseline guard",
    )
    parser.add_argument("--runs", type=int, default=3, help="independent benchmark processes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--host-id", default=os.environ.get("MOE_HOST_ID", "c500-unknown"))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--profiler-run-id", action="append", default=[])
    parser.add_argument("--skip-oj-abi", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3 for a formal candidate")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iteration <= 0:
        parser.error("--iteration must be positive")

    root = Path(__file__).resolve().parents[1]
    moe_dir = root / "benchmarks" / "tilelang-moe"
    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or f"baseline-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output or root / "data" / "benchmarks" / args.host_id / f"{run_id}.json"
    log_dir = args.log_dir or root / "logs"
    env = os.environ.copy()
    env.update({"MOE_AUTOTUNE": "0", "MOE_AUTOHEURISTIC": "0", "MOE_RECORD_RESULTS": "0"})

    process_records: list[dict[str, Any]] = []
    process_runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="moe-baseline-") as temporary:
        temporary_dir = Path(temporary)
        for process_index in range(1, args.runs + 1):
            process_json = temporary_dir / f"run-{process_index}.json"
            log_path = log_dir / f"moe-{run_id}-run-{process_index}.log"
            command = _baseline_command(sys.executable, moe_dir, process_json, args.warmup, args.iteration)
            returncode = _run(command, cwd=moe_dir, env=env, log_path=log_path)
            process_entry = {
                "process_index": process_index,
                "returncode": returncode,
                "log": _display_path(log_path, root),
                "command": command,
            }
            if process_json.exists():
                try:
                    record = json.loads(process_json.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    process_entry["result_recorded"] = False
                    process_entry["result_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    record["process_index"] = process_index
                    process_records.append(record)
                    process_entry["result_recorded"] = True
            else:
                process_entry["result_recorded"] = False
            process_runs.append(process_entry)

    aggregate = aggregate_process_records(process_records) if process_records else {
        "functional": [],
        "performance": [],
        "all_functional_passed": False,
        "all_performance_complete": False,
    }

    abi = {"status": "skipped", "log": None, "returncode": None}
    if not args.skip_oj_abi:
        abi_log = log_dir / f"moe-{run_id}-oj-abi.log"
        abi_command = [sys.executable, str(moe_dir / "test_moe_submission.py"), "--public-shape"]
        abi_returncode = _run(abi_command, cwd=moe_dir, env=env, log_path=abi_log)
        abi = {
            "status": "passed" if abi_returncode == 0 else "failed",
            "log": _display_path(abi_log, root),
            "returncode": abi_returncode,
            "command": abi_command,
        }

    fuzz_log = log_dir / f"moe-{run_id}-fuzz.log"
    fuzz_command = [sys.executable, str(moe_dir / "test_moe_submission.py"), "--fuzz-only"]
    fuzz_returncode = _run(fuzz_command, cwd=moe_dir, env=env, log_path=fuzz_log)
    fuzz = {
        "status": "passed" if fuzz_returncode == 0 else "failed",
        "log": _display_path(fuzz_log, root),
        "returncode": fuzz_returncode,
        "command": fuzz_command,
    }

    process_success = (
        len(process_records) == args.runs
        and all(run["returncode"] == 0 and run["result_recorded"] for run in process_runs)
    )
    abi_success = abi["status"] == "passed"
    fuzz_success = fuzz["status"] == "passed"
    profiler_runs = [
        {
            "run_id": profiler_run_id,
            "metadata": f"data/profiler/{args.host_id}/{profiler_run_id}/metadata.json",
            "exists": (root / "data" / "profiler" / args.host_id / profiler_run_id / "metadata.json").is_file(),
        }
        for profiler_run_id in args.profiler_run_id
    ]
    profiler_success = bool(profiler_runs) and all(item["exists"] for item in profiler_runs)
    passed = (
        process_success
        and aggregate["all_functional_passed"]
        and aggregate["all_performance_complete"]
        and abi_success
        and fuzz_success
        and profiler_success
    )
    kernel_path = moe_dir / "custom_fusedmoe.py"
    git_status = _git(root, "status", "--porcelain")
    source_snapshot = _source_snapshot(root)
    report = {
        "schema_version": 1,
        "record_type": "moe-baseline-benchmark",
        "run_id": run_id,
        "status": "passed" if passed else ("incomplete" if abi["status"] == "skipped" else "failed"),
        "recorded_at": timestamp.isoformat(),
        "baseline_tag": BASELINE_TAG,
        "host_id": args.host_id,
        "source": {
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_branch": _git(root, "branch", "--show-current"),
            "git_dirty": bool(git_status),
            "kernel_path": _display_path(kernel_path, root),
            "kernel_sha256": _sha256(kernel_path),
            "snapshot": source_snapshot,
        },
        "environment": collect_host_profile(root, args.host_id),
        "schedule": BASELINE_SCHEDULE,
        "benchmark": {
            "independent_processes": args.runs,
            "warmup": args.warmup,
            "iteration": args.iteration,
            "primary_statistic": "median_latency_ms",
            "dispersion_statistic": "mad_latency_ms",
            "tail_statistic": "p95_latency_ms",
            "percentile_method": "linear_interpolation",
        },
        "process_runs": process_runs,
        "process_records": process_records,
        "correctness": {
            "functional": aggregate["functional"],
            "oj_abi": abi,
            "fuzz": fuzz,
        },
        "performance": aggregate["performance"],
        "profiler_runs": profiler_runs,
        "policy_checks": {
            "at_least_three_processes": args.runs >= 3,
            "all_processes_succeeded": process_success,
            "all_functional_passed": aggregate["all_functional_passed"],
            "all_performance_samples_present": aggregate["all_performance_complete"],
            "oj_abi_passed": abi_success,
            "fuzz_passed": fuzz_success,
            "profiler_runs_linked": profiler_success,
            "source_snapshot_complete": len(source_snapshot) == len(SOURCE_SNAPSHOT_PATHS),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
