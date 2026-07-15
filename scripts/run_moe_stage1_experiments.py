#!/usr/bin/env python3
"""Run the canonical E0-E5 FC1/FC2 schedule matrix on independent processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MOE_DIR = ROOT / "benchmarks" / "tilelang-moe"
sys.path.insert(0, str(MOE_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

from moe_schedule import (  # noqa: E402
    EXPERIMENT_PRESETS,
    canonical_experiment_schedule,
)
from run_moe_baseline import (  # noqa: E402
    _display_path,
    _git,
    _sha256,
    _source_snapshot,
    summarize_samples,
)


def expected_schedule(experiment: str) -> dict[str, Any]:
    return canonical_experiment_schedule(experiment)


def aggregate_performance_records(
    experiment: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError(f"{experiment} requires at least one performance process record")
    expected = expected_schedule(experiment)
    grouped: dict[str, list[float]] = defaultdict(list)
    configs: dict[str, dict[str, Any]] = {}
    expected_workloads: set[str] | None = None
    for record_index, record in enumerate(records, start=1):
        if record.get("experiment") != experiment:
            raise ValueError(f"record is not labeled {experiment}")
        if record.get("schedule") != expected:
            raise ValueError(f"{experiment} record does not use its canonical schedule")
        record_workloads: set[str] = set()
        for result in record.get("results", []):
            if result.get("test_type") != "performance":
                continue
            config = result.get("config")
            if not isinstance(config, dict) or result.get("latency_ms") is None:
                raise ValueError(f"{experiment} performance result is incomplete")
            key = json.dumps(config, sort_keys=True, separators=(",", ":"))
            if key in record_workloads:
                raise ValueError(
                    f"{experiment} process record {record_index} contains a duplicate workload"
                )
            record_workloads.add(key)
            configs[key] = config
            grouped[key].append(float(result["latency_ms"]))
        if not record_workloads:
            raise ValueError(
                f"{experiment} process record {record_index} contains no performance workloads"
            )
        if expected_workloads is None:
            expected_workloads = record_workloads
        elif record_workloads != expected_workloads:
            raise ValueError(
                f"{experiment} process record {record_index} workload set does not match the other processes"
            )

    summaries = []
    for key, values in sorted(grouped.items()):
        summary = summarize_samples(values)
        summary["config"] = configs[key]
        summary["complete"] = len(values) == len(records)
        summaries.append(summary)
    return summaries


def compare_with_e0(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    def by_config(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            json.dumps(item["config"], sort_keys=True, separators=(",", ":")): item
            for item in items
        }

    baseline_by_config = by_config(baseline)
    candidate_by_config = by_config(candidate)
    if baseline_by_config.keys() != candidate_by_config.keys():
        raise ValueError("candidate workload set does not match E0")

    workloads = []
    baseline_total = 0.0
    candidate_total = 0.0
    for key in sorted(baseline_by_config):
        baseline_latency = baseline_by_config[key]["median_latency_ms"]
        candidate_latency = candidate_by_config[key]["median_latency_ms"]
        baseline_total += baseline_latency
        candidate_total += candidate_latency
        workloads.append(
            {
                "config": baseline_by_config[key]["config"],
                "e0_median_latency_ms": baseline_latency,
                "candidate_median_latency_ms": candidate_latency,
                "improvement_pct": (baseline_latency - candidate_latency) / baseline_latency * 100.0,
            }
        )
    combined_improvement = (baseline_total - candidate_total) / baseline_total * 100.0
    minimum_workload_improvement = min(item["improvement_pct"] for item in workloads)
    return {
        "workloads": workloads,
        "e0_combined_median_ms": baseline_total,
        "candidate_combined_median_ms": candidate_total,
        "combined_improvement_pct": combined_improvement,
        "minimum_workload_improvement_pct": minimum_workload_improvement,
        "meets_latency_policy": (
            combined_improvement >= 1.0 and minimum_workload_improvement >= -0.5
        ),
    }


def _command(
    python: str,
    experiment: str,
    mode: str,
    result_json: Path,
    warmup: int,
    iteration: int,
) -> list[str]:
    return [
        python,
        str(MOE_DIR / "tune_moe.py"),
        "--experiment",
        experiment,
        "--mode",
        mode,
        "--warmup",
        str(warmup),
        "--iteration",
        str(iteration),
        "--result-json",
        str(result_json),
    ]


def _run_streaming(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {label}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=MOE_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(f"[{label}] {line}", end="", flush=True)
        returncode = process.wait()
    print(f"END {label} exit={returncode}", flush=True)
    return returncode


def _load_record(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="independent performance processes per experiment")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--host-id", default=os.environ.get("MOE_HOST_ID", "c500-unknown"))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iteration <= 0:
        parser.error("--iteration must be positive")

    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or f"stage1-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output or ROOT / "data" / "benchmarks" / args.host_id / f"{run_id}.json"
    log_dir = args.log_dir or ROOT / "logs"
    env = os.environ.copy()
    env.update({"MOE_AUTOTUNE": "0", "MOE_AUTOHEURISTIC": "0", "MOE_RECORD_RESULTS": "0"})

    experiment_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="moe-stage1-") as temporary:
        temporary_dir = Path(temporary)
        for experiment in EXPERIMENT_PRESETS:
            functional_json = temporary_dir / f"{experiment}-functional.json"
            functional_log = log_dir / f"moe-{run_id}-{experiment}-functional.log"
            functional_command = _command(
                sys.executable,
                experiment,
                "functional",
                functional_json,
                0,
                1,
            )
            functional_returncode = _run_streaming(
                functional_command,
                env=env,
                log_path=functional_log,
                label=f"{experiment}/functional",
            )
            functional_record, functional_error = _load_record(functional_json)
            functional_results = (functional_record or {}).get("results", [])
            functional_passed = (
                functional_returncode == 0
                and functional_record is not None
                and functional_record.get("schedule") == expected_schedule(experiment)
                and len(functional_results) > 0
                and all(result.get("correct") is True for result in functional_results)
            )

            process_records = []
            process_runs = []
            if functional_passed:
                for process_index in range(1, args.runs + 1):
                    process_json = temporary_dir / f"{experiment}-run-{process_index}.json"
                    log_path = log_dir / f"moe-{run_id}-{experiment}-run-{process_index}.log"
                    command = _command(
                        sys.executable,
                        experiment,
                        "performance",
                        process_json,
                        args.warmup,
                        args.iteration,
                    )
                    returncode = _run_streaming(
                        command,
                        env=env,
                        log_path=log_path,
                        label=f"{experiment}/run-{process_index}",
                    )
                    record, result_error = _load_record(process_json)
                    if record is not None:
                        record["process_index"] = process_index
                        process_records.append(record)
                    process_runs.append(
                        {
                            "process_index": process_index,
                            "returncode": returncode,
                            "result_recorded": record is not None,
                            "result_error": result_error,
                            "log": _display_path(log_path, ROOT),
                            "command": command,
                        }
                    )

            try:
                performance = aggregate_performance_records(experiment, process_records)
            except ValueError as exc:
                performance = []
                aggregate_error = str(exc)
            else:
                aggregate_error = None
            processes_passed = (
                len(process_records) == args.runs
                and all(run["returncode"] == 0 and run["result_recorded"] for run in process_runs)
                and bool(performance)
                and all(item["complete"] for item in performance)
            )
            experiment_reports.append(
                {
                    "experiment": experiment,
                    "schedule": expected_schedule(experiment),
                    "functional": {
                        "passed": functional_passed,
                        "returncode": functional_returncode,
                        "result_error": functional_error,
                        "log": _display_path(functional_log, ROOT),
                        "results": functional_results,
                    },
                    "processes_passed": processes_passed,
                    "process_runs": process_runs,
                    "process_records": process_records,
                    "performance": performance,
                    "aggregate_error": aggregate_error,
                }
            )

    baseline = experiment_reports[0]["performance"]
    for experiment_report in experiment_reports:
        if baseline and experiment_report["performance"]:
            experiment_report["comparison_to_e0"] = compare_with_e0(
                baseline,
                experiment_report["performance"],
            )
        else:
            experiment_report["comparison_to_e0"] = None

    passed = all(
        report["functional"]["passed"] and report["processes_passed"]
        for report in experiment_reports
    )
    git_status = _git(ROOT, "status", "--porcelain")
    report = {
        "schema_version": 1,
        "record_type": "moe-stage1-experiment-suite",
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "recorded_at": timestamp.isoformat(),
        "host_id": args.host_id,
        "source": {
            "git_commit": _git(ROOT, "rev-parse", "HEAD"),
            "git_branch": _git(ROOT, "branch", "--show-current"),
            "git_dirty": bool(git_status),
            "kernel_sha256": _sha256(MOE_DIR / "custom_fusedmoe.py"),
            "snapshot": _source_snapshot(ROOT),
        },
        "benchmark": {
            "independent_performance_processes": args.runs,
            "functional_processes": 1,
            "warmup": args.warmup,
            "iteration": args.iteration,
            "primary_statistic": "median_latency_ms",
            "dispersion_statistic": "mad_latency_ms",
            "tail_statistic": "p95_latency_ms",
            "promotion_policy": {
                "combined_improvement_pct_min": 1.0,
                "per_workload_regression_pct_max": 0.5,
            },
        },
        "experiments": experiment_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
