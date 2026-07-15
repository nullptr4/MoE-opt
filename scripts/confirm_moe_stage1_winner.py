#!/usr/bin/env python3
"""Confirm E3 against E0 with alternating independent-process order."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_moe_stage1_experiments import (  # noqa: E402
    MOE_DIR,
    _command,
    _display_path,
    _git,
    _load_record,
    _run_streaming,
    _sha256,
    aggregate_performance_records,
    compare_with_e0,
    expected_schedule,
)
from run_moe_baseline import _source_snapshot  # noqa: E402


EXPERIMENTS = ("E0", "E3")


def confirmation_execution_complete(
    *,
    functional: dict,
    records: dict,
    runs: dict,
    performance: dict,
    comparison: dict | None,
    required_runs: int,
) -> bool:
    """Require complete functional, process, workload, and comparison evidence."""
    return (
        set(functional) == set(EXPERIMENTS)
        and all(functional[experiment].get("passed") is True for experiment in EXPERIMENTS)
        and all(len(records.get(experiment, [])) == required_runs for experiment in EXPERIMENTS)
        and all(len(runs.get(experiment, [])) == required_runs for experiment in EXPERIMENTS)
        and all(
            run.get("returncode") == 0 and run.get("result_recorded") is True
            for experiment in EXPERIMENTS
            for run in runs.get(experiment, [])
        )
        and all(bool(performance.get(experiment)) for experiment in EXPERIMENTS)
        and all(
            item.get("complete") is True
            for experiment in EXPERIMENTS
            for item in performance.get(experiment, [])
        )
        and comparison is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
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
    run_id = args.run_id or f"stage1-confirm-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output or ROOT / "data" / "benchmarks" / args.host_id / f"{run_id}.json"
    log_dir = args.log_dir or ROOT / "logs"
    env = os.environ.copy()
    env.update({"MOE_AUTOTUNE": "0", "MOE_AUTOHEURISTIC": "0", "MOE_RECORD_RESULTS": "0"})
    records = {experiment: [] for experiment in EXPERIMENTS}
    runs = {experiment: [] for experiment in EXPERIMENTS}
    functional = {}

    with tempfile.TemporaryDirectory(prefix="moe-stage1-confirm-") as temporary:
        temporary_dir = Path(temporary)
        for experiment in EXPERIMENTS:
            result_json = temporary_dir / f"{experiment}-functional.json"
            log_path = log_dir / f"moe-{run_id}-{experiment}-functional.log"
            command = _command(sys.executable, experiment, "functional", result_json, 0, 1)
            returncode = _run_streaming(
                command,
                env=env,
                log_path=log_path,
                label=f"{experiment}/functional",
            )
            record, error = _load_record(result_json)
            results = (record or {}).get("results", [])
            functional[experiment] = {
                "passed": (
                    returncode == 0
                    and record is not None
                    and record.get("schedule") == expected_schedule(experiment)
                    and bool(results)
                    and all(result.get("correct") is True for result in results)
                ),
                "returncode": returncode,
                "error": error,
                "results": results,
                "log": _display_path(log_path, ROOT),
            }

        if all(item["passed"] for item in functional.values()):
            for round_index in range(1, args.runs + 1):
                order = EXPERIMENTS if round_index % 2 else tuple(reversed(EXPERIMENTS))
                for order_index, experiment in enumerate(order, start=1):
                    result_json = temporary_dir / f"round-{round_index}-{experiment}.json"
                    log_path = log_dir / f"moe-{run_id}-round-{round_index}-{experiment}.log"
                    command = _command(
                        sys.executable,
                        experiment,
                        "performance",
                        result_json,
                        args.warmup,
                        args.iteration,
                    )
                    returncode = _run_streaming(
                        command,
                        env=env,
                        log_path=log_path,
                        label=f"round-{round_index}/{experiment}",
                    )
                    record, error = _load_record(result_json)
                    if record is not None:
                        record["round_index"] = round_index
                        record["order_index"] = order_index
                        records[experiment].append(record)
                    runs[experiment].append(
                        {
                            "round_index": round_index,
                            "order_index": order_index,
                            "returncode": returncode,
                            "result_recorded": record is not None,
                            "result_error": error,
                            "log": _display_path(log_path, ROOT),
                            "command": command,
                        }
                    )

    performance = {}
    aggregate_errors = {}
    for experiment in EXPERIMENTS:
        try:
            performance[experiment] = aggregate_performance_records(
                experiment,
                records[experiment],
            )
        except ValueError as exc:
            performance[experiment] = []
            aggregate_errors[experiment] = str(exc)
        else:
            aggregate_errors[experiment] = None
    comparison = (
        compare_with_e0(performance["E0"], performance["E3"])
        if performance["E0"] and performance["E3"]
        else None
    )
    execution_complete = confirmation_execution_complete(
        functional=functional,
        records=records,
        runs=runs,
        performance=performance,
        comparison=comparison,
        required_runs=args.runs,
    )
    promotion_status = (
        "promoted"
        if execution_complete and comparison["meets_latency_policy"]
        else "rejected"
        if execution_complete
        else "not_evaluated"
    )
    git_status = _git(ROOT, "status", "--porcelain")
    report = {
        "schema_version": 2,
        "record_type": "moe-stage1-winner-confirmation",
        "run_id": run_id,
        "status": "completed" if execution_complete else "failed",
        "promotion_status": promotion_status,
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
            "independent_processes_per_experiment": args.runs,
            "order": "alternating E0/E3 by round",
            "warmup": args.warmup,
            "iteration": args.iteration,
        },
        "functional": functional,
        "experiments": {
            experiment: {
                "schedule": expected_schedule(experiment),
                "runs": runs[experiment],
                "records": records[experiment],
                "performance": performance[experiment],
                "aggregate_error": aggregate_errors[experiment],
            }
            for experiment in EXPERIMENTS
        },
        "comparison_to_e0": comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output, flush=True)
    if not execution_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
