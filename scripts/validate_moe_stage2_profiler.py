#!/usr/bin/env python3
"""Validate the MoE Stage 2 decision table against its source evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_moe_stage2_profiler_table import (
    DEFAULT_OUTPUT,
    REQUIRED_VERSION_IDS,
    build_table,
    sha256,
)


EXPECTED_DECISIONS = {
    "official_square_double_buffer": "warp_policy_and_fc1_shared_capacity",
    "fullrow_double_buffer_row16": "fc1_shared_capacity",
    "fullrow_single_buffer_row16": "shared_memory_relief_validated",
    "E0": "control_reference",
    "E1": "reduction_loop_overhead",
    "E2": "cta_amplification",
    "E3": "memory_traffic_amplification",
    "E4": "pipeline_overhead",
    "E5": "register_spill",
}
REQUIRED_COUNTER_METRICS = {
    "total_cycles_k",
    "total_instructions",
    "compute_instructions",
    "memory_instructions",
    "memory_access_mb_s",
    "workgroups",
    "private_read_instructions",
    "private_write_instructions",
    "vl1_hit_rate_pct",
    "l2_hit_rate_pct",
    "global_memory_read_bytes",
    "global_memory_write_bytes",
    "shared_conflict_cycles_per_instruction",
    "shared_access_efficiency_pct",
    "ap_mte_duty_pct",
    "ap_mma_duty_pct",
}


def validate_table(table: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(table.get("schema_version") == 1, "schema_version must be 1")
    require(
        table.get("record_type") == "moe-stage2-profiler-decision-table",
        "record_type mismatch",
    )
    versions = table.get("versions", [])
    ids = tuple(item.get("id") for item in versions)
    require(ids == REQUIRED_VERSION_IDS, f"version order mismatch: {ids}")
    by_id = {item.get("id"): item for item in versions}

    sources = table.get("sources", [])
    source_by_path = {item.get("path"): item for item in sources}
    require(len(source_by_path) == len(sources), "source paths must be unique")
    for relative_path, source in source_by_path.items():
        path = root / relative_path
        require(path.is_file(), f"source does not exist: {relative_path}")
        if path.is_file():
            require(sha256(path) == source.get("sha256"), f"source SHA mismatch: {relative_path}")
            require(path.stat().st_size == source.get("bytes"), f"source size mismatch: {relative_path}")

    for version_id in REQUIRED_VERSION_IDS:
        if version_id not in by_id:
            continue
        version = by_id[version_id]
        require(
            version.get("decision", {}).get("primary_bottleneck")
            == EXPECTED_DECISIONS[version_id],
            f"unexpected decision for {version_id}",
        )
        e2e_source = version.get("e2e", {}).get("source")
        require(e2e_source in source_by_path, f"untracked e2e source for {version_id}")
        for stage in ("fc1", "fc2"):
            stage_record = version.get("stages", {}).get(stage, {})
            duration = stage_record.get("duration_ms", {})
            require(
                duration.get("status") in ("measured", "unavailable"),
                f"{version_id} {stage} duration lacks measured/unavailable status",
            )
            if duration.get("status") == "measured":
                require(
                    duration.get("large_ms", 0) > 0 and duration.get("small_ms", 0) > 0,
                    f"{version_id} {stage} measured duration must be positive",
                )
                for key in ("large_source", "small_source"):
                    require(
                        duration.get(key) in source_by_path,
                        f"{version_id} {stage} {key} is not hash-verified",
                    )
            static = stage_record.get("static_resources", {})
            require(
                static.get("status") in ("measured", "inferred"),
                f"{version_id} {stage} static resource status invalid",
            )
            if static.get("status") == "inferred":
                require(
                    static.get("source") in source_by_path,
                    f"{version_id} {stage} inferred static resources lack a hashed source",
                )
            counters = stage_record.get("counters", {})
            if version_id in ("E0", "E1", "E2", "E3", "E4", "E5"):
                require(
                    counters.get("status") in ("measured", "reused"),
                    f"{version_id} {stage} counter status invalid",
                )
                require(
                    REQUIRED_COUNTER_METRICS <= set(counters.get("metrics", {})),
                    f"{version_id} {stage} counter metrics incomplete",
                )
                for key in ("metadata", "targeted_report"):
                    path = counters.get(key)
                    require(path in source_by_path, f"{version_id} {stage} {key} is not hashed")
                report = counters.get("targeted_report", "")
                require(
                    Path(report).name.startswith("1_kernel_kernel"),
                    f"{version_id} {stage} used a non-targeted aggregate report",
                )
            else:
                require(
                    counters.get("status") == "unavailable",
                    f"historical {version_id} {stage} counter must be explicitly unavailable",
                )

    coverage = table.get("methodology", {}).get("plan_metric_coverage", {})
    allowed_status = {"available_direct", "available_derived", "proxy", "unavailable"}
    require(len(coverage) >= 16, "plan metric coverage is incomplete")
    for metric, record in coverage.items():
        require(record.get("status") in allowed_status, f"invalid coverage status for {metric}")
        if record.get("status") == "unavailable":
            require(bool(record.get("reason")), f"unavailable metric lacks reason: {metric}")

    rule_ids = {rule.get("id") for rule in table.get("decision_rules", [])}
    for version in versions:
        require(
            version.get("decision", {}).get("rule") in rule_ids,
            f"undocumented decision rule for {version.get('id')}",
        )

    summary = table.get("summary", {})
    require(summary.get("enter_fc1_async_pipeline") is False, "async pipeline must remain closed")
    require(summary.get("retained_default") == "E0", "E0 must remain the default")
    require(summary.get("tile_search_status") == "stopped", "tile search must be stopped")

    guardrails = table.get("guardrails", {})
    for prefix in ("submission", "autotune"):
        path = root / guardrails.get(f"{prefix}_path", "")
        require(path.is_file(), f"guardrail path missing: {prefix}")
        if path.is_file():
            current = sha256(path)
            require(current == guardrails.get(f"{prefix}_sha256"), f"{prefix} changed after table generation")
            require(current == guardrails.get(f"{prefix}_head_sha256"), f"{prefix} differs from HEAD")

    report = root / table.get("report", "")
    require(report.is_file(), "Stage2 Markdown report is missing")
    if report.is_file():
        require("stage2-decision-table.json" in report.read_text(), "report does not link the table")

    expected = build_table(root)
    expected["generated_at"] = table.get("generated_at")
    require(table == expected, "decision table differs from deterministic source reconstruction")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    table_path = args.table or root / DEFAULT_OUTPUT
    if not table_path.is_absolute():
        table_path = root / table_path
    table = json.loads(table_path.read_text())
    errors = validate_table(table, root)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(
        f"PASS: {len(table['versions'])} versions, "
        f"{len(table['sources'])} hashed sources, deterministic decisions verified"
    )


if __name__ == "__main__":
    main()
