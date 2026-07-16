#!/usr/bin/env python3
"""Merge per-host autotune observations into a deterministic heuristic index."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def workload_key(record: dict[str, Any]) -> str:
    workload = record.get("workload", {})
    fields = ("hidden", "intermediate", "experts", "group_sum", "group_count")
    return ":".join(str(workload.get(field, "?")) for field in fields)


def hardware_key(record: dict[str, Any]) -> str:
    return str(record.get("host_id") or record.get("host", {}).get("host_id") or "unknown-host")


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "data" / "autotune").glob("*/results.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if record.get("schema_version") != 1:
                continue
            if record.get("correct") is not True or record.get("latency_ms") is None:
                continue
            records.append(record)
    return records


def build_index(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(workload_key(record), hardware_key(record))].append(record)

    index: dict[str, Any] = {"schema_version": 1, "groups": {}}
    for (workload, hardware), group in sorted(groups.items()):
        by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            config_key = json.dumps(record["config"], sort_keys=True, separators=(",", ":"))
            by_config[config_key].append(record)
        ranked = []
        for config_key, samples in by_config.items():
            latencies = [float(sample["latency_ms"]) for sample in samples]
            ranked.append(
                {
                    "config": json.loads(config_key),
                    "median_latency_ms": statistics.median(latencies),
                    "samples": len(samples),
                }
            )
        ranked.sort(key=lambda item: (item["median_latency_ms"], -item["samples"]))
        index["groups"][f"{workload}|{hardware}"] = {
            "workload_key": workload,
            "hardware_key": hardware,
            "best": ranked[0],
            "candidates": ranked,
        }
    return records, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    records, index = build_index(load_records(args.root))
    output_dir = args.root / "data" / "autoheuristic"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"merged {len(records)} records into {dataset}")
    print(f"generated {len(index['groups'])} heuristic groups")


if __name__ == "__main__":
    main()
