#!/usr/bin/env python3
"""Build the evidence-backed MoE Stage 2 profiler decision table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE1_SUITE = "data/benchmarks/c500-32g/stage1-20260715T141647Z.json"
STAGE1_CONFIRM = "data/benchmarks/c500-32g/stage1-confirm-20260715T152651Z.json"
STAGE1_RESOURCE = "data/profiler/c500-32g/stage1-resource-usage.json"
GENERATED_ARCHIVE = "data/profiler/c500-32g/stage1-generated-code/metadata.json"
DEFAULT_OUTPUT = "data/profiler/c500-32g/stage2-decision-table.json"
DEFAULT_REPORT = "reports/2026-07-16-moe-stage2-profiler-decision-table.md"
REQUIRED_VERSION_IDS = (
    "official_square_double_buffer",
    "fullrow_double_buffer_row16",
    "fullrow_single_buffer_row16",
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"cannot parse numeric profiler value {value!r}")
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"profiler value contains no number: {value!r}")
    return float(match.group().replace(",", ""))


def load_json(root: Path, relative_path: str) -> dict[str, Any]:
    return json.loads((root / relative_path).read_text())


def git_blob_sha256(root: Path, relative_path: str) -> str:
    content = subprocess.check_output(
        ["git", "-C", str(root), "show", f"HEAD:{relative_path}"]
    )
    return hashlib.sha256(content).hexdigest()


def parse_e2e_log(root: Path, relative_path: str) -> dict[str, float]:
    values = [
        float(value)
        for value in re.findall(
            r"Performance test:\s*([0-9.]+)ms", (root / relative_path).read_text()
        )
    ]
    if len(values) != 2:
        raise ValueError(f"expected Large/Small timings in {relative_path}, got {values}")
    return {"large_ms": values[0], "small_ms": values[1]}


def parse_torch_stage_durations(root: Path, relative_path: str) -> dict[str, float]:
    text = (root / relative_path).read_text()
    values: dict[str, float] = {}
    for kernel_name, stage in (("kernel_kernel", "fc1"), ("kernel_kernel_1", "fc2")):
        matching_rows = [
            line.split()
            for line in text.splitlines()
            if line.split() and line.split()[0] == kernel_name
        ]
        if len(matching_rows) != 1 or len(matching_rows[0]) < 8:
            raise ValueError(f"missing {kernel_name} duration in {relative_path}")
        # Profiler columns are Name, five CPU fields, Self CUDA, Self CUDA %,
        # then CUDA total/average/calls.  Token 6 is therefore Self CUDA.
        match = re.fullmatch(r"([0-9.]+)(ms|us)", matching_rows[0][6])
        if not match:
            raise ValueError(
                f"invalid Self CUDA duration for {kernel_name} in {relative_path}"
            )
        value = float(match.group(1))
        values[stage] = value if match.group(2) == "ms" else value / 1000.0
    return values


def profiler_sections(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        section: {row["name"]: row["value"] for row in rows}
        for section, rows in document.items()
        if isinstance(rows, list)
    }


def extract_targeted_counter(document: dict[str, Any]) -> dict[str, Any]:
    sections = profiler_sections(document)
    summary = sections["Summary"]
    ce = sections["CE Statistics"]
    isu = sections["ISU Statistics"]
    memory = sections["Memory Statistics"]
    shared = sections["Workgroup Memory"]
    occupancy = sections["Occupancy"]
    throughput = sections["GPU Throughput Statistics"]
    stalls = isu["ISU stall cycles layout"]["data"]
    return {
        "total_cycles_k": numeric(summary["Total Cycles"]),
        "total_instructions": int(numeric(summary["Total Instructions"])),
        "compute_instructions": int(numeric(summary["Compute Instructions"])),
        "memory_instructions": int(numeric(summary["Memory Instructions"])),
        "memory_access_mb_s": numeric(summary["Memory Access per Second"]),
        "ap_busy_pct": numeric(summary["AP busy Duty"]),
        "compute_busy_pct": numeric(summary["Compute Instructions busy Duty"]),
        "workgroups": int(numeric(ce["WORKGROUPS"])),
        "waves": int(numeric(ce["WAVES"])),
        "average_wave_life_cycles": numeric(ce["Average Wave life cycles"]),
        "wsm_stall_cycles": stalls["wsm_stall"],
        "vls_pipeline_stall_cycles": stalls["vls_pipeline_stall"],
        "vls_wdata_stall_cycles": stalls["vls_wdata_stall"],
        "valu_stall_cycles": stalls["valu_stall"],
        "private_read_instructions": int(numeric(memory["Private Read Instructions"])),
        "private_write_instructions": int(numeric(memory["Private Write Instructions"])),
        "vl1_hit_rate_pct": numeric(memory["VL1 Hit Rate"]),
        "l2_hit_rate_pct": numeric(memory["L2C Hit Rate"]),
        "global_memory_read_bytes": int(numeric(memory["Global Memory Read bytes"])),
        "global_memory_write_bytes": int(numeric(memory["Global Memory Write bytes"])),
        "dnoc_read_average_latency_cycles": numeric(memory["Dnoc Read Average Latency"]),
        "shared_conflict_cycles_per_instruction": numeric(
            shared["average conflict cycles per instruction"]
        ),
        "shared_access_efficiency_pct": numeric(
            shared["shared memory access efficiency"]
        ),
        "achieved_waves": int(numeric(occupancy["Achieved waves"])),
        "dispatched_waves": int(numeric(occupancy["Dispatched waves"])),
        "ap_mte_duty_pct": numeric(throughput["AP MTE Duty ratio"]),
        "ap_ste_duty_pct": numeric(throughput["AP STE Duty ratio"]),
        "ap_mma_duty_pct": numeric(throughput["AP MMA Duty ratio"]),
    }


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def inferred_static(
    dynamic_shared_bytes: int, reason: str, source: str
) -> dict[str, Any]:
    return {
        "status": "inferred",
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "mt_registers": None,
        "st_registers": None,
        "stack_frame_bytes": None,
        "static_max_warps_per_peu": None,
        "reason": reason,
        "source": source,
    }


def source_entry(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def targeted_report_path(
    root: Path, metadata_path: str, kernel_name: str
) -> str:
    metadata = load_json(root, metadata_path)
    expected_kernel = (
        metadata["workload"]["kernel_names"]["fc1_gate_up"]
        if kernel_name == "kernel_kernel"
        else metadata["workload"]["kernel_names"]["fc2_down"]
    )
    if expected_kernel != kernel_name:
        raise ValueError(
            f"metadata {metadata_path} maps the stage to {expected_kernel}, not {kernel_name}"
        )
    suffix = f"/1_{kernel_name}.txt.json"
    matches = [item["path"] for item in metadata["files"] if item["path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {kernel_name} targeted report in {metadata_path}, got {matches}"
        )
    return str((Path(metadata_path).parent / matches[0]).as_posix())


def stage1_stage(
    root: Path,
    resource: dict[str, Any],
    experiment: str,
    stage: str,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matrix = resource["profile_matrix"][experiment][stage]
    metadata_path = matrix.get("counter_reference", matrix["metadata"])
    kernel_name = "kernel_kernel" if stage == "fc1" else "kernel_kernel_1"
    raw_path = targeted_report_path(root, metadata_path, kernel_name)
    for path in (metadata_path, raw_path, STAGE1_RESOURCE, GENERATED_ARCHIVE):
        sources.setdefault(path, source_entry(root, path))
    static = resource["generated_device_code"][experiment][stage]
    reused_from = matrix.get("reused_from")
    if "counter_reference" in matrix:
        reused_from = load_json(root, metadata_path)["config"]["experiment"]
    if reused_from is not None:
        if (
            resource["stage_function_sha256"][experiment][stage]
            != resource["stage_function_sha256"][reused_from][stage]
        ):
            raise ValueError(
                f"unsafe counter reuse: {experiment} {stage} differs from {reused_from}"
            )
    return {
        "duration_ms": unavailable(
            "No same-run wall-clock stage timing was captured; mcProfiler Kcycles are not milliseconds."
        ),
        "static_resources": {"status": "measured", **static},
        "counters": {
            "status": "reused" if reused_from is not None else "measured",
            "reused_from": reused_from,
            "stage_function_sha256": resource["stage_function_sha256"][experiment][stage],
            "metadata": metadata_path,
            "targeted_report": raw_path,
            "metrics": extract_targeted_counter(load_json(root, raw_path)),
        },
    }


def e2e_record(
    *,
    large_ms: float,
    small_ms: float,
    source: str,
    measurement: str,
    independent_processes: int | None,
    e0_large_ms: float,
    e0_small_ms: float,
    combined_improvement_override: float | None = None,
) -> dict[str, Any]:
    combined = large_ms + small_ms
    e0_combined = e0_large_ms + e0_small_ms
    return {
        "large_ms": large_ms,
        "small_ms": small_ms,
        "combined_ms": combined,
        "combined_improvement_vs_e0_pct": (
            (e0_combined - combined) / e0_combined * 100.0
            if combined_improvement_override is None
            else combined_improvement_override
        ),
        "measurement": measurement,
        "independent_processes": independent_processes,
        "source": source,
    }


def classify_stage1_candidate(
    experiment: str,
    version: dict[str, Any],
    e0: dict[str, Any],
) -> dict[str, Any]:
    if experiment == "E0":
        return {
            "primary_bottleneck": "control_reference",
            "next_action": "Use as the unchanged default and comparison denominator.",
            "rule": "R0",
        }

    stage = "fc1" if experiment in ("E1", "E2") else "fc2"
    metrics = version["stages"][stage]["counters"]["metrics"]
    baseline = e0["stages"][stage]["counters"]["metrics"]
    static = version["stages"][stage]["static_resources"]
    baseline_static = e0["stages"][stage]["static_resources"]

    if metrics["private_read_instructions"] or metrics["private_write_instructions"]:
        return {
            "primary_bottleneck": "register_spill",
            "next_action": "Stop this tile branch; do not enlarge the accumulator or shared tile.",
            "rule": "R1",
        }
    if (
        metrics["workgroups"] >= baseline["workgroups"] * 1.5
        and metrics["total_cycles_k"] > baseline["total_cycles_k"] * 1.02
    ):
        return {
            "primary_bottleneck": "cta_amplification",
            "next_action": "Stop the smaller-N tile branch; CTA multiplication dominates resource relief.",
            "rule": "R2",
        }
    if (
        version["schedule"].get("s2_stages") == 2
        and metrics["total_instructions"] > baseline["total_instructions"] * 1.10
    ):
        return {
            "primary_bottleneck": "pipeline_overhead",
            "next_action": "Stop FC2 BK64/stage2; synchronization and instruction overhead dominate.",
            "rule": "R3",
        }
    if (
        metrics["global_memory_read_bytes"] > baseline["global_memory_read_bytes"] * 1.5
        and metrics["l2_hit_rate_pct"] < baseline["l2_hit_rate_pct"] - 5.0
    ):
        return {
            "primary_bottleneck": "memory_traffic_amplification",
            "next_action": "Keep as a diagnostic preset only; do not promote without locality repair.",
            "rule": "R4",
        }
    if (
        static["dynamic_shared_bytes"] < baseline_static["dynamic_shared_bytes"]
        and metrics["total_instructions"] > baseline["total_instructions"] * 1.10
        and version["e2e"]["combined_improvement_vs_e0_pct"] < 0
    ):
        return {
            "primary_bottleneck": "reduction_loop_overhead",
            "next_action": "Stop BK64-only tuning; the doubled reduction loop outweighs resource relief.",
            "rule": "R5",
        }
    raise ValueError(f"no deterministic Stage2 decision rule matched {experiment}")


def build_table(root: Path) -> dict[str, Any]:
    suite = load_json(root, STAGE1_SUITE)
    confirmation = load_json(root, STAGE1_CONFIRM)
    resource = load_json(root, STAGE1_RESOURCE)
    experiments = {item["experiment"]: item for item in suite["experiments"]}
    e0_perf = experiments["E0"]["performance"]
    e0_by_shape = {
        "large": next(item for item in e0_perf if item["config"]["dhidden"] == 7168),
        "small": next(item for item in e0_perf if item["config"]["dhidden"] == 3584),
    }
    e0_large = e0_by_shape["large"]["median_latency_ms"]
    e0_small = e0_by_shape["small"]["median_latency_ms"]

    sources: dict[str, dict[str, Any]] = {}
    for path in (
        STAGE1_SUITE,
        STAGE1_CONFIRM,
        STAGE1_RESOURCE,
        GENERATED_ARCHIVE,
        "benchmarks/tilelang-moe/custom_fusedmoe.baseline.py",
        "benchmarks/tilelang-moe/custom_fusedmoe.py",
        "config/moe_stage1_profiler_large_workload.json",
        "logs/moe-baseline-20260711T094431Z.log",
        "logs/moe-fullrow-row16-functional-20260711T102741Z.log",
        "logs/moe-fullrow-row16-performance-20260711T102855Z.log",
        "logs/moe-fullrow-row16-oneweight-functional-20260711T104301Z.log",
        "logs/moe-fullrow-row16-oneweight-performance-20260711T104431Z.log",
        "logs/moe-torch-profiler-large-20260711T144843Z.log",
        "logs/moe-torch-profiler-small-20260711T144218Z.log",
        "benchmarks/tilelang-moe/submission.py",
        "data/autotune/c500-32g/results.jsonl",
        "scripts/build_moe_stage2_profiler_table.py",
        "scripts/validate_moe_stage2_profiler.py",
    ):
        sources[path] = source_entry(root, path)

    versions: list[dict[str, Any]] = []
    official_e2e = parse_e2e_log(root, "logs/moe-baseline-20260711T094431Z.log")
    versions.append(
        {
            "id": "official_square_double_buffer",
            "label": "Official Square + double weight buffer",
            "schedule": {
                "s1_bn": 128,
                "s1_bk": 128,
                "s1_stages": 1,
                "s2_bn": 128,
                "s2_bk": 128,
                "s2_stages": 1,
                "threads": 256,
                "gemm_policy": "square",
                "swizzle": 10,
                "single_weight_buffer": False,
            },
            "e2e": e2e_record(
                **official_e2e,
                source="logs/moe-baseline-20260711T094431Z.log",
                measurement="single official-entrypoint run",
                independent_processes=1,
                e0_large_ms=e0_large,
                e0_small_ms=e0_small,
            ),
            "stages": {
                "fc1": {
                    "duration_ms": unavailable("No targeted stage timing was archived."),
                    "static_resources": inferred_static(
                        65536,
                        "Two 128x128 FP16 weight buffers in the archived baseline source.",
                        "benchmarks/tilelang-moe/custom_fusedmoe.baseline.py",
                    ),
                    "counters": unavailable("No target-kernel mcProfiler run was archived."),
                },
                "fc2": {
                    "duration_ms": unavailable("No targeted stage timing was archived."),
                    "static_resources": inferred_static(
                        32768,
                        "One 128x128 FP16 down-weight buffer in the archived baseline source.",
                        "benchmarks/tilelang-moe/custom_fusedmoe.baseline.py",
                    ),
                    "counters": unavailable("No target-kernel mcProfiler run was archived."),
                },
            },
            "decision": {
                "primary_bottleneck": "warp_policy_and_fc1_shared_capacity",
                "next_action": "Historical control: FullRow first, then reduce FC1 shared residency.",
                "rule": "H0",
            },
        }
    )

    double_e2e = parse_e2e_log(root, "logs/moe-fullrow-row16-performance-20260711T102855Z.log")
    versions.append(
        {
            "id": "fullrow_double_buffer_row16",
            "label": "FullRow row16/row16 + double weight buffer",
            "schedule": {
                "s1_bn": 128,
                "s1_bk": 128,
                "s1_stages": 1,
                "s2_bn": 128,
                "s2_bk": 128,
                "s2_stages": 1,
                "threads": 256,
                "gemm_policy": "full_row",
                "swizzle_panel": 16,
                "swizzle_panel_down": 16,
                "single_weight_buffer": False,
            },
            "e2e": e2e_record(
                **double_e2e,
                source="logs/moe-fullrow-row16-performance-20260711T102855Z.log",
                measurement="single candidate run",
                independent_processes=1,
                e0_large_ms=e0_large,
                e0_small_ms=e0_small,
            ),
            "stages": {
                stage: {
                    "duration_ms": unavailable("No targeted stage timing was archived."),
                    "static_resources": inferred_static(
                        65536 if stage == "fc1" else 32768,
                        "Reconstructed from the logged 128x128 double-buffer schedule.",
                        "benchmarks/tilelang-moe/custom_fusedmoe.py",
                    ),
                    "counters": unavailable("No target-kernel mcProfiler run was archived."),
                }
                for stage in ("fc1", "fc2")
            },
            "decision": {
                "primary_bottleneck": "fc1_shared_capacity",
                "next_action": "Serialize gate/up weight use through one shared buffer.",
                "rule": "H1",
            },
        }
    )

    single_e2e = parse_e2e_log(
        root, "logs/moe-fullrow-row16-oneweight-performance-20260711T104431Z.log"
    )
    large_duration = parse_torch_stage_durations(
        root, "logs/moe-torch-profiler-large-20260711T144843Z.log"
    )
    small_duration = parse_torch_stage_durations(
        root, "logs/moe-torch-profiler-small-20260711T144218Z.log"
    )
    versions.append(
        {
            "id": "fullrow_single_buffer_row16",
            "label": "FullRow row16/row16 + single weight buffer",
            "schedule": {
                "s1_bn": 128,
                "s1_bk": 128,
                "s1_stages": 1,
                "s2_bn": 128,
                "s2_bk": 128,
                "s2_stages": 1,
                "threads": 256,
                "gemm_policy": "full_row",
                "swizzle_panel": 16,
                "swizzle_panel_down": 16,
                "single_weight_buffer": True,
            },
            "e2e": e2e_record(
                **single_e2e,
                source="logs/moe-fullrow-row16-oneweight-performance-20260711T104431Z.log",
                measurement="single candidate run",
                independent_processes=1,
                e0_large_ms=e0_large,
                e0_small_ms=e0_small,
            ),
            "stages": {
                stage: {
                    "duration_ms": {
                        "status": "measured",
                        "large_ms": large_duration[stage],
                        "small_ms": small_duration[stage],
                        "large_source": "logs/moe-torch-profiler-large-20260711T144843Z.log",
                        "small_source": "logs/moe-torch-profiler-small-20260711T144218Z.log",
                        "note": "Historical row16 single-buffer predecessor; not an E0 row8 timing.",
                    },
                    "static_resources": inferred_static(
                        32768,
                        "One 128x128 FP16 shared weight buffer in each stage.",
                        "benchmarks/tilelang-moe/custom_fusedmoe.py",
                    ),
                    "counters": unavailable("No target-kernel mcProfiler run was archived."),
                }
                for stage in ("fc1", "fc2")
            },
            "decision": {
                "primary_bottleneck": "shared_memory_relief_validated",
                "next_action": "Retain single-buffer structure; refine only stage-specific scheduling.",
                "rule": "H2",
            },
        }
    )

    for experiment in (f"E{index}" for index in range(6)):
        item = experiments[experiment]
        if experiment == "E3":
            performance = {
                result["config"]["dhidden"]: result
                for result in confirmation["experiments"]["E3"]["performance"]
            }
            e2e_source = STAGE1_CONFIRM
            e2e_measurement = "interleaved three-process confirmation median"
            improvement_override = confirmation["comparison_to_e0"][
                "combined_improvement_pct"
            ]
        else:
            performance = {
                result["config"]["dhidden"]: result for result in item["performance"]
            }
            e2e_source = STAGE1_SUITE
            e2e_measurement = "three-process median, warmup 10, iteration 100"
            improvement_override = item["comparison_to_e0"][
                "combined_improvement_pct"
            ] if item["comparison_to_e0"] else 0.0
        version = {
            "id": experiment,
            "label": {
                "E0": "Current FullRow row8/row16 single-buffer control",
                "E1": "FC1 BK64",
                "E2": "FC1 BN64",
                "E3": "FC2 BK64 stage1",
                "E4": "FC2 BK64 stage2",
                "E5": "FC2 BN256",
            }[experiment],
            "schedule": item["schedule"],
            "e2e": e2e_record(
                large_ms=performance[7168]["median_latency_ms"],
                small_ms=performance[3584]["median_latency_ms"],
                source=e2e_source,
                measurement=e2e_measurement,
                independent_processes=3,
                e0_large_ms=e0_large,
                e0_small_ms=e0_small,
                combined_improvement_override=improvement_override,
            ),
            "stages": {
                stage: stage1_stage(root, resource, experiment, stage, sources)
                for stage in ("fc1", "fc2")
            },
        }
        versions.append(version)

    by_id = {version["id"]: version for version in versions}
    for experiment in (f"E{index}" for index in range(6)):
        by_id[experiment]["decision"] = classify_stage1_candidate(
            experiment, by_id[experiment], by_id["E0"]
        )

    e0_fc1 = by_id["E0"]["stages"]["fc1"]["counters"]["metrics"]
    e1_fc1 = by_id["E1"]["stages"]["fc1"]["counters"]["metrics"]
    table = {
        "schema_version": 1,
        "record_type": "moe-stage2-profiler-decision-table",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host_id": "c500-32g",
        "methodology": {
            "workload": "Large public performance shape",
            "targeted_counter_mode": "mcProfiler single-pass, one named kernel invocation",
            "duration_policy": "Kcycles are retained as hardware counter totals and never converted to milliseconds.",
            "reuse_policy": "Counters are reused only when the generated stage-function SHA256 is identical.",
            "architecture_terms": "MetaX MT/ST registers are preserved verbatim; they are not relabeled VGPR/SGPR.",
            "unavailable_metrics": [
                "active blocks per SM",
                "time-averaged active warps",
                "VGPR/SGPR mapping",
                "VALU utilization",
                "barrier stall",
                "empty/tail CTA counts",
            ],
            "plan_metric_coverage": {
                "fc1_fc2_duration": {
                    "status": "proxy",
                    "available": "mcProfiler Total Cycles in Kcycles",
                    "reason": "Same-run wall-clock stage duration is absent except for the historical row16 single-buffer Torch profile.",
                },
                "total_kernel_duration": {
                    "status": "unavailable",
                    "reason": "FC1/FC2 Kcycles are not additive wall-clock milliseconds.",
                },
                "active_blocks_per_sm": {
                    "status": "unavailable",
                    "reason": "No runtime resident-block counter was captured.",
                },
                "active_warps": {
                    "status": "unavailable",
                    "reason": "Achieved/dispatched waves are cumulative work, not concurrent activity.",
                },
                "vgpr_sgpr": {
                    "status": "proxy",
                    "available": "mxcc MT/ST registers",
                    "reason": "The archive contains no authoritative VGPR/SGPR name mapping.",
                },
                "spill_load_store": {
                    "status": "available_direct",
                    "available": "Private Read/Write Instructions plus stack frame bytes",
                },
                "shared_bytes_per_block": {
                    "status": "available_derived",
                    "available": "generated host launch dynamic shared bytes",
                },
                "bank_conflict": {
                    "status": "available_derived",
                    "available": "shared access efficiency and conflict cycles per instruction",
                },
                "matrix_core_active": {
                    "status": "available_direct",
                    "available": "AP MMA Duty ratio",
                },
                "valu_utilization": {
                    "status": "proxy",
                    "available": "AP MTE Duty ratio",
                    "reason": "Preserved as the profiler's MTE term, not relabeled as strict VALU utilization.",
                },
                "memory_dependency_stall": {
                    "status": "unavailable",
                    "reason": "WSM/VLS stall layout cannot be mapped uniquely to memory dependency.",
                },
                "barrier_stall": {
                    "status": "unavailable",
                    "reason": "No explicit barrier counter was captured.",
                },
                "l1_l2_hit_rate": {
                    "status": "available_direct",
                    "available": "VL1 Hit Rate and L2C Hit Rate",
                },
                "dram_read_write_bandwidth": {
                    "status": "unavailable",
                    "reason": "Read/write bytes exist, but a reliable kernel duration does not.",
                },
                "aggregate_memory_rate": {
                    "status": "available_direct",
                    "available": "Memory Access per Second in MB/s",
                },
                "cta_count": {
                    "status": "available_direct",
                    "available": "CE WORKGROUPS",
                },
                "tail_empty_cta_count": {
                    "status": "unavailable",
                    "reason": "The recorded routing metadata was not archived.",
                },
            },
        },
        "decision_rules": [
            {"id": "H0", "condition": "archived official Square double-buffer control", "result": "warp_policy_and_fc1_shared_capacity"},
            {"id": "H1", "condition": "FullRow double-buffer retains 64 KiB FC1 shared", "result": "fc1_shared_capacity"},
            {"id": "H2", "condition": "single-buffer row16 cuts FC1 shared to 32 KiB and improves e2e", "result": "shared_memory_relief_validated"},
            {"id": "R0", "condition": "canonical E0 schedule", "result": "control_reference"},
            {"id": "R1", "condition": "private read or write instructions > 0", "result": "register_spill"},
            {"id": "R2", "condition": "workgroups >= 1.5x E0 and cycles > 1.02x E0", "result": "cta_amplification"},
            {"id": "R3", "condition": "FC2 stages=2 and instructions > 1.10x E0", "result": "pipeline_overhead"},
            {"id": "R4", "condition": "global reads > 1.5x E0 and L2 hit < E0-5pp", "result": "memory_traffic_amplification"},
            {"id": "R5", "condition": "shared decreases, instructions > 1.10x E0, and e2e regresses", "result": "reduction_loop_overhead"},
        ],
        "versions": versions,
        "summary": {
            "enter_fc1_async_pipeline": False,
            "async_pipeline_reasons": [
                f"E1 BK64 combined e2e change is {by_id['E1']['e2e']['combined_improvement_vs_e0_pct']:.4f}%.",
                f"E0 FC1 VLS pipeline stall counter is {e0_fc1['vls_pipeline_stall_cycles']:.1f}, while E1 rises to {e1_fc1['vls_pipeline_stall_cycles']:.1f}.",
                "No same-run FC1 wall-clock improvement exists for BK64.",
            ],
            "recommended_next_stage": "Stage 4: merge FC1 epilogue and eliminate full/tail/empty invalid work",
            "retained_default": "E0",
            "tile_search_status": "stopped",
        },
        "guardrails": {
            "submission_path": "benchmarks/tilelang-moe/submission.py",
            "submission_sha256": sha256(root / "benchmarks/tilelang-moe/submission.py"),
            "submission_head_sha256": git_blob_sha256(
                root, "benchmarks/tilelang-moe/submission.py"
            ),
            "autotune_path": "data/autotune/c500-32g/results.jsonl",
            "autotune_sha256": sha256(root / "data/autotune/c500-32g/results.jsonl"),
            "autotune_head_sha256": git_blob_sha256(
                root, "data/autotune/c500-32g/results.jsonl"
            ),
        },
        "sources": [sources[path] for path in sorted(sources)],
        "report": DEFAULT_REPORT,
    }
    if tuple(version["id"] for version in versions) != REQUIRED_VERSION_IDS:
        raise AssertionError("Stage2 version order is incomplete")
    return table


def markdown_report(table: dict[str, Any]) -> str:
    by_id = {version["id"]: version for version in table["versions"]}
    lines = [
        "# MoE Stage 2：Profiler 决策表",
        "",
        "## 结论",
        "",
        "Stage 1 的 targeted counter 不支持继续扩大 tile 搜索，也不支持立即进入 FC1 异步流水。",
        "E1 BK64 端到端退化且新增大量 VLS pipeline stall；E2 因 CTA 翻倍显著退化；",
        "E4 的两级流水增加指令；E5 明确 spill。下一步转入 FC1 epilogue 合并以及",
        "full/tail/empty 无效工作消除，默认实现继续保持 E0。",
        "",
        "## 统一对照",
        "",
        "| 版本 | Large / Small ms | FC1 cycles K | FC2 cycles K | FC1 shared | FC2 shared | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for version in table["versions"]:
        def stage_value(stage: str, group: str, key: str) -> str:
            record = version["stages"][stage][group]
            if record["status"] not in ("measured", "reused", "inferred"):
                return "—"
            if group == "counters":
                return f"{record['metrics'][key]:.2f}"
            value = record[key]
            return "—" if value is None else str(value)

        lines.append(
            "| {label} | {large:.4f} / {small:.4f} | {fc1_cycles} | {fc2_cycles} | "
            "{fc1_shared} B | {fc2_shared} B | {decision} |".format(
                label=version["label"],
                large=version["e2e"]["large_ms"],
                small=version["e2e"]["small_ms"],
                fc1_cycles=stage_value("fc1", "counters", "total_cycles_k"),
                fc2_cycles=stage_value("fc2", "counters", "total_cycles_k"),
                fc1_shared=stage_value("fc1", "static_resources", "dynamic_shared_bytes"),
                fc2_shared=stage_value("fc2", "static_resources", "dynamic_shared_bytes"),
                decision=version["decision"]["primary_bottleneck"],
            )
        )

    e0_fc1 = by_id["E0"]["stages"]["fc1"]["counters"]["metrics"]
    e0_fc2 = by_id["E0"]["stages"]["fc2"]["counters"]["metrics"]
    lines.extend(
        [
            "",
            "## 决策依据",
            "",
            f"- E0 FC1/FC2 targeted counter 为 {e0_fc1['total_cycles_k']:.2f} / {e0_fc2['total_cycles_k']:.2f} Kcycles；这些不是毫秒。",
            "- E1 虽把 FC1 shared 从 32 KiB 降到 16 KiB，但 K 循环翻倍，端到端综合退化。",
            "- E2 的 workgroups 翻倍，指令与显存读取同步增加，occupancy 静态上限改善没有转化成收益。",
            "- E3 FC2 局部 cycles 降低，但全量确认只有 0.8816%，且显存读取增加、L2 命中下降。",
            "- E4 stage2 增加指令和流量；E5 出现 private read/write 与 340 B stack frame。",
            "",
            "## 数据边界",
            "",
            "官方 Square baseline 与 FullRow 双 buffer 仅有历史端到端日志，没有 target-kernel profiler。",
            "早期 `output20260712094550/94821` 实际采集的是 PyTorch distribution kernel，未纳入本表。",
            "mcProfiler 未直接提供 active blocks/SM、VGPR/SGPR 映射、VALU utilization、barrier stall",
            "以及 empty/tail CTA 数；这些字段保持 unavailable，不使用相似计数代替。",
            "",
            "## 下一步",
            "",
            "执行 Stage 4：先用回归测试锁定行为，再合并 FC1 epilogue，并实现 full/tail/empty 三路径。",
            "只有新的同口径 profiler 证明 FC1 存在可被覆盖的权重加载等待时，才重新打开异步流水分支。",
            "",
            "机器可读数据：[`stage2-decision-table.json`](../data/profiler/c500-32g/stage2-decision-table.json)。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / DEFAULT_OUTPUT
    report = args.report or root / DEFAULT_REPORT
    table = build_table(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(table))
    print(output)
    print(report)


if __name__ == "__main__":
    main()
