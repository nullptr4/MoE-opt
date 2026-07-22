#!/usr/bin/env python3
"""Local compact-ABI validation for the remote Fused-MoE evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
from statistics import mean, median, pstdev
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from remote_contract_tools import (  # noqa: E402
    case_parameter_fingerprint,
    contract_fingerprint,
    load_contract,
    local_parity_summary,
    remote_case_specs as contract_case_specs,
    submission_abi_fingerprint,
)


def load_submission_module():
    reference = os.environ.get("MOE_SUBMISSION_SOURCE")
    source = ROOT / "submission.py" if reference is None else Path(reference)
    source = source.resolve()
    allowed_roots = (ROOT, Path.cwd().resolve())
    if not source.is_file() or not any(source.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError("submission source must be a file below the harness or current worktree")
    spec = importlib.util.spec_from_file_location("moe_submission_candidate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create submission module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_kernel", None)):
        raise RuntimeError("submission source omitted callable run_kernel")
    return module, source


submission, SUBMISSION_SOURCE = load_submission_module()
REMOTE_CONTRACT = load_contract()


BLOCK_TOKEN = 128


@dataclass
class CompactCase:
    name: str
    hidden: int
    intermediate: int
    group_sizes_list: list[int]
    tokens: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    route_weights: torch.Tensor
    group_sizes: torch.Tensor
    group_offsets: torch.Tensor
    group_padded_offsets: torch.Tensor
    group_idx_for_bx: torch.Tensor


def make_case(
    name: str,
    hidden: int,
    intermediate: int,
    group_sizes_list: list[int],
    seed: int,
    route_weights_mode: str = "random",
    terminal_offsets: bool = False,
) -> CompactCase:
    device = torch.device("cuda")
    experts = len(group_sizes_list)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    raw_offsets = [0]
    padded_offsets = [0]
    for size in group_sizes_list:
        raw_offsets.append(raw_offsets[-1] + size)
        # This is the exact formula used by the cited ee6db437 benchmark.
        blocks = math.ceil((size + 1) / BLOCK_TOKEN)
        padded_offsets.append(padded_offsets[-1] + blocks * BLOCK_TOKEN)

    group_sum = raw_offsets[-1]
    metadata_blocks = math.ceil(group_sum / BLOCK_TOKEN) + experts
    block_experts: list[int] = []
    for bx in range(metadata_blocks):
        padded_row = bx * BLOCK_TOKEN
        expert_id = 0
        for candidate in range(experts):
            if padded_row >= padded_offsets[candidate]:
                expert_id = candidate
        block_experts.append(expert_id)

    tokens = torch.randn(
        (group_sum, hidden),
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    gate = torch.randn(
        (experts, intermediate, hidden),
        device=device,
        dtype=torch.float16,
        generator=generator,
    ) / math.sqrt(hidden)
    up = torch.randn(
        (experts, intermediate, hidden),
        device=device,
        dtype=torch.float16,
        generator=generator,
    ) / math.sqrt(hidden)
    down = torch.randn(
        (experts, hidden, intermediate),
        device=device,
        dtype=torch.float16,
        generator=generator,
    ) / math.sqrt(intermediate)

    if route_weights_mode == "random":
        route_weights = torch.rand(
            (group_sum,),
            device=device,
            dtype=torch.float16,
            generator=generator,
        )
    elif route_weights_mode == "zero":
        route_weights = torch.zeros(
            (group_sum,), device=device, dtype=torch.float16
        )
    elif route_weights_mode == "one":
        route_weights = torch.ones(
            (group_sum,), device=device, dtype=torch.float16
        )
    elif route_weights_mode == "tiny":
        route_weights = torch.full(
            (group_sum,), 2**-10, device=device, dtype=torch.float16
        )
    else:
        raise ValueError(f"unsupported route_weights_mode: {route_weights_mode}")

    return CompactCase(
        name=name,
        hidden=hidden,
        intermediate=intermediate,
        group_sizes_list=group_sizes_list,
        tokens=tokens,
        gate=gate,
        up=up,
        down=down,
        route_weights=route_weights,
        group_sizes=torch.tensor(
            group_sizes_list, device=device, dtype=torch.int32
        ),
        group_offsets=torch.tensor(
            raw_offsets if terminal_offsets else raw_offsets[:-1],
            device=device,
            dtype=torch.int32,
        ),
        group_padded_offsets=torch.tensor(
            padded_offsets if terminal_offsets else padded_offsets[:-1],
            device=device,
            dtype=torch.int32,
        ),
        group_idx_for_bx=torch.tensor(
            block_experts, device=device, dtype=torch.int32
        ),
    )


def reference(case: CompactCase) -> torch.Tensor:
    result = torch.zeros_like(case.tokens)
    raw_start = 0
    for expert_id, size in enumerate(case.group_sizes_list):
        if size:
            x = case.tokens[raw_start : raw_start + size].float()
            gate = F.silu(x @ case.gate[expert_id].float().T)
            up = x @ case.up[expert_id].float().T
            intermediate = (gate * up).to(torch.float16).float()
            value = (
                intermediate @ case.down[expert_id].float().T
            ) * case.route_weights[raw_start : raw_start + size, None].float()
            result[raw_start : raw_start + size] = value.to(torch.float16)
        raw_start += size
    return result


def correctness_observation(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual32 = actual.float()
    expected32 = expected.float()
    absolute = torch.abs(actual32 - expected32)
    close = torch.isclose(actual32, expected32, atol=1e-2, rtol=1e-2, equal_nan=False)
    denominator = torch.clamp(torch.abs(expected32), min=1e-2)
    return {
        "atol": 1e-2,
        "rtol": 1e-2,
        "equal_nan": False,
        "total_values": int(expected.numel()),
        "mismatch_count": int((~close).sum().item()),
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float((absolute / denominator).max().item()),
    }


def check_case(case: CompactCase) -> tuple[dict[str, object], torch.Tensor]:
    expected = reference(case)
    out = torch.full_like(case.tokens, float("nan"))

    # First invocation compiles; the second checks the cached kernel/workspace.
    for _ in range(2):
        out.fill_(float("nan"))
        submission.run_kernel(
            case.tokens,
            case.gate,
            case.up,
            case.down,
            case.route_weights,
            case.group_sizes,
            case.group_offsets,
            case.group_padded_offsets,
            case.group_idx_for_bx,
            out,
        )
        torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=1e-2,
        rtol=1e-2,
        equal_nan=False,
    )
    observation = correctness_observation(out, expected)
    print(
        f"PASS {case.name}: E={len(case.group_sizes_list)} H={case.hidden} "
        f"I={case.intermediate} group_sum={case.tokens.shape[0]} "
        f"blocks={case.group_idx_for_bx.numel()} offsets={case.group_offsets.numel()} "
        f"dtype={case.tokens.dtype}/{case.route_weights.dtype}",
        flush=True,
    )
    return observation, out


def random_group_sizes(experts: int, group_sum: int, seed: int) -> list[int]:
    """Generate deterministic, non-uniform routed counts with an exact sum."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    assignments = torch.randint(
        0, experts, (group_sum,), generator=generator, device="cpu"
    )
    return torch.bincount(assignments, minlength=experts).tolist()


def benchmark_statistics(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    rank = 0.9 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    return {
        "samples_ms": samples,
        "count": len(samples),
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "stddev_ms": pstdev(samples),
        "p90_ms": p90,
        "outlier_policy": "none",
    }


def benchmark_case(case: CompactCase, warmup: int, iterations: int) -> dict[str, object]:
    out = torch.empty_like(case.tokens)
    args = (
        case.tokens,
        case.gate,
        case.up,
        case.down,
        case.route_weights,
        case.group_sizes,
        case.group_offsets,
        case.group_padded_offsets,
        case.group_idx_for_bx,
        out,
    )

    for _ in range(warmup):
        submission.run_kernel(*args)

    boundaries = [torch.cuda.Event(enable_timing=True) for _ in range(iterations + 1)]
    boundaries[0].record()
    for index in range(iterations):
        submission.run_kernel(*args)
        boundaries[index + 1].record()
    torch.cuda.synchronize()
    samples = [
        float(boundaries[index].elapsed_time(boundaries[index + 1]))
        for index in range(iterations)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise RuntimeError("benchmark produced a non-finite or non-positive sample")
    result = benchmark_statistics(samples)
    print(
        f"TIME {case.name}: {result['mean_ms']:.8f} ms mean "
        f"warmup={warmup} iters={iterations}",
        flush=True,
    )
    return result


def remote_case_specs() -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    return contract_case_specs(REMOTE_CONTRACT)


def environment_fingerprint() -> tuple[str, dict[str, object]]:
    import tilelang

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError("cannot capture source commit for hardware evidence")
    components = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "device": torch.cuda.get_device_name(0),
        "submission_source": SUBMISSION_SOURCE.as_posix(),
        "submission_sha256": hashlib.sha256(SUBMISSION_SOURCE.read_bytes()).hexdigest(),
        "contract_fingerprint": contract_fingerprint(REMOTE_CONTRACT),
        "source_commit": commit.stdout.strip(),
    }
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest, components


def thermal_clock_notes() -> str:
    result = subprocess.run(
        ("mx-smi",),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return f"mx-smi query failed with exit {result.returncode}"
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if "Kernel Mode Driver Version" in line
        or "MACA Version" in line
        or "MetaX C500" in line
        or ("W / 350W" in line and "C" in line and "P" in line)
    ]
    return " | ".join(lines) or "mx-smi returned no parseable C500 thermal/clock row"


def write_report(reference: str, report: dict[str, object]) -> Path:
    repository = ROOT.parents[1]
    destination = (repository / reference).resolve()
    if Path(reference).is_absolute() or not destination.is_relative_to(repository):
        raise RuntimeError("--report-json must be repository-relative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-shape",
        "--remote-shapes",
        action="store_true",
        dest="remote_shapes",
        help="run all three published remote evaluator dimensions",
    )
    parser.add_argument(
        "--fuzz", action="store_true", help="run deterministic compact metadata fuzz"
    )
    parser.add_argument(
        "--fuzz-only", action="store_true", help="run only compact metadata fuzz"
    )
    parser.add_argument(
        "--benchmark-remote",
        action="store_true",
        help="run correctness and exact published warmup/iteration timing",
    )
    parser.add_argument(
        "--report-json",
        help="write a repository-relative hardware evidence report",
    )
    args = parser.parse_args()

    if args.benchmark_remote:
        args.remote_shapes = True
    if args.report_json and not args.benchmark_remote:
        parser.error("--report-json requires --benchmark-remote complete evidence")

    if not torch.cuda.is_available():
        raise RuntimeError("the submission ABI test requires an active C500")

    case_results: list[dict[str, object]] = []
    if not args.fuzz_only:
        check_case(
            make_case(
                "compact-smoke",
                hidden=256,
                intermediate=128,
                group_sizes_list=[129, 17, 0],
                seed=20260711,
            )
        )
        if args.remote_shapes:
            for case_id, spec in enumerate(remote_case_specs(), start=1):
                name, hidden, intermediate, experts, group_sum, warmup, iterations = spec
                sizes = random_group_sizes(experts, group_sum, seed=81394 + case_id)
                case = make_case(
                    name,
                    hidden=hidden,
                    intermediate=intermediate,
                    group_sizes_list=sizes,
                    seed=81394,
                )
                print(
                    f"META {name}: group_min={min(sizes)} group_max={max(sizes)} "
                    f"group_sum={sum(sizes)}",
                    flush=True,
                )
                correctness, out = check_case(case)
                workspace = submission._get_workspace(case.tokens, intermediate)
                parameter = case_parameter_fingerprint(
                    contract=REMOTE_CONTRACT,
                    case=case,
                    out=out,
                    workspace=workspace,
                    warmup=warmup,
                    iterations=iterations,
                    routing_generator=(
                        f"torch.randint uniform expert assignments; seed={81394 + case_id}"
                    ),
                    route_weight_generator="torch.rand FP16; tensor generator seed=81394",
                    case_cache_lifecycle=(
                        "one process; correctness compilation precedes timing; "
                        "kernel/workspace caches persist; allocator cache emptied after case"
                    ),
                    compiler_path="submission.run_kernel -> tilelang.jit specialization cache",
                )
                benchmark = (
                    benchmark_case(case, warmup, iterations)
                    if args.benchmark_remote
                    else None
                )
                case_results.append(
                    {
                        "case_id": name,
                        "correctness": correctness,
                        "benchmark": benchmark,
                        "parameter_fingerprint": parameter,
                    }
                )
                del case
                gc.collect()
                torch.cuda.empty_cache()

    if args.fuzz or args.fuzz_only:
        fuzz_cases = (
            make_case(
                "fuzz-boundary-random",
                hidden=256,
                intermediate=128,
                group_sizes_list=[127, 128, 129, 0],
                seed=20260713,
            ),
            make_case(
                "fuzz-exact-one",
                hidden=256,
                intermediate=128,
                group_sizes_list=[255, 256, 257, 1],
                seed=20260714,
                route_weights_mode="one",
            ),
            make_case(
                "fuzz-skew-tiny",
                hidden=256,
                intermediate=128,
                group_sizes_list=[513, 1, 0, 0],
                seed=20260715,
                route_weights_mode="tiny",
            ),
            make_case(
                "fuzz-tail-zero-weight",
                hidden=256,
                intermediate=128,
                group_sizes_list=[1, 63, 64, 127],
                seed=20260716,
                route_weights_mode="zero",
            ),
            make_case(
                "fuzz-terminal-offset-sentinel",
                hidden=256,
                intermediate=128,
                group_sizes_list=[129, 17, 0],
                seed=20260717,
                terminal_offsets=True,
            ),
        )
        for case in fuzz_cases:
            check_case(case)

    if args.report_json:
        environment, components = environment_fingerprint()
        report = {
            "schema_version": "1.0",
            "report_type": "c500-routed-moe-submission-local-parity",
            "evidence_kind": "hardware",
            "support_level": "measured",
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "contract_fingerprint": contract_fingerprint(REMOTE_CONTRACT),
            "submission_abi_fingerprint": submission_abi_fingerprint(REMOTE_CONTRACT),
            "environment_fingerprint": environment,
            "environment": components,
            "source_commit": components["source_commit"],
            "thermal_clock_notes": thermal_clock_notes(),
            "timing_protocol": (
                "per published case: correctness/compile cache check, published warmup, "
                "then contiguous run_kernel calls with adjacent CUDA-event boundaries; "
                "all per-call samples retained and no outliers removed"
            ),
            "case_order": [item[0] for item in remote_case_specs()],
            "cases": case_results,
            "scoring": {
                "per_case_metrics": True,
                "aggregate_status": "unknown",
                "aggregate_value": None,
            },
            "parity": local_parity_summary(REMOTE_CONTRACT),
            "evidence_boundary": (
                "Real C500 measurements for local routing fixtures; not exact remote-equivalent "
                "while routing, offset sentinel, cache lifecycle, aggregate scoring, strides, "
                "workspace policy, and online toolchain remain unresolved."
            ),
        }
        destination = write_report(args.report_json, report)
        print(f"REPORT {destination.relative_to(ROOT.parents[1]).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
