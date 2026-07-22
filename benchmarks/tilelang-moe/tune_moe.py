#!/usr/bin/env python3
"""Run one MoE TileLang schedule candidate against the official race configurations.

This script intentionally lives outside the official sample tree.  It monkey-patches
only the benchmark's factory so candidate schedules can be screened without mutating
custom_fusedmoe.py.  A winning parameter set is later made the source default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOE_DIR = ROOT
sys.path.insert(0, str(MOE_DIR))

from moe_schedule import (  # noqa: E402
    EXPERIMENT_PRESETS,
    canonical_experiment_schedule,
    experiment_stage_schedule,
    resolve_stage_schedule,
)
from moe_test_config_tools import (  # noqa: E402
    load_workload_config,
    local_proxy_guard_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-target",
        choices=("local-proxy-guard",),
        required=True,
        help="explicit opt-in; this formal schedule tuner cannot evaluate submission ABI",
    )
    parser.add_argument("--block-token", type=int, default=128)
    parser.add_argument("--block-dhidden", type=int, default=128, help="legacy shared tile: FC1 BK and FC2 BN")
    parser.add_argument("--block-dexpert", type=int, default=128, help="legacy shared tile: FC1 BN and FC2 BK")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=1, help="gate/up pipeline depth")
    parser.add_argument("--num-stages-down", type=int, default=None, help="down-projection pipeline depth; defaults to --num-stages")
    parser.add_argument("--s1-bn", type=int, default=None, help="explicit FC1 output tile")
    parser.add_argument("--s1-bk", type=int, default=None, help="explicit FC1 reduction tile")
    parser.add_argument("--s1-stages", type=int, default=None, help="explicit FC1 pipeline depth")
    parser.add_argument("--s2-bn", type=int, default=None, help="explicit FC2 output tile")
    parser.add_argument("--s2-bk", type=int, default=None, help="explicit FC2 reduction tile")
    parser.add_argument("--s2-stages", type=int, default=None, help="explicit FC2 pipeline depth")
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_PRESETS),
        help="run one canonical stage-1 experiment (E0-E5)",
    )
    parser.add_argument("--swizzle-panel", type=int, default=8, help="gate/up rasterization panel")
    parser.add_argument("--swizzle-order", choices=("row", "column"), default="row", help="gate/up rasterization order")
    parser.add_argument("--swizzle-panel-down", type=int, default=16, help="down-projection rasterization panel")
    parser.add_argument("--swizzle-order-down", choices=("row", "column"), default=None, help="down-projection rasterization order; defaults to --swizzle-order")
    parser.add_argument("--gemm-policy", choices=("square", "full_row", "full_col"), default="full_row", help="gate/up GEMM warp partition")
    parser.add_argument("--gemm-policy-down", choices=("square", "full_row", "full_col"), default=None, help="down-projection GEMM warp partition; defaults to --gemm-policy")
    parser.add_argument(
        "--single-weight-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="serialize stage-1 gate/up through one shared weight tile (shipping default)",
    )
    parser.add_argument("--min-blocks-per-sm", type=int, default=None, help="stage-1 launch-bounds occupancy hint")
    parser.add_argument(
        "--compact-metadata-grid",
        action="store_true",
        help="candidate-only: use exact per-expert block counts for the host metadata grid",
    )
    parser.add_argument(
        "--combine-gate-up",
        action="store_true",
        help="candidate-only: one physical 2*BN FC1 GEMM plus SwiGLU epilogue",
    )
    parser.add_argument(
        "--candidate-id",
        help="noncanonical experiment identifier recorded with a manual candidate",
    )
    parser.add_argument("--mode", choices=("functional", "performance", "all"), default="all")
    parser.add_argument("--shape", choices=("large", "small", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=10, help="performance warmup iterations")
    parser.add_argument("--iteration", type=int, default=100, help="timed performance iterations")
    parser.add_argument("--result-json", type=Path, help="write one machine-readable process result")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iteration <= 0:
        parser.error("--iteration must be positive")

    explicit_stage_values = (
        args.s1_bn,
        args.s1_bk,
        args.s1_stages,
        args.s2_bn,
        args.s2_bk,
        args.s2_stages,
    )
    if args.experiment:
        if args.combine_gate_up:
            parser.error(
                "--experiment fixes the complete canonical schedule and cannot be combined "
                "with --combine-gate-up"
            )
        if any(value is not None for value in explicit_stage_values):
            parser.error("--experiment cannot be combined with explicit --s1/--s2 tile or stage flags")
        if (
            args.block_dhidden != 128
            or args.block_dexpert != 128
            or args.num_stages != 1
            or args.num_stages_down not in (None, 1)
        ):
            parser.error("--experiment requires the legacy tile/stage flags to remain at baseline values")
        stage_schedule = experiment_stage_schedule(args.experiment)
    else:
        try:
            stage_schedule = resolve_stage_schedule(
                block_dhidden=args.block_dhidden,
                block_dexpert=args.block_dexpert,
                num_stages=args.num_stages,
                num_stages_down=args.num_stages_down,
                s1_bn=args.s1_bn,
                s1_bk=args.s1_bk,
                s1_stages=args.s1_stages,
                s2_bn=args.s2_bn,
                s2_bk=args.s2_bk,
                s2_stages=args.s2_stages,
            )
        except ValueError as exc:
            parser.error(str(exc))

    schedule = {
        "block_token": args.block_token,
        "block_dhidden": args.block_dhidden,
        "block_dexpert": args.block_dexpert,
        "threads": args.threads,
        "num_stages": args.num_stages,
        "num_stages_down": args.num_stages_down if args.num_stages_down is not None else args.num_stages,
        **stage_schedule,
        "swizzle_panel": args.swizzle_panel,
        "swizzle_order": args.swizzle_order,
        "swizzle_panel_down": args.swizzle_panel_down,
        "swizzle_order_down": args.swizzle_order_down or args.swizzle_order,
        "gemm_policy": args.gemm_policy,
        "gemm_policy_down": args.gemm_policy_down or args.gemm_policy,
        "single_weight_buffer": args.single_weight_buffer,
        "min_blocks_per_sm": args.min_blocks_per_sm,
    }
    if args.compact_metadata_grid:
        schedule["compact_metadata_grid"] = True
    if args.combine_gate_up:
        schedule["combine_gate_up"] = True
    if args.experiment:
        canonical_schedule = canonical_experiment_schedule(args.experiment)
        if schedule != canonical_schedule:
            mismatches = ", ".join(
                f"{name}={schedule.get(name)!r} (expected {value!r})"
                for name, value in canonical_schedule.items()
                if schedule.get(name) != value
            )
            parser.error(
                f"--experiment {args.experiment} fixes the complete canonical schedule; "
                f"noncanonical flags: {mismatches}"
            )

    # Imports occur after validation so the experiment label is guaranteed to
    # represent exactly one canonical schedule. This tool always measures one
    # explicit candidate; dynamic selection belongs to the normal entrypoint.
    os.environ["MOE_AUTOTUNE"] = "0"
    os.environ["MOE_AUTOHEURISTIC"] = "0"
    os.environ["MOE_RECORD_RESULTS"] = "0"
    import fusedmoe_benchmark as benchmark  # noqa: E402
    from custom_fusedmoe import RoutedMoEKernel  # noqa: E402

    print(
        "CANDIDATE "
        f"experiment={args.experiment or 'manual'} block_token={args.block_token} "
        f"s1_bn={stage_schedule['s1_bn']} s1_bk={stage_schedule['s1_bk']} "
        f"s1_stages={stage_schedule['s1_stages']} s2_bn={stage_schedule['s2_bn']} "
        f"s2_bk={stage_schedule['s2_bk']} s2_stages={stage_schedule['s2_stages']} "
        f"threads={args.threads} "
        f"swizzle={args.swizzle_order}{args.swizzle_panel} "
        f"swizzle_down={(args.swizzle_order_down or args.swizzle_order)}"
        f"{(args.swizzle_panel_down if args.swizzle_panel_down is not None else args.swizzle_panel)} "
        f"gemm_policy={args.gemm_policy} gemm_policy_down={args.gemm_policy_down or args.gemm_policy} "
        f"single_weight_buffer={args.single_weight_buffer} min_blocks_per_sm={args.min_blocks_per_sm} "
        f"compact_metadata_grid={args.compact_metadata_grid} combine_gate_up={args.combine_gate_up} "
        f"candidate_id={args.candidate_id}",
        flush=True,
    )

    def candidate_kernel(data):
        input_tensor, weights, config = data
        routed_kernel = RoutedMoEKernel(
            config["d_hidden"],
            config["d_expert"],
            config["n_routed_experts"],
            group_sum=config["batch_size"] * config["seq_len"] * config["n_experts_per_token"],
            group_count=config["n_routed_experts"],
            block_token=args.block_token,
            block_dhidden=args.block_dhidden,
            block_dexpert=args.block_dexpert,
            threads=args.threads,
            num_stages=args.num_stages,
            num_stages_down=args.num_stages_down if args.num_stages_down is not None else args.num_stages,
            s1_bn=stage_schedule["s1_bn"],
            s1_bk=stage_schedule["s1_bk"],
            s1_stages=stage_schedule["s1_stages"],
            s2_bn=stage_schedule["s2_bn"],
            s2_bk=stage_schedule["s2_bk"],
            s2_stages=stage_schedule["s2_stages"],
            swizzle_panel=args.swizzle_panel,
            swizzle_order=args.swizzle_order,
            swizzle_panel_down=args.swizzle_panel_down,
            swizzle_order_down=args.swizzle_order_down,
            gemm_policy=args.gemm_policy,
            gemm_policy_down=args.gemm_policy_down,
            single_weight_buffer=args.single_weight_buffer,
            min_blocks_per_sm=args.min_blocks_per_sm,
        )
        routed_kernel.combine_gate_up = args.combine_gate_up
        # The sample benchmark's group metadata is always formed in 128-token
        # units; kernel block_token is a compute tile, not a metadata stride.
        moe = benchmark.MoE(
            config,
            routed_kernel,
            weights,
            padding_M=128,
            compact_metadata_grid=args.compact_metadata_grid,
        )
        return moe(input_tensor)

    benchmark.custom_kernel = candidate_kernel
    configs = local_proxy_guard_configs(
        load_workload_config(MOE_DIR / "moe_test_configs.json"),
        explicit_opt_in=args.evaluation_target == "local-proxy-guard",
    )
    modes = ("functional", "performance") if args.mode == "all" else (args.mode,)
    results = []
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        selected_configs = configs[mode]
        if args.shape == "large":
            selected_configs = selected_configs[:1]
        elif args.shape == "small":
            selected_configs = selected_configs[1:2]
        for config in selected_configs:
            results.append(
                benchmark.run_moe_test(
                    config,
                    mode,
                    warm_up=args.warmup,
                    iteration=args.iteration,
                )
            )

    if args.result_json:
        record = {
            "schema_version": 1,
            "record_type": "moe-benchmark-process",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "experiment": args.experiment or args.candidate_id,
            "schedule": schedule,
            "benchmark": {
                "warmup": args.warmup,
                "iteration": args.iteration,
                "shape": args.shape,
            },
            "results": results,
        }
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    if any(result.get("test_type") == "functional" and not result.get("correct") for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
