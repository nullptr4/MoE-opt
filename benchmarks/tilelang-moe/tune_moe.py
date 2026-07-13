#!/usr/bin/env python3
"""Run one MoE TileLang schedule candidate against the official race configurations.

This script intentionally lives outside the official sample tree.  It monkey-patches
only the benchmark's factory so candidate schedules can be screened without mutating
custom_fusedmoe.py.  A winning parameter set is later made the source default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOE_DIR = ROOT
sys.path.insert(0, str(MOE_DIR))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-token", type=int, default=128)
    parser.add_argument("--block-dhidden", type=int, required=True)
    parser.add_argument("--block-dexpert", type=int, required=True)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=1, help="gate/up pipeline depth")
    parser.add_argument("--num-stages-down", type=int, default=None, help="down-projection pipeline depth; defaults to --num-stages")
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
    parser.add_argument("--mode", choices=("functional", "performance", "all"), default="all")
    args = parser.parse_args()

    # Imports occur after parsing so candidate schedule values are fully known
    # before the benchmark factory is installed.
    # This tool measures exactly one explicit candidate; the normal benchmark
    # entrypoint is where the bounded autotune/autoheuristic loop runs.
    os.environ["MOE_AUTOTUNE"] = "0"
    import fusedmoe_benchmark as benchmark  # noqa: E402
    from custom_fusedmoe import RoutedMoEKernel  # noqa: E402

    print(
        "CANDIDATE "
        f"block_token={args.block_token} block_dhidden={args.block_dhidden} "
        f"block_dexpert={args.block_dexpert} threads={args.threads} num_stages={args.num_stages} "
        f"num_stages_down={args.num_stages_down if args.num_stages_down is not None else args.num_stages} "
        f"swizzle={args.swizzle_order}{args.swizzle_panel} "
        f"swizzle_down={(args.swizzle_order_down or args.swizzle_order)}"
        f"{(args.swizzle_panel_down if args.swizzle_panel_down is not None else args.swizzle_panel)} "
        f"gemm_policy={args.gemm_policy} gemm_policy_down={args.gemm_policy_down or args.gemm_policy} "
        f"single_weight_buffer={args.single_weight_buffer} min_blocks_per_sm={args.min_blocks_per_sm}",
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
            swizzle_panel=args.swizzle_panel,
            swizzle_order=args.swizzle_order,
            swizzle_panel_down=args.swizzle_panel_down,
            swizzle_order_down=args.swizzle_order_down,
            gemm_policy=args.gemm_policy,
            gemm_policy_down=args.gemm_policy_down,
            single_weight_buffer=args.single_weight_buffer,
            min_blocks_per_sm=args.min_blocks_per_sm,
        )
        # The sample benchmark's group metadata is always formed in 128-token
        # units; kernel block_token is a compute tile, not a metadata stride.
        moe = benchmark.MoE(config, routed_kernel, weights, padding_M=128)
        return moe(input_tensor)

    benchmark.custom_kernel = candidate_kernel
    configs = benchmark.json.load(open(MOE_DIR / "moe_test_configs.json"))
    modes = ("functional", "performance") if args.mode == "all" else (args.mode,)
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        for config in configs[mode]:
            benchmark.run_moe_test(config, mode)


if __name__ == "__main__":
    main()
