#!/usr/bin/env python3
"""Local compact-ABI validation for the remote Fused-MoE evaluator."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import submission  # noqa: E402


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


def check_case(case: CompactCase) -> None:
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
    print(
        f"PASS {case.name}: E={len(case.group_sizes_list)} H={case.hidden} "
        f"I={case.intermediate} group_sum={case.tokens.shape[0]} "
        f"blocks={case.group_idx_for_bx.numel()} offsets={case.group_offsets.numel()} "
        f"dtype={case.tokens.dtype}/{case.route_weights.dtype}",
        flush=True,
    )


def remote_case_specs() -> tuple[tuple[str, int, int, int], ...]:
    # Published remote dimensions. Every published group_sum is exactly 142 * E.
    return (
        ("remote-case-1", 2048, 8192, 16),
        ("remote-case-2", 7168, 2048, 32),
        ("remote-case-3", 7168, 2048, 64),
    )


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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the submission ABI test requires an active C500")

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
            for name, hidden, intermediate, experts in remote_case_specs():
                check_case(
                    make_case(
                        name,
                        hidden=hidden,
                        intermediate=intermediate,
                        group_sizes_list=[142] * experts,
                        seed=81394,
                    )
                )

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


if __name__ == "__main__":
    main()
