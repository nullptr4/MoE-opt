#!/usr/bin/env python3
"""Local ABI test for the standalone padded-layout MoE submission.

This emulates the interface documented for the OJ rather than the compact
layout used by fusedmoe_benchmark.py.  It deliberately covers uneven group
tails and an empty expert, checks that padding remains untouched, and invokes
the cached submission twice.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MOE_DIR = ROOT
sys.path.insert(0, str(MOE_DIR))

import submission  # noqa: E402


BLOCK_TOKEN = 128


@dataclass
class PaddedCase:
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
) -> PaddedCase:
    device = torch.device("cuda")
    experts = len(group_sizes_list)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    raw_offsets = [0]
    padded_offsets = [0]
    block_experts: list[int] = []
    for expert_id, size in enumerate(group_sizes_list):
        raw_offsets.append(raw_offsets[-1] + size)
        blocks = math.ceil(size / BLOCK_TOKEN)
        block_experts.extend([expert_id] * blocks)
        padded_offsets.append(padded_offsets[-1] + blocks * BLOCK_TOKEN)

    total_valid = raw_offsets[-1]
    total_padded = padded_offsets[-1]
    tokens = torch.zeros((total_padded, hidden), device=device, dtype=torch.float16)
    for expert_id, size in enumerate(group_sizes_list):
        if size:
            start = padded_offsets[expert_id]
            tokens[start : start + size] = torch.randn((size, hidden), device=device, dtype=torch.float16, generator=generator)

    # Match the public operator's scale range so the numerical check is useful
    # without making the small synthetic test prone to overflow.
    gate = torch.randn((experts, intermediate, hidden), device=device, dtype=torch.float16, generator=generator) / math.sqrt(hidden)
    up = torch.randn((experts, intermediate, hidden), device=device, dtype=torch.float16, generator=generator) / math.sqrt(hidden)
    down = torch.randn((experts, hidden, intermediate), device=device, dtype=torch.float16, generator=generator) / math.sqrt(intermediate)
    if route_weights_mode == "random":
        route_weights = torch.rand((total_valid,), device=device, dtype=torch.float32, generator=generator)
    elif route_weights_mode == "zero":
        route_weights = torch.zeros((total_valid,), device=device, dtype=torch.float32)
    elif route_weights_mode == "one":
        route_weights = torch.ones((total_valid,), device=device, dtype=torch.float32)
    elif route_weights_mode == "tiny":
        route_weights = torch.full((total_valid,), 2**-10, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported route_weights_mode: {route_weights_mode}")

    return PaddedCase(
        name=name,
        hidden=hidden,
        intermediate=intermediate,
        group_sizes_list=group_sizes_list,
        tokens=tokens,
        gate=gate,
        up=up,
        down=down,
        route_weights=route_weights,
        group_sizes=torch.tensor(group_sizes_list, device=device, dtype=torch.int32),
        group_offsets=torch.tensor(raw_offsets, device=device, dtype=torch.int32),
        group_padded_offsets=torch.tensor(padded_offsets, device=device, dtype=torch.int32),
        group_idx_for_bx=torch.tensor(block_experts, device=device, dtype=torch.int32),
    )


def reference(case: PaddedCase) -> torch.Tensor:
    result = torch.full_like(case.tokens, float("nan"))
    raw_offsets = case.group_offsets.cpu().tolist()
    padded_offsets = case.group_padded_offsets.cpu().tolist()
    for expert_id, size in enumerate(case.group_sizes_list):
        if not size:
            continue
        padded_start = padded_offsets[expert_id]
        raw_start = raw_offsets[expert_id]
        x = case.tokens[padded_start : padded_start + size].float()
        gate = F.silu(x @ case.gate[expert_id].float().T)
        up = x @ case.up[expert_id].float().T
        # The kernel persists this product as the FP16 up_logits workspace
        # before the down projection.
        intermediate = (gate * up).to(torch.float16).float()
        value = (intermediate @ case.down[expert_id].float().T) * case.route_weights[raw_start : raw_start + size, None]
        result[padded_start : padded_start + size] = value.to(torch.float16)
    return result


def check_case(case: PaddedCase) -> None:
    expected = reference(case)
    out = torch.full_like(case.tokens, float("nan"))

    # First call compiles/caches; the second verifies that cached workspace and
    # compiled kernel preserve the externally supplied output contract.
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

    padded_offsets = case.group_padded_offsets.cpu().tolist()
    for expert_id, size in enumerate(case.group_sizes_list):
        padded_start = padded_offsets[expert_id]
        if size:
            torch.testing.assert_close(
                out[padded_start : padded_start + size].float(),
                expected[padded_start : padded_start + size].float(),
                atol=5e-3,
                rtol=2e-2,
            )
        padding_end = padded_offsets[expert_id + 1]
        if padding_end > padded_start + size:
            assert torch.isnan(out[padded_start + size : padding_end]).all(), f"padding modified for expert {expert_id}"

    print(
        f"PASS {case.name}: H={case.hidden} I={case.intermediate} "
        f"groups={case.group_sizes_list} padded={case.tokens.shape[0]} blocks={case.group_idx_for_bx.numel()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-shape", action="store_true", help="also run the public small H/I/E shape with uneven routing")
    parser.add_argument("--fuzz", action="store_true", help="also run deterministic boundary and skewed-routing cases")
    parser.add_argument("--fuzz-only", action="store_true", help="run only deterministic fuzz cases")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the submission ABI test requires an active C500")

    if not args.fuzz_only:
        check_case(make_case("uneven-smoke", hidden=256, intermediate=128, group_sizes_list=[129, 17, 0], seed=20260711))
        if args.public_shape:
            check_case(
                make_case(
                    "public-small-shape",
                    hidden=3584,
                    intermediate=1024,
                    group_sizes_list=[129, 64, 3, 0],
                    seed=20260712,
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
                "fuzz-tail-zero",
                hidden=256,
                intermediate=128,
                group_sizes_list=[1, 63, 64, 127],
                seed=20260716,
                route_weights_mode="zero",
            ),
        )
        for case in fuzz_cases:
            check_case(case)


if __name__ == "__main__":
    main()
