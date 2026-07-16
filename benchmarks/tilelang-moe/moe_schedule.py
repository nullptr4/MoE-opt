"""Pure-Python MoE schedule resolution shared by the kernel and experiments."""

from __future__ import annotations

from typing import Any


BASELINE_STAGE_SCHEDULE = {
    "s1_bn": 128,
    "s1_bk": 128,
    "s1_stages": 1,
    "s2_bn": 128,
    "s2_bk": 128,
    "s2_stages": 1,
}

STAGE1_FIXED_SCHEDULE = {
    "block_token": 128,
    "block_dhidden": 128,
    "block_dexpert": 128,
    "threads": 256,
    "num_stages": 1,
    "num_stages_down": 1,
    "swizzle_panel": 8,
    "swizzle_order": "row",
    "swizzle_panel_down": 16,
    "swizzle_order_down": "row",
    "gemm_policy": "full_row",
    "gemm_policy_down": "full_row",
    "single_weight_buffer": True,
    "min_blocks_per_sm": None,
}

EXPERIMENT_PRESETS = {
    "E0": {},
    "E1": {"s1_bk": 64},
    "E2": {"s1_bn": 64},
    "E3": {"s2_bk": 64},
    "E4": {"s2_bk": 64, "s2_stages": 2},
    "E5": {"s2_bn": 256},
}


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_stage_schedule(
    *,
    block_dhidden: int = 128,
    block_dexpert: int = 128,
    num_stages: int = 1,
    num_stages_down: int | None = None,
    s1_bn: int | None = None,
    s1_bk: int | None = None,
    s1_stages: int | None = None,
    s2_bn: int | None = None,
    s2_bk: int | None = None,
    s2_stages: int | None = None,
) -> dict[str, int]:
    """Resolve legacy shared tiles into explicit FC1/FC2 compile-time tiles."""
    resolved = {
        "s1_bn": block_dexpert if s1_bn is None else s1_bn,
        "s1_bk": block_dhidden if s1_bk is None else s1_bk,
        "s1_stages": num_stages if s1_stages is None else s1_stages,
        "s2_bn": block_dhidden if s2_bn is None else s2_bn,
        "s2_bk": block_dexpert if s2_bk is None else s2_bk,
        "s2_stages": (
            num_stages if num_stages_down is None else num_stages_down
        ) if s2_stages is None else s2_stages,
    }
    return {name: _positive_int(name, value) for name, value in resolved.items()}


def experiment_stage_schedule(experiment: str) -> dict[str, int]:
    """Return one canonical single-variable E0-E5 experiment schedule."""
    try:
        overrides = EXPERIMENT_PRESETS[experiment]
    except KeyError as exc:
        raise ValueError(f"unknown experiment: {experiment}") from exc
    return {**BASELINE_STAGE_SCHEDULE, **overrides}


def canonical_experiment_schedule(experiment: str) -> dict[str, Any]:
    """Return the complete fixed schedule represented by an E0-E5 label."""
    return {
        **STAGE1_FIXED_SCHEDULE,
        **experiment_stage_schedule(experiment),
    }


def validate_stage_schedule(
    schedule: dict[str, int],
    *,
    d_hidden: int,
    d_expert: int,
) -> None:
    """Reject tile tails that the current unmasked N/K loads cannot handle."""
    extents = {
        "s1_bn": d_expert,
        "s1_bk": d_hidden,
        "s2_bn": d_hidden,
        "s2_bk": d_expert,
    }
    for name, extent in extents.items():
        tile = _positive_int(name, schedule[name])
        if tile % 16 != 0:
            raise ValueError(f"{name} must be a multiple of 16, got {tile}")
        if extent % tile != 0:
            raise ValueError(f"{name}={tile} must divide its extent {extent}")
