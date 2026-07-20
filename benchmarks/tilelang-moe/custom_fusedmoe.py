import math
import os
import json
import hashlib
import fcntl
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
from moe_schedule import resolve_stage_schedule, validate_stage_schedule


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_AUTOTUNE_ENABLED = _env_bool("MOE_AUTOTUNE", True)
_AUTOHEURISTIC_ENABLED = _env_bool("MOE_AUTOHEURISTIC", True)
_AUTOTUNE_WARMUP = _env_int("MOE_AUTOTUNE_WARMUP", 5)
_AUTOTUNE_REP = _env_int("MOE_AUTOTUNE_REP", 20)
_AUTOTUNE_TIMEOUT = _env_int("MOE_AUTOTUNE_TIMEOUT", 60)
_RESULT_RECORDING_ENABLED = _env_bool("MOE_RECORD_RESULTS", True)


def _shared_index_path() -> Path:
    default_root = Path(__file__).resolve().parents[2] / "data"
    return Path(os.environ.get("MOE_DATA_ROOT", default_root)) / "autoheuristic" / "index.json"


def _shared_variant(d_hidden, d_expert, n_routed_experts, group_sum, group_count):
    """Return a previously measured swizzle choice for this workload, if any."""
    if not _AUTOHEURISTIC_ENABLED:
        return None
    try:
        index = json.loads(_shared_index_path().read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    workload = f"{d_hidden}:{d_expert}:{n_routed_experts}:{group_sum}:{group_count}"
    host_id = os.environ.get("MOE_HOST_ID", "c500-unknown")
    groups = index.get("groups", {})
    exact = groups.get(f"{workload}|{host_id}")
    candidates = exact or next((value for key, value in groups.items() if key.startswith(f"{workload}|")), None)
    if not candidates:
        return None
    config = candidates.get("best", {}).get("config", {})
    if "swizzle_panel" not in config or "swizzle_panel_down" not in config:
        return None
    return int(config["swizzle_panel"]), int(config["swizzle_panel_down"])


def _append_jsonl(path: Path, record: dict) -> None:
    if not _RESULT_RECORDING_ENABLED:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _record_id(record: dict) -> str:
    payload = json.dumps(
        {"host_id": record["host_id"], "workload": record["workload"], "config": record["config"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _moe_autoheuristic_configs(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    group_count,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
    num_stages_down=1,
    swizzle_panel=8,
    swizzle_order="row",
    swizzle_panel_down=16,
    swizzle_order_down=None,
    gemm_policy="full_row",
    gemm_policy_down=None,
    single_weight_buffer=True,
    min_blocks_per_sm=None,
    metadata_m=None,
    combine_gate_up=False,
    s1_bn=None,
    s1_bk=None,
    s1_stages=None,
    s2_bn=None,
    s2_bk=None,
    s2_stages=None,
):
    """Build a safe tuning set from the C500 report's validated schedule family.

    The report ruled out generic tile/thread/pipeline changes.  Autoheuristic
    therefore dispatches only the validated row-swizzle variants, while
    TileLang autotune measures the selected small set on the actual tensors.
    """
    stage_schedule = resolve_stage_schedule(
        block_dhidden=block_dhidden,
        block_dexpert=block_dexpert,
        num_stages=num_stages,
        num_stages_down=num_stages_down,
        s1_bn=s1_bn,
        s1_bk=s1_bk,
        s1_stages=s1_stages,
        s2_bn=s2_bn,
        s2_bk=s2_bk,
        s2_stages=s2_stages,
    )
    common = {
        "s1_bn": stage_schedule["s1_bn"],
        "s1_bk": stage_schedule["s1_bk"],
        "s2_bn": stage_schedule["s2_bn"],
        "s2_bk": stage_schedule["s2_bk"],
        "threads": threads,
        "s1_stages": stage_schedule["s1_stages"],
        "s2_stages": stage_schedule["s2_stages"],
        "swizzle_order": "row",
        "swizzle_order_down": "row",
        "gemm_policy": "full_row",
        "gemm_policy_down": "full_row",
        "single_weight_buffer": True,
    }

    mode = os.environ.get("MOE_AUTOTUNE_MODE", "heuristic").lower()
    if mode == "full" or not _AUTOHEURISTIC_ENABLED:
        # These variants all passed functional validation in the report.  Do
        # not re-open variants already measured as slower or unsafe.
        variants = ((8, 16), (16, 16), (8, 8), (4, 16), (32, 16))
    elif _AUTOHEURISTIC_ENABLED and (d_hidden >= 6144 or d_expert >= 2048):
        # The large public case consistently benefits from stage-1 row8.
        variants = ((8, 16), (16, 16))
    else:
        # Keep the smaller case's measured control in the candidate set; the
        # autotuner decides if the marginal difference is real on this host.
        variants = ((16, 16), (8, 16))

    shared_variant = _shared_variant(d_hidden, d_expert, n_routed_experts, group_sum, group_count)
    if shared_variant in variants:
        variants = (shared_variant,) + tuple(variant for variant in variants if variant != shared_variant)

    return [
        {**common, "swizzle_panel": stage1_panel, "swizzle_panel_down": stage2_panel}
        for stage1_panel, stage2_panel in variants
    ]


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_tilelang_routed(
    d_hidden,
    d_expert,
    n_routed_experts,
    group_sum,
    group_count,
    block_token=128,
    block_dhidden=128,
    block_dexpert=128,
    threads=256,
    num_stages=1,
    num_stages_down=1,
    swizzle_panel=8,
    swizzle_order="row",
    swizzle_panel_down=16,
    swizzle_order_down=None,
    gemm_policy="full_row",
    gemm_policy_down=None,
    single_weight_buffer=True,
    min_blocks_per_sm=None,
    metadata_m=None,
    combine_gate_up=False,
    s1_bn=None,
    s1_bk=None,
    s1_stages=None,
    s2_bn=None,
    s2_bk=None,
    s2_stages=None,
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert
    n_routed_experts = n_routed_experts

    stage_schedule = resolve_stage_schedule(
        block_dhidden=block_dhidden,
        block_dexpert=block_dexpert,
        num_stages=num_stages,
        num_stages_down=num_stages_down,
        s1_bn=s1_bn,
        s1_bk=s1_bk,
        s1_stages=s1_stages,
        s2_bn=s2_bn,
        s2_bk=s2_bk,
        s2_stages=s2_stages,
    )
    s1_bn = stage_schedule["s1_bn"]
    s1_bk = stage_schedule["s1_bk"]
    s1_stages = stage_schedule["s1_stages"]
    s2_bn = stage_schedule["s2_bn"]
    s2_bk = stage_schedule["s2_bk"]
    s2_stages = stage_schedule["s2_stages"]
    validate_stage_schedule(stage_schedule, d_hidden=dhidden, d_expert=dexpert)

    # Existing swizzle and warp-policy parameters are already stage-specific.
    if swizzle_panel_down is None:
        swizzle_panel_down = swizzle_panel
    if swizzle_order_down is None:
        swizzle_order_down = swizzle_order
    if gemm_policy_down is None:
        gemm_policy_down = gemm_policy

    warp_policies = {
        "square": T.GemmWarpPolicy.Square,
        "full_row": T.GemmWarpPolicy.FullRow,
        "full_col": T.GemmWarpPolicy.FullCol,
    }
    try:
        gemm_policy_stage1 = warp_policies[gemm_policy]
        gemm_policy_stage2 = warp_policies[gemm_policy_down]
    except KeyError as exc:
        raise ValueError(f"unsupported GEMM warp policy: {exc.args[0]}") from exc
    if single_weight_buffer and s1_stages != 1:
        raise ValueError("single_weight_buffer currently requires s1_stages=1")
    if combine_gate_up and not single_weight_buffer:
        raise ValueError("combine_gate_up requires the single physical weight buffer path")

    # The public benchmark builds routing metadata in fixed 128-token units.
    # A smaller compute tile therefore needs multiple CTAs per metadata entry
    # rather than changing the metadata tensor's shape or stride.
    metadata_block_token = 128
    if metadata_block_token % block_token != 0:
        raise ValueError("block_token must divide the benchmark metadata tile (128)")
    tiles_per_metadata_block = metadata_block_token // block_token
    metadata_M = (
        math.ceil(group_sum / metadata_block_token) + group_count
        if metadata_m is None
        else metadata_m
    )
    if metadata_M <= 0:
        raise ValueError("metadata_m must be positive")
    M = metadata_M * tiles_per_metadata_block
    accum_dtype = T.float32

    input_shape = (group_sum, dhidden)
    intermediate_shape = (group_sum, dexpert)
    routed_expert_gate_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_up_shape = (n_routed_experts, dexpert, dhidden)
    routed_expert_down_shape = (n_routed_experts, dhidden, dexpert)
    routed_expert_weights_shape = group_sum
    group_sizes_shape = n_routed_experts

    @T.prim_func
    def kernel(
        input: T.Tensor(input_shape, dtype),  # type: ignore
        routed_expert_gate: T.Tensor(routed_expert_gate_shape, dtype),  # type: ignore
        routed_expert_up: T.Tensor(routed_expert_up_shape, dtype),  # type: ignore
        routed_expert_down: T.Tensor(routed_expert_down_shape, dtype),  # type: ignore
        routed_expert_weights: T.Tensor(routed_expert_weights_shape, dtype),  # type: ignore
        group_sizes: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_padded_offsets: T.Tensor(group_sizes_shape, T.int32),  # type: ignore
        group_idx_for_bx: T.Tensor((metadata_M,), T.int32),  # type: ignore
        up_logits: T.Tensor(intermediate_shape, dtype),  # type: ignore
        output: T.Tensor(input_shape, dtype),  # type: ignore
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(M, T.ceildiv(dexpert, s1_bn), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((block_token, s1_bk), dtype=dtype)
            if combine_gate_up:
                routed_expert_gate_up_shared = T.alloc_shared((2 * s1_bn, s1_bk), dtype=dtype)
                gate_up_logits_local = T.alloc_fragment(
                    (block_token, 2 * s1_bn), dtype=accum_dtype
                )
            else:
                routed_expert_gate_shared = T.alloc_shared((s1_bn, s1_bk), dtype=dtype)
                if single_weight_buffer:
                    routed_expert_up_shared = routed_expert_gate_shared
                else:
                    routed_expert_up_shared = T.alloc_shared((s1_bn, s1_bk), dtype=dtype)
                gate_logits_local = T.alloc_fragment((block_token, s1_bn), dtype=accum_dtype)
                up_logits_local = T.alloc_fragment((block_token, s1_bn), dtype=accum_dtype)

            T.use_swizzle(panel_size=swizzle_panel, order=swizzle_order)
            if min_blocks_per_sm is not None:
                T.annotate_min_blocks_per_sm(min_blocks_per_sm)

            m_start_padded = bx * block_token

            metadata_bx = bx // tiles_per_metadata_block
            cur_group_idx = group_idx_for_bx[metadata_bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            if combine_gate_up:
                T.clear(gate_up_logits_local)
                # One physical 2*BN weight tile and one accumulator replace
                # the two independent Gate/Up GEMMs. The two global tensors
                # remain separate to preserve the public benchmark ABI.
                # H8F1 keeps one physical buffer and the exact copy/GEMM
                # order, but lets the installed backend lower the same
                # stage-1 pipeline form that is already used by FC2.
                for k in T.Pipelined(T.ceildiv(dhidden, s1_bk), num_stages=s1_stages):
                    T.copy(
                        input[m_start : m_start + block_token, k * s1_bk : (k + 1) * s1_bk],
                        input_shared,
                    )
                    T.copy(
                        routed_expert_gate[
                            cur_group_idx,
                            by * s1_bn : (by + 1) * s1_bn,
                            k * s1_bk : (k + 1) * s1_bk,
                        ],
                        routed_expert_gate_up_shared[0:s1_bn, 0:s1_bk],
                    )
                    T.copy(
                        routed_expert_up[
                            cur_group_idx,
                            by * s1_bn : (by + 1) * s1_bn,
                            k * s1_bk : (k + 1) * s1_bk,
                        ],
                        routed_expert_gate_up_shared[s1_bn : 2 * s1_bn, 0:s1_bk],
                    )
                    T.gemm(
                        input_shared,
                        routed_expert_gate_up_shared,
                        gate_up_logits_local,
                        transpose_B=True,
                        policy=gemm_policy_stage1,
                    )

                # The combined weight tile is dead after the K mainloop. Reuse
                # its exact 256x64 shared allocation as two affine 128x64 Up
                # halves. Each Parallel region has one access pattern per
                # buffer, as required by the installed MACA layout inferencer.
                for i, j in T.Parallel(block_token, s1_bk):
                    routed_expert_gate_up_shared[2 * i, j] = (
                        gate_up_logits_local[i, j + s1_bn]
                    )
                for i, j in T.Parallel(block_token, s1_bk):
                    routed_expert_gate_up_shared[2 * i + 1, j] = (
                        gate_up_logits_local[i, j + s1_bn + s1_bk]
                    )

                for i, j in T.Parallel(block_token, s1_bk):
                    gate_up_logits_local[i, j] = gate_up_logits_local[i, j] * (
                        1.0 / (1.0 + T.exp2(-gate_up_logits_local[i, j] * scale))
                    )
                    routed_expert_gate_up_shared[2 * i, j] = (
                        routed_expert_gate_up_shared[2 * i, j] * gate_up_logits_local[i, j]
                    )
                for i, j in T.Parallel(block_token, s1_bk):
                    gate_up_logits_local[i, j + s1_bk] = (
                        gate_up_logits_local[i, j + s1_bk]
                        * (
                            1.0
                            / (
                                1.0
                                + T.exp2(-gate_up_logits_local[i, j + s1_bk] * scale)
                            )
                        )
                    )
                    routed_expert_gate_up_shared[2 * i + 1, j] = (
                        routed_expert_gate_up_shared[2 * i + 1, j]
                        * gate_up_logits_local[i, j + s1_bk]
                    )
            else:
                T.clear(gate_logits_local)
                T.clear(up_logits_local)

            if not combine_gate_up and single_weight_buffer:
                # The two copies alias by design, so this loop must remain
                # serialized rather than use TileLang's copy pipeline.
                for k in T.serial(T.ceildiv(dhidden, s1_bk)):
                    T.copy(
                        input[m_start : m_start + block_token, k * s1_bk : (k + 1) * s1_bk],
                        input_shared,
                    )
                    T.copy(
                        routed_expert_gate[
                            cur_group_idx, by * s1_bn : (by + 1) * s1_bn, k * s1_bk : (k + 1) * s1_bk
                        ],
                        routed_expert_gate_shared,
                    )
                    T.gemm(
                        input_shared,
                        routed_expert_gate_shared,
                        gate_logits_local,
                        transpose_B=True,
                        policy=gemm_policy_stage1,
                    )
                    T.copy(
                        routed_expert_up[
                            cur_group_idx, by * s1_bn : (by + 1) * s1_bn, k * s1_bk : (k + 1) * s1_bk
                        ],
                        routed_expert_up_shared,
                    )
                    T.gemm(
                        input_shared,
                        routed_expert_up_shared,
                        up_logits_local,
                        transpose_B=True,
                        policy=gemm_policy_stage1,
                    )
            elif not combine_gate_up:
                for k in T.Pipelined(T.ceildiv(dhidden, s1_bk), num_stages=s1_stages):
                    T.copy(
                        input[m_start : m_start + block_token, k * s1_bk : (k + 1) * s1_bk],
                        input_shared,
                    )
                    T.copy(
                        routed_expert_gate[
                            cur_group_idx, by * s1_bn : (by + 1) * s1_bn, k * s1_bk : (k + 1) * s1_bk
                        ],
                        routed_expert_gate_shared,
                    )
                    T.gemm(
                        input_shared,
                        routed_expert_gate_shared,
                        gate_logits_local,
                        transpose_B=True,
                        policy=gemm_policy_stage1,
                    )
                    T.copy(
                        routed_expert_up[
                            cur_group_idx, by * s1_bn : (by + 1) * s1_bn, k * s1_bk : (k + 1) * s1_bk
                        ],
                        routed_expert_up_shared,
                    )
                    T.gemm(
                        input_shared,
                        routed_expert_up_shared,
                        up_logits_local,
                        transpose_B=True,
                        policy=gemm_policy_stage1,
                    )

            if not combine_gate_up:
                for i, j in T.Parallel(block_token, s1_bn):
                    gate_logits_local[i, j] = gate_logits_local[i, j] * (
                        1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale))
                    )
                    up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            if combine_gate_up:
                for i, j in T.Parallel(block_token, s1_bk):
                    if i < actual_rows:
                        up_logits[m_start + i, by * s1_bn + j] = (
                            routed_expert_gate_up_shared[2 * i, j]
                        )
                for i, j in T.Parallel(block_token, s1_bk):
                    if i < actual_rows:
                        up_logits[m_start + i, by * s1_bn + j + s1_bk] = (
                            routed_expert_gate_up_shared[2 * i + 1, j]
                        )
            else:
                for i, j in T.Parallel(block_token, s1_bn):
                    if i < actual_rows:
                        up_logits[m_start + i, by * s1_bn + j] = up_logits_local[i, j]

        # Step 2: Compute down logits
        with T.Kernel(M, T.ceildiv(dhidden, s2_bn), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_shared((block_token, s2_bk), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((s2_bn, s2_bk), dtype=dtype)
            output_local = T.alloc_fragment((block_token, s2_bn), dtype=accum_dtype)
            routed_weight_local = T.alloc_fragment((block_token,), dtype=dtype)

            T.use_swizzle(panel_size=swizzle_panel_down, order=swizzle_order_down)

            m_start_padded = bx * block_token

            metadata_bx = bx // tiles_per_metadata_block
            cur_group_idx = group_idx_for_bx[metadata_bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, s2_bk), num_stages=s2_stages):
                T.copy(
                    up_logits[m_start : m_start + block_token, k * s2_bk : (k + 1) * s2_bk],
                    up_logits_shared,
                )
                T.copy(
                    routed_expert_down[
                        cur_group_idx, by * s2_bn : (by + 1) * s2_bn, k * s2_bk : (k + 1) * s2_bk
                    ],
                    routed_expert_down_shared,
                )
                T.gemm(
                    up_logits_shared,
                    routed_expert_down_shared,
                    output_local,
                    transpose_B=True,
                    policy=gemm_policy_stage2,
                )

            for i in T.Parallel(block_token):
                if i < actual_rows:
                    routed_weight_local[i] = routed_expert_weights[m_start + i]

            for i, j in T.Parallel(block_token, s2_bn):
                if i < actual_rows:
                    output[m_start + i, by * s2_bn + j] = output_local[i, j] * routed_weight_local[i]

    return kernel


if _AUTOTUNE_ENABLED:
    moe_forward_tilelang_routed = tilelang.autotune(
        configs=_moe_autoheuristic_configs,
        warmup=_AUTOTUNE_WARMUP,
        rep=_AUTOTUNE_REP,
        timeout=_AUTOTUNE_TIMEOUT,
        # The official functional cases remain the correctness gate.  All
        # candidates here are already known-good from the tuning report.
        skip_check=True,
        # The explicit set_autotune_inputs context supplies the same buffers
        # for every trial; disabling the tuner's secondary cache avoids false
        # shape-compatibility warnings for this multi-output ABI.
        cache_input_tensors=False,
    )(_moe_forward_tilelang_routed)
else:
    moe_forward_tilelang_routed = _moe_forward_tilelang_routed


class RoutedMoEKernel:

    def __init__(
        self,
        d_hidden: int,
        d_expert: int,
        n_routed_experts: int,
        group_sum: int,
        group_count: int,
        block_token: int = 128,
        block_dhidden: int = 128,
        block_dexpert: int = 128,
        threads: int = 256,
        num_stages: int = 1,
        backend: str = "tilelang",
        num_stages_down: int = 1,
        swizzle_panel: int = 8,
        swizzle_order: str = "row",
        swizzle_panel_down: Optional[int] = 16,
        swizzle_order_down: Optional[str] = None,
        gemm_policy: str = "full_row",
        gemm_policy_down: Optional[str] = None,
        single_weight_buffer: bool = True,
        min_blocks_per_sm: Optional[int] = None,
        s1_bn: Optional[int] = None,
        s1_bk: Optional[int] = None,
        s1_stages: Optional[int] = None,
        s2_bn: Optional[int] = None,
        s2_bk: Optional[int] = None,
        s2_stages: Optional[int] = None,
    ):
        self.d_hidden = d_hidden
        self.d_expert = d_expert
        self.n_routed_experts = n_routed_experts
        self.group_sum = group_sum
        self.group_count = group_count
        self.block_token = block_token
        self.block_dhidden = block_dhidden
        self.block_dexpert = block_dexpert
        self.threads = threads
        self.num_stages = num_stages
        self.num_stages_down = num_stages_down
        if s1_bk is None:
            s1_bk = 64
        stage_schedule = resolve_stage_schedule(
            block_dhidden=block_dhidden,
            block_dexpert=block_dexpert,
            num_stages=num_stages,
            num_stages_down=num_stages_down,
            s1_bn=s1_bn,
            s1_bk=s1_bk,
            s1_stages=s1_stages,
            s2_bn=s2_bn,
            s2_bk=s2_bk,
            s2_stages=s2_stages,
        )
        self.s1_bn = stage_schedule["s1_bn"]
        self.s1_bk = stage_schedule["s1_bk"]
        self.s1_stages = stage_schedule["s1_stages"]
        self.s2_bn = stage_schedule["s2_bn"]
        self.s2_bk = stage_schedule["s2_bk"]
        self.s2_stages = stage_schedule["s2_stages"]
        validate_stage_schedule(stage_schedule, d_hidden=d_hidden, d_expert=d_expert)
        self.swizzle_panel = swizzle_panel
        self.swizzle_order = swizzle_order
        self.swizzle_panel_down = swizzle_panel if swizzle_panel_down is None else swizzle_panel_down
        self.swizzle_order_down = swizzle_order if swizzle_order_down is None else swizzle_order_down
        self.gemm_policy = gemm_policy
        self.gemm_policy_down = gemm_policy if gemm_policy_down is None else gemm_policy_down
        self.single_weight_buffer = single_weight_buffer
        self.min_blocks_per_sm = min_blocks_per_sm
        # Internal launch metadata may be specialized by the benchmark's
        # packing path without changing the public constructor ABI.
        self.metadata_m = None
        # Aggressive candidates can select the exact Gate/Up wide-GEMM path
        # without extending the public constructor or call signatures.
        self.combine_gate_up = True
        self.backend = backend

        # Defer compilation/tuning until real tensors are available.  This
        # keeps every autotune candidate on the same routed inputs.
        self.impl = None
        self._tuning_record = None

    def _record_tuning_observation(self, correct=None, validation_source=None):
        if not _RESULT_RECORDING_ENABLED:
            return
        try:
            tuner_result = self.impl.get_tuner_result()
        except (AttributeError, ValueError):
            tuner_result = {}

        host_id = os.environ.get("MOE_HOST_ID", "c500-unknown")
        data_root = Path(os.environ.get("MOE_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
        profile_path = data_root / "hosts" / host_id / "host.json"
        try:
            profile = json.loads(profile_path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            profile = {}
        packages = profile.get("packages", {})
        config = {
            "block_token": self.block_token,
            "block_dhidden": self.block_dhidden,
            "block_dexpert": self.block_dexpert,
            "num_stages": self.num_stages,
            "num_stages_down": self.num_stages_down,
            "s1_bn": self.s1_bn,
            "s1_bk": self.s1_bk,
            "s1_stages": self.s1_stages,
            "s2_bn": self.s2_bn,
            "s2_bk": self.s2_bk,
            "s2_stages": self.s2_stages,
            "threads": self.threads,
            "swizzle_panel": self.swizzle_panel,
            "swizzle_order": self.swizzle_order,
            "swizzle_panel_down": self.swizzle_panel_down,
            "swizzle_order_down": self.swizzle_order_down,
            "gemm_policy": self.gemm_policy,
            "gemm_policy_down": self.gemm_policy_down,
            "single_weight_buffer": self.single_weight_buffer,
            "min_blocks_per_sm": self.min_blocks_per_sm,
            "combine_gate_up": self.combine_gate_up,
        }
        config.update(tuner_result.get("config") or {})
        record = {
            "schema_version": 1,
            "schedule_schema_version": 2,
            "record_type": "autotune",
            "record_id": None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "host_id": host_id,
            "host": {
                "host_id": host_id,
                "gpu_name": packages.get("gpu_name"),
                "gpu_memory_gb": packages.get("gpu_memory_gb"),
                "tilelang": packages.get("tilelang"),
                "torch": packages.get("torch"),
            },
            "workload": {
                "hidden": self.d_hidden,
                "intermediate": self.d_expert,
                "experts": self.n_routed_experts,
                "group_sum": self.group_sum,
                "group_count": self.group_count,
            },
            "config": config,
            "latency_ms": tuner_result.get("latency"),
            "correct": correct,
            "validation_source": validation_source,
            "autotune": {
                "enabled": _AUTOTUNE_ENABLED,
                "mode": os.environ.get("MOE_AUTOTUNE_MODE", "heuristic"),
                "candidate_count": len(
                    _moe_autoheuristic_configs(
                        self.d_hidden,
                        self.d_expert,
                        self.n_routed_experts,
                        self.group_sum,
                        self.group_count,
                    )
                ),
            },
            "source": {
                "git_commit": profile.get("git_commit") or os.environ.get("MOE_GIT_COMMIT"),
                "host_profile": str(profile_path.relative_to(data_root)) if profile_path.exists() else None,
            },
        }
        record["record_id"] = _record_id(record)
        if correct is not None:
            record["record_type"] = "validation"
        _append_jsonl(data_root / "autotune" / host_id / "results.jsonl", record)
        self._tuning_record = record

    def record_validation(self, correct: bool, source: str = "functional-test"):
        """Append a correctness verdict for the selected tuning result."""
        if self.impl is not None:
            self._record_tuning_observation(correct=bool(correct), validation_source=source)

    def __call__(
        self,
        input,
        routed_expert_gate,
        routed_expert_up,
        routed_expert_down,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
        output,
    ):
        if self.impl is None:
            autotune_inputs = (
                input,
                routed_expert_gate,
                routed_expert_up,
                routed_expert_down,
                routed_expert_weights,
                group_sizes,
                group_offsets,
                group_padded_offsets,
                group_idx_for_bx,
                up_logits,
                output,
            )
            compile_kwargs = {
                "d_hidden": self.d_hidden,
                "d_expert": self.d_expert,
                "n_routed_experts": self.n_routed_experts,
                "group_sum": self.group_sum,
                "group_count": self.group_count,
                "block_token": self.block_token,
            }
            if self.metadata_m is not None:
                compile_kwargs["metadata_m"] = self.metadata_m
            if self.combine_gate_up:
                compile_kwargs["combine_gate_up"] = True
            manual_schedule = {
                "s1_bn": self.s1_bn,
                "s1_bk": self.s1_bk,
                "s1_stages": self.s1_stages,
                "s2_bn": self.s2_bn,
                "s2_bk": self.s2_bk,
                "s2_stages": self.s2_stages,
                "threads": self.threads,
                "swizzle_panel": self.swizzle_panel,
                "swizzle_order": self.swizzle_order,
                "swizzle_panel_down": self.swizzle_panel_down,
                "swizzle_order_down": self.swizzle_order_down,
                "gemm_policy": self.gemm_policy,
                "gemm_policy_down": self.gemm_policy_down,
                "single_weight_buffer": self.single_weight_buffer,
                "min_blocks_per_sm": self.min_blocks_per_sm,
            }
            default_schedule = {
                "s1_bn": 128,
                "s1_bk": 128,
                "s1_stages": 1,
                "s2_bn": 128,
                "s2_bk": 128,
                "s2_stages": 1,
                "threads": 256,
                "swizzle_panel": 8,
                "swizzle_order": "row",
                "swizzle_panel_down": 16,
                "swizzle_order_down": "row",
                "gemm_policy": "full_row",
                "gemm_policy_down": "full_row",
                "single_weight_buffer": True,
                "min_blocks_per_sm": None,
            }
            # Explicit candidate values (used by tune_moe.py) bypass the
            # sweep; the default constructor enters autotune mode.
            if not _AUTOTUNE_ENABLED or manual_schedule != default_schedule:
                compile_kwargs.update(manual_schedule)

            if _AUTOTUNE_ENABLED and manual_schedule == default_schedule:
                with set_autotune_inputs(*autotune_inputs):
                    self.impl = moe_forward_tilelang_routed(**compile_kwargs)
            else:
                # A promoted/manual schedule is already fully specified. Do
                # not send it back through the autotuner wrapper: TileLang
                # would only detect duplicate tunable parameters on every new
                # RoutedMoEKernel instance, adding host-side launch gaps while
                # ultimately selecting the same direct JIT specialization.
                self.impl = _moe_forward_tilelang_routed(**compile_kwargs)

            self._record_tuning_observation()

        return self.impl(
            input,
            routed_expert_gate,
            routed_expert_up,
            routed_expert_down,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            group_idx_for_bx,
            up_logits,
            output,
        )
