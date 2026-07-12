import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def moe_forward_tilelang_routed(
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
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16

    # Parameters
    dhidden = d_hidden
    dexpert = d_expert
    n_routed_experts = n_routed_experts

    # These are compile-time schedule constants.  Keeping the two kernels
    # independently configurable lets us tune their distinct reuse patterns
    # without changing the external RoutedMoEKernel interface.
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
    if single_weight_buffer and num_stages != 1:
        raise ValueError("single_weight_buffer currently requires num_stages=1")

    # The public benchmark builds routing metadata in fixed 128-token units.
    # A smaller compute tile therefore needs multiple CTAs per metadata entry
    # rather than changing the metadata tensor's shape or stride.
    metadata_block_token = 128
    if metadata_block_token % block_token != 0:
        raise ValueError("block_token must divide the benchmark metadata tile (128)")
    tiles_per_metadata_block = metadata_block_token // block_token
    metadata_M = math.ceil(group_sum / metadata_block_token) + group_count
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
        with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_fragment((block_token, block_dhidden), dtype=dtype)
            routed_expert_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            if single_weight_buffer:
                routed_expert_up_shared = routed_expert_gate_shared
            else:
                routed_expert_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)

            T.use_swizzle(panel_size=swizzle_panel, order=swizzle_order)
            if min_blocks_per_sm is not None:
                T.annotate_min_blocks_per_sm(min_blocks_per_sm)

            m_start_padded = bx * block_token

            metadata_bx = bx // tiles_per_metadata_block
            cur_group_idx = group_idx_for_bx[metadata_bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            if single_weight_buffer:
                # The two copies alias by design, so this loop must remain
                # serialized rather than use TileLang's copy pipeline.
                for k in T.serial(T.ceildiv(dhidden, block_dhidden)):
                    T.copy(
                        input[m_start : m_start + block_token, k * block_dhidden : (k + 1) * block_dhidden],
                        input_shared,
                    )
                    T.copy(
                        routed_expert_gate[
                            cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
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
                            cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
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
            else:
                for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                    T.copy(
                        input[m_start : m_start + block_token, k * block_dhidden : (k + 1) * block_dhidden],
                        input_shared,
                    )
                    T.copy(
                        routed_expert_gate[
                            cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
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
                            cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden
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

            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]

        # Step 2: Compute down logits
        with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_fragment((block_token, block_dexpert), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.use_swizzle(panel_size=swizzle_panel_down, order=swizzle_order_down)

            m_start_padded = bx * block_token

            metadata_bx = bx // tiles_per_metadata_block
            cur_group_idx = group_idx_for_bx[metadata_bx]

            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages_down):
                T.copy(
                    up_logits[m_start : m_start + block_token, k * block_dexpert : (k + 1) * block_dexpert],
                    up_logits_shared,
                )
                T.copy(
                    routed_expert_down[
                        cur_group_idx, by * block_dhidden : (by + 1) * block_dhidden, k * block_dexpert : (k + 1) * block_dexpert
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

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]

    return kernel


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
        self.swizzle_panel = swizzle_panel
        self.swizzle_order = swizzle_order
        self.swizzle_panel_down = swizzle_panel_down
        self.swizzle_order_down = swizzle_order_down
        self.gemm_policy = gemm_policy
        self.gemm_policy_down = gemm_policy_down
        self.single_weight_buffer = single_weight_buffer
        self.min_blocks_per_sm = min_blocks_per_sm
        self.backend = backend

        self.impl = moe_forward_tilelang_routed(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            group_sum=group_sum,
            group_count=group_count,
            block_token=block_token,
            block_dhidden=block_dhidden,
            block_dexpert=block_dexpert,
            threads=threads,
            num_stages=num_stages,
            num_stages_down=num_stages_down,
            swizzle_panel=swizzle_panel,
            swizzle_order=swizzle_order,
            swizzle_panel_down=swizzle_panel_down,
            swizzle_order_down=swizzle_order_down,
            gemm_policy=gemm_policy,
            gemm_policy_down=gemm_policy_down,
            single_weight_buffer=single_weight_buffer,
            min_blocks_per_sm=min_blocks_per_sm,
        )

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
