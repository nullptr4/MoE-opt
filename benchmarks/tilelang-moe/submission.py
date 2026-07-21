"""Standalone MetaX C500 TileLang Fused-MoE submission.

This file is self-contained and exposes only the required ``run_kernel`` entry
point.  The input/intermediate/output tensors use padded expert-token storage,
while routed weights use compact storage indexed through ``group_offsets``.
"""

import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _make_moe_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    group_offsets_len,
    group_padded_offsets_len,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    block_token = 128
    stage1_bn = 128
    stage1_bk = 64
    stage2_bn = 128
    stage2_bk = 64
    threads = 256

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((group_offsets_len,), T.int32),
        group_padded_offsets: T.Tensor((group_padded_offsets_len,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # Stage 1: one combined Gate/Up GEMM.  The stage-1 K loop uses the
        # C500-validated same-buffer pipeline.  After the mainloop, the dead
        # 256x64 weight tile is reused as two affine 128x64 FP16 Up halves.
        with T.Kernel(
            num_blocks_m,
            T.ceildiv(intermediate, stage1_bn),
            threads=threads,
        ) as (bx, by):
            input_local = T.alloc_fragment((block_token, stage1_bk), dtype=dtype)
            gate_up_shared = T.alloc_shared((2 * stage1_bn, stage1_bk), dtype=dtype)
            gate_up_local = T.alloc_fragment(
                (block_token, 2 * stage1_bn),
                dtype=accum_dtype,
            )

            T.use_swizzle(panel_size=8, order="row")

            expert_id = group_idx_for_bx[bx]
            T.assume(0 <= expert_id)
            T.assume(expert_id < num_experts)
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_size - token_offset),
            )
            # H17F2's compact-row proof is adapted to the OJ's padded storage:
            # stage 1 addresses input/workspace rows through block_start.
            T.assume(0 <= block_start)
            T.assume(block_start + actual_rows <= total_padded_tokens)

            T.clear(gate_up_local)

            for k in T.Pipelined(
                T.ceildiv(hidden, stage1_bk),
                num_stages=1,
            ):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + block_token,
                        k * stage1_bk : (k + 1) * stage1_bk,
                    ],
                    input_local,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * stage1_bn : (by + 1) * stage1_bn,
                        k * stage1_bk : (k + 1) * stage1_bk,
                    ],
                    gate_up_shared[0:stage1_bn, 0:stage1_bk],
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * stage1_bn : (by + 1) * stage1_bn,
                        k * stage1_bk : (k + 1) * stage1_bk,
                    ],
                    gate_up_shared[stage1_bn : 2 * stage1_bn, 0:stage1_bk],
                )
                T.gemm(
                    input_local,
                    gate_up_shared,
                    gate_up_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # H10F5 shared-buffer reuse.  Separate Parallel regions keep one
            # affine access pattern per region for the installed MACA lowerer.
            for i, j in T.Parallel(block_token, stage1_bk):
                gate_up_shared[2 * i, j] = gate_up_local[i, j + stage1_bn]
            for i, j in T.Parallel(block_token, stage1_bk):
                gate_up_shared[2 * i + 1, j] = (
                    gate_up_local[i, j + stage1_bn + stage1_bk]
                )

            for i, j in T.Parallel(block_token, stage1_bk):
                gate_up_local[i, j] = gate_up_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_up_local[i, j] * scale))
                )
                gate_up_shared[2 * i, j] = (
                    gate_up_shared[2 * i, j] * gate_up_local[i, j]
                )
            for i, j in T.Parallel(block_token, stage1_bk):
                gate_up_local[i, j + stage1_bk] = (
                    gate_up_local[i, j + stage1_bk]
                    * (
                        1.0
                        / (
                            1.0
                            + T.exp2(
                                -gate_up_local[i, j + stage1_bk] * scale
                            )
                        )
                    )
                )
                gate_up_shared[2 * i + 1, j] = (
                    gate_up_shared[2 * i + 1, j]
                    * gate_up_local[i, j + stage1_bk]
                )

            for i, j in T.Parallel(block_token, stage1_bk):
                if i < actual_rows:
                    up_logits[
                        block_start + i,
                        by * stage1_bn + j,
                    ] = gate_up_shared[2 * i, j]
            for i, j in T.Parallel(block_token, stage1_bk):
                if i < actual_rows:
                    up_logits[
                        block_start + i,
                        by * stage1_bn + j + stage1_bk,
                    ] = gate_up_shared[2 * i + 1, j]

        # Stage 2: shared FC2 activation tile plus one cached route weight per
        # row.  Output addresses are padded; route-weight addresses are compact.
        with T.Kernel(
            num_blocks_m,
            T.ceildiv(hidden, stage2_bn),
            threads=threads,
        ) as (bx, by):
            up_shared = T.alloc_shared((block_token, stage2_bk), dtype=dtype)
            down_shared = T.alloc_shared((stage2_bn, stage2_bk), dtype=dtype)
            out_local = T.alloc_fragment(
                (block_token, stage2_bn),
                dtype=accum_dtype,
            )
            route_weight_local = T.alloc_fragment((block_token,), dtype=dtype)

            T.use_swizzle(panel_size=16, order="row")

            expert_id = group_idx_for_bx[bx]
            T.assume(0 <= expert_id)
            T.assume(expert_id < num_experts)
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_size - token_offset),
            )
            # Output/workspace rows are padded, while route weights are compact.
            # Keep both contracts explicit rather than reusing one address space.
            T.assume(0 <= block_start)
            T.assume(block_start + actual_rows <= total_padded_tokens)
            T.assume(0 <= raw_start + token_offset)
            T.assume(
                raw_start + token_offset + actual_rows <= total_valid_tokens
            )

            T.clear(out_local)

            for k in T.Pipelined(
                T.ceildiv(intermediate, stage2_bk),
                num_stages=1,
            ):
                T.copy(
                    up_logits[
                        block_start : block_start + block_token,
                        k * stage2_bk : (k + 1) * stage2_bk,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * stage2_bn : (by + 1) * stage2_bn,
                        k * stage2_bk : (k + 1) * stage2_bk,
                    ],
                    down_shared,
                )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i in T.Parallel(block_token):
                if i < actual_rows:
                    route_weight_local[i] = routed_expert_weights[
                        raw_start + token_offset + i
                    ]

            for i, j in T.Parallel(block_token, stage2_bn):
                if i < actual_rows:
                    out[block_start + i, by * stage2_bn + j] = (
                        out_local[i, j] * route_weight_local[i]
                    )
                else:
                    out[block_start + i, by * stage2_bn + j] = 0.0

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    group_offsets_len,
    group_padded_offsets_len,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
        int(group_offsets_len),
        int(group_padded_offsets_len),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _make_moe_kernel(
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def _get_workspace(stacked_expert_tokens, intermediate):
    key = (
        stacked_expert_tokens.device,
        stacked_expert_tokens.dtype,
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
    )
    up_logits = _WORKSPACE_CACHE.get(key)
    if up_logits is None:
        up_logits = torch.empty(
            (
                int(stacked_expert_tokens.shape[0]),
                int(intermediate),
            ),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = up_logits
    return up_logits


def run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
):
    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])
    group_offsets_len = int(group_offsets.shape[0])
    group_padded_offsets_len = int(group_padded_offsets.shape[0])

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
        group_offsets_len,
        group_padded_offsets_len,
    )
    kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
        out,
    )
