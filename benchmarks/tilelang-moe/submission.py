"""Conservative standalone MetaX C500 TileLang Fused-MoE submission.

The ten-argument OJ ABI has appeared with both compact storage (the cited
``ee6db437`` benchmark) and expert-padded storage (the tutorial contract).
This compatibility baseline selects the address contract from tensor shapes,
supports FP16 or FP32 route weights and E or E+1 offset tensors, and otherwise
keeps the original two-stage TileLang kernel structure.
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
    storage_rows,
    route_rows,
    num_blocks_m,
    group_offsets_len,
    group_padded_offsets_len,
    compact_storage,
    route_weight_fp32,
):
    scale = 1.44269504
    dtype = T.float16
    route_dtype = T.float32 if route_weight_fp32 else T.float16
    accum_dtype = T.float32

    block_token = 128
    block_hidden = 128
    block_intermediate = 128
    threads = 256

    input_shape = (storage_rows, hidden)
    intermediate_shape = (storage_rows, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((route_rows,), route_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((group_offsets_len,), T.int32),
        group_padded_offsets: T.Tensor((group_padded_offsets_len,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # Keep the simple kernel organization from the cited ee6db437 source.
        # In compact mode padded metadata addresses are translated back through
        # raw group offsets.  In padded mode the block address is already the
        # physical tensor address.
        with T.Kernel(
            num_blocks_m,
            T.ceildiv(intermediate, block_intermediate),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_fragment(
                (block_token, block_hidden),
                dtype=dtype,
            )
            gate_shared = T.alloc_shared(
                (block_intermediate, block_hidden),
                dtype=dtype,
            )
            up_shared = T.alloc_shared(
                (block_intermediate, block_hidden),
                dtype=dtype,
            )
            gate_local = T.alloc_fragment(
                (block_token, block_intermediate),
                dtype=accum_dtype,
            )
            up_local = T.alloc_fragment(
                (block_token, block_intermediate),
                dtype=accum_dtype,
            )

            T.use_swizzle(10)

            block_start = bx * block_token
            expert_id = group_idx_for_bx[bx]
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_size - token_offset),
            )
            if compact_storage:
                data_start = raw_start + token_offset
            else:
                data_start = block_start

            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(
                T.ceildiv(hidden, block_hidden),
                num_stages=1,
            ):
                T.copy(
                    stacked_expert_tokens[
                        data_start : data_start + block_token,
                        k * block_hidden : (k + 1) * block_hidden,
                    ],
                    input_shared,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * block_intermediate : (by + 1) * block_intermediate,
                        k * block_hidden : (k + 1) * block_hidden,
                    ],
                    gate_shared,
                )
                T.gemm(
                    input_shared,
                    gate_shared,
                    gate_local,
                    transpose_B=True,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * block_intermediate : (by + 1) * block_intermediate,
                        k * block_hidden : (k + 1) * block_hidden,
                    ],
                    up_shared,
                )
                T.gemm(
                    input_shared,
                    up_shared,
                    up_local,
                    transpose_B=True,
                )

            for i, j in T.Parallel(block_token, block_intermediate):
                gate_local[i, j] = gate_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale))
                )
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            for i, j in T.Parallel(block_token, block_intermediate):
                if i < actual_rows:
                    up_logits[
                        data_start + i,
                        by * block_intermediate + j,
                    ] = up_local[i, j]

        with T.Kernel(
            num_blocks_m,
            T.ceildiv(hidden, block_hidden),
            threads=threads,
        ) as (bx, by):
            up_shared = T.alloc_fragment(
                (block_token, block_intermediate),
                dtype=dtype,
            )
            down_shared = T.alloc_shared(
                (block_hidden, block_intermediate),
                dtype=dtype,
            )
            out_local = T.alloc_fragment(
                (block_token, block_hidden),
                dtype=accum_dtype,
            )

            T.use_swizzle(10)

            block_start = bx * block_token
            expert_id = group_idx_for_bx[bx]
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(
                0,
                T.min(block_token, group_size - token_offset),
            )
            if compact_storage:
                data_start = raw_start + token_offset
            else:
                data_start = block_start

            T.clear(out_local)

            for k in T.Pipelined(
                T.ceildiv(intermediate, block_intermediate),
                num_stages=1,
            ):
                T.copy(
                    up_logits[
                        data_start : data_start + block_token,
                        k * block_intermediate : (k + 1) * block_intermediate,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * block_hidden : (by + 1) * block_hidden,
                        k * block_intermediate : (k + 1) * block_intermediate,
                    ],
                    down_shared,
                )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                )

            if compact_storage:
                for i, j in T.Parallel(block_token, block_hidden):
                    if i < actual_rows:
                        out[data_start + i, by * block_hidden + j] = (
                            out_local[i, j]
                            * routed_expert_weights[raw_start + token_offset + i]
                        )
            else:
                for i, j in T.Parallel(block_token, block_hidden):
                    if i < actual_rows:
                        out[data_start + i, by * block_hidden + j] = (
                            out_local[i, j]
                            * routed_expert_weights[raw_start + token_offset + i]
                        )
                    else:
                        out[data_start + i, by * block_hidden + j] = 0.0

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    storage_rows,
    route_rows,
    num_blocks_m,
    group_offsets_len,
    group_padded_offsets_len,
    compact_storage,
    route_weight_fp32,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(storage_rows),
        int(route_rows),
        int(num_blocks_m),
        int(group_offsets_len),
        int(group_padded_offsets_len),
        bool(compact_storage),
        bool(route_weight_fp32),
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
            key[8],
            key[9],
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
    storage_rows = int(stacked_expert_tokens.shape[0])
    route_rows = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])
    group_offsets_len = int(group_offsets.shape[0])
    group_padded_offsets_len = int(group_padded_offsets.shape[0])

    # The cited benchmark is compact and therefore has one input row per route.
    # The tutorial-padded ABI has more physical rows than compact route weights.
    compact_storage = storage_rows == route_rows
    route_weight_fp32 = routed_expert_weights.dtype == torch.float32

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        storage_rows,
        route_rows,
        num_blocks_m,
        group_offsets_len,
        group_padded_offsets_len,
        compact_storage,
        route_weight_fp32,
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
