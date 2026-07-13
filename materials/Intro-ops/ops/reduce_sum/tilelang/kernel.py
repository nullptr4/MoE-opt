from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def reduce_sum_kernel(src, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float32
    src: T.Tensor((N, M), dtype)
    out = T.empty((N,), dtype)

    # TODO: implement a tiled row-wise reduce_sum kernel.
    #
    # Suggested steps:
    # 1. Launch one TileLang kernel over row tiles.
    # 2. Allocate fragments for the input tile and running output tile.
    # 3. Clear the running output fragment before accumulation.
    # 4. Iterate over column tiles with T.Serial(M // BLOCK_M).
    # 5. T.copy each input tile into the fragment.
    # 6. Call T.reduce_sum on the fragment and accumulate into the output fragment.
    # 7. Copy the final output fragment back to global memory.

    return out
