from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def softmax_kernel(src, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    src: T.Tensor((N, M), dtype)
    out = T.empty((N, M), dtype)

    # TODO: implement a tiled row-wise softmax kernel.
    #
    # Suggested steps:
    # 1. Launch one TileLang kernel over row tiles.
    # 2. Allocate fragments for src, out, temporary exp values, row max, row sum, and lse.
    # 3. Initialize the running log-sum-exp state.
    # 4. In a first T.Serial loop over column tiles:
    #    - copy the input tile into a fragment
    #    - reduce to get the tile max
    #    - compute exp2-based temporary values
    #    - reduce to get the tile sum
    #    - update the running lse
    # 5. In a second T.Serial loop over column tiles:
    #    - copy the input tile again
    #    - normalize with the final lse
    #    - copy the result tile to global memory

    return out
