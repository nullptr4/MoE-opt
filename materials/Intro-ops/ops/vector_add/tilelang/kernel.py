from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit
def vector_add_kernel(a, b, BLOCK_N: int, dtype):
    N = T.const("N")
    a: T.Tensor((N,), dtype)
    b: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    # TODO: implement a tile-wise vector add kernel.
    #
    # Suggested steps:
    # 1. Launch one TileLang kernel over the N // BLOCK_N tiles.
    # 2. Compute the tile base offset.
    # 3. Use T.Parallel(BLOCK_N) to fill out[base + i] = a[base + i] + b[base + i].

    return out
