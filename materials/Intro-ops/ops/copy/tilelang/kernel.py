from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit
def copy_kernel(src, BLOCK_N: int, dtype):
    N = T.const("N")
    src: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    # TODO: implement a tile-wise copy kernel.
    #
    # Suggested steps:
    # 1. Launch one TileLang kernel over the N // BLOCK_N tiles.
    # 2. Use T.copy to move one tile from src to out.

    return out
