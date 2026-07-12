from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tilelang.language as T
import torch

from ops.vector_add.tilelang.kernel import vector_add_kernel


def _tl_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return T.float16
    if dtype is torch.float32:
        return T.float32
    raise TypeError(f"unsupported TileLang dtype: {dtype}")

def _block_n(n: int) -> int:
    return 1024 if n % 1024 == 0 else n


@lru_cache(maxsize=32)
def _compiled_vector_add(n: int, block_n: int, dtype: torch.dtype):
    return vector_add_kernel.compile(N=n, BLOCK_N=block_n, dtype=_tl_dtype(dtype))


@dataclass
class TileLangVectorAddPrepared:
    out: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    kernel: object

    def run_inputs(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) not in (0, 2):
            raise ValueError(f"expected 0 or 2 input tensors, got {len(inputs)}")
        a, b = (self.a, self.b) if not inputs else inputs
        return self.kernel(a, b)

    def run_kernel(self) -> torch.Tensor:
        return self.run_inputs()

    def run(self) -> None:
        self.out.copy_(self.run_inputs())

    def destroy(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def prepare_vector_add_tl(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> TileLangVectorAddPrepared:
    n = out.numel()
    block_n = _block_n(n)
    if n % block_n != 0:
        raise ValueError("TileLang vector_add v1 requires N % BLOCK_N == 0")
    kernel = _compiled_vector_add(n, block_n, out.dtype)
    return TileLangVectorAddPrepared(out, a, b, kernel)


def vector_add_tl_(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with prepare_vector_add_tl(out, a, b) as prepared:
        prepared.run()
    return out


def vector_add_tl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    return vector_add_tl_(out, a, b)
