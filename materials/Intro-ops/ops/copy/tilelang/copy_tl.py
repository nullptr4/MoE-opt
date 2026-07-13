from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

import tilelang.language as T
import torch

from ops.copy.tilelang.kernel import copy_kernel


def _tl_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return T.float16
    if dtype is torch.float32:
        return T.float32
    raise TypeError(f"unsupported TileLang dtype: {dtype}")

def _block_n(n: int) -> int:
    return 1024 if n % 1024 == 0 else n


@lru_cache(maxsize=32)
def _compiled_copy(n: int, block_n: int, dtype: torch.dtype):
    return copy_kernel.compile(N=n, BLOCK_N=block_n, dtype=_tl_dtype(dtype))


@dataclass
class TileLangCopyPrepared:
    out: torch.Tensor
    src: torch.Tensor
    kernel: object

    def run_inputs(self, *inputs: torch.Tensor) -> torch.Tensor:
        src = self.src if not inputs else inputs[0]
        if len(inputs) not in (0, 1):
            raise ValueError(f"expected 0 or 1 input tensors, got {len(inputs)}")
        return self.kernel(src)

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


def prepare_copy_tl(out: torch.Tensor, src: torch.Tensor) -> TileLangCopyPrepared:
    n = src.numel()
    block_n = _block_n(n)
    if n % block_n != 0:
        raise ValueError("TileLang copy v1 requires N % BLOCK_N == 0")
    kernel = _compiled_copy(n, block_n, src.dtype)
    return TileLangCopyPrepared(out, src, kernel)


def copy_tl_(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    with prepare_copy_tl(out, src) as prepared:
        prepared.run()
    return out


def copy_tl(src: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(src)
    return copy_tl_(out, src)
