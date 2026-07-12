from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from ops.softmax.tilelang.kernel import softmax_kernel

@lru_cache(maxsize=32)
def _compiled_softmax(n: int, m: int, block_n: int, block_m: int):
    return softmax_kernel.compile(N=n, M=m, BLOCK_N=block_n, BLOCK_M=block_m)


def _blocks(n: int, m: int) -> tuple[int, int]:
    block_n = 16 if n % 16 == 0 else 1
    block_m = 256 if m % 256 == 0 else m
    return block_n, block_m


@dataclass
class TileLangSoftmaxPrepared:
    out: torch.Tensor
    src: torch.Tensor
    dim: int = 1
    kernel: object | None = None

    def run_inputs(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) not in (0, 1):
            raise ValueError(f"expected 0 or 1 input tensors, got {len(inputs)}")
        src = self.src if not inputs else inputs[0]
        assert self.kernel is not None
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


def prepare_softmax_tl(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> TileLangSoftmaxPrepared:
    if dim != 1:
        raise ValueError("TileLang softmax v1 supports dim=1")
    n, m = src.shape
    block_n, block_m = _blocks(n, m)
    if n % block_n != 0 or m % block_m != 0:
        raise ValueError("TileLang softmax v1 requires divisible block sizes")
    kernel = _compiled_softmax(n, m, block_n, block_m)
    return TileLangSoftmaxPrepared(out, src, dim, kernel)


def softmax_tl_(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    with prepare_softmax_tl(out, src, dim) as prepared:
        prepared.run()
    return out


def softmax_tl(src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    out = torch.empty_like(src)
    return softmax_tl_(out, src, dim)
