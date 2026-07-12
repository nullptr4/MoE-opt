from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


def _tl_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return T.float16
    if dtype is torch.float32:
        return T.float32
    raise TypeError(f"unsupported TileLang dtype: {dtype}")


@tilelang.jit
def _copy_eager_kernel(src, BLOCK_N: int, dtype):
    N = T.const("N")
    src: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        T.copy(
            src[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
            out[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )

    return out


def _block_n(n: int) -> int:
    return 1024 if n % 1024 == 0 else n


@lru_cache(maxsize=32)
def _compiled_copy_eager(n: int, block_n: int, dtype: torch.dtype):
    return _copy_eager_kernel.compile(N=n, BLOCK_N=block_n, dtype=_tl_dtype(dtype))


@dataclass
class EagerCopyPrepared:
    out: torch.Tensor
    src: torch.Tensor
    kernel: object

    def run(self) -> None:
        self.out.copy_(self.kernel(self.src))

    def destroy(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def prepare_copy_eager(out: torch.Tensor, src: torch.Tensor) -> EagerCopyPrepared:
    if out.shape != src.shape:
        raise ValueError("eager copy expects matching shapes")
    if out.dtype != src.dtype:
        raise TypeError("eager copy expects matching dtypes")
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("eager copy expects CUDA tensors")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("eager copy v1 supports contiguous tensors only")

    n = src.numel()
    block_n = _block_n(n)
    if n % block_n != 0:
        raise ValueError("eager copy v1 requires N % BLOCK_N == 0")
    kernel = _compiled_copy_eager(n, block_n, src.dtype)
    return EagerCopyPrepared(out, src, kernel)


def copy_eager_(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    with prepare_copy_eager(out, src) as prepared:
        prepared.run()
    return out


def copy_eager(src: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(src)
    return copy_eager_(out, src)
