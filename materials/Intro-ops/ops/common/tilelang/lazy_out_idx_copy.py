from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


def _tl_dtype_str(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"unsupported TileLang dtype: {dtype}")


@tilelang.jit(out_idx=[1])
def _copy_lazy_out_idx_kernel(n: int, block_n: int, dtype: str):
    @T.prim_func
    def main(src: T.Tensor((n,), dtype), out: T.Tensor((n,), dtype)):
        with T.Kernel(n // block_n, threads=256) as pid_n:
            T.copy(
                src[pid_n * block_n : (pid_n + 1) * block_n],
                out[pid_n * block_n : (pid_n + 1) * block_n],
            )

    return main


def _block_n(n: int) -> int:
    return 1024 if n % 1024 == 0 else n


@lru_cache(maxsize=32)
def _compiled_copy_lazy_out_idx(n: int, block_n: int, dtype: torch.dtype):
    return _copy_lazy_out_idx_kernel(n, block_n, _tl_dtype_str(dtype))


@dataclass
class LazyOutIdxCopyPrepared:
    out: torch.Tensor
    src: torch.Tensor
    kernel: object

    def run(self) -> None:
        # out_idx makes TileLang allocate and return the output tensor. The
        # training runtime keeps an out-variant API, so this adapter copies the
        # result into the caller-owned output buffer.
        self.out.copy_(self.kernel(self.src))

    def destroy(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def prepare_copy_lazy_out_idx(out: torch.Tensor, src: torch.Tensor) -> LazyOutIdxCopyPrepared:
    if out.shape != src.shape:
        raise ValueError("lazy out_idx copy expects matching shapes")
    if out.dtype != src.dtype:
        raise TypeError("lazy out_idx copy expects matching dtypes")
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("lazy out_idx copy expects CUDA tensors")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("lazy out_idx copy v1 supports contiguous tensors only")

    n = src.numel()
    block_n = _block_n(n)
    if n % block_n != 0:
        raise ValueError("lazy out_idx copy v1 requires N % BLOCK_N == 0")
    kernel = _compiled_copy_lazy_out_idx(n, block_n, src.dtype)
    return LazyOutIdxCopyPrepared(out, src, kernel)


def copy_lazy_out_idx_(out: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    with prepare_copy_lazy_out_idx(out, src) as prepared:
        prepared.run()
    return out


def copy_lazy_out_idx(src: torch.Tensor) -> torch.Tensor:
    n = src.numel()
    block_n = _block_n(n)
    if n % block_n != 0:
        raise ValueError("lazy out_idx copy v1 requires N % BLOCK_N == 0")
    kernel = _compiled_copy_lazy_out_idx(n, block_n, src.dtype)
    return kernel(src)
