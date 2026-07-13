from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import PreparedOp, bind_reduce_like, tensor_view
from operator_runtime.ops._common import build_prepared_op


def _check(out: torch.Tensor, src: torch.Tensor, dim: int) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("reduce_sum expects CUDA tensors")
    if src.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise TypeError("reduce_sum v1 supports float32 only")
    if src.ndim != 2 or out.ndim != 1 or dim != 1:
        raise ValueError("reduce_sum v1 supports 2D row-wise reduction over dim=1")
    if out.shape[0] != src.shape[0]:
        raise ValueError("reduce_sum output shape must be [src.shape[0]]")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("reduce_sum v1 supports contiguous tensors only")


def prepare_reduce_sum(
    out: torch.Tensor,
    src: torch.Tensor,
    dim: int = 1,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src, dim)
    if backend is Backend.TILELANG:
        from ops.reduce_sum.tilelang.reduce_sum_tl import prepare_reduce_sum_tl

        return prepare_reduce_sum_tl(out, src, dim=dim)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_reduce_like("reduce_sum", backend)
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    create_args = (ctypes.byref(out_view), ctypes.byref(src_view), ctypes.c_int64(dim))
    return build_prepared_op(funcs, create_args, (out, src), out)


def reduce_sum_(out: torch.Tensor, src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_reduce_sum(out, src, dim, backend) as prepared:
        prepared.run()
    return out


def reduce_sum(src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    if dim != 1 or src.ndim != 2:
        raise ValueError("reduce_sum v1 supports 2D row-wise reduction over dim=1")
    out = torch.empty((src.shape[0],), dtype=src.dtype, device=src.device)
    return reduce_sum_(out, src, dim, backend)
