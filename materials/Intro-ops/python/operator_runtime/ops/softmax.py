from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import PreparedOp, bind_reduce_like, tensor_view
from operator_runtime.ops._common import build_prepared_op


def _check(out: torch.Tensor, src: torch.Tensor, dim: int) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("softmax expects CUDA tensors")
    if src.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise TypeError("softmax v1 supports float32 only")
    if src.ndim != 2 or out.ndim != 2 or dim != 1:
        raise ValueError("softmax v1 supports 2D row-wise dim=1")
    if out.shape != src.shape:
        raise ValueError("softmax output shape must match input")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("softmax v1 supports contiguous tensors only")


def prepare_softmax(
    out: torch.Tensor,
    src: torch.Tensor,
    dim: int = 1,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src, dim)
    if backend is Backend.TILELANG:
        from ops.softmax.tilelang.softmax_tl import prepare_softmax_tl

        return prepare_softmax_tl(out, src, dim=dim)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_reduce_like("softmax", backend)
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    create_args = (ctypes.byref(out_view), ctypes.byref(src_view), ctypes.c_int64(dim))
    return build_prepared_op(funcs, create_args, (out, src), out)


def softmax_(out: torch.Tensor, src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_softmax(out, src, dim, backend) as prepared:
        prepared.run()
    return out


def softmax(src: torch.Tensor, dim: int = 1, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    out = torch.empty_like(src)
    return softmax_(out, src, dim, backend)
