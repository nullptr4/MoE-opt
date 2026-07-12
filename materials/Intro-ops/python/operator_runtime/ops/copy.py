from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import PreparedOp, bind_unary, tensor_view
from operator_runtime.ops._common import build_prepared_op


def _check(out: torch.Tensor, src: torch.Tensor) -> None:
    if not out.is_cuda or not src.is_cuda:
        raise ValueError("copy expects CUDA tensors")
    if out.shape != src.shape:
        raise ValueError("copy expects matching shapes")
    if out.dtype != src.dtype:
        raise TypeError("copy expects matching dtypes")
    if not out.is_contiguous() or not src.is_contiguous():
        raise ValueError("copy v1 supports contiguous tensors only")


def prepare_copy(out: torch.Tensor, src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, src)
    if backend is Backend.TILELANG:
        from ops.copy.tilelang.copy_tl import prepare_copy_tl

        return prepare_copy_tl(out, src)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_unary("copy", backend)
    out_view = tensor_view(out)
    src_view = tensor_view(src)
    create_args = (ctypes.byref(out_view), ctypes.byref(src_view))
    return build_prepared_op(funcs, create_args, (out, src), out)


def copy_(out: torch.Tensor, src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_copy(out, src, backend) as prepared:
        prepared.run()
    return out


def copy(src: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    out = torch.empty_like(src)
    return copy_(out, src, backend)
