from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import PreparedOp, bind_binary, tensor_view
from operator_runtime.ops._common import build_prepared_op


def _check(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    if not out.is_cuda or not a.is_cuda or not b.is_cuda:
        raise ValueError("vector_add expects CUDA tensors")
    if out.shape != a.shape or out.shape != b.shape:
        raise ValueError("vector_add v1 expects matching shapes")
    if out.dtype != a.dtype or out.dtype != b.dtype:
        raise TypeError("vector_add expects matching dtypes")
    if not out.is_contiguous() or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("vector_add v1 supports contiguous tensors only")


def prepare_vector_add(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    backend = normalize_backend(backend)
    _check(out, a, b)
    if backend is Backend.TILELANG:
        from ops.vector_add.tilelang.vector_add_tl import prepare_vector_add_tl

        return prepare_vector_add_tl(out, a, b)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    funcs = bind_binary("vector_add", backend)
    out_view = tensor_view(out)
    a_view = tensor_view(a)
    b_view = tensor_view(b)
    create_args = (ctypes.byref(out_view), ctypes.byref(a_view), ctypes.byref(b_view))
    return build_prepared_op(funcs, create_args, (out, a, b), out)


def vector_add_(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    with prepare_vector_add(out, a, b, backend) as prepared:
        prepared.run()
    return out


def vector_add(a: torch.Tensor, b: torch.Tensor, backend: str | Backend = Backend.NVIDIA) -> torch.Tensor:
    out = torch.empty_like(a)
    return vector_add_(out, a, b, backend)
