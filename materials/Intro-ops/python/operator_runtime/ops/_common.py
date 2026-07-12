from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from operator_runtime.backend import Backend, normalize_backend
from operator_runtime._internal import CFunctions, Descriptor, PreparedOp, bind_elementwise, check_status, tensor_view


def _broadcastable_to(src_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> bool:
    if len(src_shape) > len(out_shape):
        return False
    offset = len(out_shape) - len(src_shape)
    for out_dim, out_extent in enumerate(out_shape):
        src_dim = out_dim - offset
        if src_dim < 0:
            continue
        src_extent = src_shape[src_dim]
        if src_extent != out_extent and src_extent != 1:
            return False
    return True


def _has_broadcast_dim(tensor: torch.Tensor) -> bool:
    return any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride()))


def build_prepared_op(
    funcs: CFunctions,
    create_args: tuple[object, ...],
    tensors: tuple[torch.Tensor, ...],
    stream_tensor: torch.Tensor,
) -> PreparedOp:
    desc = Descriptor()
    check_status(funcs.create(ctypes.byref(desc), *create_args))
    workspace_size = ctypes.c_size_t()
    check_status(funcs.workspace(desc, ctypes.byref(workspace_size)))
    workspace = None
    if workspace_size.value:
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=stream_tensor.device)
    runner_args = tuple(ctypes.c_void_p(tensor.data_ptr()) for tensor in tensors)
    return PreparedOp(funcs, desc, workspace, runner_args, stream_tensor)


@dataclass(frozen=True)
class ElementwiseOpSpec:
    name: str
    input_count: int
    scalar_argtypes: tuple[object, ...] = ()
    allow_broadcast: bool = True


def check_elementwise_tensors(spec: ElementwiseOpSpec, out: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
    if len(inputs) != spec.input_count:
        raise ValueError(f"{spec.name} expects {spec.input_count} inputs, got {len(inputs)}")
    if not out.is_cuda or any(not tensor.is_cuda for tensor in inputs):
        raise ValueError(f"{spec.name} expects CUDA tensors")
    if _has_broadcast_dim(out):
        raise ValueError(f"{spec.name} expects a writable output tensor")
    for tensor in inputs:
        if spec.allow_broadcast:
            if not _broadcastable_to(tuple(tensor.shape), tuple(out.shape)):
                raise ValueError(f"{spec.name} expects inputs to be broadcastable to out")
        elif out.shape != tensor.shape:
            raise ValueError(f"{spec.name} expects matching shapes")
        if out.dtype != tensor.dtype:
            raise TypeError(f"{spec.name} expects matching dtypes")


def prepare_elementwise_op(
    spec: ElementwiseOpSpec,
    out: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    scalars: Sequence[object] = (),
    backend: str | Backend = Backend.NVIDIA,
    scalar_converters: Sequence[Callable[[object], object]] = (),
) -> PreparedOp:
    backend = normalize_backend(backend)
    if backend not in (Backend.NVIDIA, Backend.METAX):
        raise NotImplementedError(f"backend {backend.value} is not runnable")

    check_elementwise_tensors(spec, out, inputs)
    if len(scalars) != len(spec.scalar_argtypes):
        raise ValueError(f"{spec.name} expects {len(spec.scalar_argtypes)} scalar args, got {len(scalars)}")
    if scalar_converters and len(scalar_converters) != len(scalars):
        raise ValueError("scalar_converters must match scalars")

    funcs = bind_elementwise(spec.name, spec.input_count, spec.scalar_argtypes, backend)
    views = [tensor_view(out), *(tensor_view(tensor) for tensor in inputs)]
    converters: Sequence[Callable[[object], object]] = scalar_converters or tuple(lambda value: value for _ in scalars)
    converted_scalars = [
        argtype(converter(value))
        for value, argtype, converter in zip(scalars, spec.scalar_argtypes, converters)
    ]
    create_args = (*[ctypes.byref(view) for view in views], *converted_scalars)
    return build_prepared_op(funcs, create_args, (out, *inputs), out)
