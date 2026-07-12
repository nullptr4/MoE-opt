from __future__ import annotations

import ctypes

import torch

OPRT_MAX_DIMS = 8

OPRT_DTYPE_F16 = 0
OPRT_DTYPE_F32 = 1


class TensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.c_int64 * OPRT_MAX_DIMS),
        ("strides", ctypes.c_int64 * OPRT_MAX_DIMS),
    ]


def dtype_to_oprt(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return OPRT_DTYPE_F16
    if dtype is torch.float32:
        return OPRT_DTYPE_F32
    raise TypeError(f"unsupported dtype: {dtype}")


def tensor_view(tensor: torch.Tensor) -> TensorView:
    if tensor.ndim > OPRT_MAX_DIMS:
        raise ValueError(f"ndim {tensor.ndim} exceeds OPRT_MAX_DIMS={OPRT_MAX_DIMS}")
    shape = (ctypes.c_int64 * OPRT_MAX_DIMS)()
    strides = (ctypes.c_int64 * OPRT_MAX_DIMS)()
    for i, dim in enumerate(tensor.shape):
        shape[i] = dim
    for i, stride in enumerate(tensor.stride()):
        strides[i] = stride
    return TensorView(
        ctypes.c_void_p(tensor.data_ptr()),
        dtype_to_oprt(tensor.dtype),
        tensor.ndim,
        shape,
        strides,
    )


def current_stream_ptr(tensor: torch.Tensor | None = None) -> ctypes.c_void_p:
    if not torch.cuda.is_available():
        return ctypes.c_void_p()
    device = tensor.device if tensor is not None and tensor.is_cuda else None
    stream = torch.cuda.current_stream(device=device)
    return ctypes.c_void_p(stream.cuda_stream)
