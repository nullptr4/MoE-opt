from __future__ import annotations

import ctypes

import torch

from operator_runtime.backend import Backend
from operator_runtime._internal import PreparedOp
from operator_runtime.ops._common import ElementwiseOpSpec, prepare_elementwise_op


_RELU_SPEC = ElementwiseOpSpec(
    name="relu",
    input_count=1,
    scalar_argtypes=(ctypes.c_float,),
)


def prepare_relu(
    out: torch.Tensor,
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> PreparedOp:
    return prepare_elementwise_op(
        _RELU_SPEC,
        out,
        (src,),
        (negative_slope,),
        backend,
        (float,),
    )


def relu_(
    out: torch.Tensor,
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> torch.Tensor:
    with prepare_relu(out, src, negative_slope, backend) as prepared:
        prepared.run()
    return out


def relu(
    src: torch.Tensor,
    negative_slope: float = 0.0,
    backend: str | Backend = Backend.NVIDIA,
) -> torch.Tensor:
    out = torch.empty_like(src)
    return relu_(out, src, negative_slope, backend)
