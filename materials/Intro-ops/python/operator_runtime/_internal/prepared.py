from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import torch

from .bindings import CFunctions, Descriptor, check_status


@dataclass
class PreparedOp:
    funcs: CFunctions
    descriptor: Descriptor
    workspace: torch.Tensor | None
    runner_args: tuple[Any, ...]
    stream_tensor: torch.Tensor | None = None

    def run(self) -> None:
        self.run_inputs()

    def run_inputs(self, *runner_tensors: torch.Tensor) -> None:
        if runner_tensors:
            expected = len(self.runner_args) - 1
            if len(runner_tensors) != expected:
                raise ValueError(f"expected {expected} input tensors, got {len(runner_tensors)}")
            runner_args = (
                self.runner_args[0],
                *tuple(ctypes.c_void_p(tensor.data_ptr()) for tensor in runner_tensors),
            )
        else:
            runner_args = self.runner_args

        stream = torch.cuda.current_stream(device=self.stream_tensor.device if self.stream_tensor is not None else None)
        workspace_ptr = None if self.workspace is None else ctypes.c_void_p(self.workspace.data_ptr())
        workspace_size = 0 if self.workspace is None else self.workspace.numel()
        status = self.funcs.execute(
            self.descriptor,
            workspace_ptr,
            workspace_size,
            *runner_args,
            ctypes.c_void_p(stream.cuda_stream),
        )
        check_status(status)

    def run_kernel(self) -> None:
        self.run()

    def destroy(self) -> None:
        if self.descriptor:
            check_status(self.funcs.destroy(self.descriptor))
            self.descriptor = Descriptor()

    def __enter__(self) -> "PreparedOp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
