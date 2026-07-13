from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable, Sequence

from operator_runtime.backend import Backend
from operator_runtime.backend import normalize_backend

from .loader import load_library
from .tensor_view import TensorView

Status = ctypes.c_int
Descriptor = ctypes.c_void_p
OPRT_BACKEND_NVIDIA = 1
OPRT_BACKEND_METAX = 2


class OperatorRuntimeError(RuntimeError):
    pass


def _backend_code(backend: str | Backend) -> int:
    normalized = normalize_backend(backend)
    if normalized is Backend.NVIDIA:
        return OPRT_BACKEND_NVIDIA
    if normalized is Backend.METAX:
        return OPRT_BACKEND_METAX
    raise NotImplementedError(f"backend {normalized.value} is not runnable through the C ABI")


def check_status(status: int, lib=None) -> None:
    if status == 0:
        return
    if lib is None:
        lib = load_library()
    lib.oprt_status_string.argtypes = [ctypes.c_int]
    lib.oprt_status_string.restype = ctypes.c_char_p
    message = lib.oprt_status_string(status).decode("utf-8")
    raise OperatorRuntimeError(message)


def set_runtime_backend(lib, backend: str | Backend) -> None:
    lib.oprt_set_backend.argtypes = [ctypes.c_int]
    lib.oprt_set_backend.restype = Status
    check_status(lib.oprt_set_backend(_backend_code(backend)), lib)


@dataclass(frozen=True)
class CFunctions:
    create: Callable
    workspace: Callable
    execute: Callable
    destroy: Callable


def _missing_symbol_error(name: str, symbol: str, exc: AttributeError) -> OperatorRuntimeError:
    raise OperatorRuntimeError(
        f"missing symbol {symbol} for operator {name}"
    ) from exc


def _bind_lifecycle(
    name: str,
    backend: str | Backend,
    create_argtypes: Sequence[object],
    execute_argtypes: Sequence[object],
) -> CFunctions:
    lib = load_library(backend)
    set_runtime_backend(lib, backend)
    create_symbol = f"oprt_create_{name}_descriptor"
    try:
        create = getattr(lib, create_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, create_symbol, exc)
    create.argtypes = [ctypes.POINTER(Descriptor), *create_argtypes]
    create.restype = Status

    workspace_symbol = f"oprt_get_{name}_workspace_size"
    try:
        workspace = getattr(lib, workspace_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, workspace_symbol, exc)
    workspace.argtypes = [Descriptor, ctypes.POINTER(ctypes.c_size_t)]
    workspace.restype = Status

    execute_symbol = f"oprt_execute_{name}"
    try:
        execute = getattr(lib, execute_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, execute_symbol, exc)
    execute.argtypes = [Descriptor, ctypes.c_void_p, ctypes.c_size_t, *execute_argtypes, ctypes.c_void_p]
    execute.restype = Status

    destroy_symbol = f"oprt_destroy_{name}_descriptor"
    try:
        destroy = getattr(lib, destroy_symbol)
    except AttributeError as exc:
        _missing_symbol_error(name, destroy_symbol, exc)
    destroy.argtypes = [Descriptor]
    destroy.restype = Status
    return CFunctions(create, workspace, execute, destroy)


def bind_elementwise(
    name: str,
    input_count: int,
    scalar_argtypes: Sequence[object] = (),
    backend: str | Backend = Backend.NVIDIA,
) -> CFunctions:
    if input_count <= 0:
        raise ValueError("elementwise operators require at least one input")
    tensor_views = [ctypes.POINTER(TensorView)] * (input_count + 1)
    data_ptrs = [ctypes.c_void_p] * (input_count + 1)
    return _bind_lifecycle(name, backend, [*tensor_views, *scalar_argtypes], data_ptrs)


def bind_unary(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    return bind_elementwise(name, 1, backend=backend)


def bind_relu(backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    return bind_elementwise("relu", 1, [ctypes.c_float], backend)


def bind_binary(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    return bind_elementwise(name, 2, backend=backend)


def bind_reduce_like(name: str, backend: str | Backend = Backend.NVIDIA) -> CFunctions:
    return _bind_lifecycle(
        name,
        backend,
        [
            ctypes.POINTER(TensorView),
            ctypes.POINTER(TensorView),
            ctypes.c_int64,
        ],
        [ctypes.c_void_p, ctypes.c_void_p],
    )
