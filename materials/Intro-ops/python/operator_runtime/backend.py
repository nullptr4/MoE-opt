from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    NVIDIA = "nvidia"
    TILELANG = "tilelang"
    METAX = "metax"


def normalize_backend(backend: str | Backend) -> Backend:
    if isinstance(backend, Backend):
        return backend
    try:
        return Backend(backend.lower())
    except ValueError as exc:
        raise ValueError(f"unknown backend: {backend}") from exc

