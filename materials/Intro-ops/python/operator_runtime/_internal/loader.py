from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path


def _backend_key(backend: object | None) -> str | None:
    if backend is None:
        return None
    from operator_runtime.backend import normalize_backend

    return normalize_backend(backend).value


def _append_build_dir(candidates: list[Path], build_dir: str | None) -> None:
    if not build_dir:
        return
    root = Path(build_dir)
    candidates.extend([root / "libcamp_ops.so", root / "ops" / "libcamp_ops.so"])


def _candidate_library_paths(backend: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[3]

    if backend == "nvidia":
        _append_build_dir(candidates, os.environ.get("CAMP_NVIDIA_BUILD_DIR"))
        _append_build_dir(candidates, os.environ.get("CAMP_BUILD_DIR"))
        candidates.extend(
            [
                repo_root / "build" / "libcamp_ops.so",
                repo_root / "build" / "ops" / "libcamp_ops.so",
            ]
        )
    elif backend == "metax":
        _append_build_dir(candidates, os.environ.get("CAMP_METAX_BUILD_DIR"))
        _append_build_dir(candidates, os.environ.get("CAMP_BUILD_DIR"))
        candidates.extend(
            [
                repo_root / "build-metax" / "libcamp_ops.so",
                repo_root / "build-metax" / "ops" / "libcamp_ops.so",
            ]
        )
    else:
        _append_build_dir(candidates, os.environ.get("CAMP_BUILD_DIR"))
        _append_build_dir(candidates, os.environ.get("CAMP_NVIDIA_BUILD_DIR"))
        _append_build_dir(candidates, os.environ.get("CAMP_METAX_BUILD_DIR"))
        candidates.extend(
            [
                repo_root / "build" / "libcamp_ops.so",
                repo_root / "build" / "ops" / "libcamp_ops.so",
                repo_root / "build-metax" / "libcamp_ops.so",
                repo_root / "build-metax" / "ops" / "libcamp_ops.so",
            ]
        )
    return candidates


@lru_cache(maxsize=None)
def _load_library(backend: str | None) -> ctypes.CDLL:
    for path in _candidate_library_paths(backend):
        if path.exists():
            return ctypes.CDLL(str(path))
    searched = ", ".join(str(path) for path in _candidate_library_paths(backend))
    label = "default" if backend is None else backend
    raise FileNotFoundError(f"libcamp_ops.so for backend {label} not found; searched: {searched}")


def load_library(backend: object | None = None) -> ctypes.CDLL:
    return _load_library(_backend_key(backend))
