from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceResult:
    operator: str
    backend: str
    shape: str
    dtype: str
    runtime_ms: float
    torch_ms: float | None = None

    @property
    def speedup(self) -> float | None:
        if self.torch_ms is None or self.runtime_ms <= 0:
            return None
        return self.torch_ms / self.runtime_ms
