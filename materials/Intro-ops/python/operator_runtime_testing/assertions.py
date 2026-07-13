from __future__ import annotations

import pytest
import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
