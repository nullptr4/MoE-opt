from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from operator_runtime_testing import cuda_time_ms, PerformanceResult
from tests.cases import softmax as softmax_cases


def bench_softmax(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in softmax_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        out = torch.empty_like(src)
        from operator_runtime import prepare_softmax
        with prepare_softmax(out, src, dim=1, backend=backend) as prepared:
            runtime = cuda_time_ms(prepared.run_inputs, args=(src,))
        torch_ms = cuda_time_ms(lambda src: torch.softmax(src, dim=1, out=out), args=(src,))
        rows.append(
            PerformanceResult(
                "softmax",
                backend,
                str(tuple(src.shape)),
                str(src.dtype),
                runtime,
                torch_ms,
            )
        )
    return rows
