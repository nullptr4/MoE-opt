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
from tests.cases import reduce_sum as reduce_sum_cases


def bench_reduce_sum(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in reduce_sum_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        from operator_runtime import prepare_reduce_sum
        out = torch.empty(src.shape[0], dtype=src.dtype, device="cuda")
        with prepare_reduce_sum(out, src, dim=1, backend=backend) as prepared:
            runtime = cuda_time_ms(prepared.run_inputs, args=(src,))
        torch_ms = cuda_time_ms(lambda src: torch.sum(src, dim=1, out=out), args=(src,))
        rows.append(
            PerformanceResult(
                "reduce_sum",
                backend,
                str(tuple(src.shape)),
                str(src.dtype),
                runtime,
                torch_ms,
            )
        )
    return rows
