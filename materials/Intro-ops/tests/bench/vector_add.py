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
from tests.cases import vector_add as vector_add_cases


def bench_vector_add(backend: str) -> list[PerformanceResult]:
    rows: list[PerformanceResult] = []
    for case in vector_add_cases.benchmark_cases():
        a = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        from operator_runtime import prepare_vector_add
        with prepare_vector_add(out, a, b, backend=backend) as prepared:
            runtime = cuda_time_ms(prepared.run_inputs, args=(a, b))
        torch_ms = cuda_time_ms(lambda a, b: torch.add(a, b, out=out), args=(a, b))
        rows.append(
            PerformanceResult(
                "vector_add",
                backend,
                str(tuple(a.shape)),
                str(a.dtype),
                runtime,
                torch_ms,
            )
        )
    return rows
