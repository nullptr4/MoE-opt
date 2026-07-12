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
from tests.cases import relu as relu_cases


def bench_relu(backend: str) -> list[PerformanceResult]:
    if backend not in ("nvidia", "metax"):
        raise NotImplementedError("relu benchmark currently targets the C ABI elementwise framework")

    rows: list[PerformanceResult] = []
    for case in relu_cases.benchmark_cases():
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        out = torch.empty_like(src)
        negative_slope = case["negative_slope"]
        from operator_runtime import prepare_relu
        with prepare_relu(out, src, negative_slope=negative_slope, backend=backend) as prepared:
            runtime = cuda_time_ms(prepared.run_inputs, args=(src,))
        torch_ms = cuda_time_ms(
            lambda src: torch.where(src > 0, src, src * negative_slope, out=out),
            args=(src,),
        )
        rows.append(
            PerformanceResult(
                "relu",
                backend,
                str(tuple(src.shape)),
                str(src.dtype),
                runtime,
                torch_ms,
            )
        )
    return rows
