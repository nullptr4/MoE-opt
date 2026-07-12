from __future__ import annotations

import pytest
import torch

from operator_runtime import vector_add, vector_add_
from operator_runtime.ops.vector_add import prepare_vector_add
from operator_runtime_testing import assert_close, require_cuda
from tests.cases import vector_add as vector_add_cases


@pytest.mark.parametrize("case", vector_add_cases.correctness_cases(), ids=lambda c: c["name"])
def test_vector_add_correctness(case, backend):
    require_cuda()
    a = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    b = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = vector_add(a, b, backend=backend)
    assert_close(out, a + b, atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", vector_add_cases.api_error_cases(), ids=lambda c: c["name"])
def test_vector_add_api_contract(case, backend):
    require_cuda()
    if case["name"] == "shape_mismatch":
        a = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        b = torch.randn(case["other_shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="matching shapes"):
            vector_add(a, b, backend=backend)
        return
    if case["name"] == "dtype_mismatch":
        a = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        b = torch.randn(case["shape"], device="cuda", dtype=case["other_dtype"])
        with pytest.raises(TypeError, match="matching dtypes"):
            vector_add(a, b, backend=backend)
        return
    if case["name"] == "non_contiguous":
        a = torch.randn(case["shape"], device="cuda", dtype=case["dtype"]).t()
        b = torch.randn(case["shape"], device="cuda", dtype=case["dtype"]).t()
        out = torch.empty(case["shape"], device="cuda", dtype=case["dtype"]).t()
        with pytest.raises(ValueError, match="contiguous"):
            vector_add_(out, a, b, backend=backend)
        return
    raise AssertionError(f"unhandled case: {case['name']}")


def test_prepared_vector_add_reuses_descriptor(backend):
    require_cuda()
    if backend == "tilelang":
        pytest.skip("descriptor lifecycle test targets C ABI backend")
    a = torch.randn((1024,), device="cuda")
    b = torch.randn((1024,), device="cuda")
    out = torch.empty_like(a)
    prepared = prepare_vector_add(out, a, b, backend=backend)
    try:
        prepared.run()
        prepared.run()
        assert_close(out, a + b, atol=1e-6, rtol=1e-6)
    finally:
        prepared.destroy()
