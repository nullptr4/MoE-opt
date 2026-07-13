from __future__ import annotations

import pytest
import torch

from operator_runtime import reduce_sum, reduce_sum_
from operator_runtime_testing import assert_close, require_cuda
from tests.cases import reduce_sum as reduce_sum_cases


@pytest.mark.parametrize("case", reduce_sum_cases.correctness_cases(), ids=lambda c: c["name"])
def test_reduce_sum_correctness(case, backend):
    require_cuda()
    src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = reduce_sum(src, dim=1, backend=backend)
    assert_close(out, torch.sum(src, dim=1), atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", reduce_sum_cases.api_error_cases(), ids=lambda c: c["name"])
def test_reduce_sum_api_contract(case, backend):
    require_cuda()
    if case["name"] == "wrong_dim":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="dim=1"):
            reduce_sum(src, dim=case["dim"], backend=backend)
        return
    if case["name"] == "wrong_dtype":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(TypeError, match="float32"):
            reduce_sum(src, dim=case["dim"], backend=backend)
        return
    if case["name"] == "wrong_output_shape":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        out = torch.empty(case["out_shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="output shape"):
            reduce_sum_(out, src, dim=case["dim"], backend=backend)
        return
    if case["name"] == "non_contiguous":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"]).t()
        out = torch.empty((src.shape[0],), device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="contiguous"):
            reduce_sum_(out, src, dim=case["dim"], backend=backend)
        return
    if case["name"] == "wrong_rank":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="2D row-wise reduction"):
            reduce_sum(src, dim=case["dim"], backend=backend)
        return
    raise AssertionError(f"unhandled case: {case['name']}")
