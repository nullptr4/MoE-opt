from __future__ import annotations

import pytest
import torch

from operator_runtime import copy, copy_
from operator_runtime_testing import assert_close, require_cuda
from tests.cases import copy as copy_cases


@pytest.mark.parametrize("case", copy_cases.correctness_cases(), ids=lambda c: c["name"])
def test_copy_correctness(case, backend):
    require_cuda()
    src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = copy(src, backend=backend)
    assert_close(out, src, atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", copy_cases.api_error_cases(), ids=lambda c: c["name"])
def test_copy_api_contract(case, backend):
    require_cuda()
    if case["name"] == "shape_mismatch":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        out = torch.empty(case["out_shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="matching shapes"):
            copy_(out, src, backend=backend)
        return
    if case["name"] == "dtype_mismatch":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        out = torch.empty(case["shape"], device="cuda", dtype=case["out_dtype"])
        with pytest.raises(TypeError, match="matching dtypes"):
            copy_(out, src, backend=backend)
        return
    if case["name"] == "non_contiguous":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"]).t()
        out = torch.empty(case["shape"], device="cuda", dtype=case["dtype"]).t()
        with pytest.raises(ValueError, match="contiguous"):
            copy_(out, src, backend=backend)
        return
    raise AssertionError(f"unhandled case: {case['name']}")
