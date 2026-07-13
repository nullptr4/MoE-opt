from __future__ import annotations

import ctypes

import pytest
import torch

from operator_runtime import relu, relu_
from operator_runtime.ops.relu import prepare_relu
from operator_runtime_testing import assert_close, require_cuda
from tests.cases import relu as relu_cases


def _reference(src: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return torch.where(src > 0, src, src * negative_slope)


@pytest.mark.parametrize("case", relu_cases.correctness_cases(), ids=lambda c: c["name"])
def test_relu_correctness(case, backend):
    require_cuda()
    if backend not in ("nvidia", "metax"):
        pytest.skip("relu currently targets the C ABI elementwise framework")

    if case["name"] == "broadcast_input_fp32":
        src = torch.randn(case["src_shape"], dtype=case["dtype"], device="cuda")
        out = torch.empty(case["out_shape"], dtype=case["dtype"], device="cuda")
        relu_(out, src, negative_slope=case["negative_slope"], backend=backend)
        ref_src = src.expand_as(out)
    elif case["name"] == "non_contiguous_fp32":
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda").t()
        out = torch.empty_strided(src.shape, src.stride(), dtype=case["dtype"], device="cuda")
        relu_(out, src, negative_slope=case["negative_slope"], backend=backend)
        ref_src = src
    else:
        src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
        out = relu(src, negative_slope=case["negative_slope"], backend=backend)
        ref_src = src

    assert_close(out, _reference(ref_src, case["negative_slope"]), atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", relu_cases.api_error_cases(), ids=lambda c: c["name"])
def test_relu_api_contract(case, backend):
    require_cuda()
    if backend not in ("nvidia", "metax"):
        pytest.skip("relu currently targets the C ABI elementwise framework")

    if case["name"] == "shape_mismatch":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        out = torch.empty(case["out_shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="broadcastable"):
            relu_(out, src, backend=backend)
        return
    if case["name"] == "dtype_mismatch":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        out = torch.empty(case["shape"], device="cuda", dtype=case["out_dtype"])
        with pytest.raises(TypeError, match="matching dtypes"):
            relu_(out, src, backend=backend)
        return
    if case["name"] == "cpu_tensor":
        src = torch.randn(case["shape"], dtype=case["dtype"])
        out = torch.empty_like(src)
        with pytest.raises(ValueError, match="CUDA tensors"):
            relu_(out, src, backend=backend)
        return
    if case["name"] == "unsupported_dtype":
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(TypeError, match="unsupported dtype"):
            relu(src, backend=backend)
        return
    if case["name"] == "broadcasted_output":
        base = torch.empty(case["base_shape"], device="cuda", dtype=case["dtype"])
        out = base.expand(case["shape"])
        src = torch.randn(case["shape"], device="cuda", dtype=case["dtype"])
        with pytest.raises(ValueError, match="writable output"):
            relu_(out, src, backend=backend)
        return
    raise AssertionError(f"unhandled case: {case['name']}")


def test_prepared_relu_reuses_descriptor_with_new_inputs(backend):
    require_cuda()
    if backend not in ("nvidia", "metax"):
        pytest.skip("descriptor lifecycle test targets C ABI backend")

    src = torch.randn((1024,), device="cuda", dtype=torch.float32)
    out = torch.empty_like(src)
    prepared = prepare_relu(out, src, negative_slope=0.01, backend=backend)
    try:
        prepared.run()
        assert_close(out, _reference(src, 0.01), atol=1e-6, rtol=1e-6)

        next_src = torch.randn_like(src)
        prepared.run_inputs(next_src)
        assert_close(out, _reference(next_src, 0.01), atol=1e-6, rtol=1e-6)
    finally:
        prepared.destroy()


def test_prepared_relu_reports_insufficient_workspace(backend):
    require_cuda()
    if backend not in ("nvidia", "metax"):
        pytest.skip("workspace contract test targets C ABI backend")

    src = torch.randn((16,), device="cuda", dtype=torch.float32)
    out = torch.empty_like(src)
    prepared = prepare_relu(out, src, backend=backend)
    try:
        original_workspace = prepared.workspace
        prepared.workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError, match="insufficient workspace"):
            prepared.run()
        prepared.workspace = original_workspace
    finally:
        prepared.destroy()
