from __future__ import annotations

import torch


def correctness_cases():
    return [
        {"name": "relu_1k_fp32", "shape": (1024,), "dtype": torch.float32, "negative_slope": 0.0, "atol": 0, "rtol": 0},
        {"name": "relu_1k_fp16", "shape": (1024,), "dtype": torch.float16, "negative_slope": 0.0, "atol": 0, "rtol": 0},
        {"name": "leaky_relu_2d_fp32", "shape": (32, 64), "dtype": torch.float32, "negative_slope": 0.01, "atol": 1e-6, "rtol": 1e-6},
        {"name": "leaky_relu_2d_fp16", "shape": (32, 64), "dtype": torch.float16, "negative_slope": 0.01, "atol": 1e-3, "rtol": 1e-3},
        {"name": "non_contiguous_fp32", "shape": (16, 32), "dtype": torch.float32, "negative_slope": 0.01, "atol": 1e-6, "rtol": 1e-6},
        {"name": "broadcast_input_fp32", "src_shape": (64,), "out_shape": (32, 64), "dtype": torch.float32, "negative_slope": 0.01, "atol": 1e-6, "rtol": 1e-6},
    ]


def api_error_cases():
    return [
        {"name": "shape_mismatch", "shape": (16,), "out_shape": (8,), "dtype": torch.float32},
        {"name": "dtype_mismatch", "shape": (16,), "dtype": torch.float32, "out_dtype": torch.float16},
        {"name": "cpu_tensor", "shape": (16,), "dtype": torch.float32},
        {"name": "unsupported_dtype", "shape": (16,), "dtype": torch.float64},
        {"name": "broadcasted_output", "shape": (8, 16), "base_shape": (1, 16), "dtype": torch.float32},
    ]


def benchmark_cases():
    return [
        {"name": "contiguous_1001k_fp16", "shape": (1001 * 1024,), "dtype": torch.float16, "negative_slope": 0.0},
        {"name": "contiguous_4093k_fp16", "shape": (4093 * 1024,), "dtype": torch.float16, "negative_slope": 0.0},
        {"name": "contiguous_65521k_fp16", "shape": (65521 * 1024,), "dtype": torch.float16, "negative_slope": 0.0},
        {"name": "contiguous_1001k_fp32", "shape": (1001 * 1024,), "dtype": torch.float32, "negative_slope": 0.01},
        {"name": "contiguous_4093k_fp32", "shape": (4093 * 1024,), "dtype": torch.float32, "negative_slope": 0.01},
        {"name": "contiguous_65521k_fp32", "shape": (65521 * 1024,), "dtype": torch.float32, "negative_slope": 0.01},
    ]
