from __future__ import annotations

import torch


def correctness_cases():
    return [
        {"name": "contiguous_1k_fp32", "shape": (1024,), "dtype": torch.float32, "atol": 0, "rtol": 0},
        {"name": "contiguous_1k_fp16", "shape": (1024,), "dtype": torch.float16, "atol": 0, "rtol": 0},
        {"name": "contiguous_1536_fp32", "shape": (1536,), "dtype": torch.float32, "atol": 0, "rtol": 0},
        {"name": "contiguous_1536_fp16", "shape": (1536,), "dtype": torch.float16, "atol": 0, "rtol": 0},
    ]


def api_error_cases():
    return [
        {"name": "shape_mismatch", "shape": (16,), "out_shape": (8,), "dtype": torch.float32},
        {"name": "dtype_mismatch", "shape": (16,), "dtype": torch.float32, "out_dtype": torch.float16},
        {"name": "non_contiguous", "shape": (4, 4), "dtype": torch.float32},
    ]


def benchmark_cases():
    return [
        {"name": "contiguous_1001k_fp16", "shape": (1001 * 1024,), "dtype": torch.float16},
        {"name": "contiguous_4093k_fp16", "shape": (4093 * 1024,), "dtype": torch.float16},
        {"name": "contiguous_65521k_fp16", "shape": (65521 * 1024,), "dtype": torch.float16},
        {"name": "contiguous_1001k_fp32", "shape": (1001 * 1024,), "dtype": torch.float32},
        {"name": "contiguous_4093k_fp32", "shape": (4093 * 1024,), "dtype": torch.float32},
        {"name": "contiguous_65521k_fp32", "shape": (65521 * 1024,), "dtype": torch.float32},
    ]
