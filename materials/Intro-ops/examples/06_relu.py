from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import relu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia", choices=["nvidia"])
    parser.add_argument("--negative-slope", type=float, default=0.0)
    args = parser.parse_args()

    src = torch.randn((1024,), device="cuda", dtype=torch.float32)
    out = relu(src, negative_slope=args.negative_slope, backend=args.backend)
    expected = torch.where(src > 0, src, src * args.negative_slope)
    torch.testing.assert_close(out, expected)
    print(f"relu ok ({args.backend}, negative_slope={args.negative_slope})")


if __name__ == "__main__":
    main()
