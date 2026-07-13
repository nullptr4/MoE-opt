from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import reduce_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia", choices=["nvidia", "tilelang", "metax"])
    args = parser.parse_args()

    src = torch.randn((32, 128), device="cuda", dtype=torch.float32)
    out = reduce_sum(src, dim=1, backend=args.backend)
    torch.testing.assert_close(out, torch.sum(src, dim=1), atol=1e-5, rtol=1e-5)
    print(f"reduce_sum ok ({args.backend})")


if __name__ == "__main__":
    main()
