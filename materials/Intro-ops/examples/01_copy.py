from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia", choices=["nvidia", "tilelang", "metax"])
    args = parser.parse_args()

    src = torch.randn((1024,), device="cuda", dtype=torch.float32)
    out = copy(src, backend=args.backend)
    torch.testing.assert_close(out, src)
    print(f"copy ok ({args.backend})")


if __name__ == "__main__":
    main()
