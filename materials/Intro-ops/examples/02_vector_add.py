from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from operator_runtime import vector_add


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nvidia", choices=["nvidia", "tilelang", "metax"])
    args = parser.parse_args()

    a = torch.randn((1024,), device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    out = vector_add(a, b, backend=args.backend)
    torch.testing.assert_close(out, a + b)
    print(f"vector_add ok ({args.backend})")


if __name__ == "__main__":
    main()
