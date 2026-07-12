from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from ops.common.tilelang.eager_copy import copy_eager, copy_eager_
from ops.common.tilelang.lazy_out_idx_copy import copy_lazy_out_idx, copy_lazy_out_idx_


def main() -> None:
    src = torch.randn((1024,), device="cuda", dtype=torch.float32)

    eager = copy_eager(src)
    torch.testing.assert_close(eager, src)

    eager_out = torch.empty_like(src)
    copy_eager_(eager_out, src)
    torch.testing.assert_close(eager_out, src)

    lazy = copy_lazy_out_idx(src)
    torch.testing.assert_close(lazy, src)

    lazy_out = torch.empty_like(src)
    copy_lazy_out_idx_(lazy_out, src)
    torch.testing.assert_close(lazy_out, src)

    print("tilelang eager and lazy out_idx copy ok")


if __name__ == "__main__":
    main()
