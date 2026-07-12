#!/usr/bin/env bash
# Validate the task-1 TileLang/MACA toolchain without running the long benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

command -v mx-smi >/dev/null
command -v mxcc >/dev/null
[[ -d "${MACA_PATH}" ]]
[[ -c /dev/mxcd ]]

mx-smi
python - <<'PY'
import torch
import tilelang
from tvm.target import Target

assert torch.cuda.is_available(), "MetaX PyTorch CUDA-compatible device is unavailable"
target = Target("maca")
assert target.kind.name == "maca", target
print(f"Python: {__import__('sys').version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"TileLang: {tilelang.__version__}")
print(f"Target: {target}")
print(f"Device: {torch.cuda.get_device_name(0)}")
PY
