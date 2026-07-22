#!/usr/bin/env bash
# Run the standalone compact group_sum ten-argument OJ submission check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=activate-maca.sh
source "${ROOT}/scripts/activate-maca.sh"

python "${ROOT}/scripts/check_moe_submission.py" \
  "${ROOT}/benchmarks/tilelang-moe/submission.py"
python "${ROOT}/scripts/check_moe_sota_submission_sync.py" "${ROOT}"
python "${ROOT}/scripts/check_moe_remote_contract.py"

cd "${ROOT}/benchmarks/tilelang-moe"
exec python test_moe_submission.py "$@"
