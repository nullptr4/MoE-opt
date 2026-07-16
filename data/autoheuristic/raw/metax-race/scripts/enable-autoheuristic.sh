#!/usr/bin/env bash
# Usage: source scripts/enable-autoheuristic.sh after source activate.sh.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source scripts/enable-autoheuristic.sh" >&2
  exit 2
fi
ROOT="${METAX_RACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export METAX_RACE_ROOT="$ROOT"
# shellcheck source=../config/autoheuristic.env
source "$ROOT/config/autoheuristic.env"
mkdir -p "$ROOT/logs"
echo "PyTorch AutoHeuristic enabled: collect=$TORCHINDUCTOR_AUTOHEURISTIC_COLLECT use=$TORCHINDUCTOR_AUTOHEURISTIC_USE"
echo "log=$TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH"

