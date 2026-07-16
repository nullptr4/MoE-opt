#!/usr/bin/env bash
# Pull code/data, fingerprint this host, rebuild the shared heuristic index,
# and push only synchronized experiment data. Local caches are never staged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_ID="${MOE_HOST_ID:-c500-unknown}"
REMOTE="${MOE_SYNC_REMOTE:-origin}"
BRANCH="${MOE_SYNC_BRANCH:-$(git -C "${ROOT}" branch --show-current)}"

git -C "${ROOT}" pull --rebase --autostash "${REMOTE}" "${BRANCH}"
python "${ROOT}/scripts/collect_host_profile.py" --host-id "${HOST_ID}"
python "${ROOT}/scripts/merge_tuning_results.py"

git -C "${ROOT}" add \
  .gitattributes \
  data \
  reports \
  logs

if git -C "${ROOT}" diff --cached --quiet; then
  echo "no synchronized data changes"
  exit 0
fi

git -C "${ROOT}" commit -m "data(${HOST_ID}): sync tuning and profiler results"
git -C "${ROOT}" push "${REMOTE}" "HEAD:${BRANCH}"
