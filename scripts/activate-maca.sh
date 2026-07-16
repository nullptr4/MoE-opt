#!/usr/bin/env bash
# Source this file before running TileLang race tests on the MetaX C500 host.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export METAX_COMPETITION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TILELANG_HOME="${TILELANG_HOME:-${METAX_COMPETITION_ROOT}/external/tilelang-metax}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export USE_MACA=ON

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# The editable install is the primary import path.  This keeps direct source
# invocations deterministic if the package is rebuilt or the editable install
# is temporarily removed.
export PYTHONPATH="${TILELANG_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
