#!/usr/bin/env bash
# Source this file before running TileLang race tests on the MetaX C500 host.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export METAX_COMPETITION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TILELANG_HOME="${TILELANG_HOME:-/opt/tilelang-metax}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${METAX_COMPETITION_ROOT}/.cache/tilelang}"
export MOE_HOST_ID="${MOE_HOST_ID:-c500-unknown}"
export MOE_DATA_ROOT="${MOE_DATA_ROOT:-${METAX_COMPETITION_ROOT}/data}"
export MOE_PROFILER_ROOT="${MOE_PROFILER_ROOT:-${MOE_DATA_ROOT}/profiler}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export USE_MACA=ON

# MoE tuning defaults.  The shape heuristic narrows TileLang's candidate
# space; set MOE_AUTOTUNE_MODE=full for exhaustive experiments.
export MOE_AUTOTUNE="${MOE_AUTOTUNE:-1}"
export MOE_AUTOHEURISTIC="${MOE_AUTOHEURISTIC:-1}"
export MOE_AUTOTUNE_MODE="${MOE_AUTOTUNE_MODE:-heuristic}"
export MOE_AUTOTUNE_WARMUP="${MOE_AUTOTUNE_WARMUP:-5}"
export MOE_AUTOTUNE_REP="${MOE_AUTOTUNE_REP:-20}"
export MOE_AUTOTUNE_TIMEOUT="${MOE_AUTOTUNE_TIMEOUT:-60}"
export MOE_CLEAR_CACHE="${MOE_CLEAR_CACHE:-0}"

# TileLang's compilation workers are CPU-side; keep the default conservative
# on shared MACA hosts and persist successful results in TILELANG_CACHE_DIR.
export TILELANG_AUTO_TUNING_DISABLE_CACHE="${TILELANG_AUTO_TUNING_DISABLE_CACHE:-0}"
export TILELANG_AUTO_TUNING_CPU_UTILITIES="${TILELANG_AUTO_TUNING_CPU_UTILITIES:-0.75}"
export TILELANG_AUTO_TUNING_CPU_COUNTS="${TILELANG_AUTO_TUNING_CPU_COUNTS:-8}"
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT="${TILELANG_AUTO_TUNING_MAX_CPU_COUNT:-8}"

# PyTorch Inductor settings are used only by torch.compile paths.  The MoE
# benchmark remains eager/TileLang, but these defaults make its optional
# compiled helper paths use max-autotune and the built-in AutoHeuristic.
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE="${TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE:-1}"
export TORCHINDUCTOR_AUTOHEURISTIC_USE="${TORCHINDUCTOR_AUTOHEURISTIC_USE:-mixed_mm}"
export TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="${TORCHINDUCTOR_AUTOHEURISTIC_COLLECT:-}"
export TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="${TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH:-${METAX_COMPETITION_ROOT}/.cache/torchinductor-autoheuristic}"

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# The editable install is the primary import path.  This keeps direct source
# invocations deterministic if the package is rebuilt or the editable install
# is temporarily removed.
export PYTHONPATH="${TILELANG_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
