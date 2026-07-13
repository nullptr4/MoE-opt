# Shared experiment data

This directory is the Git-synchronized source of truth for experiment metadata.
Each server writes only to its own host directory; compiled kernels and local
TileLang caches stay under `.cache/` and are intentionally not synchronized.

Set the host identity before running a benchmark:

```bash
export MOE_HOST_ID=c500-32g   # on the 32 GiB server
export MOE_HOST_ID=c500-64g   # on the 64 GiB server
source scripts/activate-maca.sh
```

## Layout

```text
data/
  hosts/<host-id>/host.json                 hardware/software fingerprint
  autotune/<host-id>/results.jsonl          append-only tuning observations
  autoheuristic/dataset.jsonl               merged observations
  autoheuristic/index.json                  generated best-config index
  profiler/<host-id>/<run-id>/metadata.json profiler manifest
```

Each autotune observation records the workload, complete schedule, measured
latency, correctness status, host profile, source commit and profiler run
identifier when available. Keep records append-only so the two machines do not
edit the same JSON object.

The 32 GiB and 64 GiB machines share observations but are separate hardware
profiles. Autoheuristic lookup prefers an exact memory profile, then falls back
to another C500 result as a candidate for local autotune.

Generate the merged autoheuristic dataset after pulling both machines' data:

```bash
python scripts/merge_tuning_results.py
```

The generated index is reproducible from the JSONL records and should be
reviewed together with the records when it changes.
