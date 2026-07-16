# Profiler artifacts

Use `scripts/archive_mcprofiler.py` after an mcProfiler run. It stores a small
tracked `metadata.json` manifest and puts the raw trace under `raw/`.

```bash
python scripts/archive_mcprofiler.py \
  --host-id "$MOE_HOST_ID" \
  --run-id 20260713-row8 \
  --source /path/to/mcprofiler-output \
  --workload-json /path/to/workload.json
```

The manifest includes SHA-256 checksums, host id, Git commit, TileLang/MACA
environment variables and the original source path. Raw profiler payloads are
tracked by Git LFS; summaries and metadata remain ordinary Git files.
