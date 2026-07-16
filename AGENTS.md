# Repository instructions for coding agents

These rules apply to automated and human-assisted changes in this repository.

## Scope and sources of truth

- Treat `benchmarks/`, `scripts/`, `config/`, `docs/`, `reports/` and root governance files as maintained content.
- Treat `materials/` as archived upstream material. Avoid modifying it unless the task is explicitly about the archive.
- Preserve the public `RoutedMoEKernel` API and the competition submission ABI unless a task explicitly changes them.
- Use the official benchmark configuration and reference implementation when making correctness or performance claims.

## Performance integrity

- Never invent benchmark, profiler or hardware results.
- Record the exact workload, software versions, device profile, source commit, schedule and correctness status with each result.
- Compare candidates only under equivalent warm-up, iteration and cache conditions.
- Keep failed and rejected experiments when they explain a decision; label them clearly.
- Do not present measurements from one C500 memory profile as measurements from another profile.

## Data and secrets

- Keep build products, compiled kernels, local caches, environments and credentials out of Git.
- Store large profiler payloads through Git LFS when the repository data workflow supports them.
- Do not commit API keys, platform tokens, SSH keys, private datasets or personally identifying information.

## Validation

Run the smallest relevant checks before committing:

```bash
python scripts/check_repository.py
scripts/verify-maca.sh
scripts/run-moe.sh
scripts/test-moe-submission.sh --public-shape
```

GPU-dependent checks may be skipped only when no MACA device is available; state that limitation in the pull request.

## Change discipline

- Keep changes focused and reviewable.
- Prefer existing scripts and patterns over new dependencies.
- Update documentation and reproducibility records when behavior, schedules or data formats change.
- Use a feature branch and a pull request; do not push experimental work directly to `main`.
