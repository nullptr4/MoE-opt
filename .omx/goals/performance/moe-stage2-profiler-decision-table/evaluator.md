# Performance Evaluator: moe-stage2-profiler-decision-table

## Objective
Complete the MoE Stage 2 profiler decision table for official baseline, FullRow double-buffer, current E0 single-buffer, and Stage 1 candidates; preserve the E0 default and derive evidence-backed next actions.

## Evaluator Command
```sh
python scripts/validate_moe_stage2_profiler.py --table data/profiler/c500-32g/stage2-decision-table.json && python -m unittest discover -s tests -v
```

## Pass/Fail Contract
PASS only when every required comparison version has source-linked FC1/FC2 timing or an explicit unavailable reason, static resource and mcProfiler metrics are provenance-checked, every bottleneck/next-action claim follows documented deterministic rules, all referenced artifacts hash-verify, the E0 default and autotune data remain unchanged, and the full unit-test suite passes.

This evaluator must exist and produce concrete pass/fail evidence before the performance goal can be completed.
