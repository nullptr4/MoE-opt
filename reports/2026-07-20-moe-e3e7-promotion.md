# MoE E3+E7：FC2 BK=64 + exact metadata grid（接受并设为默认）

## 结论

**accept**。E3（FC2 `BK=64, stage=1`）单独复验的综合中位数提升为 0.9148%，不足以
替换 E0；E7（精确 metadata grid）单独只有 0.1374%。两项组合后，在 5 轮独立、交替
E0/候选进程中，Large 与 Small 均稳定超过 1%，综合提升 **1.3157%**。组合已成为
`fusedmoe_benchmark.custom_kernel` 的默认实现。

## 实现

- FC1 保持 `BM=BN=BK=128`、stage 1、row8、single weight buffer。
- FC2 保持 `BN=128, stage=1, row16`，将 K tile 从 128 改为 **64**。
- host 端以各 expert 的实际 `ceil(count/128)` 构造 `group_padded_offsets` 与 metadata
  grid；JIT 内部以相同 `metadata_m` 编译，因此不发射末尾空 CTA。
- `RoutedMoEKernel.__init__` 和 `__call__` 的公开参数与语义未改变；`metadata_m` 仅是
  benchmark packing path 设置的内部属性。
- OJ `submission.py` 保持原始 `run_kernel` ABI。它接收的 `group_idx_for_bx` 本来就是
  调用方提供的 exact grid，因此只移植 FC2 `BK=64`，不重写 caller-owned metadata。

`tune_moe.py` 保留可复现的非默认候选开关：

```bash
python tune_moe.py --candidate-id E3E7-fc2-bk64-exact-grid \
  --s2-bk 64 --compact-metadata-grid
```

## 正确性与 ABI

- 官方 functional Large / Small：2/2 PASS，记录
  `data/benchmarks/c500-64g/e3e7-combined-20260720T032700Z-functional.json`。
- 默认 `fusedmoe_benchmark.custom_kernel` 的官方 functional Large / Small：2/2 PASS。
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 PASS，日志
  `logs/moe-e3e7-promotion-submission-20260720T034500Z.log`。
  覆盖 uneven、public-small、127/128/129 边界、exact multiple、偏斜/empty expert 及
  zero route weight。

## 稳态性能

使用同一 C500，关闭 autotune 与 autoheuristic；每个独立进程均为 20 warm-up、100
iterations，连续五轮按 E0/组合候选交替执行。原始记录与进程日志：

```text
data/benchmarks/c500-64g/e3e7-combined-20260720T032700Z-e0-run-1..5.json
data/benchmarks/c500-64g/e3e7-combined-20260720T032700Z-e3e7-run-1..5.json
logs/moe-e3e7-combined-20260720T032700Z-{e0,e3e7}-run-1..5.log
```

| workload | E0 median / MAD / P95 (ms) | E3+E7 median / MAD / P95 (ms) | 相对 E0 |
| --- | ---: | ---: | ---: |
| Large | 242.1845 / 0.0212 / 242.2316 | 239.0096 / 0.0145 / 239.0240 | **+1.3110%** |
| Small | 38.0024 / 0.0168 / 38.0468 | 37.4910 / 0.0173 / 37.6056 | **+1.3458%** |
| 合计 | 280.1870 | 276.5006 | **+1.3157%** |

五组候选样本的标准差 / CV 为 Large 0.0445 ms / 0.0186%，Small 0.0566 ms / 0.1509%；
两项均未出现相对 E0 的退化。

作为组合前置证据，E3 的独立 5 进程复验记录于
`data/benchmarks/c500-64g/e3-revalidation-20260720T031900Z-*.json`，综合提升 0.9148%，
因此只作为可组合候选；E7 单独拒绝记录见
`reports/2026-07-20-moe-e7-exact-metadata-rejection.md`。

## mcProfiler 分析

使用 `--per-kernel --single-pass`，每个 workload 均保存 FC1
`1_kernel_kernel.txt.json` 与 FC2 `2_kernel_kernel_1.txt.json`：

- `mcProfilerOutput/success/e3e7-combined-large-20260720T034000Z-targeted/`
- `mcProfilerOutput/success/e3e7-combined-small-20260720T034100Z-targeted/`

对照同口径 E0 原始产物：
`e0-large-20260720T002113Z-targeted/` 与
`e0-small-20260720T014808Z-targeted/`。

| workload / kernel | E0 → E3+E7 workgroups | 变化 | E0 → E3+E7 total cycles | 变化 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Large FC1 | 16,512 → 16,464 | -0.2907% | 127,901.33 → 127,993.62 Kcycles | +0.0722% | exact grid 只减少空 CTA；FC1 不改 K tile。 |
| Large FC2 | 57,792 → 57,624 | -0.2907% | 92,403.35 → 89,059.67 Kcycles | **-3.6185%** | BK=64 的 FC2 改善超过小幅 CTA 节省。 |
| Small FC1 | 4,128 → 4,112 | -0.3876% | 16,726.47 → 16,785.66 Kcycles | +0.3539% | 同上。 |
| Small FC2 | 14,448 → 14,392 | -0.3876% | 13,211.21 → 12,632.91 Kcycles | **-4.3774%** | FC2 cycle 降低与 Small 端到端收益一致。 |

Large FC2 的 L2C hit rate 从 94.12% 升至 94.38%，Small FC2 为 93.96%→93.86%。
单次 profiler 的跨 run 流量/指令计数仍可能波动，故晋级只由五进程端到端统计决定；
workgroup 降幅与 exact M（Large 1032→1029、Small 516→514）严格一致，FC2 cycle 的
同向改善则提供了组合收益的硬件侧解释。

## 最终验证

- `scripts/verify-maca.sh` PASS（MetaX C500、MACA target）。
- `python scripts/check_repository.py` PASS。
- `pytest -q tests/test_moe_stage_schedules.py tests/test_moe_stage2_profiler.py
  tests/test_run_moe_baseline.py tests/test_run_moe_stage1_experiments.py`：36 PASS。
- Stage-2 决策表已用仓库生成器重建；它保留历史 E0–E5 判断，同时用当前文件快照检测后续
  source 漂移，不再把“尚未提交到 Git HEAD”本身视为错误。
