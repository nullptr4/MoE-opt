# MoE E7：exact metadata grid 候选（拒绝）

## 假设

E0 按 `ceil(R / 128) + expert_count` 构建 metadata grid。官方固定路由的实际
expert rows 并不恰好填满每个 128-row tile，因此该 grid 末尾含 3 个（Large）或 2 个
（Small）空 block。E7 改为由各 expert 的实际 `ceil(count / 128)` 求和，并让 kernel
编译时使用相同 metadata M；这样可直接少发射 FC1 / FC2 的无效 CTA，而不是像 E6 一样
在已经发射的 CTA 内部跳过工作。

候选仅用于筛选：`--compact-metadata-grid` 与 `--candidate-id E7-exact-metadata-grid`，
没有改动 E0 默认路径、数值类型或对外 ABI。

## 映射与正确性

官方 performance workload、官方 seed 的实际路由为：

| workload | expert counts | E0 M | E7 exact M | 减少 block |
| --- | --- | ---: | ---: | ---: |
| Large | [16519, 16399, 16337, 16520, 16306, 16277, 16353, 16361] | 1032 | 1029 | 3 |
| Small | [16486, 16398, 16375, 16277] | 516 | 514 | 2 |

- 官方 functional Large / Small：2/2 PASS。
- pre-routed compact-layout 边界：PASS，日志
  `logs/moe-e7-exact-metadata-20260720T025100Z-boundary.log`。
  覆盖 `[0, 1, 127, 128, 129, 255, 256, 257]`、8 个精确 128-row expert，及
  `[513, 1, 0, 0, 0, 0, 0, 0]` 偏斜路由。

## 性能

相同 C500 上按 E0/E7 交替运行 5 个独立进程；每个进程 20 warm-up、100 iterations，
关闭 autotune 与 autoheuristic。原始记录：

```text
data/benchmarks/c500-64g/e7-exact-metadata-20260720T025400Z-e0-run-1..5.json
data/benchmarks/c500-64g/e7-exact-metadata-20260720T025400Z-e7-run-1..5.json
```

| workload | E0 median / MAD / P95 (ms) | E7 median / MAD / P95 (ms) | E7 相对 E0 |
| --- | ---: | ---: | ---: |
| Large | 242.0776 / 0.0817 / 242.1638 | 241.7078 / 0.1041 / 241.8119 | +0.1528% |
| Small | 38.0140 / 0.0068 / 38.0374 | 37.9989 / 0.0288 / 38.0277 | +0.0396% |
| 合计 | 280.0916 | 279.7067 | **+0.1374%** |

中位数方向为正，但低于计划中 `0.3%` 的可组合候选下限，也远低于 `1%` 的默认升级门槛。

## mcProfiler

定向 `--per-kernel --single-pass` 原始产物：

- `mcProfilerOutput/success/e7-exact-metadata-large-20260720T030900Z-targeted/`
- `mcProfilerOutput/success/e7-exact-metadata-small-20260720T031100Z-targeted/`

对照 E0 的同类归档产物
`e0-large-20260720T002113Z-targeted/` 与 `e0-small-20260720T014808Z-targeted/`。
`1_kernel_kernel.txt.json` 是 FC1，`2_kernel_kernel_1.txt.json` 是 FC2。

| workload / kernel | E0 → E7 workgroups | 变化 | E0 → E7 total cycles | 变化 |
| --- | ---: | ---: | ---: | ---: |
| Large FC1 | 16,512 → 16,464 | -0.2907% | 127,901.33 → 128,175.23 Kcycles | +0.2142% |
| Large FC2 | 57,792 → 57,624 | -0.2907% | 92,403.35 → 92,240.94 Kcycles | -0.1768% |
| Small FC1 | 4,128 → 4,112 | -0.3876% | 16,726.47 → 16,810.41 Kcycles | +0.5018% |
| Small FC2 | 14,448 → 14,392 | -0.3876% | 13,211.21 → 13,241.08 Kcycles | +0.2261% |

workgroup 降幅与精确 M（Large 1032→1029、Small 516→514）完全一致，证明 E7 实际消除了
空 CTA。周期并未一致降低，说明节省的尾部 work 低于该算子的实际计时噪声和调度成本。

跨 profiler run 的 instruction/global-read 计数存在远大于 0.29%–0.39% 调度变化的差异，
不适合作为此候选的归因证据；本结论只使用稳定、可由映射严格预期的 workgroup 计数和五进程
端到端测量。

## 判断

**reject**。E7 数值正确，且确实减少 launch workgroups，但综合稳定收益仅 0.1374%，
不足以抵消 metadata 重构的额外代码路径和维护风险。候选实现、CLI 开关及编译参数已回退；
保留全部结果、日志、mcProfiler 原始文件与本记录，供后续更大粒度的调度工作复用。
