# MoE E6：空 metadata tile 跳过候选（拒绝）

## 假设

当前 kernel 以 `M = ceil(R / 128) + expert_count` 构造 metadata grid。对官方固定
seed 的实际路由，末尾仍有 `actual_rows=0` 的 block；若仅对这种 CTA-uniform 条件跳过
FC1 / FC2 的 clear、copy、GEMM 和 epilogue，可消除无效工作且不影响有效行。

## 变更

候选标识为 `E6-empty-tile-skip`。在 FC1 与 FC2 中仅增加：

```text
if actual_rows > 0:
    原有该 CTA 的计算与 store
```

开关是 compile-time 的 `skip_empty_tiles=True`，E0 默认值为 `False`。没有改变
tile、threads、stage、swizzle、布局、精度、外部 ABI 或有效行的计算。候选源码哈希为
`97a5b31c56ecfdc67cb720abe51a11718768b8cfbc4996bc3c9d303e0fc6e7c6`；本记录完成后已回退
候选代码，默认 E0 保持不变。

## 路由 / tile 覆盖

使用官方 performance workload、官方 seed 和 C500 实际执行 router 后得到：

| workload | expert counts | metadata block | full / tail / empty |
| --- | --- | ---: | ---: |
| Large | [16519, 16399, 16337, 16520, 16306, 16277, 16353, 16361] | 1032 | 1021 / 8 / 3 |
| Small | [16486, 16398, 16375, 16277] | 516 | 510 / 4 / 2 |

这说明 E6 最多影响 Large 的 3/1032、Small 的 2/516 个 M block；每个 block 仍会展开
全部 FC1 / FC2 N tile，因此值得以单变量候选实测，但理论上不应预期达到大幅收益。

## 正确性

- 官方 functional Large / Small：2/2 PASS。
- pre-routed compact-layout 边界：PASS，日志
  `logs/moe-e6-empty-tile-20260720T022400Z-boundary.log`。
  - boundary：`[0, 1, 127, 128, 129, 255, 256, 257]`
  - exact：8 个 `128`
  - skew：`[513, 1, 0, 0, 0, 0, 0, 0]`

边界参考使用与官方 harness 相同的 `1/sqrt(d)` 权重尺度、FP16 中间 workspace 及
`atol=rtol=1e-2`。一次未缩放随机权重的诊断对拍在少量大值处超出该容差，已明确排除，
不作为候选结果。

## 性能

两组均在相同 C500 上运行 5 个独立进程，按 E0/E6 交替顺序，关闭 autotune 与
autoheuristic；每进程 20 warm-up、100 iterations。原始进程记录位于：

```text
data/benchmarks/c500-64g/e6-empty-tile-20260720T022400Z-e0-run-1..5.json
data/benchmarks/c500-64g/e6-empty-tile-20260720T022400Z-e6-run-1..5.json
```

| workload | E0 median / MAD / P95 (ms) | E6 median / MAD / P95 (ms) | E6 相对 E0 |
| --- | ---: | ---: | ---: |
| Large | 242.1965 / 0.0953 / 242.2776 | 241.8430 / 0.1513 / 241.9766 | +0.1460% |
| Small | 38.0172 / 0.0510 / 38.0581 | 38.0076 / 0.0320 / 38.3109 | +0.0253% |
| 合计 | 280.2138 | 279.8506 | **+0.1296%** |

E6 Small 有一组 38.3857 ms 样本，standard deviation / CV 为 0.1787 ms / 0.4696%
（E0 为 0.0464 ms / 0.1221%）。尽管两组中位数方向均为正，合计提升远低于
0.3% 的可组合候选下限，也远低于 1% 默认升级门槛。

## mcProfiler

定向 single-pass 原始产物：

- `mcProfilerOutput/success/e6-empty-tile-large-20260720T023737Z-targeted/`
- `mcProfilerOutput/success/e6-empty-tile-small-20260720T023901Z-targeted/`

分析使用每个目录的 `1_kernel_kernel.txt.json`（FC1）和
`2_kernel_kernel_1.txt.json`（FC2）。计数中的 `WORKGROUPS` 仍等于 E0，因为
E6 只跳过已发射 CTA 的内部工作，未改变 launch grid。

| workload / kernel | cycles 相对 E0 | global read 相对 E0 | L2C hit 变化 | workgroups |
| --- | ---: | ---: | ---: | ---: |
| Large FC1 | -0.05% | -0.52% | +0.18 pp | 16,512 → 16,512 |
| Large FC2 | -0.16% | -8.07% | +0.37 pp | 57,792 → 57,792 |
| Small FC1 | +0.17% | -2.20% | +0.46 pp | 4,128 → 4,128 |
| Small FC2 | -0.24% | +0.40% | +0.01 pp | 14,448 → 14,448 |

E6 的 cycle 变化仅在约 ±0.24% 范围内，支持其端到端收益非常有限。Large FC2 的单次
global-read 差异（-8.07%）远大于 3/1032 个空 block 能解释的比例，故视为 profiler
跨 run 采样波动，不作为 E6 收益证据。

## 判断

**reject**。E6 保持数值正确，但未达到计划规定的稳定收益阈值，也没有降低 launch
workgroups。候选实现和 CLI 开关已回退；保留原始 JSON、日志、profiler 输出和本拒绝记录，
供后续 exact-grid 或 metadata 重构方案复用。

## 同轮基线

计划 P0 的正式 E0 基线已另行完成：
`data/benchmarks/c500-64g/plan-p0-e0-20260720T020300Z.json`。它包含 5 个独立进程、
20 warm-up、100 iterations、Large/Small functional 5/5、公开 ABI PASS、fuzz PASS，
并链接 Large / Small E0 的归档 mcProfiler metadata。
