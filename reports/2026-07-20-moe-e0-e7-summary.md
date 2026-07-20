# MoE E0–E7 优化方案与数据归档总览

## 最终结论

E0–E7 的单变量与结构性候选均已在 MetaX C500 64 GiB 上完成正确性、稳态性能及
mcProfiler 复验。E1、E2、E4、E5、E6 被拒绝；E3 和 E7 单独均未达到默认升级门槛，
但 E3+E7 组合在五轮交替独立进程中相对 E0 的 Large / Small 中位数分别提升
1.3110% / 1.3458%，综合提升 **1.3157%**，已设为当前默认实现。

## E0–E7 分别是什么

所有 tile 值均描述 FC1 / FC2 GEMM 的内部 schedule。公共基线统一使用
`BM=128`、256 threads、FullRow、FC1 row8、FC2 row16、stage 1 和 FC1 单
weight shared buffer。

| 编号 | 方案 | 相对 E0 的唯一变化 | 结果 |
| --- | --- | --- | --- |
| E0 | canonical 控制组 | FC1、FC2 均为 `BN=128, BK=128, stage=1` | 基线 |
| E1 | FC1 缩小 K tile | FC1 `BK: 128 → 64` | 合计 -2.9473%，拒绝 |
| E2 | FC1 缩小 N tile | FC1 `BN: 128 → 64` | 合计 -20.7560%，拒绝 |
| E3 | FC2 缩小 K tile | FC2 `BK: 128 → 64`，仍为 stage 1 | 三进程 +0.9014%；五进程复验 +0.9148%，单独未达 1% |
| E4 | FC2 双级流水 | FC2 `BK=64, stage: 1 → 2` | 合计 -1.2393%，拒绝 |
| E5 | FC2 放大 N tile | FC2 `BN: 128 → 256` | 合计 -9.3011%，拒绝 |
| E6 | 空 tile 内部跳过 | launch grid 不变；已发射 CTA 在 `actual_rows == 0` 时跳过 FC1/FC2 工作 | 五进程 +0.1296%，拒绝并回退 |
| E7 | exact metadata grid | 由各 expert 的 `sum(ceil(count/128))` 构造 grid 与 JIT `metadata_m`，不发射末尾空 CTA | 五进程 +0.1374%，单独拒绝 |

E0–E5 是 `moe_schedule.py` 中的 canonical preset。E6、E7 是后续 metadata
结构候选，不属于 `--experiment E0..E5` 的固定 schedule 表。E6 的候选代码已回退；
E7 的 `--compact-metadata-grid` 路径保留，并与 E3 组合成为当前默认。

## 分阶段性能结果

E0–E5 使用 10 warm-up、100 iterations、3 个独立性能进程；表中“合计”为 Large 与
Small median 之和：

| 实验 | Large median (ms) | Small median (ms) | 合计 (ms) | 相对 E0 |
| --- | ---: | ---: | ---: | ---: |
| E0 | 242.0556 | 37.9914 | 280.0469 | 基线 |
| E1 | 249.9671 | 38.3336 | 288.3008 | -2.9473% |
| E2 | 294.1730 | 44.0005 | 338.1735 | -20.7560% |
| E3 | 240.0410 | 37.4815 | 277.5225 | +0.9014% |
| E4 | 245.4091 | 38.1083 | 283.5174 | -1.2393% |
| E5 | 265.0633 | 41.0309 | 306.0943 | -9.3011% |

E6、E7 与最终 E3+E7 使用 20 warm-up、100 iterations、E0/候选交替的 5 个独立进程：

| 候选 | Large 相对 E0 | Small 相对 E0 | 综合相对 E0 | 判断 |
| --- | ---: | ---: | ---: | --- |
| E6 | +0.1460% | +0.0253% | +0.1296% | 拒绝 |
| E7 | +0.1528% | +0.0396% | +0.1374% | 单独拒绝 |
| E3+E7 | **+1.3110%** | **+1.3458%** | **+1.3157%** | 接受并设为默认 |

不同阶段的独立进程数和 warm-up 不同，因此上表只在各自同轮 E0 对照内计算相对收益，
不混用绝对延迟。

## 数据包索引

- E0–E5 三进程原始记录：
  `data/benchmarks/c500-64g/stage1-20260720T005300Z.json`
- E3、E6、E7、E3+E7 五进程原始记录：
  `data/benchmarks/c500-64g/`
- 结构化 profiler 摘要和采集元数据：
  `data/profiler/c500-64g/`
- mcProfiler E0–E7、E3+E7 Large / Small 原始产物：
  `mcProfilerOutput/success/`
- mcProfiler 目录说明：
  `mcProfilerOutput/README.md`

分析报告：

- `reports/2026-07-20-moe-e0-e5-mcprofiler.md`
- `reports/2026-07-20-moe-e6-empty-tile-rejection.md`
- `reports/2026-07-20-moe-e7-exact-metadata-rejection.md`
- `reports/2026-07-20-moe-e3e7-promotion.md`

mcProfiler 定向比较统一采用 `1_kernel_kernel.txt.json`（FC1）和
`2_kernel_kernel_1.txt.json`（FC2）。Kcycles 只用于同类 kernel 的方向性比较，
不换算为端到端毫秒；晋级判断始终采用独立进程的端到端 median。

## 环境与验证

- Git 实验基线：`9760e13bc51da6341cca4a96a410de20f00a3db0`
- Device：MetaX C500 64 GiB
- MACA driver `3.8.30`，MACA `3.7.1.5`
- PyTorch `2.8.0+metax3.7.1.3`
- TileLang `0.1.12+maca.gitec48829b`
- mcProfiler `3.8.1.4`（`575f5a9f6d`）

E0–E7 与组合候选均通过对应官方 Large / Small functional gate；最终默认实现另通过
公开提交 ABI、127/128/129 边界、精确倍数、偏斜/空 expert 和 zero route weight fuzz。
具体命令、样本、MAD/P95 与 profiler 计数见各分项报告。
