# MoE E0--E5 复验与 mcProfiler 分析（2026-07-20）

## 结论

E0--E5 均通过官方 Large / Small functional workload；E0 保持默认 schedule，不提升任何
候选。E3 是唯一端到端正收益候选，但 Large / Small 合计仅快 **0.9014%**，未达到
Stage-1 的 **1%** 晋级线。E1、E2、E4、E5 的合计中位数分别比 E0 慢
2.9473%、20.7560%、1.2393%、9.3011%。

本轮没有修改 kernel 源码、提交 ABI、默认 schedule 或 autotune 候选集。结论严格基于
同一设备、同一 workload、每个实验三次独立进程的 median / MAD / P95 统计；mcProfiler
只用于解释硬件侧行为，不将它的 Kcycles 换算为端到端毫秒。

## 环境、提交与工作负载

- Git 基线：`9760e13bc51da6341cca4a96a410de20f00a3db0`
- Device：MetaX C500（64 GiB）；MACA driver `3.8.30`，MACA `3.7.1.5`
- PyTorch：`2.8.0+metax3.7.1.3`；TileLang：`0.1.12+maca.gitec48829b`
- mcProfiler：`3.8.1.4`（`575f5a9f6d`）
- 官方 performance Large：`dhidden=7168`、`dexpert=2048`、8 experts、
  4 experts/token、`bs=4`、`seqlen=8192`、`seed=81394`
- 官方 performance Small：`dhidden=3584`、`dexpert=1024`、4 experts、
  2 experts/token、`bs=8`、`seqlen=4096`、`seed=81394`
- 每个性能点：10 warm-up、100 iterations、3 个独立 Python 进程；原始记录：
  `data/benchmarks/c500-64g/stage1-20260720T005300Z.json`

E0 是 `BM=BN=BK=128`、stage 1、FullRow、FC1 row8、FC2 row16、256 threads、
FC1 单 weight buffer 的基线。候选严格按技术博客定义：E1 将 FC1 `BK=64`；E2 将
FC1 `BN=64`；E3 将 FC2 `BK=64, stage=1`；E4 将 FC2 `BK=64, stage=2`；
E5 将 FC2 `BN=256`。

## 官方端到端性能

| 实验 | Large median (ms) | Small median (ms) | 合计 (ms) | 相对 E0 | 决定 |
| --- | ---: | ---: | ---: | ---: | --- |
| E0 | 242.0556 | 37.9914 | 280.0469 | 基线 | 保持默认 |
| E1 | 249.9671 | 38.3336 | 288.3008 | -2.9473% | 拒绝 |
| E2 | 294.1730 | 44.0005 | 338.1735 | -20.7560% | 拒绝 |
| E3 | 240.0410 | 37.4815 | 277.5225 | +0.9014% | 拒绝：未达 +1% |
| E4 | 245.4091 | 38.1083 | 283.5174 | -1.2393% | 拒绝 |
| E5 | 265.0633 | 41.0309 | 306.0943 | -9.3011% | 拒绝 |

样本离散度很低：Large / Small 的 MAD 分别不超过 0.0507 / 0.0239 ms。E3 的
Large、Small 均有正收益（+0.8323%、+1.3421%），但合计不足门槛，故不因边际结果改变
默认提交。

## mcProfiler 方法和产物

mcProfiler 从 `/opt/mcProfiler-ubuntu18.04` 启动（该工具需要安装目录下的相对事件表），
每个实验分别采样 Large 和 Small workload，采样设置均为 1 warm-up、1 iteration：

```bash
mcProfiler perf_exec \
  --cmdline 'python tune_moe.py --experiment E? --mode performance --shape large|small --warmup 1 --iteration 1' \
  --kernelname kernel_kernel --casename moe-e?-large|small-targeted \
  --cwd /data/MoE-opt/benchmarks/tilelang-moe \
  --per-kernel --single-pass --kernelnames kernel_kernel kernel_kernel_1
```

按仓库的 profiler 报告约定，以下定向分析使用每个目录中的
`1_kernel_kernel.txt.json`（FC1）及 `2_kernel_kernel_1.txt.json`（FC2）。
各目录同时保留原始 JSON、CSV、PDF、HTML、图及采集日志：

- `mcProfilerOutput/success/e0-large-20260720T002113Z-targeted/`
- `mcProfilerOutput/success/e1-large-20260720T004536Z-targeted/`
- `mcProfilerOutput/success/e2-large-20260720T004718Z-targeted/`
- `mcProfilerOutput/success/e3-large-20260720T002257Z-targeted/`
- `mcProfilerOutput/success/e4-large-20260720T004853Z-targeted/`
- `mcProfilerOutput/success/e5-large-20260720T005042Z-targeted/`
- `mcProfilerOutput/success/e0-small-20260720T014808Z-targeted/`
- `mcProfilerOutput/success/e1-small-20260720T015015Z-targeted/`
- `mcProfilerOutput/success/e2-small-20260720T015139Z-targeted/`
- `mcProfilerOutput/success/e3-small-20260720T015303Z-targeted/`
- `mcProfilerOutput/success/e4-small-20260720T015427Z-targeted/`
- `mcProfilerOutput/success/e5-small-20260720T015551Z-targeted/`

以下只使用稳定、可直接比对的计数：Total Cycles、WORKGROUPS、global-memory bytes、
L2C hit rate 和 private 指令数。报告中出现的个别吞吐 / occupancy 派生指标有明显异常值，
故未据其作判断。

## Large：变更 kernel 的硬件侧证据

| 候选（分析 kernel） | Cycles 相对 E0 | WORKGROUPS 相对 E0 | 全局读相对 E0 | L2C 命中变化 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| E1（FC1, BK=64） | +7.29% | 0.00% | +0.95% | -0.33 pp | K 方向切分增多，循环 / 调度开销上升。 |
| E2（FC1, BN=64） | +46.17% | +100.00% | +41.43% | +5.24 pp | CTA 数翻倍与额外流量远大于 locality 收益。 |
| E3（FC2, BK=64, stage1） | -2.54% | 0.00% | +109.60% | -5.10 pp | kernel cycle 略降，但流量翻倍、局部性变差；端到端仅边际收益。 |
| E4（FC2, BK=64, stage2） | +4.50% | 0.00% | +175.32% | -8.21 pp | stage 2 未隐藏额外数据移动成本。 |
| E5（FC2, BN=256） | +28.13% | -50.00% | +208.52% | -14.50 pp | 虽减少一半 workgroups，寄存器 / 私有访问与数据移动明显放大。 |

基线数值为：FC1 127,901.33 Kcycles、16,512 workgroups、63.865 GB global read、
64.81% L2C hit；FC2 92,403.35 Kcycles、57,792 workgroups、7.380 GB global read、
94.12% L2C hit。E5 FC2 的 private read / write instructions 为
130,725,504 / 9,246,720（E0 FC2 为 0 / 0），global write 也从 1.880 GB 增至
4.354 GB（+131.52%），与其端到端回退一致。

未改动 kernel 的另一个 GEMM 不作为候选因果分析对象；单次 profiler 跨运行计数会受到采样
扰动影响。所有性能晋级判断只采用上一节的三独立进程正式基准。

## Small：变更 kernel 的硬件侧证据

| 候选（分析 kernel） | Cycles 相对 E0 | WORKGROUPS 相对 E0 | 全局读相对 E0 | L2C 命中变化 | 与 Small 延迟的关系 |
| --- | ---: | ---: | ---: | ---: | --- |
| E1（FC1, BK=64） | +2.13% | 0.00% | -27.33% | +5.31 pp | 即使观测到更少的 global read，cycles 仍上升；Small 慢 0.9009%。 |
| E2（FC1, BN=64） | +40.55% | +100.00% | -11.19% | +9.18 pp | CTA 数翻倍的开销仍占主导；Small 慢 15.8171%。 |
| E3（FC2, BK=64, stage1） | -4.31% | 0.00% | -0.82% | +0.02 pp | FC2 cycle 降低，与 Small 快 1.3421% 的方向一致；仍不能改变合计晋级结论。 |
| E4（FC2, BK=64, stage2） | +1.36% | 0.00% | +15.65% | -0.51 pp | stage 2 的额外流量未被隐藏；Small 慢 0.3079%。 |
| E5（FC2, BN=256） | +25.74% | -50.00% | +207.97% | -13.04 pp | 减少 CTA 未抵消流量和私有访问代价；Small 慢 8.0007%。 |

Small 基线为：FC1 16,726.47 Kcycles、4,128 workgroups、4.402 GB global read、
80.21% L2C hit；FC2 13,211.21 Kcycles、14,448 workgroups、0.757 GB global read、
93.96% L2C hit。E5 FC2 的 private read instructions 从 E0 的 0 增至 16,499,616，
global write 从 0.470 GB 增至 1.089 GB（+131.55%），是其回退的直接计数证据。

Small 的 E1 / E2 虽报告了较少的 global read 和较高的 L2 hit，却没有带来更低 cycles；
这说明不能只用单个流量计数解释端到端结果。对于候选决策，继续以同条件的三独立进程延迟
统计为准，mcProfiler 用于定位与排除明显的结构性问题（例如 E2 的 CTA 扩张、E5 的私有访问
与写流量放大）。

## 正确性与验证

- E0--E5 均通过官方 Large / Small functional workload（共 12 次）。
- `scripts/test-moe-submission.sh --public-shape` 通过：uneven-smoke 与
  public-small-shape 均 PASS。
- `scripts/verify-maca.sh` 通过。
- `python scripts/check_repository.py` 通过（1,594 个 tracked files）。
- `python -m pytest -q tests/test_moe_stage_schedules.py tests/test_moe_stage2_profiler.py`
  通过：16 passed。

本地保留但未纳入本次 GitHub 数据包的被拒绝诊断采样：
`mcProfilerOutput/failed/e0-large-20260720T001939Z-per-kernel-single-pass/`。
它未筛选到目标 GEMM，而采到了 PyTorch 随机数初始化 kernel，且报告生成器报 RoofLine
除零 / `processed_data` 缺失；该目录不参与任何结论。另一个无 kernel 过滤的成功聚合
采样位于 `mcProfilerOutput/success/e0-large-20260720T001602Z-aggregate/`，同样未纳入
数据包，也不用于本报告的定向比较。
