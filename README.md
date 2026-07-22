# MoE-opt：国产 GPU 大模型推理算子优化比赛资料

本仓库整理 2026 年沐曦国产 GPU 大模型推理算子优化相关资料，重点聚焦 **MoE / Fused MoE**，同时保留比赛的完整赛题、AI Agent、FlashAttention、FlashInfer、TileLang 和平台使用资料。

> 资料归档不等于官方规则的替代品。报名、时间、评分和提交要求以赛事官方页面及最新题面为准。

## 快速入口

- [资料索引](docs/INDEX.md)：按赛题、学习阶段和任务包查找资料。
- [TileLang/MACA Fused MoE 基准](benchmarks/tilelang-moe/)：初赛 MoE kernel、参考实现、测试配置和 benchmark。

- [贡献与仓库治理](CONTRIBUTING.md)：分支、验证、性能证据、数据和安全规范。

- [MoE 调优报告](reports/2026-07-11-moe-optimization-report.md)：C500 实测候选、失败原因和最终 schedule。
- [MoE 优化阶段总结](reports/2026-07-16-moe-optimization-summary.md)：当前性能、Stage 0–2 状态、证据边界和下一步实施入口。
- [MoE 阶段 1 解耦结果](reports/2026-07-15-moe-stage1-decoupling-results.md)：E0–E5、交错确认和 FC1/FC2 profiler 对照。
- [MoE E0–E7 数据归档总览](reports/2026-07-20-moe-e0-e7-summary.md)：逐项方案、性能结论、原始基准和 mcProfiler 索引。
- [MoE 调优日志](logs/moe-tuning.md)：每个候选的 functional/performance 结果。
- [MoE 基线实验规范](docs/MOE_BASELINE_EXPERIMENTS.md)：3 个独立进程、median/MAD/P95 与统一 JSON 产物。
- [双 C500 数据同步说明](data/README.md)：主机指纹、autotune、autoheuristic 和 mcProfiler 数据布局。

- [完整比赛资料归档](materials/op_optimization/README.md)：保留原始目录关系，便于追溯来源。
- [Fused MoE 实战教程](materials/op_optimization/%E5%9F%BA%E4%BA%8EAI%20Agent%E5%BC%80%E5%8F%91%E8%8C%83%E5%BC%8F%E7%9A%84%E5%9B%BD%E4%BA%A7GPU%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E7%AE%97%E5%AD%90%E5%BA%93%E4%BC%98%E5%8C%96/Fused%20MoE%20%E7%AE%97%E5%AD%90%E5%85%A5%E9%97%A8%EF%BC%9A%E4%BB%8E%20Benchmark%20%E9%AA%8C%E8%AF%81%E5%88%B0%20XPU-OJ%20%E6%8E%A5%E5%8F%A3%E6%8F%90%E4%BA%A4.md)。
- [TileLang MoE 测试指南](materials/op_optimization/%E5%9F%BA%E4%BA%8E%E5%9B%BD%E4%BA%A7%E8%BD%AF%E4%BB%B6%E6%A0%88%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E5%89%8D%E6%B2%BF%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/race_tests_run_guide%E5%9F%BA%E4%BA%8Etilelang%E7%AE%97%E5%AD%90sample%E8%B7%91%E9%80%9A%E6%B5%8B%E8%AF%95.md)。

## 当前优化状态

截至 2026-07-20，默认实现为 C500 五进程验证通过的 E3+E7：`FullRow + FC1 row8 +
FC2 row16/BK64 + 单 weight shared buffer + exact metadata grid`。相对 E0 的 Large /
Small 中位数分别提升 1.3110% / 1.3458%，综合提升 1.3157%；官方功能、提交 ABI fuzz
及 mcProfiler 均已复验。E0–E7 的定义和归档入口见
[数据归档总览](reports/2026-07-20-moe-e0-e7-summary.md)，最终接受判断见
[E3+E7 推广报告](reports/2026-07-20-moe-e3e7-promotion.md)。FC1 异步流水仍无进入证据。

## 仓库结构

```text
benchmarks/tilelang-moe/   可直接修改的 TileLang Fused MoE 赛题基线与测试
materials/op_optimization/ 原始比赛资料、教程、题包和课程整理
scripts/                   MACA 环境检查、TileLang 构建和 MoE benchmark 脚本
docs/                      本仓库的分类索引与复现说明
```

## MoE 基准快速运行

运行环境要求：赛事专属镜像（PyTorch-Agent / 2.8.0 / Python 3.12 / maca 3.7.2.1）以及可用的沐曦 MACA GPU。仓库不提交 TileLang 的完整源码和构建产物；将其放在 `external/tilelang-metax`，或通过 `TILELANG_HOME` 指定路径。

```bash
# 在仓库根目录执行
source scripts/activate-maca.sh
scripts/verify-maca.sh
scripts/run-moe.sh
```

该默认入口执行 ten-argument `submission.run_kernel` 的三组 published dimensions，
逐 case 使用 5/30、5/20、5/20；远程 aggregate scoring 仍为 unknown，因此不计算总分。
无 GPU 检查入口选择可运行：

```bash
python scripts/run_moe_evaluation.py --describe-target
```

旧 formal Large/Small 是 11-tensor `RoutedMoEKernel` 的 local proxy guard，只有显式
选择时才运行：

```bash
scripts/run-moe.sh --local-proxy-guard
python benchmarks/tilelang-moe/tune_moe.py \
  --evaluation-target local-proxy-guard --experiment E0
```

`moe_test_configs.json` 是 role-aware 主配置：`default_evaluation_target` 固定为
`remote_submission`，remote cases 与 `remote_contract.json` 由 checker 深度校验；旧
functional/performance 位于 `local_proxy_guard`，不得作为默认 remote objective。


## 贡献与维护

提交改动前请阅读 [贡献指南](CONTRIBUTING.md) 和 [治理规则](GOVERNANCE.md)。性能结论必须附带真实硬件、软件版本、workload、正确性状态和测量方法；安全问题请按 [安全策略](SECURITY.md) 私密报告。

基础仓库检查不依赖第三方 Python 包：

```bash
python scripts/check_repository.py
git diff --check
```
当前默认 schedule 固化 E3+E7 的已验证策略：`FullRow`、FC1 stage-1
`row8/BK128`、FC2 stage-1 `row16/BK64`、gate/up 单 shared weight buffer 和 exact
metadata grid。运行时的 autotune 只在已通过 functional 的 swizzle 候选中测量；
autoheuristic 不会重新启用已排除的深 pipeline、耦合 tile、128/512 threads 或
fast-math 方案。E0–E5 显式 preset 仍保留，用于可重复的单变量对照。

需要复现单个候选时：

```bash
python benchmarks/tilelang-moe/tune_moe.py \
  --evaluation-target local-proxy-guard \
  --block-dhidden 128 --block-dexpert 128 --mode all
```

解耦后的单变量实验使用 `--experiment E0` 到 `E5`；正式三进程对比使用：

```bash
python scripts/run_moe_stage1_experiments.py \
  --evaluation-target local-proxy-guard --host-id "${MOE_HOST_ID}"
```

生成正式基线统计产物时：

```bash
python scripts/run_moe_baseline.py \
  --evaluation-target local-proxy-guard --host-id "${MOE_HOST_ID}"
```

提交 ABI 的本地对拍（compact group_sum 10-argument `submission.run_kernel`；
不要与 formal compact 11-tensor `RoutedMoEKernel` 混称）：

```bash
scripts/test-moe-submission.sh --remote-shapes
python scripts/check_moe_remote_contract.py
```

published dimensions 的本地 mirror 使用本地随机 routing，当前契约状态为
`LOCAL_PROXY_ONLY`；远程 routing/metadata sentinel/cache lifecycle/aggregate scoring/
exact toolchain 等未知项在 `benchmarks/tilelang-moe/remote_contract.json` 中逐字段记录。
因此旧 Large/Small 性能和三组本地 mirror 性能都不得冒充 remote-equivalent 分数。

两台服务器共享实验数据时，分别设置：

```bash
# C500-32G
export MOE_HOST_ID=c500-32g

# C500-64G
export MOE_HOST_ID=c500-64g
```

每次实验前后运行：

```bash
source scripts/activate-maca.sh
scripts/sync-data.sh
```

该脚本会采集主机指纹、合并两台机器的 autotune 记录生成
autoheuristic index，并只同步 `data/`、报告和 profiler 元数据；
`.cache/tilelang`、build 和编译产物保持机器本地。

## 资料来源

- `op_optimization`：`metax-maca/op_optimization`，归档提交 `c06e7fa12b2f52bf2bc4f600ff701bbbf98c1318`。
- TileLang/MACA MoE 基线：`tile-ai/tilelang-metax` 的 `race` 分支，归档提交 `ee6db4376484f2f7270183c01fd0d90f794965cb`。
- 本仓库只纳入比赛资料、MoE 基准代码和运行辅助脚本，不纳入本地 build、缓存、Python 字节码或完整第三方源码。
