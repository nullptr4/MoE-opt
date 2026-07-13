# MoE-opt：国产 GPU 大模型推理算子优化比赛资料

本仓库整理 2026 年沐曦国产 GPU 大模型推理算子优化相关资料，重点聚焦 **MoE / Fused MoE**，同时保留比赛的完整赛题、AI Agent、FlashAttention、FlashInfer、TileLang 和平台使用资料。

> 资料归档不等于官方规则的替代品。报名、时间、评分和提交要求以赛事官方页面及最新题面为准。

## 快速入口

- [资料索引](docs/INDEX.md)：按赛题、学习阶段和任务包查找资料。
- [TileLang/MACA Fused MoE 基准](benchmarks/tilelang-moe/)：初赛 MoE kernel、参考实现、测试配置和 benchmark。
- [MoE 调优报告](reports/2026-07-11-moe-optimization-report.md)：C500 实测候选、失败原因和最终 schedule。
- [MoE 调优日志](logs/moe-tuning.md)：每个候选的 functional/performance 结果。
- [双 C500 数据同步说明](data/README.md)：主机指纹、autotune、autoheuristic 和 mcProfiler 数据布局。
- [完整比赛资料归档](materials/op_optimization/README.md)：保留原始目录关系，便于追溯来源。
- [Fused MoE 实战教程](materials/op_optimization/%E5%9F%BA%E4%BA%8EAI%20Agent%E5%BC%80%E5%8F%91%E8%8C%83%E5%BC%8F%E7%9A%84%E5%9B%BD%E4%BA%A7GPU%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E7%AE%97%E5%AD%90%E5%BA%93%E4%BC%98%E5%8C%96/Fused%20MoE%20%E7%AE%97%E5%AD%90%E5%85%A5%E9%97%A8%EF%BC%9A%E4%BB%8E%20Benchmark%20%E9%AA%8C%E8%AF%81%E5%88%B0%20XPU-OJ%20%E6%8E%A5%E5%8F%A3%E6%8F%90%E4%BA%A4.md)。
- [TileLang MoE 测试指南](materials/op_optimization/%E5%9F%BA%E4%BA%8E%E5%9B%BD%E4%BA%A7%E8%BD%AF%E4%BB%B6%E6%A0%88%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E5%89%8D%E6%B2%BF%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/race_tests_run_guide%E5%9F%BA%E4%BA%8Etilelang%E7%AE%97%E5%AD%90sample%E8%B7%91%E9%80%9A%E6%B5%8B%E8%AF%95.md)。

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

也可以直接运行：

```bash
cd benchmarks/tilelang-moe
python fusedmoe_benchmark.py
```

优化边界：保持 `RoutedMoEKernel.__init__` 与 `RoutedMoEKernel.__call__` 的外部接口稳定，主要修改 `custom_fusedmoe.py` 内部 kernel 实现；每次改动后依次执行正确性测试、benchmark 并记录结果。

当前默认 schedule 已固化报告中的已验证策略：`FullRow`、stage-1
`row8`/stage-2 `row16`、gate/up 单 shared weight buffer 和 stage-1
`T.serial`。运行时的 autotune 只在这组已通过 functional 的 swizzle
候选中测量；autoheuristic 按公开 shape 选择候选集，不会重新启用报告中
已排除的深 pipeline、64/256 tile、128/512 threads 或 fast-math 方案。

需要复现单个候选时：

```bash
python benchmarks/tilelang-moe/tune_moe.py \
  --block-dhidden 128 --block-dexpert 128 --mode all
```

提交 ABI 的本地对拍：

```bash
scripts/test-moe-submission.sh --public-shape
```

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
