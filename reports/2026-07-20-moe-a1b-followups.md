# C500 A1b 后续优化收敛报告

## 结论

本轮把 target commit `0935d29a0bcbd789de8dcd499dd4e1fac15ae692`、内核 SHA-256
`581a00710cfe04b75a7da2bda8aa9fd273f9abfad9030fcf8c00191bc6fe8a27` 显式作为不可变
A1b 基线，在同一台真实 MetaX C500 64 GiB 上评估了四个 A1b 后续候选。四个候选都先通过
官方 Large/Small functional，再完成 warm-up 10、连续 100 个样本、无样本剔除的正式
Large+Small 计时，并各自取得了可解析的 mcProfiler 证据。

没有候选同时达到均值和中位数至少 0.5% 的门槛，因此没有胜者、没有带回拒绝补丁，也没有
把小于噪声的提升冒充优化。目标 `custom_fusedmoe.py` 保持 A1b 原字节不变。A1c 已有
A1b/A1c ABBA 负证据，本轮没有重复。

## 测量口径

- 官方 seed `81394`，functional `atol=rtol=1e-2`。
- functional Large：`H7168/I2048/E8/top4/bs1/seq8192`；Small：
  `H3584/I1024/E4/top2/bs2/seq4096`。
- performance Large/Small 分别为 `bs4/bs8`；每次 10 warm-up、100 timed iterations，
  保存全部有序样本，不剔除样本。
- A1b schedule：combined Gate/Up logical `BN128`、physical `BN256`、`BK64`；FC2
  `BN128/BK64/stage1`；stage-1/stage-2 row swizzle 为 `8/16`。
- 接受必须同时满足正式均值和中位数提升至少 0.5%，且 RSD 不异常；胜者还必须做第二次
  correctness+10/100+profiler 和相邻 A/B 或 ABBA。本轮没有初筛胜者，后两项不适用。
- 环境 fingerprint：
  `sha256:7ba1381313d23e9a1e1d9dcb225e5844628d2d8b0876db5adbaa1df5f85ebdc0`。
- MetaX C500 64 GiB，mx-smi `2.3.1`，driver `3.8.30`，MACA `3.7.1.5`，
  Python `3.12.11`，PyTorch `2.8.0+metax3.7.1.3`，
  TileLang `0.1.12+maca.gitec48829b`，mcProfiler `3.8.1.4`。
- 时钟没有由操作者锁定；每个运行记录了 mx-smi performance state、温度和功耗行。

## 正式结果

| 版本 | 改动 | mean (ms) | median (ms) | stddev (ms) | mean / median 相对 A1b | 决策 |
|---|---|---:|---:|---:|---:|---|
| A1b | 不可变基线 | 274.931589 | 274.851322 | 0.481305 | — | 保留 |
| A1d | FC1 `BK64→128`，physical shared 32→64 KiB | 386.015414 | 386.059658 | 0.856902 | -40.4042% / -40.4613% | 拒绝 |
| A1e | 仅 Large stage-1 row4；Small 保持 row8 | 274.611439 | 274.575226 | 0.460818 | +0.1164% / +0.1005% | 低于 0.5%，拒绝 |
| A1f | stage-1 `min_blocks_per_sm=2` | 275.138842 | 275.053057 | 0.486531 | -0.0754% / -0.0734% | 拒绝 |
| A1g | 保持 FP32 helper，消除死 Gate fragment 回写 | 275.450872 | 275.412357 | 0.538783 | -0.1889% / -0.2041% | 拒绝 |

五个版本的 Large/Small correctness 均为零 mismatch；Large/Small 最大绝对误差分别保持
`0.005859375/0.0078125`。所有正式运行 RSD 均约 `0.17%–0.22%`，没有靠异常方差作出
晋级判断。

## Profiler 归因

A1b 的两次 FC1 launch 为 `126057.01/126494.63 Kcycles`，固定 `16464` workgroups、
`65856` waves，private read/write 均为零。A1e、A1f、A1g 保持相同 workgroups/waves，
FC1 cycles 仍在约 `125.7–126.6 Mcycles`，没有形成与 0.5% 提升相称的结构性变化。

A1d 是明确的资源失败：FC1 升至 `220137.54/219667.78 Kcycles`，并在每次 FC1 launch
出现 `262304448` private reads 和 `5136768` private writes；FC2 仍约 `89 Mcycles`。
这证明 combined physical `N256×K128` 的 64 KiB shared tile 与大 FP32 fragment 导致
spill/资源压力，不能用减少 K-loop 次数抵消。

每个 profiler attempt 均为 `parsed`，包含原生 db、JSON、文本输出、调用参数和文件哈希。
机器可读索引、完整 artifact SHA-256、candidate commit/diff hash、有序样本与 profiler 路径见
[`a1b-followups-20260720T112416Z-summary.json`](../data/benchmarks/c500-64g/a1b-followups-20260720T112416Z-summary.json)。

## 搜索收敛

既有证据已经排除线程 128/512、BN 缩小/放大、深流水、FC2 `BK/stage/BN` 扩张、旧
epilogue 合并、empty-tile、低收益 exact metadata 单项，以及普通 W8A8。A1c 的既有 ABBA
中 A1b/A1c 为 `274.897197/275.332660 ms`，A1c 慢 `0.1582%`，故未重复。

本轮又覆盖了 A1b 改变 physical N 后尚未正式闭合的 K tile、swizzle、occupancy hint 和
combined epilogue 死写。剩余方向需要新的算法/ABI、低精度或跨 CTA 同步证据，已不属于
“数值风险低的相邻候选”。因此合理低风险空间在真实硬件上收敛为继续保留 A1b。

## 目标仓库出货验证

在 A1b 原哈希上执行并通过：

- `python -m pytest -q`：使用子项目文档规定的 MetaX C ABI backend 后 `86 passed`。
  首次未配置 backend 的运行因缺少 NVIDIA `libcamp_ops.so` 失败；TileLang 教学 backend
  也暴露 16 个独立 Intro-ops 错误，均如实保留为诊断，不与 MoE 性能证据混合。
- `python scripts/check_repository.py`：PASS，检查 3341 个 tracked 文件。
- `scripts/verify-maca.sh`：PASS，确认 C500/xcore1000。
- `scripts/run-moe.sh`：functional 2/2 PASS；独立出货 performance 为
  Large `238.02052734 ms`、Small `37.77364014 ms`。该次只用于出货验证，不与配对搜索混算。
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 PASS。
- `git diff --check`：PASS。

没有修改 `RoutedMoEKernel` API、submission `run_kernel` ABI、官方 shape/seed/容差，也没有
删除或覆盖用户原有未跟踪文件。
