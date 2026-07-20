# C500 autotune agent：A1b 默认升级记录

## 结论

在 MetaX C500 64 GiB 上，`/data/metax-c500-autotune-agent` 按官方 MoE 协议完成了
E3+E7 基线、候选 correctness、10 次 warm-up / 100 次计时、mcProfiler 和独立复验。
最终把 A1b 升级为 `custom_fusedmoe.py` 的默认配置：Gate/Up 使用 logical `BN128`、
physical `BN256`、`BK64` 的宽 GEMM，FC2 保持 `BK64`。公开 `RoutedMoEKernel` API 和
比赛 submission ABI 均未改变。

目标仓库从 `2563be6d004f856fb4f6089f6cf275b63911a8e6` 开始，内核升级 commit 为
`e32f43e7ed6cb29bf1e1861558c430ac759d38b3`。升级后源码 SHA-256 为
`581a00710cfe04b75a7da2bda8aa9fd273f9abfad9030fcf8c00191bc6fe8a27`。

## 官方口径与环境

- 官方资料 commit：`bed86dafbf479b9bc9ad6f71f9d0aa300bcbf4e6`（`/data/赛题`）。
- seed：`81394`；functional 容差：`atol=rtol=1e-2`。
- functional Large：`H7168/I2048/E8/top4/bs1/seq8192`；Small：
  `H3584/I1024/E4/top2/bs2/seq4096`。
- performance Large / Small 分别使用 `bs4` / `bs8`；每个候选 warm-up 10、连续记录
  100 个 CUDA-event 样本，不剔除样本，官方统计为 Large+Small 算术平均。
- Device：MetaX C500 64 GiB；实测期间 `mx-smi` 报 P9，时钟未由操作者锁定。
- mx-smi `2.3.1`，driver `3.8.30`，MACA `3.7.1.5`，BIOS `1.33.5.0`。
- Python `3.12.11`，PyTorch `2.8.0+metax3.7.1.3`，
  TileLang `0.1.12+maca.gitec48829b`，mcProfiler `3.8.1.4`。
- 环境 fingerprint：
  `sha256:0e438dd422cbfcda8e7d1caee715843c2c67fe4ca57efb5841658e1f2d70e6f3`。

## 结果

| 运行 | mean (ms) | median (ms) | mean 相对基线 | median 相对基线 | correctness | profiler |
|---|---:|---:|---:|---:|---|---|
| E3+E7 baseline | 276.448427 | 276.422268 | — | — | Large/Small 通过 | 基线不用于候选准入 |
| A1b v3 | 274.770781 | 274.697094 | **+0.606857%** | **+0.624108%** | 两形状零 mismatch | 通过，4 次目标 kernel launch |
| A1b v4 独立复验 | 274.887012 | 274.755072 | **+0.564813%** | **+0.603134%** | 两形状零 mismatch | 通过，4 次目标 kernel launch |

v3 的 Large / Small 最大绝对误差分别为 `0.005859375` / `0.0078125`，均在官方容差内。
两个完整接受实验共同生成 verified experience
`experience-85ec953dbed4cd61c94a2e26`，confidence `1.0`。

A1c 把辅助 Up fragment 改为 FP16；它相对 E3+E7 可通过准入，但 A1b/A1c 的 ABBA
对照中，A1b 两轮均值为 `274.897197 ms`，A1c 为 `275.332660 ms`，A1b 快
`0.1582%`。该增量未达到 `0.5%` 替换门槛，因此保留更简单的 A1b。

关键原始记录位于 agent 仓库：

- 基线：`artifacts/p9/baseline-edaa8be21f4fef4b48714f15.json`；
- 首次完整接受：`artifacts/p9/experiment-7c3c6b1b98e89e72d41480d6.json`；
- 独立复验：`artifacts/p9/experiment-838ccee0b4f168de6cd1065f.json`；
- ABBA：`artifacts/p9/final-abba-a1b-vs-a1c.json`；
- 完整报告：`docs/p9-c500-moe-results.md`。

## 带回目标仓库后的独立验证

在 target commit `e32f43e` 上重新执行官方入口，functional Large / Small 为 2/2 PASS；
performance Large 为 `237.32103516 ms`，Small 为 `37.11048584 ms`，合计
`274.431521 ms`。这次运行用于确认带回后的目标入口，不与上述基线/候选配对实验混算。

通过的命令：

```bash
scripts/verify-maca.sh
scripts/run-moe.sh
scripts/test-moe-submission.sh --public-shape --fuzz
python scripts/check_repository.py
python -m pytest tests/test_moe_stage2_profiler.py -q
git diff --check
```

OJ public shape 加四个 fuzz shape 共 6/6 PASS。Stage 2 决策表仅刷新当前
`custom_fusedmoe.py` 的生成时间、SHA-256 和文件大小；历史 E0–E5 数据与结论保持不变。
