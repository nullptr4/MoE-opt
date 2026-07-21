# H10F5 epilogue/exchange follow-up：真实 MetaX C500 结果

日期：2026-07-21（UTC）

## 结论

本轮没有候选同时达到 mean 和 median 至少 0.5% 的接受门。因此不允许第二次复验、
ABBA 或内核落地；H10F5 保持唯一不可变胜者，源码 SHA-256 仍为
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。

## 协议与环境

每个可编译候选先运行 official functional Large/Small，seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01`；之后是 official performance Large+Small，warmup 10、timing
100、outlier `none`，全部有序样本保存在 agent artifact。profiler workload 是
performance-large 连续两次，对应 FC1/FC2/FC1/FC2 四个 launch。

设备为 MetaX C500 64 GiB；driver `3.8.30`、MXMACA `3.7.1.5`、mxcc
`1.0.0@d9102a1572`、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MACA
`0.1.12+maca.gitec48829b@ec48829b`、mcProfiler `3.8.1.4+575f5a9f6d`。

## 正式实测

正数表示相对同 control commit 和 fingerprint H10F5 基线更快；cycles 单位 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | FC1 / FC2 cycles | 结论 |
|---|---:|---:|---|---|
| H10F5 B0 | 229.100318 / 229.186051 / 0.970742 / 230.164862 | 对照 | FC1 118695.26/118150.29；FC2 51529.45/52402.17 | H11F2–F4 同指纹基线 |
| H11F2 contiguous shared halves | 238.322195 / 238.449923 / 1.163097 / 239.788876 | -4.0253% / -4.0421% | FC1 127722.21/126875.77；FC2 51244.69/53075.34 | 拒绝 |
| H11F3 streamed halves | 232.792350 / 232.662912 / 1.185411 / 234.603903 | -1.6115% / -1.5170% | FC1 120867.78/120111.98；FC2 51912.88/51242.44 | 拒绝 |
| H11F4a bare-int hint | build failed | 不适用 | profiler 未运行 | `coalesced_width should be an IntImmNode` |
| H11F4b `T.int32(8)` | 229.274286 / 229.282429 / 1.098670 / 230.438125 | -0.0759% / -0.0421% | FC1 118096.23/118512.22；FC2 54153.68/54017.88 | 中性/慢，拒绝 |
| H10F5 B1 | 228.630523 / 228.828800 / 1.117566 / 229.951815 | 对照 | FC1 117705.83/118469.63；FC2 51286.53/54008.81 | H11F5 同指纹基线 |
| H11F5 activated-Gate shared | 229.156060 / 229.072515 / 1.109690 / 230.589820 | -0.2299% / -0.1065% | FC1 118729.94/117263.80；FC2 52188.87/52903.39 | 拒绝 |

H11F2 的 contiguous layout 将 FC1 private reads 推到约 `30.56M`，private writes 为
`1,185,408`，并与 4% 稳定回归一致。H11F3 提高 FC1 cycles；其单通 profiler 的
global/private write event 标签选到不兼容的小 scale，原始值保存但不跨 capture 解释。
H11F4b 的生成 store 仍是 `uint4`，private read/write 仍为 `921984/921984`，说明
新 hint 没有改变已有 lowering。H11F5 的 FC1 cycles 与基线交叉，private writes 不变，
无 profiler 机制支持。

H11F1、H11S1–H11S3 的静态硬边界见调研报告。H11F4a 在 build 终止，所以未伪造
correctness、timing 或 profiler。其他四个可执行候选全部 Large/Small `0/0` mismatch，
并各有独立四 launch mcProfiler。

## Provenance

完整 experiment ID、candidate commit、diff SHA-256、environment fingerprint、100 个有序样本和
raw profiler 路径在
[`h10f5-epilogue-exchange-followups-20260720.json`](../data/benchmarks/c500-64g/h10f5-epilogue-exchange-followups-20260720.json)。
受控 agent commit `d812709`、`9c9fcbf` 和 `9130654` 均早于对应硬件批次。

ABBA 未执行：没有任何可执行候选通过首次 mean+median 0.5% 双门，执行复验或 ABBA
会违反预先接受规则。本轮只落地收敛证据，未修改 `custom_fusedmoe.py`。

## 目标验证

- `CAMP_TEST_BACKEND=metax python -m pytest -q`：`86 passed in 25.23s`；
- `python scripts/check_repository.py`：通过（初次检查 3371 tracked files；新报告入 index
  后再检查）；
- `scripts/verify-maca.sh`：通过，识别 MetaX C500、driver `3.8.30`、MACA `3.7.1.5`；
- `scripts/run-moe.sh`：official functional Large/Small `2/2` 通过；diagnostic timing
  `199.24789062/31.89778809 ms`，不作正式性能对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：`6/6` 通过；
- `git diff --check`：通过。

未解决方向仅保留“未来新的公开 MetaX resident-grid/bank/occupancy 事实”、“TileLang-MACA
新增已验证 async/layout lowering”或“fixed ABI 增加 workspace/problem descriptor”。已收敛 H11
方向不应改名重跑。
