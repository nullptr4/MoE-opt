# H4F1 后续：真实 MetaX C500 收敛结果

日期：2026-07-20（UTC）

## 结论

本轮没有确定性胜者，H4F1 保持唯一正式内核。五个实际候选全部先通过官方 functional Large
和 Small（均 0 mismatch），再完成 warmup 10 / timing 100 / outlier `none` 和独立
mcProfiler；没有候选的 mean 与 median 同时达到 0.5%，所以按门禁没有执行第二次完整复验或
ABBA，也没有把候选带回 `custom_fusedmoe.py`。

- 保留的内核提交：`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`。
- 保留的内核 SHA-256：
  `2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。
- schedule：combined Gate/Up；FC1 `BN128/BK64/stage1`；FC2 `BN128/BK64/stage1`；FC2 A
  shared；route weight FP16 fragment cache。
- agent 在硬件前提交 H4F1 控制面：`aa9e252c987357af327353e01badcf83b6c6ecd3`；H5F5
  measured decomposition 注册提交为 `ebd8aa47d1315600851fba2fc37abbe168c8ee9a`，并为该提交
  重新采集 commit-matched H4F1 baseline。

## 正式协议与环境

RoutedMoEKernel API、十参数 submission `run_kernel` ABI、官方 shape/seed、FP16 输入、FP32
accumulate、`atol=rtol=0.01`、`equal_nan=false` 均未改变。每个正式 timing 保留 100 个有序
Large+Small combined samples，不剔除样本。设备为 MetaX C500 64 GiB，driver `3.8.30`，
MXMACA `3.7.1.5`，mxcc `1.0.0@d9102a1572`，PyTorch `2.8.0+metax3.7.1.3`，
TileLang-MACA `0.1.12+maca.gitec48829b@ec48829b`，mcProfiler `3.8.1.4+575f5a9f6d`。
时钟未由操作员锁定；各 artifact 保存 timing 后 P9 状态、温度和功耗。

首批环境指纹为
`sha256:1d38be1c05a0ca5d8d9503e8e3d4656660b74d4544b1f666f26f7d0c866b54f9`；H5F5
提交匹配批次为
`sha256:f9eab42b3f85480c57e0b42b0d3f3f54fc30056982958ab1b523791aa1807f8e`。
不同指纹之间不做 candidate/baseline 交叉比较。

## 新采集 H4F1 基线

| artifact | mean / median / stddev / p90 (ms) | correctness | 用途 |
|---|---:|---|---|
| `baseline-c10d645046d626f648eb64db.json` | 237.934242 / 237.862917 / 0.997732 / 239.304448 | Large/Small 0 mismatch | H5F1–H5F4 唯一对照 |
| `baseline-1e927f756da6697f4aa70681.json` | 238.132234 / 238.109701 / 1.194435 / 239.646615 | Large/Small 0 mismatch | H5F5 commit-matched 唯一对照 |

两份 baseline 均保存 100 个有序样本和完整 profiler 原始 DB/native/JSON/invocation。

## 所有真实候选

正数表示相对各自绑定的 H4F1 baseline 更快。

| 候选 | mean / median / stddev / p90 (ms) | mean / median 提升 | profiler 归因 | 决定 |
|---|---:|---:|---|---|
| H5F1 full FC2 route+store no-predicate | 238.285271 / 238.191618 / 0.882504 / 239.347329 | -0.1475% / -0.1382% | FC2 两次 `52187.60/51612.83 Kcycles`，reads `59465887/59459646`，private 0/0；采样 cycles 降低但端到端回退 | 拒绝 |
| H5F2 FC1 full-tile `T.copy` | 237.997911 / 238.133760 / 0.967824 / 239.183799 | -0.0268% / -0.1139% | 四 launch write instructions 变为 `5/160/5/158`，但 endpoint neutral/slower；private 0/0 | 拒绝 |
| H5F3 FP32 route fragment | 240.587510 / 240.550398 / 0.655282 / 241.343409 | -1.1151% / -1.1298% | FC2 `55861.05/55166.25 Kcycles`，L2 `77.95%/77.17%`，无 spill；更宽 fragment 增加代价 | 拒绝 |
| H5F4 FP32 scale + full-tile `T.copy` | 239.939415 / 239.976955 / 0.608828 / 240.694370 | -0.8427% / -0.8888% | FC2 `56096.84/54954.26 Kcycles`，write instructions `147/150`，private 0/0 | 拒绝 |
| H5F5 route-load-only no-predicate | 237.990932 / 237.723904 / 1.148941 / 239.719012 | +0.0593% / +0.1620% | FC2 `51665.72/53946.25 Kcycles`，private 0/0；read event 与 baseline 口径异常，未用作机制证明 | 拒绝，低于 0.5% |

每个候选的 Large 最大绝对误差为 `0.005859375`、Small 为 `0.0078125`，mismatch 均为 0。
完整 experiment/source/diff：

- H5F1：`experiment-b5e9cb5c416a3b87820931a1`，commit `e100c0b0`，diff
  `f0bc5e7a...`，artifact `experiment-faf4d3e081aed153694f6144.json`；
- H5F2：`experiment-9bb107378d9538cbb1357476`，commit `fe318c35`，diff
  `add1bc59...`，artifact `experiment-128698788749e7bb3b2e0916.json`；
- H5F3：`experiment-43e2342b2111415b4545929f`，commit `968935ca`，diff
  `86d207a8...`，artifact `experiment-824a64c75f794e20bc2ca51b.json`；
- H5F4：`experiment-31c6c87e5820b2b235ffc891`，commit `ef3866c1`，diff
  `f029e0c1...`，artifact `experiment-ad0541be1f6415d6d47b5009.json`；
- H5F5：`experiment-fa9b18e8ddad63d85f0c0f98`，commit `f1ee1515`，diff
  `e913f42f...`，artifact `experiment-d94361d1465eac8e0235969d.json`。

## Profiler 解释边界

driver 的 profiler workload 是 `performance-large` 连续两次 `custom_kernel`，所以 launch
0/2 是两次 FC1，1/3 是两次 FC2；不能把它误写成 Large/Small 各一次。functional Small 的
正确性仍由独立 correctness stage 证明。native export 没有可信 wall duration，性能结论只用
正式 event timing。

H5F5 批次中 baseline 的 `Global Read Instructions` 意外为 `608/2501/599/2499`，候选则回到
约 `118M/59M` 的常见量级，说明该 event 在两次 mcProfiler capture 间选择/复用不一致。原值和
原始 DB 均保留，但不把不可比 counter 当成 H5F5 的支持；该候选仅凭正式 mean/median 已明确
低于接受门。

## 搜索收敛

本轮实测关闭了：完整 tile route-load/store predicate specialization、FC1 tile-store lowering、
route fragment FP32 representation、FC2 two-phase scaled tile store，以及 H5F1 的 route-load-only
分解。静态证据关闭了 clear 写法替换、无 C500 bank 依据的手工 swizzle/padding、未证实的
async/TMA/warp-specialized/persistent/subtile。历史 BN256、BK32/BK128、深 pipeline、单 CTA
persistent、强制 occupancy 仍有 spill 或大幅 cycles 回退，而 H4F1 没有缩小造成这些问题的
accumulator，因此不重扫参数网格。

没有候选进入初胜门，所以“第二次完整 correctness+10/100+profiler”和 ABBA 均为未触发，
不是缺失证据。最终不修改 H4F1 内核。
