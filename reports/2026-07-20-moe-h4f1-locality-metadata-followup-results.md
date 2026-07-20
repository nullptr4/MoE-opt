# H4F1 locality/metadata 后续：真实 MetaX C500 收敛结果

日期：2026-07-20（UTC）

## 结论

本轮没有确定性胜者，H4F1 保持唯一正式内核，`custom_fusedmoe.py` 字节不变。H7F1–H7F4
全部先通过官方 functional Large/Small（0 mismatch），再完成 warmup 10 / timing 100 /
outlier `none` 与各自独立的四-kernel mcProfiler。没有候选的 mean 与 median 同时提升 0.5%，
所以第二次完整复验和 ABBA 没有触发；这不是缺失证据。

- 保留 kernel-origin commit：`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`。
- 保留 SHA-256：`2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。
- schedule：combined Gate/Up；FC1/FC2 `BN128/BK64/stage1`；FC2 A shared；route weight
  FP16 fragment cache。
- H7 初始控制提交：`5fd7330c793b809a7e00a68bc297ffd16a76ba17`；H7F4 在硬件前登记的
  控制提交：`813bda36864c8ed45778e12cc5c93eff79b271ca`。

## 协议与环境

RoutedMoEKernel API、submission `run_kernel` ABI、官方 seed `81394`、FP16 input、FP32
accumulate、`atol=rtol=0.01`、`equal_nan=false` 均未改变。每个正式结果包含 Large+Small
combined 的全部 100 个有序样本，没有剔除。设备是 MetaX C500 64 GiB，driver `3.8.30`，
MXMACA `3.7.1.5`，mxcc `1.0.0@d9102a1572`，PyTorch `2.8.0+metax3.7.1.3`，
TileLang-MACA `0.1.12+maca.gitec48829b@ec48829b`，mcProfiler
`3.8.1.4+575f5a9f6d`。统一环境 fingerprint 为
`sha256:83e6664bde009322d40a27b6b223217c7e84d6f3a1e395406571f206ee96d125`；P-state、
温度和功耗保存在每次 artifact 中，时钟未由操作员锁定。

## Commit-matched H4F1 基线

| 绑定候选 / agent commit | mean / median / stddev / p90 (ms) | correctness | profiler FC1 / FC2 cycles (Kcycles) | artifact |
|---|---:|---|---|---|
| H7F1–F3 / `5fd7330c` | 238.726042 / 238.819712 / 1.100336 / 240.101477 | Large/Small 0 mismatch | `126959.58,126525.98` / `53370.19,52271.97`；private 0 | `baseline-3bcebcbb48b8153e1dfa1234.json` |
| H7F4 / `813bda36` | 238.650600 / 238.622210 / 1.019817 / 239.951431 | Large/Small 0 mismatch | `126882.39,126383.89` / `51365.43,51189.43`；private 0 | `baseline-165915262a652bd3a0ea0ad7.json` |

两次 baseline 都验证相同目标 commit、kernel SHA 和 schedule；候选只与其绑定 baseline 比较。

## 所有真实候选

正数表示候选相对绑定的 H4F1 更快。

| 候选 | mean / median / stddev / p90 (ms) | mean / median 提升 | mcProfiler 归因 | 决定 |
|---|---:|---:|---|---|
| H7F1 metadata shared broadcast | 331.902600 / 331.842567 / 0.540929 / 332.543743 | -39.0307% / -38.9511% | FC1 `180566.10/180013.45 Kcycles`，新增 private read/write 各 `197568`；FC2 `91495.91/91445.08`。20-byte shared 并不轻量：barrier/shared lowering 造成 spill 和决定性回退 | 拒绝 |
| H7F2 FC1 fragment `T.copy` | 238.674207 / 238.771456 / 0.897687 / 239.800034 | +0.0217% / +0.0202% | FC1 `126492.85/126321.60 Kcycles`，private 0；生成合法但与手写 fragment move 等价，未形成资源收益 | 拒绝，低于门槛 |
| H7F3 FC2 column16 follow-up | 238.985258 / 238.932739 / 0.800247 / 239.900626 | -0.1086% / -0.0473% | FC2 `53117.45/51691.34 Kcycles`；Global Read event 异常降到 `2513/2617`，与正常约 59M 量级不兼容，不作 read 归因；endpoint 明确未赢 | 拒绝，历史方向继续关闭 |
| H7F4 route global→fragment `T.copy` | 238.516525 / 238.357506 / 0.830786 / 239.585053 | +0.0562% / +0.1109% | FC2 `51333.68/52689.10 Kcycles` 对 baseline `51365.43/51189.43`，一好一坏；private 0，reads 仍约 59.46M，机制不稳定 | 拒绝，低于门槛 |

四个候选 Large/Small 的最大绝对误差分别为 `0.005859375/0.0078125`，mismatch 均为 0。
完整 source/diff/profiler provenance：

- H7F1：experiment `experiment-3d33d6b2d0c7894a8ac9033a`，source commit
  `fc213c0d8c9899f25bcab22d7e71741105d3fbe4`，diff
  `sha256:08390e4825d9a7286fab54947c6731c0ffdf237c8ca0717717319c03e8b23ce1`，
  artifact `experiment-7b1816a96c0b8a104d9c6271.json`；
- H7F2：experiment `experiment-264fee2cbc140dba2d07c5a7`，source commit
  `9c7aaa2231f4c5e23a899954c7e3f7f133b986d6`，diff
  `sha256:f0eaa45b91fb7105c6fa144fd9cff5f0eace18b4fbb04f1078fe1ccadaceebc9`，
  artifact `experiment-19adbe3770d3c9ef194480a2.json`；
- H7F3：experiment `experiment-8116a8c3702d243c8d817139`，source commit
  `02ae77b1f94891dbbac85f5995217874b20886a5`，diff
  `sha256:f626248e5ceabe550f631cf0a0806a6068aab1c640dd9f9dff7b95fc3093a63c`，
  artifact `experiment-db1eb31cb269298204a39d00.json`；
- H7F4：experiment `experiment-92a52cfc7082f9fb87071a05`，source commit
  `bebb143c4a5ba0350d6c2f62f076cea3a96b9503`，diff
  `sha256:051d2e2c57f73a96b6a5166f8c8a60ee7753aac6f29ee104431fc383e42180d8`，
  artifact `experiment-1fd3ad3ba34cfb304eb22a69.json`。

全部工件位于 `/data/metax-c500-autotune-agent/artifacts/p9/`，并引用隔离 worktree patch、
完整 100 样本、命令 stdout/stderr、fingerprint 与 profiler raw DB/native/per-kernel JSON。

## Profiler 边界与收敛

profile workload 是 `performance-large` 连续两次 `custom_kernel`：launch 0/2 是 FC1，1/3
是 FC2，不可改标成 Large/Small。native export 的 wall duration 不可信，正式性能只来自
10/100 event timing。H7F3 的 Global Read event 出现已知量级不一致，原始值保留但不跨 capture
比较。

本轮实测关闭 CTA metadata broadcast、FC1 fragment-region lowering、H4F1 后 column locality
复审、FC2 route global→fragment copy；双 output tile/CTA 由既有 spill/serialization 硬边界排除。
结合 H5/H6，当前 fixed ABI 下的 epilogue/predicate/copy、metadata reuse/cache、layout/padding、
pipeline/stage/occupancy、persistent 和相邻 schedule 方向均已实测或有安装后端硬边界。未来仅在
TileLang-MACA 获得新的 async/L2/resident-grid/descriptor 能力，或 profiler 提供可靠新资源因果
证据时重开；不得仅重命名历史候选。
