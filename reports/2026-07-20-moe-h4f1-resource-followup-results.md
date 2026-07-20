# H4F1 资源后续：真实 MetaX C500 收敛结果

日期：2026-07-20（UTC）

## 结论

本轮没有确定性胜者，H4F1 保持唯一正式内核，`custom_fusedmoe.py` 未改。五个可编译候选
均先通过官方 functional Large/Small（0 mismatch），再完成 warmup 10 / timing 100 /
outlier `none` 与独立四-kernel mcProfiler；另两个候选在 build 硬边界失败，因此按 correctness-first
规则没有伪造 timing 或 profiler。没有候选的 mean 与 median 同时提升 0.5%，故第二次完整复验和
ABBA 均未触发。

- 保留 kernel-origin commit：`61fc0bbdb4a08c0a4ba44af278606ce362b794ce`。
- 保留内核 SHA-256：
  `2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。
- schedule：combined Gate/Up；FC1 `BN128/BK64/stage1`；FC2 `BN128/BK64/stage1`；
  FC2 A shared；route weight FP16 fragment cache。
- hardware 前控制提交：`11870e3efa92cfb2bf958afb7c1d275f9bec697e`；第二批 hypothesis 在
  hardware 前提交 `8cb3e1a1c7a4e374460d83180ac9b46de7bd8900`。

## 协议与环境

RoutedMoEKernel API、十参数 submission `run_kernel` ABI、官方 shape/seed `81394`、FP16 input、
FP32 accumulate、`atol=rtol=0.01`、`equal_nan=false` 均未改变。正式 timing 保留全部 100 个
Large+Small combined 有序样本，不剔除样本。设备为 MetaX C500 64 GiB，driver `3.8.30`，
MXMACA `3.7.1.5`，mxcc `1.0.0@d9102a1572`，PyTorch `2.8.0+metax3.7.1.3`，
TileLang-MACA `0.1.12+maca.gitec48829b@ec48829b`，mcProfiler
`3.8.1.4+575f5a9f6d`。时钟未由操作员锁定；artifact 保存 P-state、温度和功耗。

## 两个 commit-matched H4F1 基线

| 用途 / fingerprint | mean / median / stddev / p90 (ms) | correctness | artifact |
|---|---:|---|---|
| H6F1–F4；`50e67537…` | 237.879017 / 237.792130 / 0.974774 / 239.125117 | Large/Small 0 mismatch | `baseline-b1eff955dcc26fd989d2c2f3.json` |
| H6F5–F7；`5254eadc…` | 238.703931 / 238.472956 / 1.094809 / 240.270570 | Large/Small 0 mismatch | `baseline-5fd6579f75e81a9cb94a8d41.json` |

第二次 capture 比第一次慢约 0.35%，所以不同 fingerprint/commit 批次之间不做候选交叉比较。

## 所有真实候选

正数表示相对各自绑定的 H4F1 更快。

| 候选 | mean / median / stddev / p90 (ms) | mean / median 提升 | mcProfiler 归因 | 决定 |
|---|---:|---:|---|---|
| H6F1 FC2 min-blocks=2 | 238.015921 / 238.018177 / 0.947712 / 239.253423 | -0.0576% / -0.0951% | 生成 FC1 `(256,1)`、FC2 `(256,2)`；FC2 `54202.04/54067.57 Kcycles`，private 0/0；无稳定 residency 收益 | 拒绝 |
| H6F2 FC2 stride-65 padding | 490.892325 / 490.924166 / 0.523891 / 491.595749 | -106.3622% / -106.4510% | FC2 `303827.91/303494.46 Kcycles`，reads `472516488/472460736`；切片/padded layout 破坏高效 load/GEMM lowering | 拒绝，决定性回退 |
| H6F3 FC1 no-copy accumulator view | 未运行 | 未运行 | build 报 `StructuralEqual`：同一 fragment 的 `(i,1,j)` 与 `(i,0,j)` layout access 不相等；correctness/timing/profile 按规则不运行 | 拒绝，编译硬边界 |
| H6F4 FC2 stage2 | 284.358420 / 284.241022 / 0.603923 / 285.014837 | -19.5391% / -19.5334% | FC2 `98134.59/98133.58 Kcycles`，reads 仍约 `59.45M`，private 0/0；额外 stage 近乎倍增 cycles | 拒绝 |
| H6F5 FC2 serial | 239.592131 / 239.620098 / 0.707610 / 240.341988 | -0.3721% / -0.4810% | FC2 `54778.78/54817.35 Kcycles`，private 0/0；Global Read event 为异常小量级 `2433/2450`，只保留不比较 | 拒绝 |
| H6F6 metadata `T.__ldg` | 未运行 | 未运行 | build fatal：`T.__ldg expects a BufferLoad as the first argument`；后续 pass 未给 MACA codegen 保留裸 BufferLoad | 拒绝，backend 硬边界 |
| H6F7 padded-offset reuse | 238.707221 / 238.868994 / 1.120473 / 240.050072 | -0.0014% / -0.1661% | 生成代码每 kernel 从两次降为一次 padded-offset load，但 FC2 reads `59465783/59466404`、private 0/0，endpoint 无收益 | 拒绝 |

H6F1/F2/F4/F5/F7 的 Large 最大绝对误差均 `0.005859375`，Small 均 `0.0078125`，mismatch
均为 0。experiment/source/diff 与全部有序样本：

- H6F1：`experiment-baaecae916bb896eb4182dc4`，commit `c639ad11`，diff `fc57d6ed…`，
  `experiment-a7d2ba6c86e5b980cee4f4dc.json`；
- H6F2：`experiment-40792914f4597548ca641da3`，commit `b5e2250f`，diff `b52cd9aa…`，
  `experiment-5d92b4aeee67a4b3054e6f89.json`；
- H6F3：`experiment-a0005d5e9f6e1e03b82a2b17`，commit `91113549`，diff `53eeb627…`，
  `experiment-5a6a855c5f926f75166ab4eb.json`；
- H6F4：`experiment-f2b2123d47be19cfdb464a1b`，commit `f310a145`，diff `7e644cb1…`，
  `experiment-80b3a2d3f44963aa391e9afc.json`；
- H6F5：`experiment-24b2346503166ea3ea43bcdd`，commit `2da97734`，diff `652cf48b…`，
  `experiment-f7fdf72b8bd9f718f18fe881.json`；
- H6F6：`experiment-53ba641ca34c68caf26ade0a`，commit `54ab8754`，diff `0dca64c0…`，
  `experiment-82050854ab0c0f90756ca4dd.json`；
- H6F7：`experiment-3386c95dc719924d8ccea96c`，commit `6af66591`，diff `f4a27ba4…`，
  `experiment-2ed200ad372e0396f1bdf2bd.json`。

所有 artifact 位于 `/data/metax-c500-autotune-agent/artifacts/p9/`，并引用隔离 worktree 的
proposal patch、staged diff、source commit、命令 stdout/stderr、环境 fingerprint 与 profiler
raw DB/native/per-kernel JSON/invocation。

## Profiler 解释边界与收敛

profiler workload 是 `performance-large` 连续两次 `custom_kernel`：launch 0/2 为 FC1，1/3
为 FC2，不是 Large/Small 各一次。native export 没有可信 wall duration，性能只用正式 event
timing。部分 capture 的 `Global Read Instructions` 偶发为数百/数千而不是常见 118M/59M；
原值全部保留，但只在同 event scale 时作机制比较。

本轮关闭的主要 fixed-ABI 方向包括：H4F1 后 FC2 occupancy constraint、物理 padding、FC1
auxiliary-fragment elimination、stage2、stage1 serial、metadata read-only cache 和 duplicate metadata
load reuse。async copy、等价显式 swizzle、无 resident metric 的 persistent、无 MACA lowering 的
L2 persistence 均有静态/实测硬证据排除。历史 A1d–A1g、H1、H2 单体、H3、H5 和大 schedule
grid 未重复。

没有候选过初门，所以第二次 correctness+10/100+profiler 与 ABBA 未触发，不是缺失步骤。最终
不修改 H4F1 内核。
