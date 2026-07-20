# H8F1 shared-resource follow-up：真实 MetaX C500 结果

日期：2026-07-20（UTC）

## 结论

H10F5 是可重复胜者，已落地为代码提交
`e60f9b3216d008e342d285cd24f8b6ede1142387`。新内核 SHA-256 为
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`。它保留 H8F1
schedule，将 combined FC1 K mainloop 后已经死亡的 256×64 FP16 weight shared buffer 显式复用为
两个 affine 128×64 Up 半块，并按安装的 MACA inferencer 约束让每个 `T.Parallel` 只有一个访问模式。

两次正式测量相对同 fingerprint H8F1 的 mean/median 提升分别为
`1.7781%/1.8190%` 和 `1.7455%/1.7636%`。相邻 ABBA 聚合提升
`2.0129%/2.0069%`，两个 challenger 位置的 mean/median 均独立超过 0.5%。

## 正式协议与环境

所有正式 endpoint 是 official performance Large+Small combined，warmup 10、timing 100、outlier
`none`，全部有序样本保存在 agent artifact。正确性是 official functional Large/Small，seed
`81394`、FP16 input、FP32 accumulate、`atol=rtol=0.01`。设备为 MetaX C500 64 GiB；driver
`3.8.30`、MXMACA `3.7.1.5`、mxcc `1.0.0@d9102a1572`、PyTorch
`2.8.0+metax3.7.1.3`、TileLang-MACA `0.1.12+maca.gitec48829b@ec48829b`、mcProfiler
`3.8.1.4+575f5a9f6d`。profile workload 是 performance-large 的 FC1/FC2/FC1/FC2 四个 launch。

## 候选与 profiler 归因

正数表示相对各自同 fingerprint H8F1 更快；cycles 单位为 Kcycles。

| 项目 | mean / median / stddev / p90 (ms) | mean / median 提升 | FC1 / FC2 cycles | 结论 |
|---|---:|---:|---|---|
| H8F1 B0 | 233.532099 / 233.414655 / 1.166980 / 235.230923 | 对照 | FC1 121129.26/121543.71；FC2 53075.61/50921.15 | 第一批同指纹基线 |
| H10F1 | 265.362110 / 265.479809 / 0.805528 / 266.292663 | -13.6298% / -13.7374% | FC1 121564.81/121946.93；FC2 84274.24/84000.75 | FC2 workgroups 57624→115248，拒绝 |
| H10F2 | 232.719478 / 232.823300 / 0.996454 / 234.102914 | +0.3480% / +0.2533% | FC1 121185.66/120434.06；FC2 51649.69/52190.95 | 正向但低于双门，拒绝 |
| H10F3 | 295.724057 / 295.663486 / 1.051352 / 296.987724 | -26.6310% / -26.6688% | FC1 183096.61/184006.14；FC2 53000.98/51988.59 | FC1 launch 64 KiB，private 460992/460992，拒绝 |
| H8F1 B1 | 233.895931 / 233.694334 / 1.288645 / 235.621656 | 对照 | FC1 122304.04/123069.67；FC2 53791.94/52752.33 | H10F4 control commit 后重采 |
| H10F4 | build failed | 不适用 | profiler 未运行 | `StructuralEqual` 检出同 Parallel 双 offset；错误证据保留 |
| H8F1 B2 | 233.547751 / 233.564417 / 1.058403 / 234.702563 | 对照 | FC1 121629.69/122787.31；FC2 50925.64/51738.37 | H10F5 control commit 后重采 |
| H10F5 R1 | 229.395087 / 229.315837 / 1.032849 / 230.786362 | +1.7781% / +1.8190% | FC1 118785.30/117979.48；FC2 51477.94/51265.22 | 0 mismatch，初胜 |
| H10F5 R2 | 229.471245 / 229.445374 / 1.296601 / 231.160116 | +1.7455% / +1.7636% | FC1 117503.74/117449.18；FC2 51486.89/51006.23 | 0 mismatch，完整复验通过 |

H10F5 的生成 host launch 保持 FC1/FC2 dynamic shared 均为 `32768 B`。两次 profiler 中 FC1
cycles 都下降，而 FC2 基本不变，和 endpoint 机制一致。FC1 private read/write event 上升到
`921984/921984`；这是同 capture 的真实代价，但没有抵消更短的 FC1 临界路径。不同 single-pass
batch 的 global/private event label/scale 仍不跨 capture 解释，Total Cycles 与生成 launch 是主归因。

## 复验与 ABBA

ABBA artifact 为 `abba-c8a3af23b9c375b7de656015.json`，固定顺序
H8F1/H10F5/H10F5/H8F1，每段独立 warmup 10、timing 100、无样本剔除：

| 位置 | mean / median / stddev / p90 (ms) | RSD |
|---|---:|---:|
| H8F1 A1 | 233.839549 / 233.856131 / 1.107019 / 235.170156 | 0.4734% |
| H10F5 B1 | 229.247626 / 229.199871 / 1.049052 / 230.586606 | 0.4576% |
| H10F5 B2 | 229.366384 / 229.354115 / 1.196691 / 230.878612 | 0.5217% |
| H8F1 A2 | 234.195600 / 234.089087 / 1.086331 / 235.840993 | 0.4639% |

B1 相对 A1 的 mean/median 为 `+2.0383%/+2.0399%`，B2 相对 A2 为
`+1.9875%/+1.9739%`；聚合为 `+2.0129%/+2.0069%`。四段方差稳定，严格门选择 H10F5。

## Provenance 与目标验证

H10F1–H10F5 的 experiment ID、candidate commit、diff SHA、完整样本、环境指纹、correctness 和
raw mcProfiler 路径均在
[`h8f1-resource-reuse-followups-20260720.json`](../data/benchmarks/c500-64g/h8f1-resource-reuse-followups-20260720.json)
中。H10F5 R1/R2 使用相同 diff SHA
`a8902e9472353f8135446e21e94edbc7608c8669e5b3983c1738230aed41267e`，candidate commits
分别为 `9eed5becdd9642075365d141cb30acebb017a03c` 与
`bb500056076e49e97d1f302265486260874db612`。

目标仓库从新代码执行：

- `CAMP_TEST_BACKEND=metax python -m pytest -q`：`86 passed in 25.06s`；
- `python scripts/check_repository.py`：passed（3365 tracked files）；
- `scripts/verify-maca.sh`：MetaX C500 passed；
- `scripts/run-moe.sh`：official Large/Small functional 2/2 passed；diagnostic timing
  `198.72535156/31.86319824 ms`，不作正式对照；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：6/6 passed；
- Stage-2 table：9 versions、32 hashed sources，确定性重建通过；
- `git diff --check`：passed。

未解决方向只包括未来 TileLang-MACA 新增可验证的 async/layout lowering、可靠 resident-grid 或
bank/occupancy counters，以及 fixed ABI 新增 problem descriptor/workspace。H10F1–H10F4 和 H10S1
已有硬件或 ABI 证据收敛，不应改名重跑。
