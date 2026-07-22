# 使用说明


## 1. 文件定位

`custom_fusedmoe.py` 作为对外暴露的 MoE fused kernel 接入文件。

调用和改写：

1. `RoutedMoEKernel` 的初始化和__call__接口固定；
2. 内部实现可重写调整。

## 2. 使用
默认命令运行 ten-argument submission 的三组 published remote dimensions/timing：
```bash
bash run.sh
```
旧 formal 11-tensor Large/Small 只作为显式 local proxy guard：
```bash
python fusedmoe_benchmark.py --local-proxy-guard
```
## 3. 测试用例
`moe_test_configs.json` schema v2 直接列出默认 `remote_submission` 三 case，并把旧
功能/性能各两个 case 放在 `local_proxy_guard`。checker 要求前者与
`remote_contract.json` 的 order/dimensions/timing 完全一致；后者不得静默回落为默认目标。
当前默认：input float16, output float32

功能测试正确输出
```bash
✅ Functional test passed for config:
```
功能测试错误输出
```bash
❌ Functional test failed for config:
```

## 3. 需安装
torch \
tilelang \
apache-tvm-ffi

## 4. 自动调优

`scripts/activate-maca.sh` 会默认开启 TileLang autotune，并启用面向当前
MoE shape 的 autoheuristic，结果缓存到仓库的 `.cache/tilelang`。默认只
测量报告中已经通过 functional 的 `row8/row16` 与 `row16/row16` schedule，
不会重新搜索已证明较慢或不安全的 tile、thread、pipeline 组合。

```bash
source scripts/activate-maca.sh
scripts/run-moe.sh
```

formal local proxy 调优必须显式选择；需要扩大其搜索空间时使用：

```bash
MOE_AUTOTUNE_MODE=full scripts/run-moe.sh --local-proxy-guard
```

调优报告和原始候选记录位于仓库根目录的 `reports/` 与 `logs/`。正式
`custom_fusedmoe.RoutedMoEKernel` 是带调用方 `up_logits` 的 compact 11-tensor
ABI；OJ 独立入口 `submission.run_kernel` 是私有 workspace 的 compact
group_sum 10-argument ABI。padded scheduling metadata 不代表 padded data storage。
本地 smoke/remote-dimension 对拍使用：

```bash
scripts/test-moe-submission.sh --remote-shapes
scripts/test-moe-submission.sh --benchmark-remote \
  --report-json data/benchmarks/c500-64g/remote-submission-parity.json
```

三组 published dimensions/timing 是 `(H,I,E,group_sum,warmup,iterations)`：
`(2048,8192,16,2272,5,30)`、`(7168,2048,32,4544,5,20)`、
`(7168,2048,64,9088,5,20)`。单一事实源是 `remote_contract.json`，运行
`python ../../scripts/check_moe_remote_contract.py` 可验证 canonical contract/ABI
fingerprint。当前 `group_sizes`/route weights 是确定性的本地随机 fixture；远程
routing、offset sentinel、stride/contiguity、workspace policy、case cache lifecycle、
评分聚合与 exact toolchain 仍未知，因此即使三组本地 C500 correctness 通过也只能称
`LOCAL_PROXY_ONLY`，不能称 submit-ready、remote-equivalent 或 remote SOTA。旧
`moe_test_configs.json` 的默认目标是上述 remote submission matrix；旧 Large/Small
明确位于 `local_proxy_guard`，不是远程 workload。

本次 remote-first 默认入口的真实 C500 5/30、5/20、5/20 重测保存在
`data/benchmarks/c500-64g/remote-submission-default-v2-20260722.json`；报告绑定
submission、role-aware config 与 harness 哈希，并诚实记录运行时 tracked source 为 dirty。

可用环境变量：`MOE_AUTOTUNE=0` 关闭实测调优，`MOE_AUTOHEURISTIC=0`
关闭 shape 启发式但保留 autotune，`TILELANG_AUTO_TUNING_DISABLE_CACHE=1`
关闭 autotune 磁盘缓存，`MOE_CLEAR_CACHE=1` 清空已有缓存。正确性仍由
functional test 独立验证。

调优结果会追加到 `data/autotune/$MOE_HOST_ID/results.jsonl`。如果另一台
C500 已经上传过相同 workload 的结果，autoheuristic 会优先排列该机器
记录中的最佳 swizzle 候选，再由本机 autotune 实测确认。两台机器的
`MOE_HOST_ID` 必须不同，不能共用同一个结果文件。

## 5. FC1/FC2 解耦实验

新 schedule 在内部使用六个独立字段：`s1_bn`、`s1_bk`、
`s1_stages`、`s2_bn`、`s2_bk`、`s2_stages`。旧的
`block_dhidden` / `block_dexpert` / `num_stages*` 参数保留为兼容入口；
新实验应使用显式阶段参数或固定 preset，避免 FC1/FC2 变量混杂。

| 编号 | 相对 E0 的变化 | 类型 / 结论 |
|---|---|---|
| E0 | FC1/FC2 均为 BN128 / BK128 / stage1 | canonical 对照 |
| E1 | FC1 BK64 | canonical，拒绝 |
| E2 | FC1 BN64 | canonical，拒绝 |
| E3 | FC2 BK64 / stage1 | canonical，单独未达 1% |
| E4 | FC2 BK64 / stage2 | canonical，拒绝 |
| E5 | FC2 BN256 | canonical，拒绝 |
| E6 | 已发射的空 metadata CTA 内跳过 FC1/FC2 工作，grid 不变 | 非 canonical，拒绝 |
| E7 | 按各 expert 实际 128-row tile 数构造 exact metadata grid，减少空 CTA | 非 canonical，单独拒绝；与 E3 组合接受 |

E0–E5 可直接通过 `--experiment` 复现；E6 是已回退的诊断候选，E7 使用
`--compact-metadata-grid`。逐项定义、测量口径与原始数据入口见
[`reports/2026-07-20-moe-e0-e7-summary.md`](../../reports/2026-07-20-moe-e0-e7-summary.md)。

单组功能对拍：

```bash
python tune_moe.py --evaluation-target local-proxy-guard \
  --experiment E4 --mode functional --warmup 0 --iteration 1
```

硬件 profiler 需要隔离某个规格时，增加 `--shape large` 或 `--shape small`。

## 6. 当前 C500 默认

`fusedmoe_benchmark.custom_kernel` 当前使用正式接受的 H8F1：在 H4F1 的 combined Gate/Up、
FC1/FC2 `BN128/BK64/stage1`、FC2 activation shared tile 和 FP16 route-weight fragment cache
基础上，combined FC1 K loop 使用同 buffer 的 stage-1 `T.Pipelined`。源码 SHA-256 为
`6543398edf4e48c95c896b5a703c1a58f3f49a8a7dec543cb892cb0467aea7a3`。H8F1 已通过两次
完整 correctness+10/100+mcProfiler 及相邻 H4F1/H8F1/H8F1/H4F1 复核；ABBA mean/median
分别提升 `1.9755%/1.9047%`。不得回退到 H4F1、H4、E3+E7 或 A1b 作正式对照。接受证据见
[`reports/2026-07-20-moe-h4-followup-results.md`](../../reports/2026-07-20-moe-h4-followup-results.md)
和
[`reports/2026-07-20-moe-h4f1-pipeline-pair-followup-results.md`](../../reports/2026-07-20-moe-h4f1-pipeline-pair-followup-results.md)。

六组正式实验（每组 3 个独立性能进程，输出 median/MAD/P95）：

```bash
python ../../scripts/run_moe_stage1_experiments.py \
  --evaluation-target local-proxy-guard --host-id "${MOE_HOST_ID}"
```

runner 只在单组 functional 通过后启动该组性能进程，并校验 JSON 中的
schedule 与 E0–E5 canonical preset 完全一致。当前 kernel 的 N/K 读取没有
tail mask，所以 tile 必须 16 对齐且整除对应维度。
