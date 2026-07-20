# 使用说明


## 1. 文件定位

`custom_fusedmoe.py` 作为对外暴露的 MoE fused kernel 接入文件。

调用和改写：

1. `RoutedMoEKernel` 的初始化和__call__接口固定；
2. 内部实现可重写调整。

## 2. 使用
如下命令可以自动跑功能和性能测试,结果输出在终端
```bash
python fusedmoe_benchmark.py
```
或者
```bash
bash run.sh
```
## 3. 测试用例
可以在`moe_test_configs.json`中直接添加测试case，目前有性能和功能各两个
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

默认配置适合日常迭代；需要扩大搜索空间时使用：

```bash
MOE_AUTOTUNE_MODE=full scripts/run-moe.sh
```

调优报告和原始候选记录位于仓库根目录的 `reports/` 与 `logs/`。OJ 的
独立入口是 `submission.py`，其 padded-token ABI 对拍使用：

```bash
scripts/test-moe-submission.sh --public-shape
```

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
python tune_moe.py --experiment E4 --mode functional --warmup 0 --iteration 1
```

硬件 profiler 需要隔离某个规格时，增加 `--shape large` 或 `--shape small`。

## 6. 当前 C500 默认

`fusedmoe_benchmark.custom_kernel` 使用经过五进程验证的 E3+E7 组合：FC1 保持
`BK=128`，FC2 使用 `BK=64`，并按每个 expert 的实际 128-row tile 数构造 metadata
grid，避免末尾空 CTA。该组合相对 E0 的 Large / Small 中位数分别快 1.3110% / 1.3458%。
原始基准、mcProfiler 和提交 ABI 结果见
[`reports/2026-07-20-moe-e3e7-promotion.md`](../../reports/2026-07-20-moe-e3e7-promotion.md)。

六组正式实验（每组 3 个独立性能进程，输出 median/MAD/P95）：

```bash
python ../../scripts/run_moe_stage1_experiments.py --host-id "${MOE_HOST_ID}"
```

runner 只在单组 functional 通过后启动该组性能进程，并校验 JSON 中的
schedule 与 E0–E5 canonical preset 完全一致。当前 kernel 的 N/K 读取没有
tail mask，所以 tile 必须 16 对齐且整除对应维度。
