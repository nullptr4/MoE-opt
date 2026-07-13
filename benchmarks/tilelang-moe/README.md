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
