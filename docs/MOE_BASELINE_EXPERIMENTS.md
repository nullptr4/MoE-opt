# MoE 基线实验规范

正式候选统一使用 `scripts/run_moe_baseline.py`。该入口把当前固定
schedule 放在三个独立 Python 进程中运行，并生成一份可审计 JSON：

```bash
source scripts/activate-maca.sh
python scripts/run_moe_baseline.py \
  --host-id c500-32g \
  --profiler-run-id <run-id>
```

默认规范为：

- 3 个独立进程；
- 每个性能规格 warmup 10 次、计时 100 次；
- 主指标为进程间 latency median；
- 同时记录 MAD、P95、min、max 和所有原始进程样本；
- 每个进程都运行公开 functional cases；
- 汇总结束后运行 padded OJ ABI public-shape smoke；
- 独立运行 127/128/129、exact、skewed routing 及 0/1/tiny route weight fuzz；
- 至少关联一个已经归档且 metadata 可读的 profiler run-id；
- 固定 schedule 为 `FullRow + FC1 row8 + FC2 row16 + 单 weight buffer`。

默认产物位置：

```text
data/benchmarks/<host-id>/<run-id>.json
logs/moe-<run-id>-run-<n>.log
logs/moe-<run-id>-oj-abi.log
```

JSON 包含 Git SHA、branch、dirty 状态、kernel SHA256、TileLang SHA、
MACA/PyTorch/TileLang 版本、GPU 型号和显存、完整 schedule、warmup、
iteration、每个独立进程结果、median、MAD、P95、functional/OJ ABI
结果及关联的 profiler run-id。为使脏工作树上的测量仍可重建，JSON 还会
内嵌 kernel、benchmark harness、submission、测试和配置文件的路径、
SHA256 与完整内容；`git_dirty` 不能再掩盖实际测量源码。

`--skip-oj-abi` 只用于诊断。使用该选项的产物会标为 `incomplete` 并以
非零状态退出，不能晋级为正式基线。

## 基线身份

基线标签固定为：

```text
v1-fullrow-single-buffer-row8-row16
```

进入默认实现的候选必须通过 functional、OJ ABI 和 fuzz，并满足计划中的
统计门槛：综合中位数至少提升 1%，任一公开规格退化不超过 0.5%，且无
新增 spill。低于 1% 的单次差异不作为晋级证据。

## 阶段 1 解耦实验

E0–E5 使用独立 runner，共享相同的 warmup / iteration 与统计方法：

```bash
source scripts/activate-maca.sh
python scripts/run_moe_stage1_experiments.py \
  --host-id c500-32g \
  --runs 3 --warmup 10 --iteration 100
```

每组先运行一个 functional 进程，通过后再运行 3 个独立性能进程。产物默认位于
`data/benchmarks/<host-id>/stage1-<timestamp>.json`，包含每组原始进程记录、
median/MAD/P95、相对 E0 的逐 workload 变化与晋级门槛判定。
