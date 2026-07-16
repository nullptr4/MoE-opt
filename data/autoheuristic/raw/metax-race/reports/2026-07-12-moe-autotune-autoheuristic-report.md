# MoE AutoTuner 扫描与 AutoHeuristic 训练记录（2026-07-12）

## 1. 目标与范围

本次工作在 MetaX C500 卡上使用 TileLang `AutoTuner` 扫描赛题一 routed-MoE，
并将扫描结果与官方端到端复测结果整理成 `moe_autoheuristic.json`。覆盖两种官方性能形状：

| 名称 | hidden | expert | experts | tokens | 来源 |
|---|---:|---:|---:|---:|---|
| small | 3584 | 1024 | 4 | 65,536 | `bs=8, seq=4096, top-k=2` |
| large | 7168 | 2048 | 8 | 131,072 | `bs=4, seq=8192, top-k=4` |

每个形状扫描 10 个编译安全候选，参数包括 `block_dhidden`、`block_dexpert`、
stage-1/stage-2 swizzle panel、GEMM warp policy、单/双 weight shared buffer、
以及 `min_blocks_per_sm`。线程数固定为 256，`block_token=128`，保持比赛元数据布局一致。

## 2. 扫描流程

1. **修正 AutoTuner 接口**：比赛 kernel 是 `@tilelang.jit` 包装器，直接返回
   `JITKernel` 会被 `AutoTuner.from_kernel` 拒绝；脚本改为调用
   `moe_forward_tilelang_routed.func.orig_func`，让 AutoTuner 接收 `PrimFunc`。
2. **修正序列化**：kernel factory 不再闭包捕获不可序列化的 `JITImpl`，只捕获整数形状
   和输入路由模式，可以生成稳定的 autotuner cache key。
3. **输入构造**：`--routing-mode official` 复用
   `fusedmoe_benchmark.generate_input` 和真实 top-k 路由、非均匀 expert counts、
   128-token padding 元数据；另保留 `balanced` 模式用于快速 smoke test。
4. **直接扫描**：每个候选编译后 warmup 5、重复 20 次用于最终官方路由记录；原始 JSON、
   控制台输出和 TileLang cache 均留在 `logs/`、`.cache/tilelang/`。
5. **端到端复测**：AutoTuner 只测 routed kernel，另外用官方 benchmark 对候选执行
   functional + performance，确认 scatter/reduce 和真实 routing 不改变结论。

## 3. 官方路由 AutoTuner 结果

### small（65,536 tokens，直接 kernel，warmup=5、rep=20）

| 排名 | dhidden tile | dexpert tile | stage-1/down panel | GEMM | single buffer | min blocks | latency (ms) |
|---:|---:|---:|---|---|---|---:|---:|
| 1 | 128 | 128 | 8 / 8 | FullRow / FullRow | yes | - | 25.127 |
| 2 | 128 | 128 | 16 / 16 | FullRow / FullRow | yes | - | 25.136 |
| 3 | 128 | 128 | 8 / 16 | FullRow / FullRow | yes | - | 25.149 |
| 4 | 128 | 128 | 8 / 16 | FullRow / FullRow | yes | 2 | 25.152 |
| 5 | 128 | 128 | 4 / 16 | FullRow / FullRow | yes | - | 25.159 |
| 6 | 128 | 64 | 8 / 16 | FullRow / FullRow | yes | - | 30.612 |
| 7 | 64 | 128 | 8 / 16 | FullRow / FullRow | yes | - | 32.682 |
| 8 | 128 | 128 | 8 / 16 | FullRow / FullRow | no | - | 36.833 |
| 9 | 128 | 128 | 8 / 16 | Square / Square | yes | - | 40.122 |
| 10 | 128 | 128 | 8 / 16 | FullCol / FullRow | yes | - | 59.642 |

### large（131,072 tokens，直接 kernel，warmup=5、rep=20）

| 排名 | dhidden tile | dexpert tile | stage-1/down panel | GEMM | single buffer | min blocks | latency (ms) |
|---:|---:|---:|---|---|---|---:|---:|
| 1 | 128 | 128 | 4 / 16 | FullRow / FullRow | yes | - | 201.232 |
| 2 | 128 | 128 | 8 / 16 | FullRow / FullRow | yes | 2 | 201.982 |
| 3 | 128 | 128 | 16 / 16 | FullRow / FullRow | yes | - | 202.609 |
| 4 | 128 | 128 | 8 / 16 | FullRow / FullRow | yes | - | 202.819 |
| 5 | 128 | 128 | 8 / 8 | FullRow / FullRow | yes | - | 203.465 |
| 6 | 128 | 64 | 8 / 16 | FullRow / FullRow | yes | - | 251.286 |
| 7 | 64 | 128 | 8 / 16 | FullRow / FullRow | yes | - | 266.570 |
| 8 | 128 | 128 | 8 / 16 | Square / Square | yes | - | 321.106 |
| 9 | 128 | 128 | 8 / 16 | FullRow / FullRow | no | - | 422.666 |
| 10 | 128 | 128 | 8 / 16 | FullCol / FullRow | yes | - | 480.002 |

直接 kernel 的绝对时间不等于官方端到端时间，因此只用来产生候选，不能直接替代比赛成绩。

## 4. 官方端到端复测与取舍

以下候选的 functional case 均通过（2/2）；性能是官方 benchmark 重复测量：

| 候选 | large (ms) | small (ms) | 结论 |
|---|---:|---:|---|
| shipping：row8 / down-row16，FullRow，single buffer | **243.5285** | **38.5460** | small 稳定，作为 fallback |
| row4 / down-row16，FullRow，single buffer | **243.0021** | **38.6018** | large 最快，small 与 shipping 持平 |
| row8 / down-row8，FullRow，single buffer | 310.6963（一次冷启动） | 52.6914 | 失败故事：直接 kernel 看似接近，端到端明显退化，拒绝 |

此前控制组和重复测量记录在 `logs/moe-tuning.md`；单 buffer、FullRow 和 128×128 tile
是稳定收益来源，Square/FullCol、双 buffer、较小 tile 均明显变慢。

## 5. AutoHeuristic 训练产物

```bash
cd /data/metax-race
python tools/train_moe_autoheuristic.py
PYTHONPATH=tools python tools/verify_moe_autoheuristic.py
```

训练器收集 50 条观测（40 条官方路由 direct-kernel + 10 条官方端到端复测），
以端到端 latency 优先、direct-kernel 作为没有端到端样本时的 fallback。生成：

```text
config/moe_autoheuristic.json
```

当前规则：

- small：`row8 / down-row16`、FullRow、single buffer，官方复测 38.5460 ms；
- large：`row4 / down-row16`、FullRow、single buffer，官方复测 243.0021 ms；
- 未见过的相近形状采用 log-distance 最近规则，过远形状回退 shipping schedule。

查询接口位于 `tools/moe_autoheuristic.py`：

```python
from moe_autoheuristic import choose_schedule
cfg = choose_schedule(7168, 2048, 8, 131072)
```

该 heuristic 是显式 opt-in，不会悄悄改变比赛默认源码；拿到 `cfg` 后传给
`RoutedMoEKernel` 即可复现选择。模型、全部观测和来源日志均可审计。

## 6. 失败与修复记录

- 第一次 AutoTuner 扫描闭包捕获 `JITImpl`，cache-key 序列化直接失败；改为 module
  global + `orig_func` 后恢复。
- 第二次扫描返回 `JITKernel` 而不是 `PrimFunc`，10 个候选全部在 compile/profile
  阶段失败；切换到 `func.orig_func` 后 10/10 编译并完成测量。
- 初始 balanced 路由的 down-row8 结果与官方端到端不一致，说明只看 kernel latency
  会漏掉 scatter/reduce、非均匀 group 和 launch 交互；训练器强制端到端优先，并把
  该候选保留为失败样本。
- 一次 shipping 复测出现 316/59 ms 的冷启动异常；无代码变化的重复运行恢复到
  243.5/38.5 ms，因此没有把单次异常写入最优规则。
