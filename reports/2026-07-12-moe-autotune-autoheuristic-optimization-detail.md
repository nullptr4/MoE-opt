# MoE 路由算子优化记录（赛题一）

- 日期：2026-07-12（UTC+8）
- 机器：有卡后的本地环境（maca 3.7.2.1）
- 任务：基于跑分结果做 baseline / AutoTuner / AutoHeuristic 对比，补齐失败故事与下一步优化方向，并同步资料到 GitHub。

## 1. 这轮优化目标与边界

1. 完整跑通 `autotune` 与 `autoheuristic` 流程（含官方端到端验证）。
2. 使用 `mcProfiler` 做性能采样（baseline、autotune、autoheuristic）。
3. 记录所有失败案例与复现实据，形成可提交报告。
4. 将“本赛题已用到的资料”同步到 `git@github.com:nullptr4/MoE-opt.git`。

## 2. 已做的工作（按步骤）

### 步骤 A：梳理并修正执行路径
- 发现 `mcprof_profile_once.sh` 中 `source ./activate.sh` 被 sh 解释导致 `source: not found`。
- 将 profile 命令改为 `bash -lc '... source ./activate.sh ...'` 并确认 `source` 能执行。
- 结果：`mcProfiler` profile 流水线恢复。

### 步骤 B：准备统一的基准脚本
- 使用 `scripts/mcprof_moe_benchmark.py`，统一输出 `PROFILE_CASE|...` 格式，包含：
  - 模式（baseline / autotune / autoheuristic）
  - 形状（small / large）
  - 吞吐时延 `ms`
- 使用 `tools/moe_autoheuristic.py` 的 `choose_schedule()` 做在线 schedule 选择。

### 步骤 C：AutoTuner 扫描（直接 kernel）
- 使用 `tools/autotune_moe_sweep.py` 做官方路由输入下参数搜索。
- 搜索维度：`block_dhidden`、`block_dexpert`、`swizzle_panel`、`swizzle_panel_down`、`gemm_policy`、`min_blocks_per_sm`、`single_weight_buffer`。
- 输出：
  - `/data/metax-race/logs/moe-autotune-sweep-20260712T092830Z.json`
  - `/data/metax-race/logs/moe-autotune-sweep-20260712T092904Z.json`

### 步骤 D：AutoHeuristic 训练/验证
- 依据 `train_moe_autoheuristic.py` 规则：优先官方端到端观测（`official_e2e`），无则回退 direct-kernel。
- 产物：`/data/metax-race/config/moe_autoheuristic.json`
- 规则结果（两条）：
  - `small`: `swizzle_panel=8, swizzle_panel_down=16`
  - `large`: `swizzle_panel=4, swizzle_panel_down=16`

### 步骤 E：mcProfiler 分析
- 已完成三批采集：
  1. `output20260712094550`（baseline small）
  2. `output20260712094706`（small autotune+autoheuristic）
  3. `output20260712094821`（large autotune+autoheuristic）
- 额外现象：auto+small 图像中曾出现一次 `RoofLine` 脚本除零异常（`ZeroDivisionError`），已定位为绘图阶段稳定性问题，不影响采样主数据提取。

## 3. 具体实验结果（e2e 测量）

### 3.1 `autotune_moe_benchmark.py`（baseline / autotune / autoheuristic）

#### iters=2, warmup=1

| shape | baseline(ms) | autotune(ms) | autoheuristic(ms) | 改善（vs baseline） |
|---|---:|---:|---:|---:|
| small | 37.098751 | 37.083138 | 37.018112 | `-0.042%` / `-0.217%` |
| large | 237.302017 | 236.849411 | 237.095932 | `-0.191%` / `-0.087%` |

#### iters=5, warmup=1（更稳定复测）

| shape | baseline(ms) | autotune(ms) | autoheuristic(ms) |
|---|---:|---:|---:|
| small | 36.932504 | 36.980121 | 36.888217 |
| large | 236.801636 | 236.828882 | 236.729907 |

### 3.2 关键判断

- 在当前规则空间下，`small` 与 `large` 都**没有出现显著加速**（幅度属于测量抖动范围）
- 目前 autoheuristic 的价值偏向“稳定配置选择”而非明显降时延。

## 4. 成功故事

1. **流程可复现**：`autotune -> autoheuristic -> mcProfiler` 一条链路全部闭环。
2. **配置选择可审计**：自动化 artifact 文件 (`moe_autoheuristic.json`) 可追踪来源日志与证据。
3. **失败不留空白**：将 `autotune` 的闭包序列化问题、PrimFunc 识别问题、RoofLine 除零问题写入运行记录。

## 5. 失败故事（已记录）

1. **`source` 在非 bash shell 失败**
   - 现象：`source: not found`
   - 处理：改为 `bash -lc`。
2. **AutoTuner 一些候选阶段异常**（早期尝试中出现构造参数/闭包序列化问题）
   - 现象：候选构建失败、candidate 无法完成编译/计时
   - 处理：统一通过 primfunc 工厂与参数注入方式稳定化。
3. **mcProfiler RoofLine 计算除零**
   - 现象：小样本图生成脚本抛 `ZeroDivisionError`
   - 处理：保留其余指标采集，标注该异常为“采样脚本级别异常”。

## 6. 下一步计划

- **继续深化参数搜索**：扩大搜索空间（例如更多 `threads`、`num_stages`、更多 swizzle 组合）做第二轮，判断是否有真正可跨形状收益配置。
- **基于统计化验证推进**：每轮至少跑 3 次独立 run，保留中位数，避免只看单次波动。
- **分离 kernel 与端到端瓶颈**：通过 `moe_torch_profiler` 细化 scatter/gather 阶段占比，优先对瓶颈阶段做手工改造。

## 7. 与 git 仓库同步

将包含本轮结果与模型产物的材料提交到 `git@github.com:nullptr4/MoE-opt.git`：
- `reports/2026-07-12-moe-autotune-autoheuristic-optimization-detail.md`
- `config/moe_autoheuristic.json`（如提交路径建立）
- `logs/moe-autotune-sweep-20260712T092830Z.json`
- `logs/moe-autotune-sweep-20260712T092904Z.json`

