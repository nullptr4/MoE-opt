# MetaX 赛题一 Fused MoE 调优报告

- **日期**：2026-07-11
- **项目目录**：`/data/metax-race`
- **优化对象**：赛题一初赛的 Fused MoE GEMM
- **主实现**：`tilelang-metax/race_tests/moe/custom_fusedmoe.py`
- **提交实现**：`tilelang-metax/race_tests/moe/submission.py`
- **结论**：在当前 C500 卡实例上，官方本地性能用例从 `425.8014 / 62.3782 ms` 降至 `243.4768 / 38.6093 ms`，分别为 **1.75× / 1.62× 加速**（延迟下降 **42.82% / 38.10%**）。

> 版本边界：当前卡实例实测为 MACA 3.7.1.5，而比赛材料指定 MACA 3.7.2.1。以下数字是当前实例上的本地结果，切换到 3.7.2.1 后必须重新验证，不能直接当作 OJ 最终成绩。

---

## 1. 目标、计算语义与不可破坏的约束

每个 expert 的计算语义是：

```text
gate = x @ gate_w[expert]^T
up   = x @ up_w[expert]^T
hidden = silu(gate) * up
out = hidden @ down_w[expert]^T
out *= routed_expert_weights
```

本轮调优遵循以下约束：

1. `RoutedMoEKernel` 的原有调用方式和 `__call__` 参数顺序不能破坏；特别是 `backend` 保持原来的第 11 个位置参数。
2. 本地 benchmark 的 `group_idx_for_bx`、`group_padded_offsets` 以 **128 token** 为元数据单位；不能只把 `block_token` 从 128 改成 64。
3. 用官方两个 functional case 做正确性门禁；性能候选只有在功能 2/2 通过后才保留。
4. 不能用降低精度、跳过计算或改 benchmark wrapper 的方式换取分数。
5. 对 OJ，`submission.py` 必须暴露无类型标注的固定 `run_kernel(...)`，并遵循教程所述的 padded token ABI。

### 1.1 本地 benchmark 与 OJ ABI 不是同一个封装层

这是调优中最容易踩坑的一点。

| 项目 | 本地 `fusedmoe_benchmark.py` | OJ `submission.py` 教程 ABI |
|---|---|---|
| token 存储 | 按有效 token 紧凑存放 | 按 expert padding 到 128 token 后存放 |
| `routed_expert_weights` | FP16 | FP32 |
| `group_offsets` / `group_padded_offsets` | 长度 E | 长度 E+1 |
| 结果后处理 | benchmark 还会 `scatter_reduce` | `run_kernel` 直接写入 `out` |

因此，`custom_fusedmoe.py` 用于本地官方 benchmark 调优；`submission.py` 以同样的 kernel 策略重写为 OJ 的 padded ABI，不能直接 import 前者。

---

## 2. 环境与测试方法

### 2.1 实测环境

| 项目 | 实测值 |
|---|---|
| 加速卡 | MetaX C500，sGPU 50%，32 GiB quota |
| Python | 3.12.11 |
| PyTorch | `2.8.0+metax3.7.1.3` |
| TileLang | `0.1.9+maca.gitee6db437` |
| MACA | **3.7.1.5** |
| TileLang target | `maca`，`max_shared_memory_per_block=65536`，warp size=64 |

环境自检日志：[`../logs/preflight-final-gpu.json`](../logs/preflight-final-gpu.json)。

### 2.2 官方测试用例

官方 JSON 中每类各有两个 case：

| 类型 | 大规格 | 小规格 |
|---|---|---|
| Functional | H=7168, I=2048, E=8, top-k=4, BS=1, Seq=8192 | H=3584, I=1024, E=4, top-k=2, BS=2, Seq=4096 |
| Performance | H=7168, I=2048, E=8, top-k=4, BS=4, Seq=8192 | H=3584, I=1024, E=4, top-k=2, BS=8, Seq=4096 |

性能采用官方 `warm_up=10`、`iteration=100`，计时区域包括官方 MoE wrapper，而不是只测单个 TileLang kernel。

### 2.3 验证纪律

每个候选依次执行：

1. 编译和两组 functional case；
2. 仅在 2/2 成功时跑两组官方 performance case；
3. 把结果写入 [`../logs/moe-tuning.md`](../logs/moe-tuning.md)；
4. 最终默认代码使用 `./run-race-test.sh moe` 重新回归；
5. 独立提交文件再用 padded ABI 工具对拍。

---

## 3. 初始实现与瓶颈判断

初始实现的 tile 是 `M=128, N=128, K=128`、`threads=256`、`num_stages=1`。第一阶段同时计算 gate 与 up：

- 为 gate 和 up 分别分配一个 `128 × 128 × FP16` shared tile；
- 两个 tile 共用 **64 KiB** 动态 shared memory，正好触及 C500 单 block 上限；
- 每线程还保存两个 FP32 累加 fragment：`gate_logits_local` 与 `up_logits_local`；
- 第二阶段再用中间 `up_logits` 做 down projection。

生成的 MACA C++ kernel 显示第一阶段高 shared-memory 占用、寄存器压力和 warp 分工是主要可控点。由此制定了从低风险到高风险的顺序：

1. pipeline、tile、thread 数量等常规参数；
2. rasterization/swizzle 和 GEMM warp policy；
3. shared-memory 结构优化；
4. 仅在必要时尝试 fast math 或更激进的 tile 变化。

---

## 4. 调优过程：每一步的假设、代码改动与结果

下表记录所有实际跑过的主要候选。性能单位均为毫秒；“通过”均指 functional 2/2 通过。

| # | 候选与改动 | 思路 | 功能 | 大规格 | 小规格 | 结论 |
|---:|---|---|---|---:|---:|---|
| 0 | 原始 M128/N128/K128、256T、s1 | 官方参考点 | 通过 | 425.8014 | 62.3782 | 基线 |
| 1 | gate/up 全局 `num_stages=2` | 用更深软件流水遮蔽访存 | 编译拒绝 | — | — | 动态 shared 变为 131072 B，超过 C500 65536 B 限制 |
| 2 | `block_dhidden=64` | 减少 K tile 和 shared/寄存器压力 | 通过 | 491.5795 | 68.6550 | 更小 K tile 的额外循环/同步成本更高 |
| 3 | `block_dexpert=64` | 减少 N tile，改善占用 | 通过 | 482.9604 | 68.3492 | CTA 数翻倍、权重/调度开销上升 |
| 4 | `threads=128` | 更少线程可能提高每线程资源 | 通过 | 792.7937 | 105.1254 | 2 个 64-lane warp 不能充分喂饱 GEMM |
| 5 | 仅 down `num_stages=2` | 不增加第一阶段 64 KiB shared，尝试第二阶段流水 | 通过 | 449.1695 | 65.2812 | pipeline 开销大于收益 |
| 6 | row swizzle panel 10→16 | 让同一 bx 的更多 by 连续执行，复用 input/up tile | 通过 | 424.8234 | 61.9322 | 仅约 0.2%/0.7%，方向正确但不足够 |
| 7 | 三个 GEMM 全部 `FullRow`，row10 | 改变 4 个 MACA warp 的 M/N 划分 | 通过 | 340.2105 | 50.4333 | 首个大收益：20.1%/19.2% 降延迟 |
| 8 | 仅 stage-1 `FullRow` | 区分收益来自 gate/up 还是 down | 通过 | 402.4672 | 58.0526 | 有益但弱于三处都改 |
| 9 | 仅 stage-2 `FullRow` | 同上 | 通过 | 361.8487 | 54.0935 | 收益更大，但仍弱于组合 |
| 10 | `FullRow` + row16 | 将 policy 与最优 row panel 组合 | 通过 | 337.7548 | 50.1424 | 相比 row10 再小幅提升 |
| 11 | down row panel=28 | 希望完整覆盖 down 的较大 by 网格 | 通过 | 338.5263 | 50.2019 | 略差，拒绝 |
| 12 | stage-1 `FullCol` | 尝试更多 weight L2 复用 | 通过 | 578.7929 | 80.0066 | 分工方向不匹配，显著退化 |
| 13 | `TL_ENABLE_FAST_MATH` | 期待加快 SiLU 的 `exp2` 路径 | 通过 | 338.5914 | 50.1863 | 既不更快，又增加数值/复现风险，删除该实验分支 |
| 14 | **stage-1 单 weight shared buffer** | gate/up 串行复用同一 shared tile，64 KiB→32 KiB | 通过 | **244.9299** | **38.6086** | 主成功：提升 block 并发/资源余量，较基线下降 42.48%/38.11% |
| 15 | 单 buffer + stage-1 column16 | 尝试优先复用 gate/up 权重 | 通过 | 244.1342 | 38.7586 | 大规格快 0.32%，小规格慢 0.39%，不作为全局默认 |
| 16 | 单 buffer + down column16 | 尝试复用 down 权重 | 通过 | 247.5920 | 38.8365 | 两规格均不如 row，拒绝 |
| 17 | 单 buffer + `min_blocks_per_sm=2` | 用 launch bounds 压寄存器，争取双 block 驻留 | 通过 | 244.7720 | 38.5064 | 小规格略好、大规格略差，量级接近测试波动，未保留 |
| 18 | 单 buffer + K=64 | 利用更低 shared/寄存器试图提高占用 | 通过 | 313.2079 | 46.1666 | reduction loop 翻倍，拒绝 |
| 19 | 单 buffer + 512 threads | 8 warp 代替 4 warp | 通过 | 316.0550 | 48.4101 | block 更重、并发下降，拒绝 |
| 20 | 单 buffer + down row8 | 更小 down swizzle panel | 通过 | 244.9369 | 38.5949 | 与 row16 基本持平，无稳健优势 |
| 21 | 单 buffer + M=64 计算 tile | 通过 2:1 元数据映射降低每 CTA accumulator | 通过 | 309.1540 | 44.7300 | CTA/权重加载翻倍，拒绝 |
| 22 | 最终默认代码，官方入口 | 完整回归 | 通过 | **245.0176** | **38.4679** | 保留 |
| 23 | 最终默认代码，重复测量 | 验证调优脚本默认参数与出货实现一致 | 通过 | 245.5890 | 38.6898 | 正常 timing variation，配置一致 |
| 24 | 单 buffer + K=256 | 减少 K 循环次数 | 通过 | 484.2256 | 67.9936 | 64 KiB stage tile 降低并发，拒绝 |
| 25 | 单 buffer + down N=64 | 缩小 down 输出 tile | 通过 | 305.0344 | 45.7394 | CTA 数翻倍，拒绝 |
| 26 | 单 buffer + down `num_stages=2` | 仅让 down 更深流水 | 通过 | 269.3212 | 41.8366 | pipeline 开销/资源占用大于收益，拒绝 |
| 27 | 单 buffer + down `min_blocks_per_sm=2` | 控制 down block 驻留 | 通过 | 244.8288 | 38.5300 | 与默认差异不稳健，拒绝 |
| 28 | 单 buffer 但 stage-1 `T.Pipelined(..., 1)` | 让编译器调度 alias buffer 的 barrier | 编译拒绝 | — | — | pipeline planner 检测到 alias buffer 的重叠写入，恢复 `T.serial` |
| 29 | stage-1 row8 / down row16 | 缩小 gate/up 的 raster panel | 通过（两次） | 243.4042 / 243.2124 | 38.5252 / 38.5337 | 大规格稳定约快 0.7%；小规格差异接近波动 |
| 30 | stage-1 row32 / down row16 | 扩大 gate/up 的 raster panel | 通过 | 245.1068 | 38.5451 | 无收益，拒绝 |
| 31 | **最终：stage-1 row8 / down row16，官方入口** | 对最终源码完整回归 | **通过** | **243.4768** | **38.6093** | **保留** |
| 32 | stage-1 row4 / down row16 | 继续缩小 gate/up raster panel | 通过 | 243.1514 | 38.6111 | 大规格差异在波动内，小规格无收益，拒绝 |
| 33 | stage-1 row8 / down row8 | 缩小 down raster panel | 通过 | 243.5149 | 38.6994 | 小规格退化，拒绝 |
| 34 | stage-1 row8 / down row16 + `min_blocks_per_sm=2` | 与 row8 schedule 组合 occupancy hint | 通过 | 243.5154 | 38.5250 | 大规格更慢，小规格变化轻微，拒绝 |
| 35 | gate activation 存 FP16、复用 FP32 gate accumulator 计算 up | 减少一个 FP32 output fragment | 通过 | 308.7221 | 47.1000 | 分两次 reduction 后 input 被重复读取，严重退化，拒绝 |
| 36 | down 阶段单独使用 512 threads | 减小每线程 output fragment，尝试减轻寄存器压力 | 通过 | 275.2145 | 42.4989 | block 过重、CTA 并发下降，拒绝 |
| 37 | down 阶段单独使用 128 threads | 增加 CTA 数，尝试增加驻留 | 通过 | 281.0836 | 42.6746 | 两个 64-lane warp 不足以喂饱 GEMM，拒绝 |

下面按“成功故事”和“失败故事”解释这些数据背后的因果。

### 4.1 成功一：`FullRow` warp policy

**假设**：默认 `Square` 会将 256 线程（4 个 64-lane warp）按 2×2 的 M/N 方向切分。对于 `128×128`、`transpose_B=True` 的三次 GEMM，`FullRow` 让所有 warp 沿 M 方向切分、完整覆盖 N，可能更符合 C500 的访存和 MMA 布局。

**代码位置**：

- policy 映射：`custom_fusedmoe.py:50-59`；
- gate/up 的 `T.gemm(... policy=gemm_policy_stage1)`：`139-157`；
- down 的 `T.gemm(... policy=gemm_policy_stage2)`：`230-236`；
- 最终默认：`gemm_policy="full_row"`，见 `23-30` 与构造函数 `261-268`。

**效果**：仅 policy 从默认 Square 改为 FullRow（row10）即把性能从 `425.8014 / 62.3782 ms` 降到 `340.2105 / 50.4333 ms`。单独测试 stage-1 或 stage-2 都有收益，三次 GEMM 全部使用 FullRow 才最好，证明这不是某一个阶段的偶然现象。

### 4.2 成功二：分别为两个阶段选择 row swizzle panel

**假设**：row swizzle 会让固定的逻辑 `bx` 在连续 `by` 间执行。第一阶段的 `by` 是 expert hidden tile，连续执行可重复使用同一个 token input。最初的 FullRow 试验中 panel=16 优于原始 panel=10；在单 weight buffer 已降低 shared-memory 压力后，再单独搜索第一阶段，panel=8 对大规格更合适，而 down 阶段仍保留 panel=16。

**代码位置**：

- stage-1：`T.use_swizzle(panel_size=swizzle_panel, order=swizzle_order)`，`108`；
- stage-2：`T.use_swizzle(panel_size=swizzle_panel_down, order=swizzle_order_down)`，`206`；
- 最终默认 `swizzle_panel=8`、`swizzle_panel_down=16`：`23-26`。

**效果**：在原始 Square layout 上，panel 10→16 只有很小提升；在 FullRow 上由 `340.2105 / 50.4333` 进一步到 `337.7548 / 50.1424 ms`。最终的受控比较中，row8/row16 两次为 `243.4042 / 38.5252` 和 `243.2124 / 38.5337 ms`；对应 row16/row16 控制组为 `244.8464 / 38.4821 ms`。因此固化为 stage-1 row8、stage-2 row16：大规格约快 0.7%，小规格无实质差异。这仍是辅助优化，主导收益来自 shared-memory 重用。

### 4.3 成功三：gate/up 共享一个 shared-memory weight tile（最终核心）

**观察**：初始 gate/up 阶段分别有一个 `128×128 FP16` 权重 shared tile。两块 tile 正好是 64 KiB，达到 C500 单 CTA shared-memory 限制；生成代码中也能看到两个大 shared buffer。

**假设**：如果同一 K iteration 先计算 gate、再覆盖同一个 tile 计算 up，可以把 shared 从 64 KiB 降到 32 KiB。虽然失去 gate/up 的双权重预取，较小 shared 可能允许更好的资源调度和占用，收益有机会更大。

**代码位置**：

1. 分配一个 gate tile：`custom_fusedmoe.py:97-103`；
2. `single_weight_buffer=True` 时，让 `routed_expert_up_shared` 指向同一个 buffer：`100-101`；
3. 用 `T.serial` 明确保证同一 buffer 的顺序：先 copy gate→GEMM，再 copy up→GEMM，见 `124-158`；
4. 默认开启：`single_weight_buffer=True`，见 `29` 和 `267`。

**为什么不是直接 alias 后仍使用 pipeline**：pipeline 可将下一次 copy 前移。两个 copy alias 时可能出现覆盖尚未消费的数据，或让编译器拒绝不安全的重叠写入。因此单 buffer 路径显式用 `T.serial`，并限制该路径 `num_stages=1`（`60-61`）。

**效果**：这是决定性优化。FullRow + row16 的 `337.7548 / 50.1424 ms` 降到 `244.9299 / 38.6086 ms`。最终通过官方入口复测为 `245.0176 / 38.4679 ms`，改善明显且可重复。

### 4.4 成功四：安全地支持 M=64 实验，但不把它作为最终策略

直接把 `block_token=64` 会出错，因为 benchmark 仍按 128 构造 `group_idx_for_bx`。为验证这一假设，代码增加了 2:1 元数据映射：

```text
metadata block 128 tokens
    ├── compute CTA 0: rows [0, 63]
    └── compute CTA 1: rows [64, 127]
```

**代码位置**：`custom_fusedmoe.py:63-71` 计算 `metadata_M` 和 `tiles_per_metadata_block`；两个 kernel 用 `group_idx_for_bx[bx // tiles_per_metadata_block]`，见 `114-115`、`210-211`。

**结果**：功能正确，但性能为 `309.1540 / 44.7300 ms`。原因是 M=64 减少了单 CTA accumulator，却把 CTA 数和同一权重 tile 的加载次数翻倍。该改动保留为可控实验能力，默认仍是 M=128。

---

## 5. 失败故事与排除理由

### 5.1 更深 pipeline 不是免费收益

全局 `num_stages=2` 在编译阶段直接失败：第一阶段 shared 需要 128 KiB，超过 C500 的 64 KiB/block 上限。仅给 down 阶段加 stage 2 虽可运行，却在单 buffer 后仍为 `269.3212 / 41.8366 ms`。还尝试过在别名 shared buffer 的 stage-1 上写 `T.Pipelined(..., num_stages=1)`；这次不是数值错误，而是 TileLang pipeline planner 正确拒绝了 `routed_expert_up_shared` 的重叠写入。最终严格使用 `T.serial`，避免用未定义的流水顺序换性能。

### 5.2 小 tile 并未改善吞吐

K=64 或 N=64 都功能正确，但性能分别落在 `491.5795 / 68.6550 ms` 和 `482.9604 / 68.3492 ms`。单 buffer 后再试 K=64 仍是 `313.2079 / 46.1666 ms`；反向放大为 K=256 则是 `484.2256 / 67.9936 ms`，因为单个 stage tile 已到 64 KiB，反而压低 CTA 并发。down 的 N=64 也为 `305.0344 / 45.7394 ms`。原因是 reduction loop/CTA 数增加或 block 资源过重，权重访存和同步成本高于理论占用收益。

### 5.3 少线程和多线程都不如 256

- 128 threads：`792.7937 / 105.1254 ms`，warp 数不足；
- 512 threads（在单 buffer 后）：`316.0550 / 48.4101 ms`，block 过重、并发下降；
- 最终 256 threads 是经实测最平衡的配置。

### 5.4 `FullCol` 与 column swizzle 的边界

- stage-1 FullCol 的性能是 `578.7929 / 80.0066 ms`，明显不匹配本 GEMM 的 tile/layout；
- stage-1 column16 swizzle 对大规格略有利（`244.1342 ms`），但小规格变慢（`38.7586 ms`）；
- stage-2 column16 两个用例都更慢；
- 由于比赛给定两个规格，最终选择两者都稳健的 row16，而不是针对单一大规格过拟合。

### 5.5 fast math 与 launch-bounds 没有形成稳健收益

- MACA `-use-fast-math` 的功能测试通过，但性能 `338.5914 / 50.1863 ms`，略慢且引入数值路径变化，因此最终删除环境变量控制分支；
- `T.annotate_min_blocks_per_sm(2)` 在大规格略慢、小规格略快，差异接近测量波动，未保留为默认；
- down row8 与 row16 基本持平，row28 略慢，也未增加复杂度；
- 相比之下，单 buffer 后第一阶段 row8 的大规格收益经两次重复出现，故只将第一阶段改为 row8，第二阶段仍为 row16。
- 将 `min_blocks_per_sm=2` 与最终 row8 schedule 组合后得到 `243.5154 / 38.5250 ms`，仍是大规格变慢、小规格轻微变化，故不引入 launch-bounds 约束。
- gate activation 压缩为 FP16 后复用 gate 累加器虽通过正确性，却必须在第二个 reduction loop 重新读取 input，性能退化到 `308.7221 / 47.1000 ms`，说明本题第一阶段优先保证一次 input load 供 gate/up 两次 GEMM 共享。
- down 阶段单独扩大到 512 threads 也退化至 `275.2145 / 42.4989 ms`，说明该阶段同样以 256 threads 的 CTA 并发更平衡。
- down 阶段单独缩小到 128 threads 同样退化至 `281.0836 / 42.6746 ms`；至此上下界均被排除，保留 256 threads。

---

## 6. 最终代码结构

### 6.1 本地 benchmark 接入

`custom_fusedmoe.py` 的最终默认值：

```python
block_token = 128
block_dhidden = 128
block_dexpert = 128
threads = 256
num_stages = 1
swizzle_panel = 8
swizzle_panel_down = 16
gemm_policy = "full_row"
single_weight_buffer = True
```

额外调优参数都放在 `backend` 之后，原有的 positional constructor prefix 不变。`tools/tune_moe.py` 的默认值已经和最终默认内核对齐；如需复现失败候选，可用显式参数，例如：

```bash
cd /data/metax-race
source ./activate.sh

# 最终默认配置
python tools/tune_moe.py --block-dhidden 128 --block-dexpert 128 --mode all

# 对照：关闭单 shared buffer
python tools/tune_moe.py --block-dhidden 128 --block-dexpert 128 \
  --no-single-weight-buffer --mode performance
```

### 6.2 独立 OJ 提交文件

`race_tests/moe/submission.py` 不依赖 benchmark、reference 或相对导入，包含：

1. `_moe_forward_kernel`：同样的 FullRow + stage-1 row8 / stage-2 row16 + 单 weight tile 策略；
2. `_KERNEL_CACHE`：按 `(H, I, E, padded token 总数, valid token 总数, M blocks)` 缓存编译结果；
3. `_WORKSPACE_CACHE`：缓存 FP16 `up_logits` workspace；
4. `run_kernel(...)`：固定 10 参数、无 Tensor 类型标注、不做同步、原地写 `out`。

### 6.3 OJ ABI 本地对拍

新增 `tools/test_moe_submission.py`，覆盖：

- 不等长 expert group（包括非 128 的尾部）；
- 空 expert；
- padded 行不应被写入；
- FP32 routed weights；
- E+1 offsets；
- 重复调用验证 kernel/workspace cache；
- 一个公开小规格 H=3584/I=1024/E=4 的 ABI smoke case。

已通过日志：[`../logs/moe-submission-abi-row8-20260711T151056Z.log`](../logs/moe-submission-abi-row8-20260711T151056Z.log)。

---

## 7. 最终验证结果

### 7.1 官方入口

执行：

```bash
cd /data/metax-race
./run-race-test.sh moe
```

最终官方日志：[`../logs/moe-final-row8-official-20260711T150830Z.log`](../logs/moe-final-row8-official-20260711T150830Z.log)。

| 性能用例 | 基线 | 最终默认 | 加速比 | 延迟下降 |
|---|---:|---:|---:|---:|
| H=7168, I=2048 | 425.8014 ms | **243.4768 ms** | **1.7488×** | **42.82%** |
| H=3584, I=1024 | 62.3782 ms | **38.6093 ms** | **1.6156×** | **38.10%** |

两组官方 functional case 均通过。row8/row16 在调优脚本的两次重复为 `243.4042 / 38.5252` 与 `243.2124 / 38.5337 ms`，官方入口为 `243.4768 / 38.6093 ms`；大规格收益稳定，小规格差异处于正常 timing variation。

### 7.2 静态与接口检查

已执行：

```bash
python -m py_compile \
  tilelang-metax/race_tests/moe/custom_fusedmoe.py \
  tilelang-metax/race_tests/moe/submission.py \
  tools/tune_moe.py tools/test_moe_submission.py

git -C tilelang-metax diff --check
```

同时检查：

- `RoutedMoEKernel` 的旧 positional 参数前缀仍将 `backend` 保持在原位置；
- `submission.run_kernel` 只有教程要求的 10 个参数且没有类型标注；
- `./preflight.sh --require-gpu --json` 返回 `software_ready=true`、`gpu_ready=true`。

---

## 8. 复现、提交与后续工作

### 8.1 复现最终本地成绩

```bash
cd /data/metax-race
source ./activate.sh
./preflight.sh --require-gpu
./run-race-test.sh moe
```

### 8.2 验证提交 ABI

```bash
cd /data/metax-race
source ./activate.sh
python tools/test_moe_submission.py --public-shape
```

提交时使用：

```text
tilelang-metax/race_tests/moe/submission.py
```

不要提交本地 benchmark、日志或 `custom_fusedmoe.py` 的 import wrapper；OJ 只需要 `submission.py` 暴露的 `run_kernel`。

### 8.3 尚存风险与建议

1. **SDK 版本风险**：当前是 MACA 3.7.1.5，不是比赛指定的 3.7.2.1。切换镜像/挂载后首先重建 TileLang，再跑完整验证。
2. **OJ 隐藏 shape 风险**：当前已验证公开的 128 对齐 H/I 形状和不等长 token group；若 OJ 增加非 128 对齐 H/I，需要确认 TileLang copy/GEMM 尾块语义后再泛化。
3. **性能波动**：最优候选之间的 0.1%～0.5% 差异可能属于时钟、缓存或 sGPU 干扰。最终选择的是大规格重复出现收益、且小规格差异不显著的 stage-1 row8 / stage-2 row16 + FullRow + 单 buffer 方案。
4. **下一轮方向**：若需要继续冲榜，应先在 MACA 3.7.2.1 上重建并重跑；gate accumulator 压缩、stage-2 单独改线程数已被本轮数据排除。后续只值得探索能避免重复读取 input 的融合方式、经公开/隐藏 shape 驱动的专用 dispatch，或更低层的 MACA 代码生成改动。
