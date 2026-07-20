# MetaX C500 TileLang Fused MoE 激进优化方案

> 目标算子：`race_tests/moe/custom_fusedmoe.py`
> 目标硬件：MetaX C500
> 重点方向：矩阵融合、低精度模拟高精度、中间张量压缩、跨 Kernel 流水融合

---

# 1. 方案总览

进一步可研究的激进优化主要分为三类：

1. 跨矩阵、跨 Kernel 的融合；
2. 使用多个低精度分量模拟较高精度计算；
3. 改变中间张量和权重的数值表示。

需要注意：

```text
Gate → SiLU → 与 Up 相乘
```

中间存在非线性，因此不能简单预计算：

```text
W_gate × W_up × W_down
```

把整个专家 MLP 精确合并成一次普通 GEMM。真正可行的是：

```text
执行层融合
+ 数据布局融合
+ Epilogue 融合
+ 流水融合
+ 低精度计算重构
```

---

# 2. FC1 Gate/Up 超宽矩阵融合

## 2.1 原始计算

```text
G = X @ W_gate^T
U = X @ W_up^T
Z = SiLU(G) × U
```

## 2.2 融合形式

在输出 N 方向拼接权重：

```text
W_gu = concat(W_up, W_gate)
Y = X @ W_gu^T
```

得到：

```text
Y = [U | G]
Z = U × SiLU(G)
```

最终变为：

```text
一次宽 GEMM
+ SwiGLU Epilogue
```

## 2.3 推荐 Tile

```text
A[BM, BK]
B_gu[2BN, BK]
C_gu[BM, 2BN]
```

Epilogue：

```text
Z[i,j] = C_gu[i,j] × SiLU(C_gu[i,j+BN])
```

推荐起点：

```text
BM128
logical BN64
physical fused BN128
BK64
threads256
stage1
```

## 2.4 与当前实现对比

当前：

```text
两个 [128,128] FP32 accumulator
+ 两次 GEMM
```

候选：

```text
一个 [128,128] FP32 accumulator
+ 一次更宽 GEMM
```

## 2.5 可继续组合

```text
Gate/Up interleave
+ combined GEMM
+ SwiGLU Epilogue
+ intermediate quantization
```

## 2.6 风险

- 更宽 GEMM 的 MACA lowering 未必更优；
- 权重拼接和 prepack 必须在模型加载阶段完成；
- physical BN 增大可能降低 occupancy；
- 需要检查 Gate/Up 对应列在线程和 fragment 中的映射。

---

# 3. FC1 Epilogue 直接量化，FC2 使用 INT8 GEMM

## 3.1 当前路径

```text
FC1 FP16 输入
→ FP32 累加
→ SwiGLU
→ FP16 up_logits 写显存
→ FP16 up_logits 读显存
→ FP16 FC2
```

## 3.2 激进路径

```text
FC1 FP16 输入
→ FP32 累加
→ SwiGLU
→ 计算 block scale
→ INT8 量化
→ 按 FC2 layout 打包
→ INT8 up_logits 写显存
→ W8A8 FC2
→ INT32/FP32 累加
→ 反量化
→ route weight
```

数学表示：

```text
Z ≈ s_z × q_z
W_down ≈ s_w × q_w

Output ≈ s_z × s_w × INT8_GEMM(q_z, q_w)
```

## 3.3 潜在收益

- `up_logits` FP16→INT8，读写流量约减半；
- Down 权重 FP16→INT8，缓存占用约减半；
- FC2 可走 W8A8 路径；
- 量化和 packing 融入 FC1 Epilogue；
- FC2 直接读取目标 layout。

## 3.4 推荐量化粒度

```text
up_logits：per-row 或 per-row×BK
W_down：per-output-channel 或 per-BK block
```

## 3.5 推荐组合

```text
FC1：
FP16 combined Gate/Up GEMM
+ FP32 SwiGLU
+ INT8 intermediate quantization

FC2：
INT8 activation
+ INT8 prepacked W_down
+ INT32 accumulation
+ FP32 scale/dequant Epilogue
+ route weight
```

## 3.6 风险

- 动态量化有额外开销；
- scale 生成需要 reduction；
- INT8 实际吞吐必须显著高于 FP16；
- 需要验证 C500 W8A8 lowering；
- 可能无法通过默认误差容限。

---

# 4. 两个低精度数模拟较高精度数

## 4.1 基本表示

```text
x = x_hi + x_lo
```

构造：

```text
x_hi = quantize_low_precision(x)
x_lo = quantize_low_precision(x - x_hi)
```

矩阵乘法：

```text
(A_hi + A_lo)(B_hi + B_lo)
```

展开：

```text
A_hi B_hi
+ A_hi B_lo
+ A_lo B_hi
+ A_lo B_lo
```

即用四次低精度 GEMM 模拟一次较高精度 GEMM。

## 4.2 三项近似

忽略最小的低低项：

```text
C ≈ A_hi B_hi
   + A_hi B_lo
   + A_lo B_hi
```

由四次 GEMM 降为三次。

## 4.3 对当前算子的意义

当前已经使用：

```text
FP16 × FP16 → FP32 accumulate
```

所以把 FP16 再拆成两份 FP16 通常不值得。更有价值的是：

```text
使用两个 INT8 slice 模拟一个 FP16 值
```

以更高吞吐的 INT8 路径恢复接近 FP16 的数值质量。

---

# 5. 双 INT8 残差表示 FP16

## 5.1 数值表示

```text
A ≈ s_A(q_A0 + αq_A1)
B ≈ s_B(q_B0 + βq_B1)
```

乘法展开：

```text
AB ≈ s_A s_B [
    q_A0 q_B0
  + β q_A0 q_B1
  + α q_A1 q_B0
  + αβ q_A1 q_B1
]
```

## 5.2 INT8×1：普通 W8A8

```text
C ≈ q_A0 q_B0
```

特点：一次 INT8 GEMM，最快，误差最大。

## 5.3 INT8×2：一侧残差

只分解权重：

```text
B ≈ B_hi + B_lo
C ≈ A_hi B_hi + A_hi B_lo
```

组合：

```text
Activation：单 INT8
Weight：双 INT8
```

特点：

- 两次 INT8 GEMM；
- Weight residual 可离线预处理；
- Activation 只量化一次；
- Intermediate 只保存一份 INT8；
- 最容易落地。

## 5.4 INT8×3：双侧一阶补偿

```text
C ≈ A_hi B_hi
   + A_hi B_lo
   + A_lo B_hi
```

忽略：

```text
A_lo B_lo
```

特点：三次 INT8 GEMM，精度更高，但对吞吐比要求更高。

## 5.5 INT8×4：完整双残差

```text
C = A_hi B_hi
  + A_hi B_lo
  + A_lo B_hi
  + A_lo B_lo
```

特点：最接近原始 FP16，但开销最大。

---

# 6. 推荐的双低精度组合

## 6.1 R1：权重双 INT8，激活单 INT8

```text
Z ≈ s_z q_z

W_down ≈ s_w0 q_w0
       + s_w1 q_w1
```

FC2：

```text
C0 = INT8_GEMM(q_z, q_w0)
C1 = INT8_GEMM(q_z, q_w1)

C = s_z s_w0 C0
  + s_z s_w1 C1
```

完整组合：

```text
FP16 combined Gate/Up
→ FP32 SwiGLU
→ INT8 activation
→ 两次 INT8 FC2 GEMM
→ FP32 residual merge
→ route weight
```

风险：高。

## 6.2 R2：激活双 INT8，权重单 INT8

```text
Z ≈ s_z0 q_z0 + s_z1 q_z1
W_down ≈ s_w q_w
```

FC2：

```text
C0 = INT8_GEMM(q_z0, q_w)
C1 = INT8_GEMM(q_z1, q_w)
```

特点：两个 INT8 intermediate 合计约等于一个 FP16 intermediate，权重只保存一个 INT8 版本。

风险：高。

## 6.3 R3：双侧 INT8×3

```text
Z = Z_hi + Z_lo
W = W_hi + W_lo

C ≈ Z_hi W_hi
  + Z_hi W_lo
  + Z_lo W_hi
```

风险：极高。

## 6.4 R4：动态精度选择

```text
普通 tile：INT8×1
残差较大：INT8×2
高精度风险：INT8×3
异常 tile：回退 FP16
```

选择特征：

```text
tile max_abs
exponent span
quantization residual norm
estimated relative error
activation saturation ratio
expert identity
K tile index
```

风险：极高。

---

# 7. 性能成立条件

粗略要求：

```text
INT8×2：INT8 有效吞吐明显超过 FP16 的约 2 倍
INT8×3：INT8 有效吞吐明显超过 FP16 的约 3 倍
INT8×4：INT8 有效吞吐明显超过 FP16 的约 4 倍
```

实际还需覆盖：

```text
动态量化
scale 计算
residual 生成
多次 GEMM
FP32 merge
反量化
额外 shared/register
额外同步
```

必须先做：

```text
1. 同 shape FP16 GEMM vs INT8 GEMM
2. FC1 Epilogue INT8 量化吞吐
3. 单/双/三项 INT8 精度
4. 双 INT8 FC2 总时间
5. scale/dequant/merge Epilogue 时间
6. INT8 kernel 资源占用
```

---

# 8. FC1 与 FC2 Streaming Superkernel

## 8.1 不可直接单 CTA 完全融合

若：

```text
H=7168
BN2=128
```

FC2 需要 56 个 N tile。若每个 FC2 tile 自己重算 FC1，则 FC1 可能被重复 56 次。

## 8.2 Producer-Consumer Superkernel

```text
FC1 producer CTAs：
生成 Z tile
→ 写入 L2-resident ring buffer
→ 更新 semaphore

FC2 consumer CTAs：
等待 Z tile ready
→ 读取 Z tile
→ 更新 FC2 accumulator
→ 释放 ring-buffer slot
```

## 8.3 流水过程

```text
FC1 tile 0 → slot 0
FC1 tile 1 → slot 1
FC2 消费 tile 0
FC1 tile 2 → 重用 slot 0
FC2 消费 tile 1
```

## 8.4 潜在收益

- 减少 FC1/FC2 launch 间隔；
- intermediate 尽量停留在 L2；
- 不保存完整 `R×D` intermediate；
- FC1 与 FC2 可跨 tile 重叠；
- 可控制 ring-buffer 大小。

## 8.5 Ring Buffer Slot

```text
data
scale
expert_id
m_tile_id
k_tile_id
ready_flag
consumer_count
```

## 8.6 风险

- CTA 间 semaphore；
- global atomic；
- memory ordering；
- 多消费者同步；
- persistent worker 负载平衡；
- 跨 CTA 同步能力未知；
- 死锁风险。

风险：极高。

---

# 9. W4A16 权重压缩路线

## 9.1 FC1

```text
W_gate：INT4
W_up：INT4
X：FP16
```

组合：

```text
Gate/Up INT4 interleaved prepack
+ combined GEMM
+ fused dequant
+ FP32 SwiGLU
```

## 9.2 FC2

```text
up_logits：FP16 或 INT8
W_down：INT4
```

组合：

```text
W4A16 FC2
+ per-group scale
+ fused dequant
+ FP32 route-weight Epilogue
```

## 9.3 风险

- 解量化和 INT4 unpack 开销；
- scale load；
- 实际 MMA 支持未知；
- 随机权重下误差较大。

风险：高。

---

# 10. 数值近似矩阵融合

## 10.1 低秩分解

```text
W ≈ U V
X @ W ≈ (X @ U) @ V
```

只适合固定且确实低秩的真实模型权重，不适合随机 benchmark。

## 10.2 结构化稀疏

```text
2:4 sparsity
N:M sparsity
block sparsity
```

需要权重剪枝、重训练和 sparse MMA 支持，通常不适合严格 reference。

## 10.3 SiLU 多项式近似

```text
SiLU(x) ≈ p(x)
```

可组合：

```text
combined Gate/Up GEMM
+ polynomial SiLU
+ fused quantization
```

风险：中高。

---

# 11. 不推荐的极端方案

## 11.1 Strassen/Winograd

会增加矩阵加法、临时 buffer、调度复杂度和误差，通常难以击败原生 FP16 MMA。

## 11.2 双 accumulator 补偿

当前已有 FP32 accumulator。使用 `sum_hi/sum_lo` 或 Kahan 会增加寄存器、标量指令和依赖链，不适合作为性能主线。

## 11.3 FC1+FC2 单 CTA 完全融合

会造成 FC1 对不同 FC2 N tile 重算，除非硬件支持高效 CTA cluster 和共享中间 tile，否则不应采用。

---

# 12. 推荐实验优先级

| 顺序 | 方案 | 数学等价性 | 潜力 | 风险 |
|---|---|---|---|---|
| R1 | Gate/Up 宽矩阵 + SwiGLU Epilogue | 等价 | 高 | 中 |
| R2 | FC1 生成 INT8 intermediate + W8A8 FC2 | 近似 | 很高 | 高 |
| R3 | Down 权重双 INT8 residual | 近似高精度 | 高 | 高 |
| R4 | INT8×1/2/3 动态精度 | 近似可控 | 很高 | 极高 |
| R5 | FC1/FC2 producer-consumer ring buffer | 等价 | 很高 | 极高 |
| R6 | W4A16 Gate/Up/Down | 近似 | 高 | 高 |
| R7 | INT8 intermediate + ring buffer | 近似 | 极高 | 极高 |

---

# 13. 推荐组合配置

## 13.1 X1：宽矩阵融合版

```text
FC1:
BM128
logical BN64
physical BN128
BK64
threads256
stage1
Gate/Up combined GEMM
FP32 SwiGLU Epilogue

FC2:
BM128
BN128 或 256
BK64
threads256
stage1
FP16 Down GEMM
```

风险：中。

## 13.2 X2：INT8 Intermediate 版

```text
FC1:
BM128
logical BN64
physical BN128
BK64
combined Gate/Up
FP32 SwiGLU
per-row INT8 quantization

FC2:
BM128
BN256
BK64
INT8 activation
INT8 W_down
INT32 accumulation
FP32 dequant + route weight
```

风险：高。

## 13.3 X3：双 INT8 Weight Residual 版

```text
FC1:
combined Gate/Up
FP32 SwiGLU
INT8 intermediate

FC2:
W_down_hi INT8
W_down_lo INT8

C0 = A8 @ W8_hi
C1 = A8 @ W8_lo

Output = scale0 × C0 + scale1 × C1
```

风险：高。

## 13.4 X4：Streaming Superkernel 版

```text
Persistent producer:
combined Gate/Up
+ SwiGLU
+ 写入 L2 ring buffer

Persistent consumer:
读取 ring buffer
+ FC2
+ route weight

Scheduler:
expert-major
weight-stationary
reference-counted slots
```

风险：极高。

## 13.5 X5：终极组合版

```text
GPU counting partition
→ exact persistent scheduler
→ Gate/Up combined W4A16 或 FP16 GEMM
→ FP32 SwiGLU
→ dynamic INT8 residual encoding
→ L2 ring buffer
→ INT8×1/2/3 FC2 动态精度
→ FP32 merge
→ route weight
→ custom scatter/combine
```

风险：极高。

---

# 14. 最终推荐主线

优先顺序：

```text
第一步：Gate/Up 宽矩阵融合
第二步：FC1 Epilogue 生成 INT8 intermediate
第三步：普通 W8A8 FC2
第四步：W_down 双 INT8 residual
第五步：INT8×1/2/3 动态精度
第六步：FC1/FC2 persistent streaming superkernel
第七步：GPU routing + custom combine 全链路融合
```

最推荐首先验证的激进组合：

```text
Combined Gate/Up FP16 GEMM
→ FP32 SwiGLU
→ INT8 intermediate
→ 双 INT8 residual W_down
→ 两次 INT8 FC2 GEMM
→ FP32 scale merge
→ route weight
```

它同时使用：

```text
矩阵融合
Epilogue 融合
中间张量压缩
INT8 matrix unit
双低精度恢复精度
权重预处理
输出融合
```
