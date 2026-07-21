# 当前线上 OJ 合同对齐：FP16 route weight、padded zero 与官方 case 1

日期：2026-07-21（UTC）

## 结论

standalone `submission.py` 与本地 OJ 模拟器已按当前线上题目说明对齐：

- `routed_expert_weights` 从本地旧测试的 FP32 改为线上声明的 FP16；
- input/workspace/output 按 expert-grouped M=128 padded storage 访问；
- route weight 继续通过 compact `group_offsets` 访问；
- tail/padding 输出不再保留调用方 sentinel，而是明确写为精确 0；
- metadata tensor annotation 按实际 `.shape[0]` 专门化，因此兼容 E 个 start offsets 和带
  terminal sentinel 的 E+1 形式；
- 静态提交检查禁止除 `torch.empty` workspace allocation 以外的 PyTorch 调用，并强制
  route-weight `T.Tensor` 使用 FP16 submission dtype。

这次变更只调整 standalone OJ 提交面和测试/provenance，不修改正式 compact performance
harness 的 `custom_fusedmoe.py`，也不构成新的性能晋级。正式胜者仍是 H10F5。

## 来源与冲突审查

当前线上说明由用户提供，入口为固定十参数 `run_kernel`，要求 TileLang GPU compute、FP16
input/weights/workspace/output、INT32 metadata 和 padding output=0。其引用的历史 TileLang-MetaX
commit 为 `ee6db4376484f2f7270183c01fd0d90f794965cb`。

直接审计该 commit 的 `race_tests/moe/custom_fusedmoe.py` 后发现它实际上使用 compact
input/intermediate/output：

```text
m_start = m_start_padded - group_padded_offsets[e] + group_offsets[e]
```

且 offsets annotation 长度为 E。它与当前文字合同的 padded input/output 不同，因此只作为算法
参考，不能覆盖当前 OJ 文字合同。

当前线上文字本身仍有未决歧义：公开 `group_sum` 2272/4544/9088 与 M=128 padded row count
不能直接同时成立；两专家示例 offsets `[0, 3]` 暗示 E 长度，而旧本地指南曾写 E+1。当前实现
不在 host 读取 metadata 值，只从允许的 tensor shape 做 annotation 专门化；真实线上 fixture
应作为最终权威。Agent 知识条目 `knowledge/curated/p9-moe-official-contract.md` 保留完整 provenance、
已确认事实和未决项。

## 实现变化

`benchmarks/tilelang-moe/submission.py`：

1. route-weight annotation 改为 FP16 `dtype`；
2. offsets annotation 使用运行时 tensor shape 产生的 JIT specialization；
3. FC2 tail store 增加 `else: out[...] = 0.0`；
4. 其余 H10F5 schedule 保持不变。

`benchmarks/tilelang-moe/test_moe_submission.py`：

1. route weights 全部改为 FP16；
2. reference/output padding 初始化为 0，但被测 `out` 每轮先填 NaN，证明 kernel 主动清零；
3. correctness 改用线上 `atol=rtol=0.01, equal_nan=false`；
4. 默认 metadata 使用 E 个 start offsets；另有 E+1 sentinel case；
5. `--public-shape` 使用当前表中完整可解释的 case 1：16 experts、H=2048、I=8192、
   valid group sum=2272。本地 padded-storage 解释采用每专家 142 valid rows，得到 4096 storage
   rows和 32 个 M blocks。

## 真实 C500 证据

MetaX C500 上执行：

```bash
bash scripts/test-moe-submission.sh --fuzz-only
bash scripts/test-moe-submission.sh --public-shape
```

通过：

```text
fuzz-boundary-random
fuzz-exact-one
fuzz-skew-tiny
fuzz-tail-zero
fuzz-terminal-offset-sentinel
uneven-smoke
official-case-1: E=16 H=2048 I=8192 valid=2272 padded=4096 blocks=32
```

每个 case 调用两次 submission；`out` 在调用前填 NaN，所有 valid rows 对 FP32 reference 在
`atol=rtol=0.01` 下通过，所有 padding rows 与 FP16 zero 做 `atol=rtol=0` 精确比较通过。
E 和 E+1 offsets 均成功编译并运行。

## 回归与策略验证

- submission policy：通过；
- policy targeted tests：`3 passed`，覆盖 checked-in source、FP32 route annotation 拒绝、
  `torch.matmul` 拒绝；
- MetaX pytest：`89 passed`；
- deterministic Stage 2 provenance：9 versions / 32 hashed sources，通过；
- `custom_fusedmoe.py` 未修改，正式 H10F5 性能证据不变。

由于这次变更改变 OJ route dtype 和 padding store，旧 submission timing 不能直接作为新版本线上
性能结论。当前只声明真实 C500 correctness/compile 通过，不声明新的 10/100 或 OJ 排名收益。
