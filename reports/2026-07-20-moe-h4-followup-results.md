# H4 后续：真实 MetaX C500 实测与 H4F1 落地

日期：2026-07-20（UTC）

## 结论

唯一确定性胜者是 H4F1：保留 H4 的 shared FC2 A tile，并把每个 token 的 route weight
在 FC2 epilogue 前加载到 fragment，一次加载后跨输出列复用。它保持
`RoutedMoEKernel` API、submission ABI、官方 shape/seed、FP16 输入、FP32 accumulate 与
`atol=rtol=0.01` 不变。两次完整正式运行和相邻 ABBA 的 mean、median 均超过 0.5%，
且 workload-matched mcProfiler 一致显示 Large FC2 global-read instructions 从约
62.68M 降至约 59.47M，无 private spill。因此将 exact candidate diff 落地。

- H4 不可变起点：目标提交 `d945c4ee9a46d9c48e864eb1bd17e61acdaf684e`，源码
  SHA-256 `c70e328a8d69ca5278fac4ec2ebf85fac1d099696a7f98c392c66b1ff7be1c78`。
- H4F1 落地源码 SHA-256：
  `2be93d4210d7d3bdda857887c70540c6c4da964946033fa95e58b114dd1fe47e`。
- agent 控制面先行提交：`72ce88049088ce8fc35465ae87abc8fb0697a841`；所有新硬件
  artifact 均在该提交之后产生，并绑定 campaign `h4-followups-20260720`。

## 正式环境与协议

- 设备 MetaX C500 64 GiB，PCI `0000:35:00.0`；104 AP（外部资料），driver
  `3.8.30`，BIOS `1.33.5.0`，MXMACA `3.7.1.5`。
- mxcc `1.0.0 (d9102a1572)`；PyTorch `2.8.0+metax3.7.1.3`；TileLang-MACA
  `0.1.12+maca.gitec48829b@ec48829bb61fcd55a366407c69355581733cceef`；
  mcProfiler `3.8.1.4+575f5a9f6d`。
- 环境指纹：
  `sha256:dad31bd899b7cbf62576ce37b7f11c1b9ab31ac1454fb1c7fbd44fe7fdd6152e`。
- workload 为官方 `functional-large`、`functional-small`、`performance-large`、
  `performance-small`；每次正式 timing 为 warmup 10 / timing 100，保留全部有序样本，
  outlier policy `none`。时钟未人为锁定；每次 timing 后保存 `mx-smi` 热/钟状态。
- 每个实际候选都按 correctness → 10/100 → 独立 mcProfiler 执行。四个 workload-matched
  profiler kernel、原始 DB、native output、解析 JSON 和 invocation manifest 均在相应 agent
  artifact 中，不以一个 profile 代替另一 workload/candidate。

## 新采集 H4 基线

证据：`/data/metax-c500-autotune-agent/artifacts/p9/baseline-6927e48958a498a9637f9d4b.json`。

- Large/Small correctness 均 0 mismatch；最大绝对误差分别 `0.005859375`、`0.0078125`。
- mean `240.61594650268555 ms`；median `240.86566352844238 ms`；stddev
  `1.1966599755275875 ms`；p90 `242.07406539916994 ms`；RSD `0.0049733`。
- Large FC1/FC2 cycles：`126929.95 / 54721.85 Kcycles`；Large FC2 global-read
  instructions `62679130`；private read/write `0/0`。
- Small FC1/FC2 cycles：`125889.21 / 54708.56 Kcycles`；Small FC2 reads
  `62671696`。

## 所有实际候选

提升以新采集 H4 为唯一对照；正数表示更快。

| 候选 | mean / median / stddev / p90 (ms) | mean / median 提升 | correctness | Large profiler 归因 | 决定 |
|---|---:|---:|---|---|---|
| H4F1 run 1 | 238.627467 / 238.780930 / 0.983709 / 239.810767 | +0.8264% / +0.8655% | Large/Small 0 mismatch | FC2 `51334.64 Kcycles`，reads `59466925`，private 0/0；FC1 `127079.01 Kcycles` | 进入复验 |
| H4F1 run 2 | 238.688547 / 238.639359 / 1.060628 / 240.104020 | +0.8010% / +0.9243% | Large/Small 0 mismatch | FC2 `54018.25 Kcycles`，reads `59466199`，private 0/0；FC1 `126463.26 Kcycles` | 进入 ABBA |
| H4F2 FC1 A shared | 247.757614 / 247.988224 / 1.135105 / 248.906676 | -2.9681% / -2.9571% | Large/Small 0 mismatch | FC1 `132997.94 Kcycles`；vectorized lowering 使 read instruction 计数改变，但 48 KiB shared/residency 代价使 cycles 增加；private 0/0 | 拒绝 |
| H4F3 FC2 BK32 | 278.959370 / 278.885763 / 1.627511 / 280.936729 | -15.9355% / -15.7848% | Large/Small 0 mismatch | FC2 `94667.79 Kcycles`，reads `62682362`；K-loop 翻倍/局部性损失远超 shared-footprint 收益；private 0/0 | 拒绝 |
| H4F4 copy width8 | 240.854584 / 240.983294 / 1.098943 / 242.163383 | -0.0992% / -0.0488% | Large/Small 0 mismatch | FC2 `56020.02 Kcycles`，reads `62685775`，未减少读取；默认 lowering 已等价；private 0/0 | 拒绝 |

Small profiler 同样独立采集：H4F1 run1 FC2 `53469.95 Kcycles`/reads `59466399`，
run2 `53467.74`/`59459334`；H4F2 FC1/FC2 `132850.53/54869.32 Kcycles`；
H4F3 FC2 `100127.29 Kcycles`；H4F4 FC2 `52876.11 Kcycles`。

候选可追溯信息：

- H4F1 run1：experiment `experiment-79b6f91bf716c4069a665c53`，source commit
  `fb21c4041becc32bb0c3149a57687bb359f2e890`，artifact
  `experiment-16147ea217c235ce0d038223.json`。
- H4F1 run2：experiment `experiment-e33197458d0aae1b791f9d5b`，source commit
  `da1e1f456d2e6bf39bf6aacb4304c0be51f4c7e3`，artifact
  `experiment-f0856c382369e698014e9d30.json`。
- 两个 H4F1 exact diff 均为
  `sha256:c3d0cf04bcb356646df68decfa52867047089b546591ff2e1dcdfc5dd6bd459e`。
- H4F2：experiment `experiment-33331c81f09335a0cf4ab5b4`，source commit
  `99ca2cca77d8ee047439b2f6c63dad2960ba1992`，diff
  `sha256:79a0be25382928b79578401915fb94cc6060a3ea81ab577467e7759656760a00`，
  artifact `experiment-4227c73ca46cbe4368faaa1a.json`。
- H4F3：experiment `experiment-e86a7396659551d1519f054f`，source commit
  `132a34e30d549d0714c02c59c468477bc5b69be3`，diff
  `sha256:b2a97554368bcbdd5335b81beb9b349030064f2701a7d07372c6344ca9ba5148`，
  artifact `experiment-a3dc6135afcf1e7553b742b8.json`。
- H4F4：experiment `experiment-8541d0772c5caa314c0ba5cf`，source commit
  `81e6544f4116872de4834e33d2012f72e964598e`，diff
  `sha256:c68480dd3435713a135d5a014e099592661666b2f6229bf12e29513ca607b631`，
  artifact `experiment-7571d6528c1df42f9e8be3ad.json`。

## 相邻 ABBA

证据：`/data/metax-c500-autotune-agent/artifacts/p9/abba-1653a1522cbebe426954d179.json`。
严格顺序 H4-r1、H4F1-r1、H4F1-r2、H4-r2；每段均 10/100、无样本剔除：

| 段 | mean (ms) | median (ms) | stddev (ms) |
|---|---:|---:|---:|
| H4-r1 | 240.089259 | 240.091518 | 1.188056 |
| H4F1-r1 | 238.112119 | 237.924089 | 1.006752 |
| H4F1-r2 | 238.597599 | 238.641661 | 1.090009 |
| H4-r2 | 240.453201 | 240.697086 | 1.195532 |

H4 aggregate mean/median `240.271230 / 240.394302 ms`；H4F1 为
`238.354859 / 238.282875 ms`；聚合提升 `+0.7976% / +0.8783%`。两个相邻
challenger 段分别获得 mean `+0.8986%/+0.6966%`，median
`+1.0276%/+0.7291%`；stable gate 为 true。

## 解释边界

H4F1 的收益不能外推为“所有 metadata cache 都有效”：本结论只覆盖当前 C500、上述软件
版本、官方 Large+Small 组合和 H4 shared FC2 A tile。H4F2 证明 FC1 的 48 KiB shared
placement 在当前 combined Gate/Up footprint 下回退；H4F3 排除当前 BK32；H4F4 表明显式
width-8 没有改善默认 copy lowering。未执行方向的能力/ABI 边界见增量调研报告。

## 目标仓库落地门禁

均从 `/data/MoE-opt` 运行，先设置 `TILELANG_HOME=/opt/tilelang-metax`，再执行
`source scripts/activate-maca.sh`；未改写 `HOME`：

- MetaX backend `python -m pytest -q`：`86 passed in 25.01s`；
- `python scripts/check_repository.py`：`3350 tracked files inspected`；
- `scripts/verify-maca.sh`：C500、MXMACA、TileLang MACA target 全部核验通过；
- `scripts/run-moe.sh`：官方 Large/Small functional 均 PASS；脚本内诊断 timing 为
  `206.42572266 / 33.44409668 ms`，不属于正式 10/100 接受数据；
- `scripts/test-moe-submission.sh --public-shape --fuzz`：uneven smoke、public small 与
  4 个 fuzz/boundary case 全部 PASS；
- stage2 profiler decision table 已由仓库确定性生成器按新源码 SHA 刷新；
- `git diff --check`：通过。

第一次 pytest 诊断命令错误地在 `source` 之后才设置 `TILELANG_HOME`，因此 collection
阶段出现 3 个 `ModuleNotFoundError: tilelang`；该运行没有执行测试、不是代码失败，也没有
被计为通过记录。随后按 activation 脚本的正确环境顺序重跑上述完整测试集。
