# Remote compact range alignment

Date: 2026-07-21

## Authoritative range clarification

The remote evaluator publishes three compact workloads:

| ID | E | H | I | group_sum |
|---:|---:|---:|---:|---:|
| 1 | 16 | 2048 | 8192 | 2272 |
| 2 | 32 | 7168 | 2048 | 4544 |
| 3 | 64 | 7168 | 2048 | 9088 |

All input, weights, private `up_logits` workspace and output tensors are FP16. Group metadata is
INT32 and `block_token=128`. Each published group sum is exactly `142 * E`; group sum is a physical
compact row count, not the sum of separately padded expert buffers.

## Corrected local contract

The standalone source is now compact-only:

```text
block_start = bx * 128
expert = group_idx_for_bx[bx]
token_offset = block_start - group_padded_offsets[expert]
data_start = group_offsets[expert] + token_offset
```

Both FC1 input/workspace stores and FC2 workspace loads/output stores use `data_start`. Route weights
use the same compact row identity. There is no padded storage branch and no FP32 route branch.

The local driver exactly reproduces the cited benchmark metadata:

```text
expert padded blocks = ceil((group_size + 1) / 128)
M metadata blocks = ceil(group_sum / 128) + E
```

E and E+1 offset lengths remain supported by shape specialization, but values are never read on the
host. `run_kernel` has no explicit synchronization and allocates/caches only the private FP16
`up_logits` workspace.

## Real C500 result

All three published dimensions compiled and passed two consecutive invocations against the FP32
PyTorch reference at `atol=rtol=0.01`:

```text
remote-case-1: E=16 H=2048 I=8192 group_sum=2272 blocks=34
remote-case-2: E=32 H=7168 I=2048 group_sum=4544 blocks=68
remote-case-3: E=64 H=7168 I=2048 group_sum=9088 blocks=135
```

Compact boundary, exact-block, skew/tiny, zero-route-weight and E+1 sentinel fuzz also passed. The
formal H17F2 research kernel remains unchanged; its optimized schedule and measured speedup are not
claimed for this conservative standalone source.

Warmup/iteration counts are retained as evaluator timing policy rather than correctness dimensions.
The supplied text unambiguously shows case 1 as warmup 5 / iteration 30, but the pasted formatting
does not preserve both values for cases 2 and 3. No missing timing values are invented in the local
report.
