# Online compact/padded compatibility baseline

> **Superseded by published remote ranges:** input, private workspace and output are FP16 compact
> tensors with exactly `group_sum` rows (2272/4544/9088). The dual-layout version was locally
> correct but unnecessarily broad and was never claimed as exact-online validated. See
> `2026-07-21-moe-remote-compact-range-alignment.md`.

Date: 2026-07-21

## Trigger and corrected diagnosis

The online evaluator ran two standalone submissions and both emitted `OJCHAL v1`, then crashed in
the Python traceback handler with exit code 11:

1. H17 assumptions adapted to the padded submission, SHA-256 `dd21d094...`;
2. the exact pre-assumption bytes, SHA-256 `c44f7f4d...`.

The second failure disproves the claim that removing `T.assume` fixes the incident. No Python
exception body, core file, exact online package inventory, or failing tensor metadata was supplied,
so a unique root cause is not proven.

Direct inspection of the cited TileLang-MetaX commit
`ee6db4376484f2f7270183c01fd0d90f794965cb` found a material contract conflict. Its benchmark
allocates input, workspace, route weights and output with exactly `group_sum` compact rows and maps
metadata blocks back to data with:

```text
m_start = m_start_padded - group_padded_offsets[e] + group_offsets[e]
```

The tutorial contract previously supplied by the user describes padded input/workspace/output.
Local validation had tested only that padded interpretation, so it could not reject a compact-online
fixture. The optimized H10F5/H17 code also targets the locally installed newer lowerer rather than
the cited source's conservative schedule.

## Compatibility implementation

The standalone ten-argument `run_kernel` now specializes without host metadata reads:

- `input_rows == route_rows`: compact address `raw_start + token_offset`;
- otherwise: padded address `block_start` and exact zero stores for tail rows;
- route tensor dtype selects a controlled FP16 or FP32 prim-func annotation;
- E and E+1 offset tensor lengths remain independent specialization dimensions;
- the GPU code uses the original two-stage, separate Gate/Up shared-buffer organization from the
  cited commit instead of H10F5 combined Gate/Up or H17 assumptions.

The formal compact H17F2 research source remains unchanged. No formal speedup is attributed to this
online compatibility baseline.

## Real C500 validation

The new source compiled and passed all tested address/dtype variants:

- padded uneven smoke, FP16 route;
- compact uneven smoke with extra empty metadata blocks, FP16 route;
- compact boundary smoke, FP32 route;
- Case 1 padded interpretation: 4096 storage rows, 2272 routes, 32 metadata blocks;
- Case 1 cited-commit compact interpretation: 2272 storage rows/routes, 34 metadata blocks;
- padded boundary 127/128/129, exact-block, skew/tiny, zero-tail, and E+1 sentinel fuzz.

Every test invokes the submission twice and compares valid output against the FP32 PyTorch
reference at `atol=rtol=0.01`; padded tails must be exact zero.

## Remaining uncertainty

Only another exact online run can prove the challenge wrapper accepts this source. If it still exits
11, the next isolation step must remove one host-side feature at a time, beginning with private
`torch.empty` workspace allocation and module-level caches, or obtain the hidden exception preceding
the wrapper's traceback-handler crash. Those possibilities are not claimed as causes without exact
online evidence.
