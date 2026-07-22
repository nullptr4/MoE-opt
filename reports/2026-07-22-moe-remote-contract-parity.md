# C500 Routed-MoE remote-contract / local-parity audit

Date: 2026-07-22 UTC

This report replaces parameter-parity assumptions with a versioned contract and a
deterministic local fingerprint boundary. It does not change the byte-identical H17F2
formal kernel and it does not claim that local C500 execution reproduces the online OJ.

## Contract identities

- Contract: `benchmarks/tilelang-moe/remote_contract.json`
- Contract version: `2026-07-22.1`
- Canonical fingerprint: `sha256:76213668b712444f98522210534bd01faba7b1d4c09bee17f44ff9fe76c6b1d8`
- Ten-argument submission ABI fingerprint: `sha256:f4151e2be522b52490ef26bf2c204e1e0b66bb2fff69f84950de01be124f8fa2`
- Formal ABI: compact 11-tensor `RoutedMoEKernel`, including caller-owned `up_logits`
- Submission ABI: compact group_sum 10-argument `submission.run_kernel`, with private
  `up_logits` workspace

Each contract field carries one of `published`, `remotely_observed`, `derived`,
`local_assumption`, or `unknown`, plus sources, access date, limitations, and (for
unknowns) the evidence needed to close it. The two online exit-11 observations retain
`root_cause_status=unknown`; removing `T.assume` did not establish a cause.

## Deterministic mismatch matrix

| Property | Old local Large | Old local Small | Published remote cases |
|---|---:|---:|---|
| hidden | 7168 | 3584 | 2048, 7168, 7168 |
| intermediate | 2048 | 1024 | 8192, 2048, 2048 |
| experts | 8 | 4 | 16, 32, 64 |
| group_sum | 131072 | 65536 | 2272, 4544, 9088 |
| ABI | formal compact 11 tensors | formal compact 11 tensors | submission compact 10 arguments |
| routing source | local softmax/top-k, seed 81394 | local softmax/top-k, seed 81394 | remote unknown; local mirror uses uniform `torch.randint` |
| metadata grid | exact compact `sum(ceil(size/128))` | same | local mirror uses derived `ceil(group_sum/128)+E`; remote construction not fully proved |
| timing | warmup 10 / repeat 100 | warmup 10 / repeat 100 | 5/30, 5/20, 5/20 |
| scoring use | old two-case arithmetic sum/mean | same objective | per-case metrics known; aggregate unknown |

Thus the old Large/Small results remain useful as guard/proxy evidence only. A coincidentally
matching hidden/intermediate dimension does not make either workload a remote case.

## Known and unresolved

Published/observed facts include the three dimension rows and order, ten argument names,
compact group_sum input/output rows, FP16 data and INT32 metadata dtypes, and the published
warmup/iteration counts. Local/cited implementation evidence derives compact private
workspace rows and metadata formulas; those derivations are not silently promoted to
remote facts.

Unresolved policy fields include exact routing and route-weight generation/distribution,
offset sentinel convention, tensor stride/contiguity requirements, private workspace
policy, correctness `equal_nan`, case-order cache/compiler lifecycle, aggregate scoring,
and the exact online compiler/runtime inventory. Closing them requires evaluator source or
schema, an organizer statement, or captured remote invocation/toolchain evidence as stated
in the manifest.

## Commands and evidence boundary

```bash
python scripts/check_moe_remote_contract.py
python scripts/check_moe_sota_submission_sync.py
scripts/test-moe-submission.sh --remote-shapes
scripts/test-moe-submission.sh --benchmark-remote \
  --report-json data/benchmarks/c500-64g/remote-submission-parity.json
```

The hardware report retains per-case correctness, every contiguous-call timing sample,
mean/median/stddev/p90, warmup and iteration counts, environment and source hashes, all
tensor shapes/dtypes/strides, routing/offset/metadata hashes, and contract/ABI fingerprints.
It reports `aggregate_status=unknown` and `parity.status=LOCAL_PROXY_ONLY`. Local success is
real C500 evidence for these local fixtures, not evidence of exact online equivalence.
