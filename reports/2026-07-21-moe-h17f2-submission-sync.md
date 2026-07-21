# H17F2 formal-SOTA to online-submission synchronization

Date: 2026-07-21

## Purpose

A formal SOTA promotion is no longer complete when only
`benchmarks/tilelang-moe/custom_fusedmoe.py` changes. The accepted mechanism must be adapted to the
standalone ten-argument online ABI in `submission.py`, bound by exact source hashes, and validated on
C500 before integration.

## Synchronized SOTA

- Formal winner: `H17F2`
- Deployment commit: `2a94659f4a0c7c65bd191367fd48d50fef47ce2f`
- Formal source SHA-256: `e9f83f74fe702d7adea33e3754fa618b5adfe7e103e68daaa58499e806fd0de9`
- Submission source SHA-256: `dd21d0940706ddb30023819d69c2a1405bf88a00433915ff330e03763114388b`
- Sync manifest: `benchmarks/tilelang-moe/sota_submission_sync.json`

## ABI-aware adaptation

The compact formal kernel proves that each mapped expert is valid and each compact valid-row
interval remains inside `group_sum`. The online submission uses a different address contract:

- input, intermediate workspace, and output are padded;
- route weights remain compact;
- `group_padded_offsets` and `group_offsets` must not be interchanged.

The submission therefore synchronizes H17F2 semantically:

1. both stages assert `expert_id` in `[0, num_experts)`;
2. both stages assert valid padded row intervals inside `total_padded_tokens`;
3. FC2 separately asserts compact route-weight intervals inside `total_valid_tokens`;
4. H18F1's sub-threshold positive-row assumption and H18F3's regressing scalar rewrite are not
   imported.

## Fail-closed promotion gate

`scripts/check_moe_sota_submission_sync.py` verifies exact SHA-256 identities for the formal kernel
and submission plus required adapted mechanism markers. It is invoked by
`scripts/test-moe-submission.sh`. Any later formal-source or submission change fails until the sync
manifest and evidence are deliberately refreshed.

Unit tests prove rejection of:

- a formal SOTA source change without submission synchronization;
- a submission change without manifest refresh;
- removal of an adapted SOTA mechanism.

## C500 validation

The following completed successfully on the MetaX C500 64 GiB system:

- submission AST policy: PASS;
- SOTA/submission hash and mechanism sync: PASS;
- sync and policy unit tests: `7 passed`;
- padded online official-shape plus fuzz: `7/7 passed`;
  - includes official Case 1 (`16/2048/8192`, valid token sum `2272`);
  - includes boundary, exact-block, skew, tail-zero, E-offset, and E+1-offset cases;
- deterministic Stage 2 provenance: `9 versions / 32 hashed sources` PASS;
- scoped MetaX repository pytest: `47 passed`;
- repository check: `3308 tracked files` PASS;
- `verify-maca.sh`: C500 and installed TileLang-MACA PASS;
- compact formal functional: Large/Small `2/2 passed` with seed `81394`;
- `git diff --check`: PASS.

An unscoped repository-root pytest was also attempted. It collected unrelated
`materials/Intro-ops` NVIDIA C-ABI tests and failed because their optional `libcamp_ops.so` was not
built. Those failures are outside the MoE/MetaX validation scope. The same run correctly detected
that the Stage 2 decision table had to be regenerated after `submission.py` changed; the table was
regenerated and its deterministic validator then passed.

## Performance scope

This change synchronizes the accepted H17F2 mechanism and proves padded-ABI correctness. It does not
claim that the online padded submission independently reproduces the compact harness's
`+4.0891%/+4.1297%` ABBA result. A standalone performance claim still requires its own OJ-ABI
10/100 replication and adjacent comparison.
