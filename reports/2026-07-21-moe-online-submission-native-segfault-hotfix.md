# Online evaluator native-segfault compatibility hotfix

Date: 2026-07-21

## Incident

The first reported online evaluation after H17F2 range contracts were adapted to the standalone
submission terminated with a native segmentation fault. The traceback contained CPython evaluator
and exception/traceback machinery but no recoverable Python exception, TileLang diagnostic, source
line, device fault, or core file.

The submitted source identity was:

```text
dd21d0940706ddb30023819d69c2a1405bf88a00433915ff330e03763114388b
```

The only executable delta from the previously C500-validated online adapter was ten `T.assume`
calls: expert-index bounds in FC1/FC2, padded valid-row intervals in FC1/FC2, and a compact route
interval in FC2. The formal H17F2 compact kernel uses those contracts successfully with the local
installed TileLang-MACA build, but the online evaluator's exact compiler/lowering build is not
available locally.

## Diagnosis and confidence

`T.assume` incompatibility is the strongest differential diagnosis, not a proven root cause. A
native crash cannot be attributed conclusively without the exact online toolchain, native error
preceding the traceback, or a core dump. The fix therefore minimizes executable change rather than
adding speculative guards or changing the MoE algorithm.

## Hotfix

- Restore `submission.py` exactly to the pre-assume byte sequence:
  `c44f7f4dcffe67d63a7ce1eed76c5570ac73654b76e6fb55fbf4721db64a1b9d`.
- Keep formal `custom_fusedmoe.py` and its measured H17F2 SOTA unchanged.
- Keep the padded input/workspace/output, compact route-weight indexing, FP16 route dtype, tail
  predicates, and padding-zero stores unchanged.
- Reject every `T.assume` call in the standalone online compatibility policy.
- Record H17F2 assumptions as an online-incompatible fallback in
  `sota_submission_sync.json`; do not claim their formal ABBA gain for this submission.

Re-enabling range assumptions requires a successful test under the exact online compiler/runtime,
not merely the local `0.1.12+maca.gitec48829b` build.

## Validation

After the exact rollback:

- submission AST policy and SOTA/fallback manifest: PASS;
- policy and synchronization unit tests: `8 passed`;
- real C500 padded online cases: `7/7 passed`;
- official Case 1 (`16 experts`, hidden `2048`, intermediate `8192`, valid token sum `2272`): PASS;
- boundary, exact-block, skew/tiny, tail-zero, E-offset, and E+1-offset cases: PASS;
- all tested padding output rows remained zero;
- both compact route and padded tensor offsets retained their prior semantics.

This local evidence establishes that the hotfix compiles and runs correctly on the available C500.
Only a new online evaluation can confirm that it removes the reported native failure.
