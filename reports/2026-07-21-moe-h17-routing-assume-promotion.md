# C500 Routed-MoE H17 routing-assumption promotion

Date: 2026-07-21. Device: MetaX C500 64 GiB. Formal comparator: H10F5 only at target commit
`ba3747b3b03349ec3a67c40c71d7dc92ac63f4f8`, source SHA-256
`338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`.

## Incremental first-hand sources

- [TileLang instruction semantics](https://tilelang.com/programming_guides/instructions.html),
  [transform API](https://tilelang.com/autoapi/tilelang/transform/index.html), and
  [pass configuration source](https://github.com/tile-ai/tilelang/blob/main/tilelang/engine/pass_config.py),
  MIT: explicit shared synchronization and the automatic ThreadSync opt-out motivated H17F1.
- [TileLang v0.1.10 release](https://github.com/tile-ai/tilelang/releases/tag/v0.1.10) and
  [`T.assume` API](https://tilelang.com/autoapi/tilelang/language/tir/op/index.html), MIT: the release
  records removal of redundant BufferLoad bounds checks and the API defines assumptions as true
  constraints for simplification. Direct PR 2122 access failed, so no claim relies on that body.
- [TileLang GEMM source](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/gemm_op.py),
  MIT: first-GEMM `clear_accum` plus exact installed MACA support motivated the resource-distinct
  FC1 follow-up H17F3.

The actual bodies and licenses were opened. CUDA persistent/TMA/WGMMA behavior was not transferred
to C500. The exact installed TileLang-MACA revision `ec48829b` was separately audited for
ThreadSync, `T.sync_threads`, `T.assume`, InjectAssumes and MACA `clear_accum` support.

## Formal hardware evidence

The fresh H10F5 baseline passed official Large and Small with zero mismatches. Its 100 ordered
samples after warmup 10, outlier policy `none`, measured `228.832898/228.806015 ms` mean/median,
standard deviation `1.361969 ms`, p90 `230.395290 ms`. Four workload-matched profiler launches
measured FC1 `117535.05/119011.17` and FC2 `51432.83/53807.40 Kcycles`.

| ID | Mechanism | Correctness and formal 10/100 | Profiler/generated-code conclusion | Decision |
|---|---|---|---|---|
| H17F1 | Replace automatic ThreadSync with explicit required barriers and contract the FC1 exchange | Large/Small 0 mismatch; `230.968402/230.843647 ms`; mean `-0.9332%`, median `-0.8906%`; stddev `1.187476 ms` | Generated barrier count fell from 11–12 to 10, but FC1 measured `121555.13/120444.17 Kcycles` and endpoint regressed. | Rejected; mechanism realized but slower. |
| H17F2 | State official routing expert and compact-row range invariants with `T.assume` while retaining safe tail legalization | R1 0 mismatch; `219.080630/219.331071 ms`, `+4.2617%/+4.1410%`; R2 0 mismatch; `218.867519/218.969728 ms`, `+4.3549%/+4.2990%` | Generated MACA source shrank from about 33.4 KB to 27.6 KB. R1 FC1 `112321.79/111502.84`, FC2 `49507.62/50286.59`; R2 FC1 `112212.17/112074.77`, FC2 `50517.68/50456.58 Kcycles`. Private traffic increased slightly, but cycle and endpoint reductions repeated. | Accepted twice; advanced to ABBA. |
| H17F3 | Peel FC1 k=0 and define the combined accumulator in first GEMM | Large/Small 0 mismatch; `231.296374/231.314175 ms`; mean `-1.0765%`, median `-1.0962%`; stddev `1.275160 ms` | FC1 private writes fell to `790272`, but FC1 cycles rose to `121040.79/120346.17 Kcycles`; native global-read event scale changed and is not cross-compared. | Rejected; lower private writes did not improve cycles or endpoint. |

All formal runs used the same environment fingerprint
`sha256:4de1de3d1bb5cd6321d5440a3c3041c044a115312f35e20d7f9bc4325b708019`.
Each correctness-passing candidate retained all 100 ordered samples and an independent four-launch
mcProfiler artifact. H17S1–H17S3 statically excluded a generated-source 256-bit-vector no-op,
unsafe blanket bounds removal, and blind synchronization/bounds/clear-accumulator grids.

## Adjacent confirmation and promotion

Adjacent H10F5/H17F2/H17F2/H10F5 ABBA used four independent warmup-10/repeat-100 runs with no
sample removal. Incumbent aggregate mean/median was `228.891424/228.986748 ms`; challenger was
`219.531875/219.530177 ms`, improving `4.0891%/4.1297%`. Challenger position 1 improved
mean/median `4.1019%/4.1667%`; position 2 improved `4.0762%/4.0928%`. Every individual position
passed the 0.5 percent gate and relative standard deviations remained below 0.72 percent.

Only H17F2 was deployed. The new kernel SHA-256 is
`e9f83f74fe702d7adea33e3754fa618b5adfe7e103e68daaa58499e806fd0de9`. It adds four truthful
range assumptions to each of FC1 and FC2; `RoutedMoEKernel`, the OJ `run_kernel` ABI, workloads,
seed, dtype, accumulation dtype, and tolerances are unchanged. The deterministic stage2 table was
rebuilt to carry the new source SHA and size.

From this repository, MetaX pytest passed `89/89`; repository/device checks passed; C500 driver
`3.8.30`, MACA `3.7.1.5`, and TileLang-MACA `0.1.12+maca.gitec48829b` were verified; official
functional passed `2/2`; submission policy, uneven smoke, public shape, and all five fuzz/terminal
offset cases passed. The standalone diagnostic performance run measured
`190.442598/30.954004 ms`; it is not formal H17 timing. `git diff --check` passed.

Machine-readable evidence and exact artifact identifiers are in
`reports/2026-07-21-moe-h17-routing-assume-results.json`; the raw ordered samples, source commits,
diffs, environment captures, and mcProfiler databases remain in the agent repository.
