# C500 Routed-MoE H19 warp/copy/vector convergence

Date: 2026-07-21. Device: MetaX C500 64 GiB. The only formal comparator was H17F2 at target
evidence commit `696b5d4f45be33092507ec8cb7eaa64baf1f44e9`, kernel-origin commit
`2a94659f4a0c7c65bd191367fd48d50fef47ce2f`, and source SHA-256
`e9f83f74fe702d7adea33e3754fa618b5adfe7e103e68daaa58499e806fd0de9`.

## Incremental first-hand sources

- [MetaX-MACA C500 architecture guide](https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/guide/ch2.%E6%9B%A6%E4%BA%91C500%E8%8A%AF%E7%89%87%E6%9E%B6%E6%9E%84.md)
  records the 64-thread lockstep warp, cross-warp CTA synchronization, 64-KB shared memory, and a
  warning that some direct 128-bit vector-load tests regress. Its LICENSE URL was inaccessible;
  facts only were summarized and no code was copied. This produced H19F1/H19F3/H19F4.
- TileLang's official [warp intrinsic](https://raw.githubusercontent.com/tile-ai/tilelang/main/tilelang/language/builtin.py),
  [parallel-loop frontend](https://raw.githubusercontent.com/tile-ai/tilelang/main/tilelang/language/loop.py),
  and [vector planner](https://raw.githubusercontent.com/tile-ai/tilelang/main/src/op/parallel.cc)
  (MIT) establish `T.sync_warp` and exact integer `coalesced_width` planning. These bounded all
  four executable hypotheses.
- [CUTLASS GEMM 3.x collective API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md)
  (BSD-3-Clause) separates cooperative copy/MMA/synchronization and fusion boundaries. This
  motivated H19F2, but CUDA copy atoms, clusters, residency, TMA and SM assumptions were excluded.
- The MetaX developer [TileLang-MetaX Task6 report](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf)
  records 104 APs, wavefront 64, 64-KB group memory, a slower tested async path and stage-3
  regression/overflow. It has no explicit reuse license; facts only were used to retain H19S2's
  async/deeper-stage exclusions.

All actual bodies were opened on 2026-07-21. Exact installed TileLang-MACA revision `ec48829b`
and existing H17 generated/profile evidence remained the C500 authority; no CUDA/ROCm hardware
constant or primitive was transferred.

## Formal C500 results

The first fresh H17F2 comparator passed official Large/Small at zero mismatches and measured
`219.324237/219.380615 ms` mean/median, standard deviation `1.478042 ms`, from every one of 100
ordered samples after 10 warmups and no removal. Its four workload-matched profiles were FC1
`113378.19/111648.35` and FC2 `47189.13/49928.35 Kcycles`.

| ID | Distinct mechanism | Official correctness and formal 10/100 | Workload-matched profiler | Decision |
|---|---|---|---|---|
| H19F1 | Warp-scope barriers only around H17F2's warp-private FC1 Up exchange | Large/Small 0 mismatch; `246.347064/246.492029 ms`; `-12.3209%/-12.3582%`; stddev `1.154221 ms` | FC1 `138260.19/138115.19`, FC2 `50390.67/51262.83 Kcycles`; both kernels worsened. | Rejected; closes the named H17F1 decomposition. |
| H19F2 | One cooperative traversal for separate Gate/Up shared copies | Large/Small 0 mismatch; `221.164290/221.409533 ms`; `-0.8390%/-0.9248%`; stddev `1.442565 ms` | FC1 `112190.08/113948.60` mixed; FC2 `48717.15/49469.29 Kcycles` higher. | Rejected below both gates; no combination. |
| H19F3 | Request width two only for two FC1 FP32 activation/product loops | Large/Small 0 mismatch; `219.625705/219.510273 ms`; `-0.1375%/-0.0591%`; stddev `1.338915 ms` | FC1 `111300.05/114149.59`, FC2 `46260.92/49545.67 Kcycles` mixed; all four generated sources were byte-identical to H17F2. | Rejected; exact installed-backend no-op, so no grid. |
| H19F4 | Request four-FP16/64-bit loads only for FC1 Gate/Up weight copies | Fresh H17F2 `220.262114/220.591226 ms`; candidate Large/Small 0 mismatch and `290.547789/290.633085 ms`; `-31.9100%/-31.7519%`; stddev `1.590387 ms` | Generated counts changed from 24 `uint4` + 40 `uint2` to 16 `uint4` + 48 `uint2`; FC1 regressed from `113152.48/114020.23` to `183207.02/183633.02 Kcycles`, FC2 remained `49074.47/46553.63`. | Rejected severe realized-mechanism regression; width decomposition closed. |

Quick 2/7 timings in evaluator artifacts are diagnostics only; every number in the table is the
formal 10/100 result. All 100 ordered samples, candidate commits/diffs, identical environment
fingerprint, correctness records, generated source and independent native profile databases live
in the agent artifacts. Native instruction-event scales changed across some captures, so the
cross-run causal claims use generated types and workload-position-matched `Total Cycles` only.

H19S1 was statically excluded because exact synchronous MACA lowering ignores eviction hints.
H19S2 retained the established async, deeper-stage, broad schedule grid, persistent, Split-K,
descriptor and workspace boundaries because the incremental sources exposed no new fixed-ABI
capability.

## Conclusion

No candidate passed both initial 0.5% gates, so no second run, ABBA, combination or deployment was
authorized. H17F2 remains the sole immutable kernel with unchanged API, submission ABI, shapes,
seed, dtype, tolerances, schedule and SHA-256. The machine summary is
`reports/2026-07-21-moe-h19-warp-copy-vector-results.json`.

Final target validation passed MetaX pytest `89/89`, repository check over 3397 tracked files,
C500/MACA verification, official functional `2/2`, submission policy/public/fuzz, and
`git diff --check`. The standalone `run-moe.sh` diagnostic was
`190.39873047/30.86630859 ms`; it is not formal H19 timing evidence.
