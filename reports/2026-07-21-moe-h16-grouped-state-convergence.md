# C500 Routed-MoE H16 grouped-state convergence

Date: 2026-07-21. Device: MetaX C500 64 GiB. Formal comparator: H10F5 only, source
SHA-256 `338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`.

## Incremental first-hand sources

- [TileLang official grouped GEMM](https://github.com/tile-ai/tilelang/blob/main/examples/grouped_gemm/example_grouped_gemm_fwd.py),
  MIT: scalar-local group index, size and offset motivated H16F1.
- [TileLang `T.gemm` source](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/gemm_op.py),
  MIT: its `clear_accum` contract plus exact installed MACA lowering motivated H16F2.
- [TileLang release history](https://github.com/tile-ai/tilelang/releases), MIT: recent scalar-binding
  and grouped-compilation context; PR 979 itself was inaccessible and no claim depends on it.
- [FlashInfer grouped-GEMM plus combine PR 2944](https://github.com/flashinfer-ai/flashinfer/pull/2944),
  Apache-2.0: device epilogue routing reuse motivated H16F3, without copying CuTe/CUDA machinery.
- [CUTLASS grouped GEMM example 57](https://github.com/NVIDIA/cutlass/blob/main/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu),
  BSD-3-Clause: pointer-array/workspace and architecture-specific scheduler requirements prove the
  fixed-ABI exclusion for descriptor, persistent and Stream-K variants.

The actual source/document bodies and licenses were opened. Exact queries, access status,
applicability and the installed TileLang-MACA `ec48829b` audit are retained in the agent's
`kernels/moe/external_research.json`; search summaries are not evidence and no CUDA performance
assumption is transferred to C500.

## Formal hardware evidence

The commit-matched H10F5 baseline passed official Large and Small with zero mismatches. All 100
ordered samples after warmup 10 and outlier policy `none` measured
`228.599621/228.585093 ms` mean/median, standard deviation `1.133878 ms`, p90 `230.076729 ms`.
Its four workload-matched profiler launches measured FC1 `118450.08/118807.31` and FC2
`54046.69/50551.81 Kcycles`.

| ID | Mechanism | Correctness and formal timing | Profiler/generated-code conclusion | Decision |
|---|---|---|---|---|
| H16F1 | Cache FC1/FC2 routing scalars with `T.alloc_var` | Large/Small 0 mismatch; `228.738261/228.749058 ms`; mean `-0.0606%`, median `-0.0717%`; stddev `1.109623 ms` | FC1 `118359.67/116784.43`, FC2 `54073.39/52488.57 Kcycles` crossed. Native Global Read Instructions changed scale to `636/2490/646/2554`, so no cross-capture read claim is made. | Rejected regression below both gates. |
| H16F2 | Peel FC2 k=0 and fuse accumulator clear into first GEMM | Large/Small 0 mismatch; `228.941795/228.980607 ms`; mean `-0.1497%`, median `-0.1730%`; stddev `1.049539 ms` | FC2 `52460.68/52925.32 Kcycles` did not improve consistently versus baseline `54046.69/50551.81`; an unchanged FC1 private-read counter also changed native scale. | Rejected regression below both gates. |
| H16F3 | Broadcast 128 route weights through an FC2 shared vector | Large/Small 0 mismatch; `231.201423/231.120765 ms`; mean `-1.1381%`, median `-1.1093%`; stddev `0.587310 ms` | Both FC2 captures regressed to `55204.31/54460.75 Kcycles`. Generated source already maps `routed_weight_shared` and dead `up_logits_shared` to offset zero and the weight tile to offset 16384. | Rejected stable regression; manual aliasing is already realized and is not a new follow-up. |

H16S1–H16S4 statically exclude descriptor/workspace scheduling, MACA-no-op `k_pack`, runtime
transpose/offline repack, and an unchanged generic compiler/wave grid. H16F1/H16F2 were negative,
and H16F3 regressed by over one percent despite automatic lifetime packing, so there was no pair of
positive mechanisms to combine. No candidate passed both initial 0.5 percent gates; replication
and adjacent ABBA were therefore not authorized.

## Outcome

H10F5 remains byte-identical and is the only deployed kernel. Each executable H16 candidate retains
its candidate source commit/SHA, diff and patch SHA, identical environment fingerprint, official
correctness result, all ordered 10/100 samples, and an independent four-launch mcProfiler artifact
in the agent repository. Native event-scale inconsistencies are preserved rather than converted
into unsupported counter claims. The machine-readable target summary is
`reports/2026-07-21-moe-h16-grouped-state-results.json`.

From this target repository, MetaX backend pytest passed `89/89`; repository check inspected `3390`
tracked files after both H16 reports were tracked; `verify-maca.sh` confirmed C500, driver `3.8.30`,
MACA `3.7.1.5`, and TileLang-MACA `ec48829b`; official functional passed `2/2`; submission policy,
public shape, uneven smoke and all five fuzz/terminal-offset cases passed; and `git diff --check`
passed. The standalone diagnostic performance run reported `197.450840/31.326743 ms`; it is not
formal candidate timing and was not used for an acceptance decision.
