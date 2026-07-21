# C500 Routed-MoE H15 native MACA convergence

Date: 2026-07-21. Device: MetaX C500 64 GiB. Formal comparator: H10F5 only, source
SHA-256 `338b4d93cadc69a12e49690efc5d3030630b8cf412ee0a0691a6ef44db8bedfe`.

## Incremental first-hand sources

- [TileLang-MetaX PR 109](https://github.com/tile-ai/tilelang-metax/pull/109), MIT: native
  MACA persisting-L2 access-window lowering and best-effort runtime calls.
- [CUDA Best Practices, L2 access window](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#l2-cache-access-window),
  NVIDIA documentation terms: bounded-window/hit-ratio concept only; no NVIDIA performance or
  hardware constant was transferred to C500.
- [TileLang-MetaX PR 67](https://github.com/tile-ai/tilelang-metax/pull/67), MIT: native
  `maca_async_copy`, barrier wait and a full-tile stage-2 GEMM example.
- [TileLang MLA swizzle notes](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md),
  MIT: CTA adjacency/locality mechanism; another swizzle search was excluded by the exact current
  mapping and measured H7F3.
- [MetaX-hosted TileLang C500 report](https://developer.metax-tech.com/forum/media/attachments/c7/ec/p921bRvdtxGdMGVlejqB3MWt5vRdjo5Vd9Mdb21bQ3YTuk9JSnNrARaoU14kszzE/yu-han-tilelang-.pdf),
  public facts only: 64 KiB group-memory boundary and the requirement to measure async copy.

Exact installed TileLang-MACA `ec48829b` source, under its MIT license, was also audited. Search
queries, access status, license boundaries and hypothesis mappings are recorded in the agent's
`kernels/moe/external_research.json`; search summaries were not used as evidence.

## Formal hardware evidence

The first commit-matched H10F5 baseline (`00a0e427`) passed official Large and Small with zero
mismatches, then retained all 100 ordered samples after warmup 10 with outlier policy `none`:
`228.899139/228.795525 ms` mean/median, standard deviation `1.121979 ms`, p90
`230.475925 ms`. A second source-identical baseline was required after the agent registered H15F4;
the evaluator refused cross-commit comparison before running the candidate. It again passed zero
mismatch correctness and measured `229.042132/229.240829 ms`, standard deviation `1.268867 ms`,
p90 `230.638491 ms`, with its own four-launch profiler.

| ID | Mechanism | Correctness and formal timing | Profiler/runtime conclusion | Decision |
|---|---|---|---|---|
| H15F1 | One 0.8 persisting-L2 window on FC2 `up_logits` | Large/Small 0 mismatch; `228.857662/228.911360 ms`; mean `+0.0181%`, median `-0.0506%` | All evaluator stages emitted 243 `mcStreamSetAttribute: mcErrorInvalidValue` diagnostics. FC2 L2/cycle variation was not consistent across captures and a later unannotated baseline reached still higher L2 rates, so it cannot evidence an applied policy. | Rejected: median regression, below gate, runtime did not apply the mechanism reliably. |
| H15F2 | One 0.8 persisting-L2 window on FC1 `input` | Large/Small 0 mismatch; `229.209793/229.169286 ms`; mean `-0.1357%`, median `-0.1634%` | The same 243 runtime invalid-value diagnostics occurred. FC1 cycles were `117438.91/119603.35 Kcycles` versus baseline `117466.67/118931.89`, not a consistent improvement. | Rejected regression and invalid runtime policy. |
| H15F3 | Async-copy both FC2 operands, one barrier, stage 2 | Build failed before correctness/timing | Exact compiler diagnostic: `InternalError: Check failed: (a.dtype().is_bool()) is false`. The fixed ABI's shortened final routed activation tile requires safe-access predication absent from the upstream full-tile example. No profiler can exist for a non-built kernel. | Deterministic compile rejection. |
| H15F4 | Keep activation copy synchronous; async-copy only full expert-weight tile | Build failed before correctness/timing | Same exact compiler assertion, showing that current lowering also cannot accept the dynamic routed-expert address. No profiler can exist for a non-built kernel. | Deterministic compile rejection. |

H15S1 (multiple L2 windows), H15S2 (cross-kernel persistence) and H15S3 (renamed swizzle
search) were excluded by exact runtime reset/one-window code or prior measured evidence. The
formal acceptance gate remained mean and median both at least 0.5 percent, stable variance and a
matching profiler mechanism. No candidate passed it, so replication and adjacent ABBA were not
authorized.

## Validation and outcome

H10F5 remains byte-identical and is the only deployed kernel. From this repository, the MetaX
suite passed `89` tests, final repository check inspected `3388` tracked files, `verify-maca.sh` passed
on C500/driver `3.8.30`/MACA `3.7.1.5`, both official functional cases passed, the diagnostic
performance run reported `198.330684/32.663997 ms`, submission public-shape plus all fuzz cases
passed, and `git diff --check` passed. Diagnostic timings are not formal candidate evidence.

The unresolved directions require a backend/runtime capability change: a working C500 access
policy-window call, or async-copy lowering that supports uniformly dynamic routed addresses and
safe tail predication (alternatively an ABI-sanctioned padded intermediate). They must not be
retested under renamed candidates while the exact installed capability is unchanged.
