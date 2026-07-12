# MoE tuning log

| Variant | Functional result | Large performance | Small performance | Decision |
|---|---|---:|---:|---|
| baseline: M128/N128/K128, threads256, stages1 | pass (2/2) | 425.80136719 ms | 62.37817871 ms | reference |
| stages2 | rejected before functional completion: 131072 B dynamic shared memory > C500 65536 B limit | — | — | reverted |
| K64/N128 (block_dhidden=64, block_dexpert=128), stages1 | pass (2/2) | 491.57949219 ms | 68.65500977 ms | rejected: slower |
| K128/N64 (block_dhidden=128, block_dexpert=64), stages1 | pass (2/2) | 482.96042969 ms | 68.34916504 ms | rejected: slower |
| M128/N128/K128, threads128, stages1 | pass (2/2) | 792.79367187 ms | 105.12538086 ms | rejected: slower |
| M128/N128/K128, gate/up stage1 + down stage2 | pass (2/2) | 449.16949219 ms | 65.28123535 ms | rejected: slower |
| M128/N128/K128, row swizzle panel16 (both stages) | pass (2/2) | 424.82343750 ms | 61.93219727 ms | provisional: +0.23% / +0.72%; repeat before retaining |
| M128/N128/K128, GEMM FullRow (all 3 GEMMs), row10 | pass (2/2) | 340.21050781 ms | 50.43333008 ms | provisional winner: +20.10% / +19.15%; repeat and refine |
| stage-1 GEMMs FullRow; down GEMM Square, row10 | pass (2/2) | 402.46718750 ms | 58.05255859 ms | improves baseline but worse than FullRow all |
| stage-1 GEMMs Square; down GEMM FullRow, row10 | pass (2/2) | 361.84867187 ms | 54.09350586 ms | improves baseline but worse than FullRow all |
| GEMM FullRow all; row swizzle stage1=16, down=16 | pass (2/2) | 337.75480469 ms | 50.14238281 ms | provisional winner: +20.68% / +19.62%; test down panel 28 |
| GEMM FullRow all; row swizzle stage1=16, down=28 | pass (2/2) | 338.52632813 ms | 50.20187500 ms | rejected: slower than down panel16 |
| stage-1 GEMMs FullCol; down GEMM FullRow, row16 | pass (2/2) | 578.79289063 ms | 80.00660645 ms | rejected: much slower |
| GEMM FullRow all; row16; MACA fast math | pass (2/2) | 338.59144531 ms | 50.18633789 ms | rejected: slower than precise math |
| GEMM FullRow all; row16; serialized single stage-1 weight buffer | pass (2/2) | 244.92994141 ms | 38.60862305 ms | provisional winner: +42.48% / +38.11% vs baseline; refine and repeat |
| one-weight, FullRow; stage1 column16; down row16 | pass (2/2) | 244.13417969 ms | 38.75856201 ms | tradeoff: large +0.32%, small -0.39% vs row/row |
| one-weight, FullRow; stage1 row16; down column16 | pass (2/2) | 247.59197266 ms | 38.83649414 ms | rejected: slower than row/row |
| one-weight, FullRow, row16, stage1 min-blocks=2 | pass (2/2) | 244.77203125 ms | 38.50635498 ms | mixed/noise vs unannotated; keep unannotated pending repeat |
| one-weight, FullRow, row16, K=64 | pass (2/2) | 313.20787109 ms | 46.16655762 ms | rejected: slower |
| one-weight, FullRow, row16, threads=512 | pass (2/2) | 316.05500000 ms | 48.41013184 ms | rejected: slower |
| one-weight, FullRow, row16; down row8 | pass (2/2) | 244.93691406 ms | 38.59494873 ms | essentially tied / no material gain |
| final source defaults via official `run-race-test.sh moe` | pass (2/2) | 245.01759766 ms | 38.46790771 ms | retained; 42.46% / 38.33% lower latency vs baseline |
| final tuner defaults (fresh repeat) | pass (2/2) | 245.58896484 ms | 38.68975586 ms | confirms tuner defaults match shipping schedule; normal timing variance |
| one-weight, FullRow, row16, K=256 | pass (2/2) | 484.22558594 ms | 67.99356934 ms | rejected: 64 KiB stage tiles severely reduce occupancy |
| one-weight, FullRow, row16, down output tile N=64 | pass (2/2) | 305.03435547 ms | 45.73944824 ms | rejected: smaller down tile doubles CTA work and is slower |
| one-weight, FullRow, row16, down stages=2 | pass (2/2) | 269.32121094 ms | 41.83664063 ms | rejected: stage-2 pipeline overhead/shared usage outweighs benefit |
| one-weight, FullRow, row16, down min-blocks=2 | pass (2/2) | 244.82880859 ms | 38.53001221 ms | mixed/noise vs default; rejected to keep a single global schedule |
| one-weight, FullRow, row16, stage-1 `T.Pipelined(..., num_stages=1)` | rejected at compile | — | — | reverted: pipeline planner detects overlapping writes to the intentionally aliased `routed_expert_up_shared` buffer |
| one-weight, FullRow, stage-1 row8 / down row16 | pass (2/2) | 243.40417969 ms | 38.52518799 ms | tradeoff: large improves, small regresses; repeat against row16 before choosing |
| one-weight, FullRow, stage-1 row32 / down row16 | pass (2/2) | 245.10683594 ms | 38.54509277 ms | rejected: no improvement over the retained row16 schedule |
| one-weight, stage-1 row16 / down row16 (repeat) | performance repeat | 244.84636719 ms | 38.48209717 ms | control run for the final swizzle comparison |
| one-weight, FullRow, stage-1 row8 / down row16 (repeat) | performance repeat | 243.21238281 ms | 38.53373291 ms | confirms roughly 0.7% large-case gain; small-case difference is within timing variation |
| final source defaults: one-weight, FullRow, stage-1 row8 / down row16 | pass (2/2), official entrypoint | 243.47679687 ms | 38.60929443 ms | retained: 1.7488x / 1.6156x faster than the original baseline; OJ ABI smoke also passed |
| one-weight, FullRow, stage-1 row4 / down row16 | pass (2/2) | 243.15136719 ms | 38.61110352 ms | rejected: large-case difference from row8 is within noise, while small case is no better |
| one-weight, FullRow, stage-1 row8 / down row8 | pass (2/2) | 243.51494141 ms | 38.69943604 ms | rejected: smaller down panel slows the small configuration |
| one-weight, FullRow, stage-1 row8 / down row16, min-blocks=2 | pass (2/2) | 243.51541016 ms | 38.52495850 ms | rejected: trades a slower large case for a marginal small-case change; retain unannotated schedule |
| gate activation FP16 + reuse gate FP32 accumulator for up | pass (2/2) | 308.72207031 ms | 47.09996582 ms | rejected: the separate gate/up reductions reread input, and bandwidth cost overwhelms the register saving |
| one-weight, FullRow, row8/row16, down threads=512 | pass (2/2) | 275.21447266 ms | 42.49893555 ms | rejected: a wider down block lowers CTA concurrency more than it reduces per-thread work |
| one-weight, FullRow, row8/row16, down threads=128 | pass (2/2) | 281.08361328 ms | 42.67457520 ms | rejected: two 64-lane warps cannot sustain the down GEMM efficiently |
