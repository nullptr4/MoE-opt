# C500 Routed-MoE H20 compiler-pass convergence

Date: 2026-07-21. Device: MetaX C500 64 GiB. The sole formal kernel remained H17F2,
originating at `2a94659f4a0c7c65bd191367fd48d50fef47ce2f`, with SHA-256
`e9f83f74fe702d7adea33e3754fa618b5adfe7e103e68daaa58499e806fd0de9`.
The campaign control was synchronized to target `6a92944cd1a4d649341f7f3ee4a105d72ae4803e`
before the bounded H20F1 feasibility run. Older A1b, H4 and H10F5 results were not used as
comparators.

## Incremental first-hand sources

- TileLang's official
  [pass configuration](https://raw.githubusercontent.com/tile-ai/tilelang/main/tilelang/transform/pass_config.py)
  and [MIT license](https://github.com/tile-ai/tilelang/blob/main/LICENSE) expose separate
  constraint-aware Simplify, replayable-Bind, loop-unswitch and shared-lifetime controls.
- TileLang's official
  [Simplify implementation](https://raw.githubusercontent.com/tile-ai/tilelang/main/src/transform/simplify.cc)
  maps the constraint flags to transitive inequality, branch-constraint and known-value proving.
- TileLang's official [v0.1.10 release](https://github.com/tile-ai/tilelang/releases/tag/v0.1.10)
  records scalar-bind scope/replay and redundant safe-memory-bound work. The linked PR and raw
  loop-unswitch/IfStmtBinding bodies were inaccessible to Web Search and were not used as evidence.
- The exact installed TileLang-MACA revision `ec48829b` was inspected locally. Its MACA pipeline
  contains the same controls, and safe loads already default to zero. No CUDA/ROCm performance or
  architecture assumption was transferred to C500.

All external bodies were opened on 2026-07-21. Source concepts only were used under the upstream
MIT boundary; vendor MACA code was not copied. The agent's source registry records the queries,
URLs, access status, licenses and hypothesis mappings.

## Formal C500 evidence

The primary H17F2 comparator passed official Large and Small correctness with zero mismatches and
measured `219.865344/219.819519 ms` mean/median, standard deviation `1.324687 ms`, from all 100
ordered samples after 10 warmups with no sample removal. Its workload-matched FC1/FC2/FC1/FC2
profiles were `113670.70/47116.08/111945.94/48443.90 Kcycles`.

| ID | Mechanism | Correctness and formal result | mcProfiler attribution | Decision |
|---|---|---|---|---|
| H20F1 | Enable full transitive/branch/known-value Simplify over existing H17F2 facts | The source-identical 240-second-boundary baseline passed Large/Small with zero mismatches and measured `219.220800/219.202559 ms`; the isolated candidate did not build within 240 seconds. | No candidate profile exists because no executable was produced. The timeout command result and empty build-artifact count are retained. | Rejected at the deterministic build-feasibility boundary. No timing or profiler result is fabricated. |
| H20F2 | Preserve replayable scalar Bind scope instead of substituting it into guarded statements | Large/Small zero mismatch; `219.038607/219.091203 ms`; stddev `1.288409 ms`; mean/median gains `0.3760%/0.3313%`. | FC1 private writes fell from `987840/987840` to `5/5`, but cycles were mixed at `113192.54/112720.16` versus `113670.70/111945.94`; FC2 was also mixed at `47273.07/47709.66` versus `47116.08/48443.90 Kcycles`. Global reads were essentially unchanged. | Rejected below both 0.5% gates and without consistent cycle direction. |
| H20F3 | Disable loop unswitching | Large/Small zero mismatch; `219.578414/219.555707 ms`; stddev `1.408755 ms`; mean/median gains `0.1305%/0.1200%`. | Private/global instruction counts remained effectively unchanged; cycles were `114078.12/48797.52/112207.29/50367.80 Kcycles`, with three of four workload positions worse than the comparator. | Rejected below both gates and profiler-inconsistent. |

H20S1 was excluded because explicit zero safe values reproduce the exact installed default. H20S2
was excluded because the live FC2 shared operands overlap and FC1 has no profitable implicit
disjoint lifetime; disabling reuse cannot improve the explicit H10F5/H17F2 alias and can only be a
no-op or increase pressure.

H20F2 was not combined with H18F1 or H20F3. H18F1 itself reached only `0.0830%/0.2248%` with mixed
profiles; even the additive mean estimate with H20F2 is below 0.5%. H20F3's three-of-four cycle
regression supplies no positive mechanism. Combining rejected, profiler-inconsistent candidates
would be an unguided noise search.

## Conclusion

No H20 candidate passed the initial dual 0.5% gate. No replication, ABBA, integration change or
submission adaptation was authorized. `custom_fusedmoe.py` therefore remains exact H17F2 at the
SHA above. The target's remote-aligned compact FP16 `submission.py` remains byte-identical to
target `6a92944`; H20 does not claim online deployment of any compiler-pass experiment. All ordered
samples, environment fingerprints, commits/diffs, correctness, profiler databases and the H20F1
timeout are preserved under the agent repository.
