# C500 Routed-MoE H18 compact-proof convergence

Date: 2026-07-21. Device: MetaX C500 64 GiB. The only formal comparator was H17F2 at target
commit `2a94659f4a0c7c65bd191367fd48d50fef47ce2f`, source SHA-256
`e9f83f74fe702d7adea33e3754fa618b5adfe7e103e68daaa58499e806fd0de9`.

## Incremental first-hand sources

- [TVM S-TIR transform API](https://tvm.apache.org/docs/reference/api/python/s_tir/transform.html)
  (Apache-2.0) and [TileLang safe-memory legalizer
  source](https://raw.githubusercontent.com/tile-ai/tilelang/main/src/transform/legalize_safe_memory_access.cc)
  (ASF-header file in the MIT TileLang repository): assumptions can support analyzer proof, while
  unproved global indices retain runtime protection. This motivated H18F1 without disabling safe
  memory.
- [TileLang `T.likely` API](https://tilelang.com/autoapi/tilelang/language/tir/op/index.html)
  (MIT) and the TVM transform API above: likely annotations can inform control-flow transforms.
  H18F2 tested whether the exact installed MACA backend implements that frontend claim.
- [CUTLASS grouped GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu)
  (BSD-3-Clause) and [Triton persistent matmul
  tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py)
  (MIT): grouped schedulers distinguish host/device metadata work and exact tile counts still retain
  residue masks. Fixed ABI excluded their descriptor/persistent structures, but motivated the
  narrow metadata reuse audit H18F3/H18F4.

All linked bodies were opened on 2026-07-21. CUDA SM counts, residency, TMA, WGMMA, and async-copy
behavior were not transferred to C500. The exact installed TileLang-MACA revision `ec48829b` was
audited separately and remained authoritative.

## Formal hardware evidence

The initial H17F2 baseline passed official Large/Small with zero mismatches. All 100 ordered
samples after 10 warmups and no removal measured `219.721774/219.975165 ms` mean/median, standard
deviation `1.640725 ms`; four workload-matched profile launches measured FC1
`112346.52/113316.92` and FC2 `47408.01/49198.32 Kcycles`.

| ID | Mechanism | Correctness and formal 10/100 | Profiler conclusion | Decision |
|---|---|---|---|---|
| H18F1 | Add compact positive local-row contracts while retaining safe-memory legalization | Large/Small 0 mismatch; `219.539329/219.480576 ms`; `+0.0830%/+0.2248%`; stddev `1.602196 ms` | FC1 `113361.20/110670.73`, FC2 `46184.39/49095.09 Kcycles`; changes were mixed. | Rejected below both 0.5% gates; no combination. |
| H18F2 | Mark existing FC1/FC2 tail predicates with `T.likely` | Build failed before correctness | Installed MACA emitted unresolved `tirx.likely`; profiler was not run or fabricated. | Rejected at deterministic backend capability boundary. |
| H18F3 | Reuse padded-offset scalar in both FC1 and FC2 | Large/Small 0 mismatch; `221.695667/221.916799 ms`; `-0.8984%/-0.8827%`; stddev `1.429104 ms` | FC1 regressed to `115474.09/115660.88`, while FC2 fell to `46292.83/47635.42 Kcycles`. | Rejected; directional split authorized one FC2-only decomposition. |
| H18F4 | Reuse padded-offset scalar only in FC2 | Large/Small 0 mismatch; `219.418581/219.687807 ms`; `-0.0736%/-0.0823%`; stddev `1.549763 ms` against a fresh same-commit baseline `219.257195/219.507194 ms` | FC2 regressed in both positions to `50482.25/51140.28 Kcycles`; H18F3's apparent FC2 reduction did not reproduce. | Rejected; metadata decomposition closed. |

Every correctness-passing candidate retained all ordered samples and its independent native
mcProfiler database. No candidate passed both initial gates, so replication and ABBA were not
authorized. H18S1 excluded unsafe padded-tail overcompute; H18S2 retained fixed-ABI boundaries for
descriptors, metadata packing, persistent scheduling, and Split-K workspace; H18S3 retained prior
closures for full-tile duplication, blind schedule grids, unsupported async/TMA/WGMMA/FP4, and
safe-memory opt-out.

## Conclusion

No kernel change is deployed. H17F2 remains the sole immutable winner with the same SHA-256 and
schedule. Full raw samples, source commits/diffs, environment manifests, build error, and native
profile artifacts are retained in the agent repository; the compact machine summary is
`reports/2026-07-21-moe-h18-compact-proof-results.json`.

Final target validation passed MetaX pytest `89/89`, repository check over 3394 files, C500/MACA
verification, official functional `2/2`, submission public/fuzz, and `git diff --check`. The
standalone diagnostic run measured `190.584551/30.857212 ms`; it is not formal H18 timing.
