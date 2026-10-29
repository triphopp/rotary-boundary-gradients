# Results

| Path | Content | Produced by |
|---|---|---|
| `final_study/{shape}_{seed}_{slow}.json` | frozen study: 4 shapes × seeds 1009/1013/1019 × primary/slow-decay. Each file covers 7 methods with errors (relative L2, max abs, cosine, sign disagreement), captured timings (9 rounds), eager timings, promotion cost and protocol SHA-256 | `experiments/run_final_study.py` |
| `resource_e2e/{shape}_{method}.json` | warmed peak tensor memory and complete-primitive (forward+backward+conversion) timing, one fresh process each | `experiments/run_resource_e2e.py` |
| `fma_shape_audit/` | post-study FMA-enabled local adjoint on the four shapes (supplementary; not part of the frozen estimates) | `experiments/benchmark_fma_shape.py` |
| `terminal_study/` | earlier frozen 36-case study of direct terminal evaluation (negative result) | earlier iteration |
| `validation/` | CPU and GPU correctness checks (identity, adjoints, checkpoint replay, local Triton adjoint, policy) | `reference/`, `experiments/validate_*` |
| `summaries/` | `final_study_summary.json`, `final_study_table.csv`, development probe summaries, `analysis_macros.tex` | `analysis/` |
| `figures/` | figures from the analysis pipeline (the paper's Fig. 1 is `paper/images/error_vs_length.pdf`) | `analysis/` |
| `development/` | earlier isolated timing probe (chunk 32/64/128 × methods), not part of the frozen estimates | earlier iteration |

Method keys: `bf16_sum` (BF16 suffix), `bf16_fp32_intermediates`, `bf16_boundary_legacy`, `bf16_boundary_policy` (proposed), `fp32_storage_tf32_dots`, `fp32_sum` (FP32/tf32x3), `local_fp32` (direct local-coordinate adjoint).

The cold-start `extra_peak_MB` field in `final_study/` includes autotuner allocations. It is kept for completeness, but the paper does not use it. Use `resource_e2e/` for memory.
