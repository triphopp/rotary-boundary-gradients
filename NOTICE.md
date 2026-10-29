# Notice and provenance

The directories under `kernels/` are derived from the `mamba_ssm` package of [state-spaces/mamba](https://github.com/state-spaces/mamba) (upstream snapshot `e9594ce`). That package is licensed under Apache-2.0, and its copyright headers are kept in every file. These trees are modified research variants, not official releases.

| Tree | Modification |
|---|---|
| `terminal_fused` | fused boundary evaluation with comparison modes (`none` = ordinary BF16 suffix sum, `legacy` = last-chunk κ estimate); FP32 cumulative angles; fixed BF16 backward launch configurations |
| `boundary_policy` | explicit boundary buffer; κ = 0 when final states are not returned, legacy estimator otherwise |
| `midprecision_baseline` | FP32 intermediate Q/K gradients, BF16 elsewhere |
| `tf32_baseline` | FP32 operand/state storage with single-pass TF32 dot products |
| `fp32_tuned` | FP32 storage with tf32x3 dot products and four-choice backward autotuning |

`src/m3env.py` loads a tree without building the CUDA extensions of `mamba_ssm`. The other code under `src/`, `reference/`, `experiments/`, `analysis/` and `verification/` is original to this repository.

`notes/prior_sources_*.json` record the URLs, revisions and hashes of the third-party sources inspected for the prior-art audit. The sources themselves are not redistributed.
