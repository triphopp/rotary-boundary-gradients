# รอบ 03 — practical boundary policy and strong-baseline validation

Started after the completed iteration02 tests, including the negative direct-terminal ragged result. This is a versioned narrowing of the contribution and application, not a reinterpretation of the failed threshold.

## Concrete problem / proposed policy

Problem: angle-rate gradients in a mixed-precision rotary SISO state-space layer can lose accuracy when many already-inaccurate per-token derivatives are summed across chunks. The target use is the numerical backward primitive used in ordinary sequence-model training, not a general ODE solver or a claim of better language-model quality.

Candidate policy, fixed without looking at any new test results:

1. Use boundary state/adjoint contractions for cross-chunk carry.
2. When final states are not outputs of the primitive (`return_final_states=False`), set κ=0 exactly. This follows from the proved identity, not an empirical threshold. The actual `Mamba3` module requests final states only when its external state/cache is present (`mamba3.py` lines249–263 in the pinned local source), so this is relevant to the regular output-only training path.
3. When final states are returned, retain the established last-chunk κ estimator by default. The optional direct-terminal estimator remains an experimental ablation and is not automatically enabled, because its superiority failed in iteration02. A future conditioned selector would be a separate contribution with new tests.
4. Preserve fallback behavior for unsupported varlen/MIMO; do not claim to improve them in the paper. Empty/one-chunk cases must still be correct.

This policy is metadata-based. It does not inspect reference error or select a winning estimator retrospectively. The full terminal identity remains a theorem supporting correct treatment of boundaries, but the article must disclose the empirical state-error limitation of direct evaluation.

## Required comparisons

- Original BF16/FP32-angle implementation, FP32 intermediate-gradient storage, legacy boundary implementation, the proposed zero/legacy policy, full FP32/tf32x3, and where feasible direct local-rotation adjoint.
- Check optimized/tuned launch choices rather than assuming one fixed BF16-oriented launch configuration is optimal for FP32. Initial four-choice FP32 probe completed, selecting different warps/stages; the large observed cost did not disappear on its H2/T4096 case. This does not settle large-head geometry.
- Report wall-call/backward latency and capture/replay compute latency separately if CUDA graph measurement is used. Include kernel/terminal work; no comparing a captured candidate against an uncaptured baseline.
- Account for forward costs, precision conversions and memory separately before end-to-end assertions.

## Evidence needed before marking ready

1. Implement metadata policy with exact fallback/shape behavior and independently verify it.
2. Test realistic head parallelism and memory, with a reference that fits in memory (checkpointed sequential adjoint is acceptable after validation).
3. Stress slow decay, large/multiple chunk remainders, initial/final-state cases and conditioning; give clear limitations instead of promising universal length independence.
4. Freeze final unseen seeds/configurations after development and test useful achieved-accuracy cost, including stronger baselines.
5. Complete focused manuscript and references, claims audit and reproducibility package. G0–G6 do not pass merely because a useful candidate exists.

## Novelty position to be tested

The original complex/RoPE equivalence and general state-adjoint calculus are established. Johnsonms already diagnosed the cancellation problem and suggested cross-cut algebraic evaluation. The proposed contribution is the explicit boundary/terminal numerical realization, a qualified error decomposition and a verified implementation/application trade-off. Do not claim first discovery or new state-space dynamics. A reviewer may still regard this as an implementation refinement; significance must come from the demonstrated problem and evidence, not labels.
