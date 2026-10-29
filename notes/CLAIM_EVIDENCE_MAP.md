# Claim-to-evidence map

This file controls what can appear in the manuscript. “Supported” means supported in the stated scope, not guaranteed novel or accepted by a journal.

| Claim | Evidence | Current assessment / limit |
|---|---|---|
| A new Lag ODE solver is generally faster than standard methods | Existing standard benchmarks and iteration01 tuned/factorized study | **REJECTED. Do not use.** |
| Increment/expm1/Kahan/refresh by themselves are new | Primary-source matrix | **REJECTED. Established tools.** |
| A suffix-angle derivative equals a boundary state–adjoint contraction plus terminal terms | `BOUNDARY_THEORY.md`; independent 256-case / 4,800-cut checks; finite differences | Supported for scalar-decay/commuting rotary recurrence, locally differentiable outputs; not a priority claim |
| The boundary state includes the folded previous-input trapezoidal term | Derivation and sequential reference | Required; using the pre-fold state alone is wrong |
| Output-only loss permits κ=0, even with nonzero initial states | Derivation; independent initial-state checks; policy implementation | Supported; applies when final states are not loss outputs; policy conservatively falls back if API returns them |
| Boundary evaluation removes the explicit sum of noisy local terms over all later chunks | Exact error decomposition and GPU noisy-term-sum diagnostic | Supported; boundary/adjoint/phase errors remain; no unconditional length-independent error |
| BF16 boundary evaluation improves long-sequence angle-gradient accuracy in tested cases | 36 frozen V2 cases and full-chunk development data | Supported in reported shapes/objectives/precision; not every recurrence or trained model |
| Direct terminal-state evaluation always improves on last-chunk κ | 12 old development cases vs 36 fresh ragged cases | **REJECTED.** 17/36 worse, one >=20% worse; inaccurate forward final states dominate diagnostic case |
| Boundary method is faster than full FP32 at a stated achieved gradient-error tolerance | Frozen24-case/seven-method study, corrected resource study and direct-adjoint FMA audit | Supported only for stated moderate tolerance, tested implementations and RTX3080; no global optimum or model-training throughput claim |
| Boundary method has negligible/no overhead | Initial timing shows >20% overhead in some cases | **Not supported.** Report measured trade-off; no broad speed claim |
| Gradient error is caused solely by cumsum rounding | FP64 summation of noisy terms still fails | **REJECTED** for diagnosed cases; per-term errors are central |
| Kernel outputs/gradients are bitwise deterministic | K atomic-reduction variation repeats even with same method | **Not supported** for inherited GQA/terminal path; disclose it |
| This improves training loss/perplexity/task accuracy | No end-to-end training outcome evidence | **Not claimed.** Numerical primitive quality is the target unless new application evidence is added |
| Supports/improves varlen and MIMO | New policy preserves legacy fallback, no new varlen/MIMO validation | Do not claim improvement or generalization |
| First discovery of rotary equivalence or cancellation problem | Mamba-3 paper and Johnsonms public comment precede work | **REJECTED.** Credit primary sources |

Novelty assessment remains a separate gate: the potentially defensible contribution is a concrete numerical realization, qualified analysis and demonstrated application trade-off beyond the known broad cancellation idea. It is not a newly discovered differentiation principle.
