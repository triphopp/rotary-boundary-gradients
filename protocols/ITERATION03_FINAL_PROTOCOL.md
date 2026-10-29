# Frozen out-of-sample protocol — iteration03

Frozen after development seed503 and before running any seeds below. Existing direct-terminal iteration02 held-out negatives are not replaced or relabeled. This study evaluates the narrowed metadata policy, not that rejected incremental-terminal claim.

## Scope / data

Output-only SISO rotary state-space backward, zero initial state, D and Z enabled, BF16 Q/K/V/bias/Trap/Z and FP32 Angles/ADT/DT/D. Reference differentiates the mathematical recurrence on these identical quantized operands, with identical BF16 output cotangent. No improved language-model accuracy claim follows from this primitive benchmark.

Seeds1009,1013,1019. Four shapes `(B,T,H,Hq,N,P)`:
`(1,1023,8,1,64,64)`, `(1,2048,16,4,64,64)`, `(2,4097,8,2,64,32)`, `(1,8192,16,4,64,32)`.

12 primary cases use the existing upstream-test-like generator unchanged. 12 separately labeled slow-decay stress cases multiply ADT by0.01; all other operands and cotangents are matched. Stress cases test the limitation of state/adjoint conditioning and are not silently removed from the report. Cotangent RNG seed is input seed+91000. Every case, pass or fail, is retained.

## Frozen competitors

BF16 ordinary suffix, FP32 intermediate-gradient storage, legacy boundary, proposed metadata policy: chunk64. Full FP32/tf32x3: chunk64 with the four-choice backward autotune. FP32 storage/single TF32 dots: chunk32 (chunk64/128 exceeded shared memory in independently repeated development on this GPU). Direct local-coordinate FP32 adjoint: Vtile8, selected from4/8/16 development sweep. These are development-selected implementations, not a proof of globally optimal kernels. Chunk32/64/128 development results are reported alongside them. Direct local adjoint includes every input gradient and all reduction/cast work.

## Measurement

Independent checkpointed NumPy FP64 adjoint, validated against autograd and a bitwise-equivalent uncheckpointed implementation. Report relative L2, max absolute, reference norms, forward relative L2, angle-gradient cosine and sign disagreements for components exceeding0.001 times the reference RMS. Near-zero gradient cases use absolute errors and are not called wrong from relative error alone.

Forward and backward use the same non-default stream. Numerical errors are checked before timing and again after capture replay. Capture one full backward per method; retain outputs and check finite values. Nine shuffled rounds, three replays per round. Separately record nine ordinary synchronized wall/event backward measurements, five forward measurements, and isolated forward+backward peak allocated tensor memory (excluding inputs, excluding graph storage). Do not equate allocator reservation with tensor memory or captured device time with ordinary user latency. Compilation and CPU reference are excluded from timing. Precision promotion cost is recorded separately where required.

## Predeclared decision rules (internal, not IEEE requirements)

1. Primary angle-error reduction: median ratio of the smaller error of ordinary BF16 and FP32-intermediate baseline to policy error must be≥2. No primary policy error may exceed that smaller baseline error by>20%. Report legacy separately because its underlying boundary realization is part of the credited lineage.
2. Moderate-accuracy target: angle relative L2≤0.025 and DT relative L2≤0.01. Policy must pass≥10 of12 primary cases. Report1%,2.5%,5% angle targets as sensitivity, with DT bound fixed, without choosing the best threshold after testing.
3. Among primary cases that pass, compare policy cost with the fastest qualifying non-boundary competitor among ordinary, intermediate, fullFP32, TF32, local adjoint. Median captured speed advantage must be≥1.5×. Report every infeasible configuration and all ordinary timings. Median policy/ordinary BF16 captured overhead must be≤1.25×. Do not hide a slower individual case.
4. Stress is a limitation/robustness study: report every error and failure with the same metrics and targets. It cannot rescue a failed primary result. Any correctness or capture mismatch blocks readiness and requires a documented fix plus a fresh test protocol if the candidate changes.
5. Bootstrap primary case clusters by shape (4 groups, explicitly a small population) for descriptive confidence intervals; publish raw timings. Do not present timing repetitions as independent scientific cases or make population-wide significance claims.

Passing this study supports a measured cost–accuracy niche on one RTX3080, not global superiority or IEEE acceptance. Novelty/source audit, reproducibility and a complete appropriately qualified manuscript remain separate gates.
