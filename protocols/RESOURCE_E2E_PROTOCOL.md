# Warmed memory correction and complete primitive timing

The24-case frozen numerical/captured-backward study passed its predeclared primary criteria. Its first-call peak-memory field included autotuner/cold-start allocations and is **not used as algorithm memory evidence**. Preserve that field as raw diagnostic data. This correction changes measurement only, not the candidate, competitors, tolerances, primary decision rules or seeds.

Rerun one existing seed1013 per primary shape, seven methods, each in a separate process. Five full forward/backward warmups precede three peak-allocated-tensor measurements. Report peak minus caller input/cotangent allocation. No CUDA graph is present during peak measurement. Fixed-shape allocation does not depend on these random values; do not count the three readings as independent scientific cases.

Then measure15 ordinary complete forward/backward calls (event and synchronized wall), followed by15 captured rounds of3 full-step replays. Include BF16→FP32 promotion and gradient conversion back to caller leaves for FP32 methods, all input gradients, output cast, state computation and temporary reductions. This is a complete numerical primitive, not a full model/optimizer training step. Validate Angles/DT against the exact same independent reference and the corresponding frozen-study result before and after capture.

Do not change the final study's backward timing from this supplementary check. Report single-process method-order timing limitations: cross-method full-step comparisons here are supplementary, not the paired main backward speed claim.
