# Targeted source audit for the boundary-gradient candidate

28 October 2026. Read-only audit; no comments, PRs or messages sent. The GitHub connector returned “Unknown tool”; public REST retrieval was used as fallback. Downloaded source was inspected as text, not executed.

## Known diagnosis and idea before this project

[Johnsonms comment, 13 October 2026](https://github.com/state-spaces/mamba/pull/1059#issuecomment-6138351445) identifies inaccurate per-token angle gradients amplified by reverse cumsum and proposes cross-cut interaction evaluation for algebraic cancellation. Credit that diagnosis and high-level remedy explicitly. Our candidate is an explicit boundary-state/adjoint evaluation including terminal outputs and its numerical implementation, subject to the remaining prior-art review. Do not claim first discovery of the failure mode or first idea of cancellation before summation.

Snapshot: `prior_sources/github1059_comments.json`, source URL and hash in `prior_sources/manifest.json`. Raw API data contains the discussion for provenance; manuscript text should summarize and cite rather than reproduce it.

## phanerozoic/mamba3

Public API gives revision `595fa7cfc9b040fa3dbc207cae137f5697b47f1d`, lastModified `2026-07-30T13:01:28Z`. This metadata predates our work, but is not a detailed per-file first-publication chronology. Four relevant files were fetched at that revision.

- [Python wrapper](https://huggingface.co/kernels/phanerozoic/mamba3/blob/595fa7cfc9b040fa3dbc207cae137f5697b47f1d/torch-ext/mamba3/__init__.py), lines 72–78: `_angle_adjoint` forms an inclusive reverse cumsum of local terms. Lines 169–179 invoke the CUDA backward and feed its `dang` into this wrapper.
- [CUDA backward](https://huggingface.co/kernels/phanerozoic/mamba3/blob/595fa7cfc9b040fa3dbc207cae137f5697b47f1d/mamba3_cuda/mamba3_bwd.cu), lines 729–737: stores per-token/per-pair angle terms. The inspected wrapper does not replace their suffix carry with a boundary contraction.
- The implementation supports a broader MIMO setting and different precision/kernels, so card accuracy numbers are not a matched benchmark against our SISO prototype.

Conclusion: the inspected angle-gradient path is not the same boundary-state reformulation. It remains an alternative implementation relevant to strong baselines. This limited source audit cannot establish global novelty or guarantee priority.

## Closest mathematical baseline still required

A direct recurrent adjoint in local rotation coordinates differentiates the original per-step state update without expressing angles as a global cumulative sum. The fundamental tools are standard reverse-mode differentiation. A paper must compare both this formulation and precision upgrades, accounting for parallelism, memory and performance. Showing an advantage over a cancellation-prone cumulative-angle implementation alone does not establish that a new mathematical gradient algorithm was required.

The original manual FP64 sequential adjoint is a correctness reference in cumulative-angle coordinates. A separate `local_rotation_adjoint.py` now implements direct local-coordinate derivatives and validates every input gradient against autograd (32 cases, maximum normalized error4.58e-16). `local_rotation_triton.py` supplies a persistent FP32 GPU baseline, validates all gradients on18 small cases, and includes all input gradients, reductions and casts in timing. Development Vtile4/8/16 selects8. It is a concrete strong comparator, not a claim to exhaust all optimized implementations of local-coordinate differentiation.

## Original Mamba-3 formulation

Inspected [Mamba-3, arXiv v1, §3.2 and Appendix B.2](https://arxiv.org/html/2603.15569v1). Proposition 3 and equations (17)–(23) establish the rotation/product factorization and change of state coordinates for scalar decay and commuting rotations. That equivalence is prior art and underlies the present suffix-rotation proof. The inspected passages do not give our terminal-cotangent evaluation or its mixed-precision carry comparison. Nevertheless, differentiating an existing change of coordinates is a standard operation; the paper's claim should focus on a numerically useful realization and validated error/cost behavior, not a new underlying SSM equivalence.

## Access limits / scope of completed targeted audit

- Equation-level inspection of Mamba-3 section3.2 and AppendixB.2, plus pinned Phanerozoic and burn-mamba implementation paths, is completed as documented below. This closes the planned targeted audit for the narrowly framed realization; it does not establish global priority or the absence of other implementations.
- Search results for other direct-complex recurrent implementations are leads, not evidence that they do or do not contain this numerical reformulation.
- No “search not found, therefore new” conclusion is permitted.

## burn-mamba targeted inspection

Pinned revision `e44d42b7ab12f4b0ea3d0674f5fbb98b75071179`, retrieved28 October2026. Saved seven source/document files under `prior_sources/burn_*`; hashes recorded separately. This snapshot does not establish the first-publication date of each idea.

The [Complex2D branch](https://github.com/swfsql/burn-mamba/blob/e44d42b7ab12f4b0ea3d0674f5fbb98b75071179/src/mamba3/rotation/mod.rs#L1004) constructs cumulative angles with a blocked `prefix_sum` and rotates the inputs through ordinary tensor operations. The [RoPE helpers](https://github.com/swfsql/burn-mamba/blob/e44d42b7ab12f4b0ea3d0674f5fbb98b75071179/src/mamba3/rotation/rope.rs) discuss low-precision phase accumulation and maintain the autodiff graph through wrapping. This is relevant existing precision/phase management, not evidence of our particular SISO boundary-angle carry.

The inspected single/double SSD combined backward files differentiate decay cumsums. Those variables are not angular cumsums: do not cite their suffix sums as proof about the rotary gradient path. The [quaternion-scan backward](https://github.com/swfsql/burn-mamba/blob/e44d42b7ab12f4b0ea3d0674f5fbb98b75071179/src/mamba3/quat_scan/backward.rs) derives a parallel quaternion prefix-product adjoint involving suffix sums, a different operator and broader noncommuting setting. Its presence limits any broad claim that efficient rotational adjoints are new.

The inspected `rotation-as-optimization.md` concerns interpretations of the forward recurrence as optimization, not this gradient-error correction. No absence or global novelty conclusion is drawn from keyword search alone. The closest explicit public numerical diagnosis remains Johnsonms's cross-cut cancellation discussion, which must be credited in the manuscript.
