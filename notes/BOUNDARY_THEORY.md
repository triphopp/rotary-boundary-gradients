# Boundary identity and terminal correction — derivation and limitations

Status: mathematical derivation independently checked numerically; not a priority/novelty claim. Indices are zero-based. The formulas concern differentiation of a rotary state-space recurrence, not a new ODE solver.

## 1. Exact problem and the state at a cut

Consider one head with row state coordinates in S_t ∈ R^(P×N), N even. Let q_t=R(θ_t) q_t^raw and k_t=R(θ_t) k_t^raw in column convention, with independent 2D rotations on selected coordinate pairs. With scalar a_t, β_t, γ_t,

    X_t = S_(t−1) + β_t v_(t−1) k_(t−1)^T,
    S_t = a_t X_t + γ_t v_t k_t^T,
    y_t = S_t q_t.

The trapezoidal realization has a_t=exp(ADT_t), β_t=DT_t(1−sigmoid(Trap_t)), γ_t=DT_t sigmoid(Trap_t). Initial S_(−1), k_(−1), v_(−1) are independent inputs. Output-only gates/skip terms are allowed if their θ dependence is through y_t. The scalar transition and pairwise rotations commute. A general noncommuting matrix transition is NOT covered by this proof.

At a cut immediately before t=u, the incoming state is **X_u**, not S_(u−1) alone. Omitting the previous-input trapezoidal term changes the problem. The adjoint P_u=∂L/∂X_u follows only the suffix computation through X_u. Using ∂L/∂S_(u−1) from an unmodified tape would also include the output at u−1 and is generally wrong for this identity.

For pair i define J_i with 2×2 block [[0,−1],[1,0]]. On row states, (S J_i)_(2i)=S_(2i+1), (S J_i)_(2i+1)=−S_(2i). Define

    B_(u,i) = <P_u, X_u J_i>_F.

## 2. Exact identity including terminal losses

Let g_(t,i)=∂L/∂θ_(t,i), treating the cumulative θ values as independent variables. Include any terminal-angle cotangent in the last g term. Then

    G_(u,i) := Σ_(t=u)^(T−1) g_(t,i) = B_(u,i) + κ_i,

where

    κ_i = −<bar S_T, S_(T−1) J_i>_F
          + <bar k_T, J_i k_(T−1)>
          + bar θ_(T−1),i.

Here bar S_T and bar k_T mean cotangents supplied directly to the returned final states, not total adjoints that also include the last output. There is no v term because v is not rotated. For output-only losses all terminal cotangents vanish, so κ=0, even if the initial states are nonzero. The terminal angle modulo 2π is locally differentiable away from its wrap discontinuities; no theorem is claimed at a wrap.

### Proof by a suffix rotation

Shift θ_(t,i) by ε for every t≥u, holding the prefix fixed. Let U=exp(ε J_i). For suffix q,k, this is q→Uq and k→Uk. If instead one holds suffix q,k fixed and changes the incoming row state X_u→X_u U, induction on the recurrence shows that the shifted-angle state equals the incoming-state-perturbed state times U^T. Consequently suffix y values agree. Differentiating the incoming state contributes <P_u,X_u J_i>. The returned final state carries the additional right factor U^T and contributes −<bar S_T,S_(T−1)J_i>. The returned final k contributes <bar k_T,J_i k_(T−1)> and the final θ contributes its cotangent. The prefix contributes zero. This proves the identity.

For θ_t=θ_initial+Σ_(j≤t) δ_j, the chain rule gives ∂L/∂δ_u=G_u. For δ_u=π DT_u tanh(A_u), multiply G_u by π DT_u sech²(A_u) for the raw angle-rate gradient, and by π tanh(A_u) for its DT contribution. Other DT paths remain separate.

## 3. Chunk implementation

For u within a chunk ending just before b, evaluate the at-most-q local terms and use G_b=B_b+κ for the rest:

    G_u = Σ_(t=u)^(b−1) g_t + B_b + κ.

The final chunk uses its usual terminal-angle carry; do not double-count final S/K effects already present in its local g terms. A boundary formulation removes the explicit sum of noisy local terms over later chunks. It does not improve the local terms themselves automatically.

The old prototype estimates κ by G_last_start−B_last_start, where G_last_start comes from the last chunk's local terms. This propagates the error of that estimate to all earlier chunks. Direct evaluation from terminal states avoids those local terms. It may still be inaccurate when the terminal contractions are badly conditioned; it is a candidate improvement, not an unconditional dominance result.

## 4. Forward-error decomposition (deterministic)

For an exact reference problem, write computed local terms as g_hat_t=g_t+ε_t and computed contractions as B_hat=B+ΔB, κ_hat=κ+Δκ. If η_local denotes arithmetic error of the chosen local reduction and additions, then

    |G_hat_u−G_u| ≤ Σ_(t=u)^(b−1)|ε_t| + |ΔB_b| + |Δκ| + |η_local|.

There are at most q local ε terms. An ordinary suffix sum instead contains terms from u through T−1. This is an algebraic removal of one accumulation channel, not a claim that the total error is independent of T. There is no assumption that rounding errors are independent or zero mean. A sqrt(T) observation is empirical unless stochastic assumptions are supplied.

For exact B=<P,XJ>, computed inputs P+E_P, X+E_X and contraction rounding η_B give

    |ΔB| ≤ ||P||_F ||E_X||_F + ||X||_F ||E_P||_F
           + ||E_P||_F ||E_X||_F + |η_B|,

because ||J||_2=1 on its pair. The contraction rounding is bounded using the actual reduction graph: a sequential non-FMA sum of 2P rounded products permits the conservative γ_(2P+1) times the sum of absolute products, assuming standard normal arithmetic and (2P+1)u<1. Tree/FMA/tensor-core paths need their own operation/precision bounds; this note does not claim the GPU reduction satisfies a sequential graph bound unchanged. Terminal contractions admit the same input-perturbation and rounding decomposition. Near cancellation, relative error can be large even when absolute errors are small.

Uniform-in-T behavior requires separately bounded state/adjoint/terminal errors and scales. Near-unit decay, poorly conditioned terminal objectives, exploding adjoints, quantized forward states, or long accumulated phase errors can violate those assumptions. Input/parameter quantization changes the reference problem and must be measured separately.

## 5. Independent validation

`check_boundary_identity.py` constructs the O(T) recurrence directly in FP64 and compares it with the project's separate O(T²) attention-form reference. It does not reuse the GPU boundary code. Grid: T∈{1,2,7,65}, zero/nonzero initial states, full/partial rotation, all 16 subsets of final-angle/SSM/K/V cotangents, with nonzero output loss; 256 cases and 4,800 cuts.

Observed maximum normalized errors (denominator max(1,max|reference|)):

| Check | Maximum |
|---|---:|
| Boundary identity | 9.86e-16 |
| Sequential vs attention forward | 1.67e-16 |
| All input gradients vs attention reference | 7.78e-16 |
| 24 central finite-difference checks, h=1e-4/1e-5/1e-6 | 1.22e-9 |
| Wrong formula omitting terminal correction | 1.69 |

These tests validate implementation and catch terminal/sign/index errors; they are not substitutes for the proof, a machine-checked proof, or validation of the GPU implementation. Raw results: `boundary_identity_results.json`.

## 6. Prior art and permissible claim

[Johnsonms, PR #1059 comment, 13 Oct 2026](https://github.com/state-spaces/mamba/pull/1059#issuecomment-6138351445) already identifies inaccurate local angle terms amplified by reverse summation and suggests computing cross-cut interactions to cancel algebraically. That diagnosis and broad strategy belong to prior art. State-adjoint differentiation and symmetry arguments are also established tools. The candidate contribution must be limited to the particular derived boundary/terminal realization, numerical analysis, and demonstrated implementation utility if the remaining source audit supports it. No priority claim follows from this derivation alone.
