/-
  Machine-checked core of "Boundary Evaluation of Rotary State-Space Gradients in Mixed Precision"
  (independent verification, 2026-10-29).

  What is formalized (all `sorry`-free, checked by Lean 4 + Mathlib v4.35.0):
    §1  rotation transport of outer products and readouts;
    §2  the suffix-rotation EQUIVARIANCE theorem used in the manuscript's proof
        (shifted-angle states = perturbed-incoming-state states * Uᵀ, outputs equal,
         terminal state/key transform), for any commutative ring and any U with U Uᵀ = 1;
    §3  concrete 2×2 rotations: R(θ+ε) = R(ε) R(θ), Rᵀ R = 1, R(0) = 1;
    §4  derivative glue: d/dε F(ε) = Ψ'(w) when F = Ψ ∘ c (chain rule step of the proof);
    §5  cumulative-angle chain rule: d/dδ_u (L ∘ cum) = Σ_{t ≥ u} ∂L/∂θ_t;
    §6  chunk split  G_u = Σ_{t=u}^{b-1} g_t + B_b + κ;  zero-initial-state corollary G_0 = κ;
    §7  error decomposition (eq. 6) and bilinear contraction bound (eq. 7) with ‖J‖ ≤ 1;
    §8  local-coordinate change of variables H_t = S_t R(θ_t) (supplement eq. 8).

  Not formalized: trig derivatives (d/dε R(ε)|₀ = J is a hypothesis), differentiability of the
  whole recurrence, floating-point rounding models (γ_n bounds), tanh/sech² chain rule.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

open Matrix Finset

set_option linter.unusedSectionVars false

namespace BoundaryProof

/-! ## §1 Rotation transport -/

section Transport
variable {R : Type*} [CommRing R] {p n : Type*} [Fintype p] [Fintype n] [DecidableEq n]

/-- `v (U k)ᵀ = (v kᵀ) Uᵀ`. -/
lemma outer_rot (v : p → R) (k : n → R) (U : Matrix n n R) :
    vecMulVec v (U *ᵥ k) = vecMulVec v k * Uᵀ := by
  rw [vecMulVec_mul]
  congr 1
  ext j
  simp [vecMul, mulVec, dotProduct, transpose_apply, mul_comm]

/-- `(S Uᵀ)(U q) = S q` when `Uᵀ U = 1`. -/
lemma read_rot (S : Matrix p n R) (q : n → R) (U : Matrix n n R) (hU : Uᵀ * U = 1) :
    (S * Uᵀ) *ᵥ (U *ᵥ q) = S *ᵥ q := by
  rw [mulVec_mulVec, Matrix.mul_assoc, hU, Matrix.mul_one]

end Transport

/-! ## §2 Suffix equivariance (the induction in the manuscript's proof)

Relative index `m = t - u` on the suffix. `X0` is the folded incoming state `X_u`.
`states … k X0 m` is `S_{u+m}` computed from keys `k` (already rotated by the current angles). -/

section Equivariance
variable {R : Type*} [CommRing R] {p n : Type*} [Fintype p] [Fintype n] [DecidableEq n]

/-- Suffix recurrence  S_u = a_u X_u + γ_u v_u k_uᵀ,
    S_{t+1} = a_{t+1}(S_t + β_{t+1} v_t k_tᵀ) + γ_{t+1} v_{t+1} k_{t+1}ᵀ. -/
def states (a β γ : ℕ → R) (v : ℕ → p → R) (k : ℕ → n → R) (X0 : Matrix p n R) :
    ℕ → Matrix p n R
  | 0 => a 0 • X0 + γ 0 • vecMulVec (v 0) (k 0)
  | m + 1 => a (m + 1) • (states a β γ v k X0 m + β (m + 1) • vecMulVec (v m) (k m))
              + γ (m + 1) • vecMulVec (v (m + 1)) (k (m + 1))

/-- **Equivariance.** Rotating every suffix key by `U` gives the same states as keeping the keys
and replacing the incoming folded state `X0` by `X0 U`, up to the right factor `Uᵀ`. -/
theorem equivariance (a β γ : ℕ → R) (v : ℕ → p → R) (k : ℕ → n → R) (X0 : Matrix p n R)
    (U : Matrix n n R) (hU : U * Uᵀ = 1) (m : ℕ) :
    states a β γ v (fun t => U *ᵥ k t) X0 m = states a β γ v k (X0 * U) m * Uᵀ := by
  induction m with
  | zero =>
    simp only [states, outer_rot, Matrix.add_mul, Matrix.smul_mul, Matrix.mul_assoc, hU,
      Matrix.mul_one]
  | succ m ih =>
    simp only [states, ih, outer_rot, Matrix.add_mul, Matrix.smul_mul]

/-- Suffix outputs agree: `y'_t = S'_t (U q_t) = S''_t q_t`. -/
theorem outputs_agree (a β γ : ℕ → R) (v : ℕ → p → R) (k q : ℕ → n → R) (X0 : Matrix p n R)
    (U : Matrix n n R) (hU : U * Uᵀ = 1) (hU' : Uᵀ * U = 1) (m : ℕ) :
    states a β γ v (fun t => U *ᵥ k t) X0 m *ᵥ (U *ᵥ q m)
      = states a β γ v k (X0 * U) m *ᵥ q m := by
  rw [equivariance a β γ v k X0 U hU m, read_rot _ _ _ hU']

/-- Same statement with a scalar output gate / skip term that does not see the angles. -/
theorem gated_outputs_agree (a β γ : ℕ → R) (v : ℕ → p → R) (k q : ℕ → n → R)
    (X0 : Matrix p n R) (U : Matrix n n R) (hU : U * Uᵀ = 1) (hU' : Uᵀ * U = 1) (m : ℕ)
    (gate : (p → R) → (p → R)) :
    gate (states a β γ v (fun t => U *ᵥ k t) X0 m *ᵥ (U *ᵥ q m))
      = gate (states a β γ v k (X0 * U) m *ᵥ q m) := by
  rw [outputs_agree a β γ v k q X0 U hU hU' m]

end Equivariance

/-! ## §3 Concrete pair rotation -/

section Rotation
open Real

/-- Column-convention pair rotation `R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] = exp(θ J)`. -/
noncomputable def rot (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![cos θ, -sin θ; sin θ, cos θ]

/-- Generator `J = [[0,-1],[1,0]]`. -/
def Jgen : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

lemma rot_add (θ ε : ℝ) : rot (θ + ε) = rot ε * rot θ := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Fin.sum_univ_two, cos_add, sin_add] <;> ring

lemma rot_orth (θ : ℝ) : (rot θ)ᵀ * rot θ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Fin.sum_univ_two] <;> nlinarith [sin_sq_add_cos_sq θ]

lemma rot_orth' (θ : ℝ) : rot θ * (rot θ)ᵀ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Fin.sum_univ_two] <;> nlinarith [sin_sq_add_cos_sq θ]

lemma rot_zero : rot 0 = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rot]

/-- Shifting a cumulative angle by `ε` rotates the already-rotated key by `U = R(ε)`. -/
lemma shift_is_left_rotation (θ ε : ℝ) (k0 : Fin 2 → ℝ) :
    rot (θ + ε) *ᵥ k0 = rot ε *ᵥ (rot θ *ᵥ k0) := by
  rw [rot_add, mulVec_mulVec]

/-- The suffix-shift equivariance specialised to genuine rotations. -/
theorem rotary_equivariance {p : Type*} [Fintype p] (a β γ : ℕ → ℝ) (v : ℕ → p → ℝ)
    (θ : ℕ → ℝ) (k0 : ℕ → Fin 2 → ℝ) (X0 : Matrix p (Fin 2) ℝ) (ε : ℝ) (m : ℕ) :
    states a β γ v (fun t => rot (θ t + ε) *ᵥ k0 t) X0 m
      = states a β γ v (fun t => rot (θ t) *ᵥ k0 t) (X0 * rot ε) m * (rot ε)ᵀ := by
  have h : (fun t => rot (θ t + ε) *ᵥ k0 t) = fun t => rot ε *ᵥ (rot (θ t) *ᵥ k0 t) := by
    funext t; exact shift_is_left_rotation _ _ _
  rw [h]
  exact equivariance a β γ v _ X0 (rot ε) (rot_orth' ε) m

end Rotation

/-! ## §4 Derivative glue (chain-rule step of the proof) -/

section Deriv
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- If the angle-shifted loss factors as `F ε = Ψ (c ε)` (equivariance), the suffix angular
derivative is `Ψ'(c'(0))`; with `c ε = (X_u U(ε), …)` and `U'(0) = J` this is `⟨P_u, X_u J⟩ + κ`. -/
theorem boundary_deriv (F : ℝ → ℝ) (Ψ : E → ℝ) (c : ℝ → E) (w : E) (Ψ' : E →L[ℝ] ℝ)
    (hF : ∀ ε, F ε = Ψ (c ε)) (hc : HasDerivAt c w 0) (hΨ : HasFDerivAt Ψ Ψ' (c 0)) :
    HasDerivAt F (Ψ' w) 0 := by
  have hfun : F = Ψ ∘ c := funext hF
  rw [hfun]
  exact hΨ.comp_hasDerivAt 0 hc

end Deriv

/-! ## §5 Cumulative-angle chain rule -/

section Cum
variable {T : ℕ}

/-- `θ_t = Σ_{j ≤ t} δ_j` (initial angle omitted: it only translates). -/
def cum (δ : Fin T → ℝ) : Fin T → ℝ := fun t => ∑ j ∈ univ.filter (fun j => j ≤ t), δ j

def cumLin : (Fin T → ℝ) →ₗ[ℝ] (Fin T → ℝ) where
  toFun := cum
  map_add' x y := by ext t; simp [cum, Finset.sum_add_distrib]
  map_smul' c x := by ext t; simp [cum, Finset.mul_sum]

lemma cum_single (u : Fin T) : cum (Pi.single u (1 : ℝ)) = fun t => if u ≤ t then 1 else 0 := by
  ext t
  simp only [cum, Pi.single_apply]
  rw [Finset.sum_ite_eq']
  simp

lemma indicator_eq_sum (u : Fin T) :
    (fun t : Fin T => if u ≤ t then (1 : ℝ) else 0)
      = ∑ t ∈ univ.filter (fun t => u ≤ t), Pi.single t (1 : ℝ) := by
  ext s
  rw [Finset.sum_apply]
  simp only [Pi.single_apply]
  rw [Finset.sum_ite_eq]
  simp

/-- For any differentiable loss `L` of the cumulative angles,
`∂(L ∘ cum)/∂δ_u = Σ_{t ≥ u} ∂L/∂θ_t`. -/
theorem suffix_chain_rule (L : (Fin T → ℝ) → ℝ) (L' : (Fin T → ℝ) →L[ℝ] ℝ) (δ : Fin T → ℝ)
    (hL : HasFDerivAt L L' (cum δ)) (u : Fin T) :
    ∃ D : (Fin T → ℝ) →L[ℝ] ℝ, HasFDerivAt (L ∘ cum) D δ ∧
      D (Pi.single u 1) = ∑ t ∈ univ.filter (fun t => u ≤ t), L' (Pi.single t 1) := by
  let C : (Fin T → ℝ) →L[ℝ] (Fin T → ℝ) :=
    ContinuousLinearMap.pi fun t => ∑ j ∈ univ.filter (fun j => j ≤ t), ContinuousLinearMap.proj j
  have hcum : (cum : (Fin T → ℝ) → (Fin T → ℝ)) = C := by
    funext x; ext t
    simp [cum, C]
  have hC : HasFDerivAt (cum : (Fin T → ℝ) → (Fin T → ℝ)) C δ := by
    rw [hcum]; exact C.hasFDerivAt
  refine ⟨L'.comp C, hL.comp δ hC, ?_⟩
  rw [ContinuousLinearMap.comp_apply, ← hcum, cum_single, indicator_eq_sum, map_sum]

end Cum

/-! ## §6 Chunk split and terminal corollaries -/

section Chunk

/-- If the boundary identity holds at chunk start `b` (`Σ_{t ≥ b} g_t = B_b + κ`), then
`G_u = Σ_{t=u}^{b-1} g_t + B_b + κ` (manuscript eq. 5). -/
theorem chunk_split (g : ℕ → ℝ) {u b T : ℕ} (hub : u ≤ b) (hbT : b ≤ T) (Bb κ : ℝ)
    (hid : ∑ t ∈ Ico b T, g t = Bb + κ) :
    ∑ t ∈ Ico u T, g t = ∑ t ∈ Ico u b, g t + Bb + κ := by
  rw [← Finset.sum_Ico_consecutive g hub hbT, hid]
  ring

/-- Terminal correction (eq. 4): `κ = -⟨S̄, S_{T-1}J⟩ + k̄ᵀ(J k_{T-1}) + θ̄`. -/
noncomputable def kappa {p n : Type*} [Fintype p] [Fintype n] (Sbar SJ : Matrix p n ℝ)
    (kbar Jk : n → ℝ) (thbar : ℝ) : ℝ :=
  -(∑ i, ∑ j, Sbar i j * SJ i j) + kbar ⬝ᵥ Jk + thbar

/-- Output-only loss: all direct terminal cotangents vanish, hence κ = 0 (eq. 4),
independently of the (possibly nonzero) initial and final states. -/
theorem kappa_output_only {p n : Type*} [Fintype p] [Fintype n]
    (SJ : Matrix p n ℝ) (Jk : n → ℝ) : kappa 0 SJ 0 Jk 0 = 0 := by
  simp [kappa]

/-- Zero initial state ⇒ `X_0 = 0` ⇒ `B_0 = ⟨P_0, X_0 J⟩ = 0`, so `G_0 = κ`
(for output-only losses the total angular derivative vanishes: global rotation invariance). -/
theorem G0_eq_kappa {p n : Type*} [Fintype p] [Fintype n] (P0 : Matrix p n ℝ)
    (J : Matrix n n ℝ) (κ G0 : ℝ)
    (hid : G0 = (∑ i, ∑ j, P0 i j * ((0 : Matrix p n ℝ) * J) i j) + κ) : G0 = κ := by
  simpa using hid

end Chunk

/-! ## §7 Error decomposition -/

section Errors

/-- Manuscript eq. (6): computed carry minus exact carry is bounded by the local-term errors in
the current chunk only, plus boundary, terminal and local arithmetic errors. -/
theorem carry_error_bound {ι : Type*} (s : Finset ι) (g e : ι → ℝ) (B dB κ dκ η : ℝ) :
    |(∑ t ∈ s, (g t + e t) + (B + dB) + (κ + dκ) + η) - (∑ t ∈ s, g t + B + κ)|
      ≤ ∑ t ∈ s, |e t| + |dB| + |dκ| + |η| := by
  have h : (∑ t ∈ s, (g t + e t) + (B + dB) + (κ + dκ) + η) - (∑ t ∈ s, g t + B + κ)
      = ∑ t ∈ s, e t + dB + dκ + η := by
    rw [Finset.sum_add_distrib]; ring
  rw [h]
  calc |∑ t ∈ s, e t + dB + dκ + η| ≤ |∑ t ∈ s, e t + dB + dκ| + |η| := abs_add_le _ _
    _ ≤ |∑ t ∈ s, e t + dB| + |dκ| + |η| := by gcongr; exact abs_add_le _ _
    _ ≤ |∑ t ∈ s, e t| + |dB| + |dκ| + |η| := by gcongr; exact abs_add_le _ _
    _ ≤ ∑ t ∈ s, |e t| + |dB| + |dκ| + |η| := by gcongr; exact Finset.abs_sum_le_sum_abs _ _

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- Manuscript eq. (7): for `B = ⟨P, J X⟩` with a linear `J` of norm ≤ 1 (right multiplication
by `J_i` in Frobenius geometry), perturbed inputs and a contraction rounding error `η`. -/
theorem contraction_error_bound (J : F →ₗ[ℝ] F) (hJ : ∀ x, ‖J x‖ ≤ ‖x‖) (P X EP EX : F) (η : ℝ) :
    |(inner ℝ (P + EP) (J (X + EX)) + η) - inner ℝ P (J X)|
      ≤ ‖P‖ * ‖EX‖ + ‖X‖ * ‖EP‖ + ‖EP‖ * ‖EX‖ + |η| := by
  have expand : (inner ℝ (P + EP) (J (X + EX)) + η) - inner ℝ P (J X)
      = inner ℝ P (J EX) + inner ℝ EP (J X) + inner ℝ EP (J EX) + η := by
    rw [map_add, inner_add_left, inner_add_right, inner_add_right]; ring
  have b1 : |inner ℝ P (J EX)| ≤ ‖P‖ * ‖EX‖ :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left (hJ EX) (norm_nonneg _))
  have b2 : |inner ℝ EP (J X)| ≤ ‖X‖ * ‖EP‖ := by
    rw [mul_comm]
    exact (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left (hJ X) (norm_nonneg _))
  have b3 : |inner ℝ EP (J EX)| ≤ ‖EP‖ * ‖EX‖ :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left (hJ EX) (norm_nonneg _))
  rw [expand]
  calc |inner ℝ P (J EX) + inner ℝ EP (J X) + inner ℝ EP (J EX) + η|
      ≤ |inner ℝ P (J EX)| + |inner ℝ EP (J X)| + |inner ℝ EP (J EX)| + |η| := by
        refine (abs_add_le _ _).trans ?_
        gcongr
        refine (abs_add_le _ _).trans ?_
        gcongr
        exact abs_add_le _ _
    _ ≤ ‖P‖ * ‖EX‖ + ‖X‖ * ‖EP‖ + ‖EP‖ * ‖EX‖ + |η| := by gcongr

end Errors

/-! ## §8 Local-coordinate change of variables (supplement eq. 8) -/

section Local
variable {R : Type*} [CommRing R] {p n : Type*} [Fintype p] [Fintype n] [DecidableEq n]

/-- With `R_t = R_{t-1} R(δ_t)` orthogonal and `H = S R`, the global-angle step
`S_t = a(S_{t-1} + β v_{t-1} k_{t-1}ᵀ) + γ v_t k_tᵀ` (rotated keys) becomes
`H_t = a (H_{t-1} + β v_{t-1} k⁰_{t-1}ᵀ) R(δ_t) + γ v_t k⁰_tᵀ`. -/
theorem local_coordinates (a β γ : R) (S : Matrix p n R) (vprev v : p → R) (k0prev k0 : n → R)
    (Rprev Rd : Matrix n n R) (hprev : Rprevᵀ * Rprev = 1)
    (hcur : (Rprev * Rd)ᵀ * (Rprev * Rd) = 1) :
    (a • (S + β • vecMulVec vprev (Rprev *ᵥ k0prev)) + γ • vecMulVec v ((Rprev * Rd) *ᵥ k0))
        * (Rprev * Rd)
      = a • ((S * Rprev + β • vecMulVec vprev k0prev) * Rd) + γ • vecMulVec v k0 := by
  rw [outer_rot, outer_rot]
  simp only [Matrix.add_mul, Matrix.smul_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc (Rprevᵀ) Rprev Rd, hprev, Matrix.one_mul, hcur, Matrix.mul_one]

/-- Readout is unchanged: `S_t q_t = H_t q⁰_t` with `q_t = R_t q⁰_t`. -/
theorem local_readout (S : Matrix p n R) (Rt : Matrix n n R) (q0' : n → R) :
    S *ᵥ (Rt *ᵥ q0') = (S * Rt) *ᵥ q0' := by
  rw [mulVec_mulVec]

end Local

end BoundaryProof
