# GAS-BayesSHAP — Full Spec Analysis & Requirement Inventory (v11.0)

**Source documents analysed (current HEAD = `206ad00 "spec update"` on `main`):**
1. `GAS_BayesSHAP_Implementation_Spec (4).md` — **v11.0 Gold-Standard Final** implementation spec (authoritative engineering spec; 860 lines incl. inline certified reference implementation + 10-tier test suite)
2. `GAS_BayesSHAP_Paper_Structure (3).md` — **v11.0 Gold-Standard Final** academic paper structure (theoretical framing, domain games, experiments)

**Previous versions (superseded):** `GAS_BayesSHAP_Implementation_Spec (2).md` (v7.1, 629 lines) and `GAS_BayesSHAP_Paper_Structure (1).md` (v7.1, 201 lines) were deleted in the last commit and replaced by the v11.0 documents above.

**Repository state:** no implementation code exists yet; Section 4 of the implementation spec contains the complete inline "certified reference implementation" (source of truth). The spec's own test file `test_gas_bayesshap.py` does not yet exist and must be created. **The inline reference + 10-tier suite were extracted and executed during this analysis: all 10 tests pass** (one typo fixed, see §7).

---

## 1. Executive Summary

GAS-BayesSHAP estimates Shapley values for black-box set functions `v(S)` over `M` features with two decoupled modules plus a projection stage:

- **Module A — Active Bounded-Linear GP Control Variate:** learns a GP-based linear surrogate `m_b(S) = c + λ·h(S)` on actively-selected coalitions (`h(S) = k_D(S)ᵀα`, `α = (K_DD + η²I)⁻¹y`), **linearly shrunk** so that `m_b(S) ∈ [L,U]` for *all* `2^M` coalitions — replacing the old v7.1 nonlinear clipping. Its Shapley attribution `φ(m_b) = λ·K_{φ,D}·α` is computed in closed form with zero sampling variance via exact analytical kernel–Shapley covariances (Lemma D O(M²) + Lemma E O(M³)).
- **Module B — Stratified Residual Certifier:** measures the residual game `r_D(S) = v(S) − m_b(S)` on an unbiased Neyman-allocated stratified sample (both Add-One and Remove-One marginals) and certifies the combined estimator with an **anytime, distribution-free empirical-Bernstein confidence sequence** (Theorem B). Extreme strata s=0 and s=M−1 are exact deterministic singletons (Lemma G, 0 width).
- **Stage 3 — Posterior-Diagonal Projection:** projects the raw estimator onto the efficiency hyperplane weighted by the (λ²-scaled) posterior diagonal, with analytical post-projection certificate (Theorem C / Corollary C.1) enabling **Sign-Certified Feature Importance** (Definition 1: `|φ_i*| > W_i^proj`).

---

## 2. What Changed in the Last Commit (`206ad00`) — v7.1 → v11.0 Delta

### 2.1 Methodology / theory changes
| # | v7.1 (old) | v11.0 (enhanced) |
|---|---|---|
| D1 | Clipped GP posterior `clip(m+E_base, L, U)` — nonlinear, biased | **Bounded Linear Surrogate** `m_b(S) = c + λ·h(S)` with conservative global bounds `h_lb ≤ h(S) ≤ h_ub`, `λ = min(1, (U−L)/(h_ub−h_lb))`, `c = L − λ·h_lb`. Bounded **globally on all 2^M subsets, no clipping bias**, exact residual range `R_Δ^res = 4(U−L)`. |
| D2 | Lemma D computed per-feature (M × O(M²)) | Lemma D upgraded to **O(M²) via V_in(r)/V_out(r) symmetry** — all `i ∈ S_j` share `V_in(r)`, all `i ∉ S_j` share `V_out(r)`. |
| D3 | Remark 2.1 (extreme-stratum init) | Promoted to **Lemma G (Exact Singleton Extreme-Stratum Identification)**. |
| D4 | Theorem A: joint ℓ₂ Neyman allocation with closed-form `K_s* ∝ ‖σ_s‖₂` | **Theorem A: Coupled Adjacent-Stratum Neyman Allocation Program** — convex program over draw sizes `q ∈ {1..M−1}`: `min Σ_s ‖σ_s^r‖₂² / ((M−s)K_s + (s+1)K_{s+1})` s.t. `ΣK_q = K_cert` (accounts for add-one/remove-one joint accumulation; draw size q=M−1 supplies stratum M−2). Solved numerically (SLSQP) in `_solve_coupled_neyman_allocation`. |
| D5 | Theorem B: "Residual Bernstein Supermartingales" | **Theorem B: Anytime Stratified Empirical-Bernstein Confidence Sequences** — identical formula; adds the denominator-M interpretation note (W bounds the full M-term stratified estimator). |
| D6 | Theorem C: "Uncertainty-Weighted MAP Efficiency Projection" | **Theorem C: Posterior-Diagonal Uncertainty-Weighted Projection** with `v = diag(Σ_{m_b|D_gp})` (λ²-scaled). Same projection formula. |
| D7 | — | **NEW Definition 1 (Sign-Certified Feature Importance):** feature is certified important iff `|φ_i*| > W_i^proj`; grey out in waterfall plots otherwise. |
| D8 | §3 "Software Architecture & Query Accounting" (prose section) | Removed as a numbered section; query accounting (2 meters) still implemented in code. New §3 is **Domain Game Formulations for Explainable Clustering**. |
| D9 | Remark 2.2 fallback: `R_Δ = 4·max(1, |Δ_total|)` | Fallback derives **L = min(E_base, v_N) − |Δ_total|, U = max(E_base, v_N) + |Δ_total|**, `R_Δ = 4(U−L)` (always ≥ the old heuristic, still flagged `range_bound_is_heuristic=True`). |
| D10 | Complexity: GP prediction O(D)/coalition; active search O(P·(D²+MD)); total `O(K·B·Cost + D²P + M³)` | Complexity: GP prediction **O(DM)/coalition**; active search **O(P·D³) total**; total `O(K_coal·B·Cost(f) + P·D³ + M³)`, memory `O(D²+DM+M²)` — strictly polynomial. |
| D11 | 6-tier verification suite | **10-tier verification suite** with complete inline code (see §5). |
| D12 | Target: generic ML Shapley | Target domain added: **explainable clustering, air-quality regime monitoring, high-dimensional lagged attributions** (5 domain games with explicit `R_Δ^res`: membership=4, contrastive=8, archetype=4, silhouette=8, group-lag M=66→M_group=11). |

### 2.2 Code-level changes (reference implementation)
- Constructor gains `self._surrogate_scale = 1.0` and `self._surrogate_shift = 0.0`.
- `_closed_form_cross_cov(S_j)`: rewritten to compute `V_in(r) = eval_cross_scalar(r−1, +1)` (if r>0) and `V_out(r) = eval_cross_scalar(r, −1)` (if r<M), then `K_phi_j[S_j] = V_in; K_phi_j[~S_j] = V_out`.
- **NEW** `_solve_coupled_neyman_allocation(sigma_res) → probs[M]`: SLSQP solve of the coupled convex program; `probs[0] = 0` always (extreme strata never drawn in Stage 2); fallback uniform over q∈{1..M−1} if solver fails; `A[s] = Σ_i σ²_{i,s} + 1e−8`; objective uses `max(d_s, 1e−12)`; returns zeros for M ≤ 2.
- `_predict_gp_fast(S)`: returns `shift + scale·h(S)` — **no clipping** (boundedness now structural). Returns 0.0 when GP empty (note: 0.0 is now an absolute-level prediction; see edge case E4).
- `explain`: after Stage 1 computes `pos_alpha_sum/neg_alpha_sum`, `h_lb/h_ub`, sets scale/shift; `phi_m_D = scale·(K_phi_D @ alpha)`; `phi_cov_mb = scale²·phi_cov_h`; `posterior_variances = max(diag(phi_cov_mb), 1e−10)`.
- Stage 2 surrogate evaluations use `_predict_gp_fast(S)` **without** `+ E_base` (v7.1 added E_base; the v11.0 surrogate is absolute through c/λ).
- Neyman sampling guard: `s_target = choice(M, p=neyman_probs)` if `Σp > 0` else `s_target = 1`.
- Docstring of `_predict_gp_fast` updated ("O(DM)", "Bounded Linear Shrinkage Surrogate").
- All other Stage-2/Stage-3 logic, budget guard, 5M refresh, return contract: **unchanged from v7.1**.

---

## 3. Requirement Inventory — Implementation Spec v11.0

### 3.1 Architecture (Spec §1, §2)

| # | Requirement | Detail |
|---|---|---|
| R1 | Dual-module decoupling | Module A on `D_gp` (active), Module B on `D_cert` (unbiased random Neyman strata). |
| R2 | Frozen surrogate | Surrogate frozen after Stage 1 (Remark 2.3) — conditional i.i.d. residual marginals for the filtration. |
| R3 | Closed-form surrogate attribution | `φ_i(m_b) = λ·[K_{φ,D}(K_DD+η²I)⁻¹ y]_i` — exact, zero sampling variance. |
| R4 | Boundedness without clipping | `h_lb ≤ h(S) ≤ h_ub` conservative (ρ^M contraction of Σα); `λ = min(1, (U−L)/(h_ub−h_lb))`; `c = L − λh_lb`; ⇒ `m_b(S) ∈ [L,U] ∀ S`; `R_Δ^res = 4(U−L)` exact. |
| R5 | Exact prior covariance | `K_{φ,φ}` analytical (Lemma E), precomputed in constructor. |
| R6 | Vectorized fast inference | `D_matrix` + `α` precomputed; O(DM) `_predict_gp_fast`. |
| R7 | Symmetrized posterior covariance | `Σ_h = K_{φ,φ} − K_{φ,D}K_DD⁻¹K_{φ,D}ᵀ`; `Σ_{m_b} = λ²Σ_h`; floor 1e−10 on diag. |
| R8 | Deterministic extreme-stratum identification (Lemma G) | s=0, s=M−1 singletons; exact residual means, σ=0, 0 width; 2M+2 calls. |
| R9 | Residual marginals both directions | Add-One → stratum s; Remove-One → stratum s−1 (Lemma F). |
| R10 | Unbiased stratified residual estimator | `φ̂_i(r_D) = (1/M) Σ_s μ̂_{i,s}(R)` over per-cell means. |
| R11 | Anytime Bernstein certification | `|φ̂_i^raw − φ_i| ≤ W_i^res(n_{i,s})` ∀n, ∀i with prob ≥ 1−δ (Theorem B). |
| R12 | Unified raw estimator | `φ̂_i^raw = φ_i(m_b) + φ̂_i(r_D)`. |
| R13 | MAP projection | `φ_i* = φ̂_i^raw + v_i·[(v(N)−v(∅) − Σφ̂_j^raw)/Σv_j]`, `v = diag(Σ_{m_b|D_gp})`. |
| R14 | Post-projection certificate | `|φ_i* − φ_i| ≤ W_i^proj ≡ W_i^res + (v_i/Σv_j)·Σ_j W_j^res` (Corollary C.1). |
| R15 | Sign certification (Definition 1) | `|φ_i*| > W_i^proj` ⇒ feature certified important at (1−δ). |

### 3.2 Mathematical core

**Lemma D (R16) — exact O(M²) hypergeometric cross-covariance.** Kernel `k(S,S') = σ₀²ρ^{|SΔS'|}`, `ρ = e^{−1/ℓ}`. For observed coalition `S_j` of size `r`, all `i ∈ S_j` share `V_in(r)` (r>0), all `i ∉ S_j` share `V_out(r)` (r<M):
```
V_in(r)  = (σ₀²(1−ρ)/M) · Σ_{s=0}^{M−1} Σ_l [C(r−1,l)·C(M−r,s−l)/C(M−1,s)]·ρ^{r−1+s−2l}   (l: max(0,s−M+r)…min(s,r−1))
V_out(r) = −(σ₀²(1−ρ)/M) · Σ_{s=0}^{M−1} Σ_l [C(r,l)·C(M−1−r,s−l)/C(M−1,s)]·ρ^{r+s−2l}   (l: max(0,s−M+1+r)…min(s,r))
```
`V_in(0) ≡ 0`, `V_out(M) ≡ 0` by definition.

**Lemma E (R17) — exact prior Shapley covariance.** `K_{φ,φ} = (V_diag − V_off)·I_M + V_off·1_M·1_Mᵀ`; `V_diag = (2σ₀²(1−ρ)/M²)·Σ_sΣ_tΣ_l [C(s,l)C(M−1−s,t−l)/C(M−1,t)]ρ^{s+t−2l}` (l: `max(0,s+t−M+1)…min(s,t)`); `V_off = σ₀²(1−ρ)²·Σ_sΣ_t Δw_sΔw_t·Σ_l C(M−2,s)C(s,l)C(M−2−s,t−l)ρ^{s+t−2l}` (s,t∈[0,M−2], l: `max(0,s+t−M+2)…min(s,t)`); `Δw_s = s!(M−2−s)!(M−2−2s)/M!`. Symmetrized in code.

**Lemma G (R18)** — exact singleton extreme-stratum identification (see R8).

**Lemma F (R19)** — conditional stratum uniformity of Add-One/Remove-One marginals (see R9).

**Theorem A (R20) — coupled adjacent-stratum Neyman allocation program.** Expected count backing interior stratum s: `n_s(K) = ((M−s)/M)K_s + ((s+1)/M)K_{s+1}`; solve `min_K (1/M)Σ_{s=1}^{M−2} ‖σ_s^r‖₂² / ((M−s)K_s + (s+1)K_{s+1})` s.t. `Σ_{q=1}^{M−1} K_q = K_cert, K_q ≥ 0`; q=M−1 included to feed stratum M−2. Dynamic re-solve every 5M evaluations (R21).

**Theorem B (R22) — anytime stratified empirical-Bernstein CS.** With frozen bounded-linear surrogate, residuals `∈ [−2(U−L), 2(U−L)]`, `R_Δ^res = 4(U−L)`:
```
W_i^res(n_i) = (1/M) · Σ_{s=1}^{M−2}
   [ sqrt(2·(σ̂^r_{i,s})²·log(π²M²n²_{i,s}/(3δ))/n_{i,s})
   + 7·R_Δ^res·log(π²M²n²_{i,s}/(3δ)) / (3·(n_{i,s}−1)) ]
```
- Coverage: `P(∃n≥2·1_interior, ∃i: |φ̂_i^raw − φ_i| > W_i^res) ≤ δ` (R23).
- Extreme strata 0-width once `n_{i,s} ≥ 1` (R24); interior cells need `n ≥ 2`, else `W_i^res = ∞` (R25).
- **Remarks:** 2.1 = Lemma G; 2.2 = heuristic-bounds fallback (D9) with `range_bound_is_heuristic=True` (R26); 2.3 = frozen surrogate (R2).

**Theorem C / Corollary C.1 (R27, R28)** — see R13/R14; v = λ²-scaled posterior diagonal; Sign-Certification Definition 1 (R15).

### 3.3 Query accounting (kept from v7.1, code-level)

| # | Requirement | Detail |
|---|---|---|
| R29 | `num_coalition_evals` | Per-`explain` delta of `total_coalition_evals` (counts calls, incl. duplicates). |
| R30 | `num_model_evals` | Per-`explain` delta of `total_model_evals`: B per hybrid coalition, 0 for ∅ shortcut, 1 for full-set shortcut. |
| R31 | Cumulative meters | `total_coalition_evals` / `total_model_evals` accumulate across calls on the instance. |
| R32 | Init accounting | `E_base` = mean of B background forward passes, charged to cumulative `total_model_evals` only. |

### 3.4 Domain games (Spec §3 — new in v11.0)

| Game | Formula | Range | R_Δ^res |
|---|---|---|---|
| Membership attribution | `v̂_{x,c}(S) = (1/B)Σ_b g_c(x_S, z̄_S^{(b)})` | [0,1] | **4** |
| Contrastive regime | `v̂_{x,c,c′}(S) = (1/B)Σ_b [g_c − g_{c′}]` | [−1,1] | **8** |
| Global archetype | mean over archetype set `Ĩ_c` × background | [0,1] | **4** |
| Intrinsic silhouette | `v̂_sil(S) = Silhouette(Cluster(X_S))`, `v̂_sil(∅)=0`, deterministic init | [−1,1] | **8** |
| Group-lag spatiotemporal | M=66 features → M_group=11 macro-players, block background sampling, exact ground truth at 2^11=2048 | — | — |

Convention 1: interventional (marginal) imputation, strictly deterministic oracle; coverage is w.r.t. `φ(v̂)`; background discretization error O(B^{−1/2}) is orthogonal.

### 3.5 Reference implementation (Spec §4) — functional requirements

Constructor and helpers unchanged from v7.1 except: `_surrogate_scale/_shift` init (R33); `_closed_form_cross_cov` O(M²) V_in/V_out (R16); `_predict_gp_fast` linear shrinkage, no clipping, empty-GP → 0.0 (R34); **`_solve_coupled_neyman_allocation`** SLSQP convex solver, `probs[0]=0`, uniform-q fallback, zeros for M≤2 (R20/R35).

`explain` stages (R36–R50), unchanged semantics from v7.1 except where listed in §2.2:
- **Stage 1:** seeds {∅, N} ∪ one random subset per size s∈{1..M−1}; rank-1 Sherman–Morrison updates with near-duplicate guard (`schur < η²` → skip, `ok=False`); A-optimal pool `max(32, 2M)` with score `Σ(cov_phi)²/(max(v_post,1e−8)+η²)`; freeze α, D_matrix; then compute h_lb/h_ub, λ, c; `φ(m_b) = λ·K_φD·α`; `Σ_{m_b} = λ²Σ_h`.
- **Stage 2:** Lemma G extreme init (2M+2 calls, σ=0); `n_pilot` pilots per interior stratum; `safe_std` (ddof=1, default 0.5, preserves true 0); coupled Neyman solve; adaptive loop: Theorem-B widths (`∞` for invalid cells), stop when `max(W) ≤ ε`; strict budget guard `(spent) + (1+M) > max_budget → break` (spent measured after extreme-init+pilot); Neyman re-solve every 5M iterations; draw `s_target ~ p` (fallback 1 if degenerate), evaluate, append Add-One → (s,·) / Remove-One → (s−1,·), refresh `sigma_res` on interior cells.
- **Stage 3:** `φ̂(r_D)` stratum-mean average; raw = surrogate + residual; projection; Corollary C.1 widths.
- **Return contract (R51) — 12 keys, unchanged:** `shapley_values`, `surrogate_shapley`, `residual_shapley`, `raw_confidence_widths`, `certified_projected_widths`, `posterior_std`, `num_coalition_evals`, `num_model_evals`, `converged_early`, `certificate_is_rigorous` (= bounds supplied AND all certified widths finite), `range_bound_is_heuristic` (= bounds None), `uncertified_features`.

### 3.6 Complexity (Spec §5)

| R | Operation | Time | Space |
|---|---|---|---|
| R52 | Exact `K_{φ,φ}` | O(M³) upfront | O(M²) |
| R53 | Lemma D cross-cov per subset | **O(M²)** (V_in/V_out symmetry) | O(M) |
| R54 | Sherman–Morrison per active query | O(D²) | O(D²) |
| R55 | Total active search (D steps) | **O(P·D³)** total | O(P·M) |
| R56 | Vectorized GP prediction | **O(DM)** per coalition | O(DM) |
| R57 | Bernstein CS check | O(M²) per iteration | O(M) |
| R58 | Projection + Corollary C.1 | O(M) | O(M) |
| R59 | Total engine | O(K_coal·B·Cost(f) + P·D³ + M³) | O(D²+DM+M²) — strictly polynomial |

---

## 4. Formula ↔ Code Cross-Reference (v11.0)

| Spec element | Code site | Notes |
|---|---|---|
| `ρ = e^{−1/ℓ}` | `__init__` | `sigma0`, `eta` stored |
| Kernel `k = σ₀²ρ^{d_H}` | `_kernel_val` | |
| Lemma D (O(M²) V_in/V_out) | `_closed_form_cross_cov` | `K_phi_j[S_j]=V_in; ~S_j=V_out` |
| Lemma E `K_{φ,φ}` | `_compute_exact_K_phi_phi` | V_diag, V_off, Δw_s; symmetrized |
| Sherman–Morrison | `_rank1_inverse_update` | Schur guard `< η²` → skip |
| Bounded linear surrogate | `_predict_gp_fast` + scale/shift | `shift + scale·h`; no clipping |
| h_lb/h_ub, λ, c | `explain` Stage-1 tail | `ρ^M` contraction of α sums |
| Theorem A coupled Neyman | `_solve_coupled_neyman_allocation` | SLSQP; probs[0]=0; 5M re-solve |
| Theorem B width | Stage-2 width check | 0 for extremes; ∞ invalid cells |
| Theorem C projection | Stage-3 `phi_final` | v = λ²·posterior diag |
| Corollary C.1 widths | Stage-3 `certified_widths` | inflation `v_i·ΣW/Σv` |

---

## 5. 10-Tier Verification Suite (Spec §6) — Acceptance Criteria

The spec includes complete runnable test code. Extracted and executed: **10/10 pass** (after fixing the driver typo `run_all_tests()` → `test_all()`). Acceptance criteria per tier:

| Tier | Test | Assertion / acceptance |
|---|---|---|
| T1 | Lemma D sign & exact enumeration | `_closed_form_cross_cov` vs 2^M brute force (M=4) `allclose(atol=1e−10)` |
| T2 | Lemma E raw pair counts | analytical vs 4^M double enumeration, M ∈ {2,3,4,5,6}, `max|Δ| < 1e−10` |
| T3 | Null-player certified containment | true 0.0 ∈ [φ̂_null − W_null^proj, φ̂_null + W_null^proj] (M=5, weights w/ 0.0 player) |
| T4 | Coverage calibration (R=30) | coverage-given-finite ≥ 90% (M=3, exact φ = [1.5, 2.5, −1.0]; seeded per trial) |
| T5 | Corollary C.1 tightness | prints mean inflation `W^proj/W^res` (≈2.2× in run; **no assert** in reference) |
| T6 | Query isolation & budget guard | two `explain` calls: per-call `num_coalition_evals > 0`; strict Stage-2 budget containment |
| T7 | Surrogate global boundedness | `_predict_gp_fast(S) ∈ [L,U]±1e−10` for ALL 2^M subsets (M=4, L=0, U=1) |
| T8 | Zero extreme-stratum allocation | `_solve_coupled_neyman_allocation(ones)` ⇒ `probs[0] == 0.0` |
| T9 | M=2 exact certification | `certified_projected_widths` all exactly `0.0` |
| T10 | Surrogate linearity / additive fit | linear model w=[1,2,3]: `shapley_values ≈ w` `allclose(atol=0.2)` |

---

## 6. Edge Cases & Degenerate Configurations (v11.0)

1. **M=1:** extremes s=0/s=M−1 coincide; empty interior; immediate W=0 convergence; guard in `_solve_coupled_neyman_allocation` (M≤2 → zeros) and pilot loop `range(1, max(1, M−1))`.
2. **M=2:** no interior strata; extreme init enumerates everything; `certified_widths == 0.0` exactly (T9) — note posterior variances ≥ 1e−10 but inflation is 0 since ΣW=0.
3. **`output_bounds=None`:** derived L/U from `min/max(E_base, v_N) ± |Δ_total|`; `R_Δ = 4(U−L)`; `range_bound_is_heuristic=True`, `certificate_is_rigorous=False`.
4. **Empty GP (`D_gp` empty):** `_predict_gp_fast` → 0.0 — **now absolute level** (c-shiftless); surrogate attribution 0; all signal from Module B. (In v7.1 the 0.0 was centred.)
5. **Budget exhaustion:** break cleanly; `converged_early=False`; widths may be ∞; `uncertified_features` lists them.
6. **Near-duplicate active queries:** rank-1 update `ok=False`; candidate skipped, matrices intact.
7. **Zero-variance cells:** `safe_std` preserves true 0 (not NaN/0.5).
8. **Coupled Neyman solver failure (SLSQP `res.success=False`):** fallback uniform over q ∈ {1..M−1}; `probs[0]=0` always.
9. **Degenerate Neyman probs (Σp=0):** `s_target = 1` fallback.
10. **Large M:** `comb`/`factorial` float64 overflow risk; spec tests cap Lemma E at M=6; practical target M up to 66 (group-lag game uses M_group=11).
11. **h_ub ≤ h_lb (degenerate α):** `λ` forced to 1.0, `c = L − h_lb` (guard in code).

---

## 7. Ambiguities, Bugs & Caveats in the Enhanced Spec (flagged for the build phase)

1. **Spec bug (test driver):** the inline suite ends with `if __name__ == "__main__": run_all_tests()` but defines `test_all()` — the created test file must call `test_all()`. (Fixed during verification; without it the suite does nothing.)
2. **T5 has no assertion** — only prints the inflation ratio; the build must decide whether to assert a range (paper spec says ≈1.5–2.0×; observed 2.19× at T3 settings).
3. **T5/T6 reuse variables from T3** (`res`, `eng`) — fine inside one driver function, but brittle if tests are split into separate functions/files; the delivered `test_gas_bayesshap.py` should keep them in one driver or explicitly thread the objects.
4. **T9 asserts exact `== 0.0`** on floats — verified to hold because widths are built from exact 0.0 additions; keep the arithmetic order (start `W_i_res = 0.0`, add only `(1/M)·w_s` for interior strata) or the test could break.
5. **Seeding:** the reference engine uses global `np.random`; T4 seeds per trial (`np.random.seed(trial)`); other tests rely on implicit state. Delivered tests should seed deterministically for CI reproducibility.
6. **`max_budget` scope:** Stage-2 adaptive loop only, measured after extreme-init+pilot; per-round upper cost 1+M coalition evals.
7. **Duplicate coalition evaluations counted** by design (v_empty/v_full appear across stages).
8. **Extreme-cell duplicates from pilot/adaptive sampling** (Remove-One from s=1, Add-One into s=M−2) — harmless; means unchanged; widths ignore extremes after n≥1.
9. **`certificate_is_rigorous` requires both** supplied `output_bounds` AND finite certified widths.
10. **Coverage semantics:** Theorem B certifies the raw estimator's anytime stopping; Corollary C.1 lifts to the projected estimator on the same (1−δ) event.
11. **`_predict_gp_fast` empty-GP return value changed meaning** (0.0 absolute vs centred in v7.1) — Stage-2 residual arithmetic and T7 boundedness rely on the new semantics; keep them consistent.
12. **Paper spec (3) is narrative** but its RQ2 (≥95% empirical coverage over R=500), RQ5 (query reduction vs ShaplEIG/OddSHAP), RQ6 (kernel-misspecification robustness on parity games) define the future experimental phase; the implementation must preserve the required hooks (per-call meters, `posterior_std`, certified widths, frozen-surrogate flag).

---

## 8. Build-Phase Checklist (v11.0)

- [ ] Module with `GASBayesSHAP` class: exact constructor signature/defaults + `_surrogate_scale/_shift`.
- [ ] Query-accounted `_eval_coalition` (0/1/B shortcuts, hybrid imputation, Convention 1).
- [ ] `_kernel_val`, `_closed_form_cross_cov` (O(M²) V_in/V_out), `_compute_exact_K_phi_phi` (Lemma E).
- [ ] `_rank1_inverse_update` with near-duplicate guard; `safe_std`.
- [ ] `_predict_gp_fast` (linear shrinkage, no clipping, empty-GP → 0.0).
- [ ] `_solve_coupled_neyman_allocation` (SLSQP, probs[0]=0, fallbacks).
- [ ] `explain` Stage 1: seeds → active A-optimal loop → freeze → h_lb/h_ub/λ/c → `φ(m_b)=λK_φDα` → `λ²Σ_h` posterior.
- [ ] `explain` Stage 2: Lemma G init → pilot → coupled Neyman → anytime stopping → 5M re-solve → budget guard.
- [ ] `explain` Stage 3: residual estimator → Theorem C projection → Corollary C.1 widths.
- [ ] 12-key return contract incl. flags/meters; `range_bound_is_heuristic` per Remark 2.2.
- [ ] 10-tier test suite `test_gas_bayesshap.py` (T1–T10) with `test_all()` driver, seeding, exact-0.0 M=2 assertion, boundedness sweep over 2^M.
- [ ] Complexity conformance (O(K·B·Cost + P·D³ + M³); O(D²+DM+M²) memory).
