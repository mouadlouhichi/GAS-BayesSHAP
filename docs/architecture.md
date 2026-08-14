# GAS-BayesSHAP Architecture

```
                    +--------------------------------------------------+
                    |             GAS-BayesSHAP (v11.0)                |
                    +--------------------------------------------------+
                                      |
            +-------------------------+--------------------------+
            |                                                    |
            v                                                    v
 [MODULE A: Active GP Control Variate]         [MODULE B: Stratified Residual Certifier]
   gas_bayesshap/gp/ + acquisition/              gas_bayesshap/residual/ + certification/
   - seeds -> active A-optimal acquisition      - Lemma G extreme strata (s=0, M-1)
   - bounded linear surrogate m_b = c + λh       - add-one / remove-one marginals (Lemma F)
   - φ(m_b) = λ K_φ,D α  (closed form)           - coupled adjacent-stratum Neyman program (Theorem A)
   - λ² scaled posterior covariance              - anytime empirical-Bernstein CS (Theorem B)
            |                                                    |
            +-------------------------+--------------------------+
                                      v
                      [UNIFIED CERTIFIED ESTIMATOR]
                      φ̂^raw = φ(m_b) + φ̂(r_D)
                                      |
                                      v
              [POSTERIOR-DIAGONAL EFFICIENCY PROJECTION]
              Theorem C + Corollary C.1 + sign certification
```

## Layer map

| Layer | Module | Responsibility |
|---|---|---|
| kernel | `kernels/hamming.py` | exponential Hamming kernel `k = σ₀²ρ^{d_H}`, `ρ = e^{-1/ℓ}` |
| kernel | `kernels/covariance.py` | Lemma D (O(M²) V_in/V_out), Lemma E (V_diag/V_off with Δw_s) |
| game | `game/oracle.py` | deterministic interventional oracle, exact query accounting |
| game | `game/brute_force.py` | independent 2^M / 4^M reference for validation |
| game | `game/domain_games.py` | membership / contrastive / archetype / silhouette / group-lag |
| gp | `gp/updates.py` | Sherman–Morrison rank-1 inverse updates + Schur guard |
| gp | `gp/posterior.py` | O(DM) vectorized prediction, posterior covariance |
| gp | `gp/control_variate.py` | bounded linear surrogate fit (h_lb/h_ub, λ, c) |
| acquisition | `acquisition/` | candidate pool (max(32,2M)), attribution-aware score |
| residual | `residual/strata.py` | per-cell (i, s) residual storage with metadata |
| residual | `residual/neyman.py` | coupled adjacent-stratum allocation program (SLSQP) |
| certification | `certification/bernstein.py` | Theorem-B width formula |
| certification | `certification/projection.py` | Theorem C, Corollary C.1, sign certification |
| core | `core/estimator.py` | stage orchestration, budget guard, resume, result schema |
| core | `core/results.py` | result schema + `results/runs/<id>/` writer + status model |
| engineering | `checkpointing/`, `logging/`, `cache/`, `utils/` | atomic checkpoints, JSONL events, keyed cache, hashing/provenance |

## Stages

`PREFLIGHT → ORACLE_VALIDATION → MATHEMATICAL_VALIDATION → GP_INITIALIZATION
→ ACTIVE_GP → BOUNDED_SURROGATE → SURROGATE_SHAPLEY → RESIDUAL_PILOT
→ NEYMAN_ALLOCATION → ADAPTIVE_CERTIFICATION → EFFICIENCY_PROJECTION
→ FINAL_RESULT → BENCHMARK → REPORT → COMPLIANCE_AUDIT`

Each stage has explicit inputs/outputs/dependencies/completion criteria,
checkpoint hooks and validation functions (see `core/estimator.py`).
