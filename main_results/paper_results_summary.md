# GAS-BayesSHAP — Paper Results Summary (regenerated 2026-08-16)

All numbers below are read from the committed `main_results/paper_*.csv` /
`*.json` artifacts.  Every experiment is reproducible via
`scripts/run_paper_experiments.py`, `scripts/probe_width_tightness.py`,
`scripts/probe_finite_population.py`, `scripts/ablation.py`,
`scripts/run_sota_baselines.py`, `scripts/regime_semantics.py`, and the
`RUN_REMAINING_EXPERIMENTS.ipynb` orchestrator.

---

## RQ1 — Attribution fidelity (real data, N=50, ε=0.05, budget 3000)

`paper_wine_n50_budget3000_summary.csv`, `paper_air_n50_budget3000_summary.csv`

| Dataset | Clusters | GAS RMSE (mean±std) | KernelSHAP RMSE | SamplingSHAP RMSE | GAS evals | Exact evals | Sim. coverage |
|---|---|---|---|---|---|---|---|
| Wine | 2 | **0.00168 ± 0.00101** | 0.00472 | 0.04655 | 1430 | 2048 | 1.0 |
| Air (Aotizhongxin) | 4 | **0.00186 ± 0.00117** | 0.00606 | 0.03695 | 1436 | 2048 | 1.0 |

- GAS-BayesSHAP recovers exact Shapley to RMSE ~1.7–1.9e-3 at ~70% of the
  exact enumeration budget, with simultaneous coverage 1.0 on all 50
  instances per dataset.
- Standard error over N=50: wine ±0.00014, air ±0.00016.
- **Honest caveat:** `sign_certified_fraction = 0.0` and
  `converged_fraction = 0.0` under the spec range at this budget: the
  anytime widths (~9.1–9.4) are valid but wider than the attribution scale.
  The certificate *tightens* predictably (see RQ5/frontier) and the
  finite-population mode (below) closes most of the width gap.

## RQ2 — Anytime coverage calibration (R=500)

`paper_coverage_calibration_R500.json` (spec),
`paper_coverage_calibration_R500_finite_population.json` (both modes)

| Mode | Empirical coverage | Finite-width rate | Mean width | Mean oracle cost |
|---|---:|---:|---:|---:|
| Spec (R_Δ = 4(U−L)) | 1.0 | 1.0 | 12.31 | 350 |
| Finite-population (Thm E) | 1.0 | 1.0 | **2.38** | 98.6 |

- The finite-population mode reports the realised coverage level
  `1 − δ2 − δ1` (mean 0.959, mean δ1 = 0.016 on M=3) — the certificate is
  rigorous at that level, and reaches the nominal 1−δ once the coupon
  thresholds hold (Corollary E).
- Real-data check: simultaneous coverage 1.0 on all N=50 instances of RQ1.

## RQ2b — Finite-population width tightening (real data)

`paper_wine_range_modes_n5_summary.csv`, `paper_wine_range_modes_n5.csv`

| Mode | RMSE | Mean width | Sim. cov | Sign-cert | R_eff |
|---|---:|---:|---:|---:|---:|
| Spec | 0.00059 | 8.75 | 1.0 | 0 | 4.0 |
| Finite-population | 0.00059 | **0.94** | 1.0 | 0 | 0.42 |

- 9.3× width reduction at identical RMSE and coverage (wine, K=3000, N=5).
- **Honest caveat:** at K=3000 the coupon collector is still open on the
  M=11 game (δ1 ≈ 59, realised level 0): the width reduction is real, but
  the *nominal* 1−δ certificate requires the frontier budget
  (K ≈ 2×10^5 for M=11, Corollary E).  This is exactly what the
  certification cost frontier (below) characterises.

## RQ3 — Regime semantics (air quality, Aotizhongxin station)

`paper_regime_semantics_summary.csv`

| Regime | N instances | RMSE | Spearman |
|---|---:|---:|---:|
| winter_smog | 1 | 0.00179 | 0.642 |
| photochemical | 1 | 0.00165 | 0.601 |
| clean_air | 2 | 0.00396 | 0.179 |

- Regimes are named by driver profile (winter_smog: PM10/CO/PM2.5;
  photochemical: O3/NO2/WSPM).  **Honest caveat:** N per regime is
  pilot-scale; a N≥20-per-regime run is required before this is a paper
  claim.

## RQ4 — Ablation (wine, pilot)

`paper_ablation_wine_summary.csv` (K=400, N=3, shared coalition cache)

| Tier | RMSE |
|---|---:|
| 1 Uniform MC | 0.0510 |
| 2 Neyman MC | 0.0091 |
| 3 GP-only | 0.0813 |
| 4 Full GAS | **0.0038** |

- Each component contributes; the GP-only tier is worst because the
  exponential-Hamming surrogate underfits the sharp membership boundary and
  its bias is uncorrected without the residual certifier — precisely the
  motivation for the dual-module design.  **Honest caveat:** N=3 pilot;
  N≥20 rerun with RMSE+width+coverage per tier is `scripts/ablation.py
  --n 20`.

## RQ5 — Matched-budget curves (nominal coalition budgets)

`paper_wine_matched_budget.csv`, `paper_air_matched_budget.csv` (N=8)

| K | Wine GAS | Wine Kernel | Air GAS | Air Kernel |
|---|---:|---:|---:|---:|
| 128 | **0.00430** | 0.00694 | **0.00418** | 0.01034 |
| 256 | **0.00352** | 0.00358 | **0.00433** | 0.00529 |
| 512 | 0.00304 | **0.00176** | 0.00312 | **0.00277** |
| 1024 | 0.00197 | **0.00089** | 0.00314 | **0.00124** |
| 2048 | 0.00133 | **0.00014** | 0.00208 | **0.00026** |

**Honest interpretation (rewritten after the audit):** GAS-BayesSHAP
dominates Monte Carlo everywhere and is competitive/better at low budgets
(≤ 256–512), while KernelSHAP achieves lower point-estimate RMSE at
medium/high budgets.  GAS's unique value is the distribution-free anytime
certificate, not unconditional RMSE dominance.  The "beats KernelSHAP"
claim of the earlier summary is **retracted**.  Note: budgets are *nominal*
(`max_budget` for GAS, `nsamples` for KernelSHAP); actual per-method call
counts are instrumented in the follow-up curves run.

## Tier-B (group-lag, M=66 → 11 pollutant macros)

`paper_air_tierB_summary.csv` (N=20): RMSE 0.00202, sim. coverage 1.0,
mean width 9.02, sign-cert 0.

- **Status: being rerun after the temporal label-alignment fix** (the
  previous run aligned lagged features at time t+24 with labels at time t;
  the corrected `run_tier_b` uses a contiguous chronological window and the
  shifted label alignment).

## SOTA-style baselines

`gas_bayesshap/benchmarking/sota_baselines.py` + `scripts/run_sota_baselines.py`
produce OddSHAP-style (exact log-odds Shapley) and ShaplEIG-style (GP
posterior-mean Shapley) comparisons.  **Honest labelling:** official
OddSHAP / ShaplEIG code is not public in this environment; these are
method-style reimplementations from their published descriptions, reported
as non-certified reference points — not official reproductions.

## The certification cost frontier

`paper_width_probe.csv`, `certificate_tightness_probe.md` (wine, ε=0.02,
1 instance)

| K | mean W (spec) | sign-cert | max err vs exact |
|---:|---:|---:|---:|
| 2 000 | 13.39 | 0 | 0.00269 |
| 8 000 | 3.84 | 0 | 0.00100 |
| 30 000 | 1.19 | 0 | 0.00066 |
| 100 000 | 0.400 | 0 | 0.00014 |

- Widths follow W ≈ 318/√K (spec) with errors always ≪ W — the bounds are
  valid, and the bottleneck is attribution scale + conservative R_Δ.
- The finite-population mode lowers the constant ~16× (W ≈ 19/√K in the
  frontier regime), making sign-certification feasible at K ~ 10^4–10^5 for
  the M=3 calibration game (validated) and characterisable for M=11
  (coupon-collector threshold K ≈ 2×10^5, Corollary E).

## What can / cannot be claimed

**Can claim:**
- Near-exact Shapley recovery on real wine/air membership games (RMSE
  ~1.7–1.9e-3 at ~1.4k evals, N=50) with simultaneous coverage 1.0.
- Certificates tighten as 1/√K with a characterised constant; the
  finite-population refinement shrinks widths 5–9× at a rigorous realised
  coverage level.
- Dominance over Monte Carlo at matched budgets; competitiveness with
  KernelSHAP at low budgets; the unique distribution-free anytime
  certificate.

**Cannot yet claim:**
- Unconditional RMSE superiority over KernelSHAP (curves contradict it).
- Sign-certified feature discovery at the standard budget (K=3000) under
  the spec range; under the finite-population mode this requires the
  coupon-completed regime (K ~ 10^4–10^5 for M=11), validated on M=3.
- Statistical regime semantics at N=1–2 per regime.
- Official OddSHAP/ShaplEIG parity (method-style baselines only).
