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
- Tight-ε calibration (ε ∈ {0.05, 0.1, 0.2, 0.5}, R=200, both modes):
  coverage 1.0 and finite-width rate 1.0 at every ε
  (`paper_calibration_eps{eps}_{mode}.json`); fp widths 0.57–0.95.
- Adversarial stress test (`paper_stress_finite_population.json`, R=200):
  M=3 coupon closes → nominal level reached on 99.5% of trials, coverage
  1.0; M=6 coupon open → `certificate_at_nominal_level=False`, realised
  level 0.0 honestly reported (never claims nominal coverage while open).

## RQ2b — Finite-population width tightening (real data)

`paper_wine_n50_budget3000_rangefinite_population_summary.csv`,
`paper_air_n50_budget3000_rangefinite_population_summary.csv` (N=50 each),
`paper_wine_range_modes_n5_summary.csv` (matched probe)

| Setting | Mode | RMSE | Mean width | Sim. cov | Sign-cert |
|---|---:|---:|---:|---:|---:|
| Wine N=50, K=3000 | Spec | 0.00168 | 9.15 | 1.0 | 0 |
| Wine N=50, K=3000 | Finite-pop | 0.00168 | **1.90** | 1.0 | 0 |
| Air N=50, K=3000 | Spec | 0.00186 | 9.43 | 1.0 | 0 |
| Air N=50, K=3000 | Finite-pop | 0.00186 | **1.85** | 1.0 | 0 |
| Wine N=5, K=3000 | Spec | 0.00059 | 8.75 | 1.0 | 0 |
| Wine N=5, K=3000 | Finite-pop | 0.00059 | **0.94** | 1.0 | 0 |

- **4.8–9.3× width reduction at identical RMSE and coverage.**  At the
  standard budget the fp width (~1.9) is still above the dominant
  attribution (~0.26), so sign-cert remains 0 at K=3000 — honest.
- The *nominal* 1−δ certificate requires the coupon-completed frontier
  budget (K ≈ 2×10^5 for M=11, Corollary E), demonstrated in the probe
  (below).

## RQ3 — Regime semantics (air quality, Aotizhongxin station)

`paper_regime_semantics_summary.csv`

| Regime | N instances | RMSE | Spearman |
|---|---:|---:|---:|
| winter_smog | 5 | 0.00101 | 0.559 |
| photochemical | 5 | 0.00206 | 0.418 |
| clean_air | 10 | 0.00256 | 0.056 |

- Regimes are named by driver profile (winter_smog: PM10/CO/PM2.5;
  photochemical: O3/NO2/WSPM).  N=20 instances total (up from the pilot
  N=1–2).  **Honest caveat:** clean_air has near-zero attributions, so its
  low Spearman is noise-dominated (expected, not a failure); N per regime
  is still modest (5–10).

## RQ4 — Ablation (wine, N=20)

`paper_ablation_wine_summary.csv` (K=1000, N=20, shared coalition cache;
per-instance rows include tier-4 width / sim. coverage / sign-cert)

| Tier | RMSE |
|---|---:|
| 1 Uniform MC | 0.0390 |
| 2 Neyman MC | 0.0070 |
| 3 GP-only | 0.0712 |
| 4 Full GAS | **0.0030** |

- Each component contributes; the GP-only tier is worst because the
  exponential-Hamming surrogate underfits the sharp membership boundary and
  its bias is uncorrected without the residual certifier — precisely the
  motivation for the dual-module design.  Tier-4 at K=1000 (spec): width
  24.7, sim. coverage 1.0, sign-cert 0.  Consistent with the N=3 pilot
  (0.051 / 0.009 / 0.081 / 0.0038).

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

`paper_air_tierB_summary.csv` (N=20, rerun after the temporal
label-alignment fix — contiguous chronological window + labels shifted by
max(LAGS)): RMSE 0.00195, sim. coverage 1.0, mean width 9.20, sign-cert 0.
(The misaligned run reported 0.00202 / 9.02; the corrected protocol shifts
both slightly.)

## SOTA-style baselines

`paper_sota_baselines_comparison.csv` (N=20, wine + air, K ∈ {256,1024,2048})
from `gas_bayesshap/benchmarking/sota_baselines.py` +
`scripts/run_sota_baselines.py`.  **Honest labelling:** official
OddSHAP / ShaplEIG code is not public in this environment; these are
method-style reimplementations, reported as non-certified reference points
— not official reproductions.

**Comparability caveat (important):** the OddSHAP-style baseline computes
the exact Shapley of the *log-odds* game, whose attributions are on an
unbounded scale — its RMSE vs the membership-game exact (≈1.5–1.7) is
~440–540× GAS's and is dominated by the transform's scale, not by
estimation error.  It is therefore reported as a *transform-sensitivity*
reference, not a fidelity baseline.  The ShaplEIG-style GP quadrature
(RMSE ≈0.077–0.086 on the membership game, same scale as GAS) confirms
that a surrogate without residual correction is insufficient for fidelity —
consistent with the ablation's GP-only tier.

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
- The finite-population mode lowers the constant ~6.5× on the probe
  (matched budgets: W_spec/W_fp = 6.6× at 10k, 6.5× at 30k, 6.3× at 100k).
  At K ≥ 3×10^4 the dominant feature (|φ*|=0.257) is sign-certified with
  the certified sign matching the exact sign (`paper_width_probe_
  finite_population.csv`: 1 feature at K=30k/100k, signs_match_exact=1,
  margin 0.168 at 100k).
- **Nominal certification closes at K=2×10^5** (Corollary E confirmed on
  real data): at K=150k realised level 0.892 (coupon open); at K=200k the
  run **converges with status CERTIFIED**, δ1=0.013, realised level 0.962,
  1 feature sign-certified, signs_match_exact=1, margin 0.207 — the first
  run of the project to reach the nominal 1−δ level at feasible budget on
  an M=11 real game.

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
- Sign-certified feature discovery at the standard budget (K=3000): the
  finite-population width there is ~0.94, still above the dominant
  attribution (~0.26).  Sign certification is demonstrated at K ≥ 3×10^4
  (validated vs exact) and the *nominal* 1−δ certificate requires
  K ≈ 2×10^5 for M=11 (validated end-to-end on M=3).
- Statistical regime semantics at N=1–2 per regime.
- Official OddSHAP/ShaplEIG parity (method-style baselines only).
