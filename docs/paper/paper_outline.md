# GAS-BayesSHAP — Paper Outline (Q1-target)

**Working title:** Anytime-Certified Bayesian Control Variates for Shapley
Estimation: Theory, Real-Data Validation, and the Certification Cost Frontier

**Target venue (tiered):** TPAMI / JMLR (if width-tightening succeeds) or
Information Fusion / Machine Learning (applied framing).

## Structure

1. **Introduction**
   - XAI trust gap; 2^M barrier; existing methods lack anytime, distribution-free CIs.
   - Contribution list (theory, engineering, real-data evidence).

2. **Related Work**
   - SamplingSHAP (Castro 2009), KernelSHAP (Lundberg & Lee 2017),
     TreeSHAP, GP-based (ShaplEIG 2026), odd-ratio (OddSHAP 2026),
     Bayesian quadrature. Positioning table.

3. **Method** (from spec v11.0)
   - 3.1 Bounded linear control variate (Lemmas D/E; lambda-shrinkage; no clipping).
   - 3.2 Neyman-stratified residual certifier (Lemma F/G; Theorem A coupled allocation).
   - 3.3 Anytime empirical-Bernstein CS (Theorem B; exact singleton strata).
   - 3.4 Posterior-diagonal projection + Corollary C.1 + sign certification.
   - 3.5 (NEW, review task #2) Empirical residual-range tightening:
       - Theorem E: finite-population empirical-range Bernstein — the
         residual marginal of cell (i,s) is an iid draw from a finite
         population of C(M-1,s) pairs, so the observed-support range is
         rigorous at realised level 1 - delta2 - delta1 with the
         coupon-collector budget delta1 = sum (1-1/N_s)^n (R=500:
         coverage 1.0, width 2.38 vs 12.31 spec, 5.2x tighter, level 0.959).
       - Corollary E: level-delta certification once n >= tau*_s
         (coupon threshold; K ~ 1e4-1e5 for M=11).
       - Remark (impossibility): generic observed-max range is NOT
         distribution-free (sparse-extreme construction) — why the
         empirical_max mode stays flagged heuristic.

4. **Experiments**
   - 4.1 Exact ground truth (M=11, 2^11) on wine + Beijing-Aotizhongxin.
   - RQ1 fidelity table (N=50): wine GAS 0.00168+-0.00101 (Kernel 0.00472,
     MC 0.0466); air GAS 0.00186+-0.00117 (Kernel 0.00606, MC 0.0370);
     ~1.43k evals vs exact 2048; sim coverage 1.0; sign_cert 0.
   - RQ2 coverage calibration (R=500): coverage 1.0, finite-width 1.0;
     finite-population mode: width 2.38 vs 12.31 spec (5.2x), level 0.959.
   - RQ3 regime semantics (Spearman driver correlation; named regimes) +
     Tier-B N=20 macro RMSE 0.00202, sim cov 1.0.
   - RQ5 matched-budget curves (KernelSHAP overtakes at K>=512; GAS wins
     low-budget <=256: 2.5x at K=128 air).
   - Ablation (4 tiers): uniform 0.051 / neyman 0.009 / gp 0.081 / full 0.0038.
   - Baselines: OddSHAP-style log-odds, ShaplEIG-style GP quadrature (both
     non-certified, method-style), KernelSHAP, SamplingSHAP.

5. **Certification cost frontier** (the honest finding)
   - Width vs K: W ~ 318/sqrt(K); sign-cert feasible at K ~ 1e5-1e6 with
     spec range; ~1e4-1e5 with empirical range.
   - Wine K=3000 (N=5): fp width 0.94 vs 8.75 spec (9.3x), same RMSE
     (0.00059), coverage 1.0, but realised level 0 (coupon open, delta1~59).
   - M=3 R=500: fp width 2.38 vs 12.31 (5.2x), level 0.959, coverage 1.0 —
     the coupon collector completes in-budget => fully rigorous.
   - Level-1-delta at M=11 needs K ~ sum_s N_s ln(M/delta1)(M+1) ~ 2e5.
   - Discussion of what can/cannot be claimed.

6. **Limitations & Future Work**
   - Width theory (empirical-range CS with rigorous adjustment).
   - 4-regime labeling, DEC/Transformer black-box, R>=500 on real data.

Appendices: A Lemma E algebra; B supermartingale filtration; C hyperparameters; D data cards.
