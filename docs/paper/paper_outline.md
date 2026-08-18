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
     finite-population mode: width 2.38 vs 12.31 spec (5.2x), level 0.959,
     at_nominal_level fraction 0.782.
   - RQ2b finite-pop N=50 reruns (spec-vs-fp at standard budget): width
     ~0.97/1.12 vs 9.15/9.43, same RMSE/coverage, sign_cert 0 at K=3000
     (fp width still > |phi| ~ 0.26) — honest.
   - RQ3 regime semantics (Spearman driver correlation; named regimes) +
     Tier-B N=20 (alignment-fixed protocol): macro RMSE 0.00195, sim cov 1.0.
   - RQ5 matched-budget curves (KernelSHAP overtakes at K>=512; GAS wins
     low-budget <=256: 2.5x at K=128 air); instrumented actual call counts.
   - Ablation N=20 (K=1000): uniform 0.039 / neyman 0.007 / gp 0.071 /
     full 0.0030; tier4 width 24.7, sim cov 1.0.
   - Baselines: OddSHAP-style log-odds (exact), ShaplEIG-style GP
     quadrature (both non-certified, method-style), KernelSHAP,
     SamplingSHAP; matched-budget comparison CSV.
   - Frontier: fp probe K=2k..200k — width ~6.5x tighter than spec;
     sign-cert of the dominant feature (|phi|=0.257) at K>=30k, signs
     validated vs exact; **nominal 1-delta CERTIFIED at K=200k** (converged,
     realised level 0.962, delta1=0.013, margin 0.207) — Corollary E
     confirmed on real data.
   - **Multi-instance nominal certification (N=3 per dataset, K=200k):**
     5/6 instances at nominal 1-delta (wine 3/3 levels 0.962-0.974, air
     2/3 levels 0.952-0.968); **13 sign-certified features total, every
     sign validated vs exact** (RMSE ~1e-4, sim cov 1.0).  Air inst 0
     honest outlier: coupon open (delta1=4.42), at_nominal=False, level 0,
     though width converged — convergence vs nominality are distinct axes,
     both reported (paper_nominal_certification_{wine,air}.csv).
   - Adversarial stress test (R=200): M=3 coupon closes -> nominal reached
     on 99.5% of trials; M=6 coupon open -> flag False, level 0.0 honest.
   - Tight-eps calibration (0.05..0.5, R=200, both modes): coverage 1.0.

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
