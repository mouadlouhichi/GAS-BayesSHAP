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
   - 3.5 (NEW, review task #2) Empirical residual-range tightening (opt-in;
       flags heuristic) — width reduction ~20x demonstrated.

4. **Experiments**
   - 4.1 Exact ground truth (M=11, 2^11) on wine + Beijing-Aotizhongxin.
   - RQ1 fidelity table (N=20): GAS RMSE 1.6-1.9e-3 @ ~1.4k evals vs exact 2048.
   - RQ2 coverage calibration (R=500): coverage 1.0, finite-width 1.0.
   - RQ3 regime semantics (Spearman driver correlation; named regimes).
   - RQ5 matched-budget curves (KernelSHAP overtakes at high K; GAS wins <=512).
   - Ablation (4 tiers): uniform 0.051 / neyman 0.009 / gp 0.081 / full 0.0038.
   - Baselines: OddSHAP-style log-odds, ShaplEIG-style GP quadrature, KernelSHAP, SamplingSHAP.

5. **Certification cost frontier** (the honest finding)
   - Width vs K: W ~ 318/sqrt(K); sign-cert feasible at K ~ 1e5-1e6 with
     spec range; ~1e4-1e5 with empirical range.
   - Discussion of what can/cannot be claimed.

6. **Limitations & Future Work**
   - Width theory (empirical-range CS with rigorous adjustment).
   - 4-regime labeling, DEC/Transformer black-box, R>=500 on real data.

Appendices: A Lemma E algebra; B supermartingale filtration; C hyperparameters; D data cards.
