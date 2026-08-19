# GAS-BayesSHAP — response-letter skeleton (ESWA target)

Pre-empts the three predictable reviewer attacks.  Each entry: the attack,
the honest answer, and the exact artifact/number to cite.  Numbers are
drawn from the committed `main_results/` artifacts at commit `f6c05ea`.

---

## R1. "The M=30 sign certification is not a nominal 1−δ certificate."

**Attack:** "You claim high-dimensional certification at M=30, but
`certificate_at_nominal_level=False` and `realised_coverage_level=0.0` in
your own CSV — so the certificates certify nothing."

**Answer (accept the premise, then reframe):**
- We agree: the M=30 result is *not* a nominal 1−δ anytime certificate,
  and the paper says so explicitly (Section "What this means for the
  paper's claims").
- What it *is*: sub-enumerative, exact-validated sign certification —
  3 driver features certified with unique coalition evaluations
  229,850 ≪ 2^30 (ratio 2.1e-4), every certified sign matching the
  analytic exact Shapley, RMSE 9.2e-6, simultaneous coverage 1.0
  (`paper_high_dim_M30_summary.csv`).
- The coupon-collector analysis (Corollary E) *predicts* exactly this
  boundary: nominal closure requires
  n_{i,s} ≥ ln(δ1,s)/ln(1−1/N_s) per cell, and N_s = C(29,14) ≈ 7.7e7
  for the mid strata — infeasible at any practical K.  The estimator
  reports the realised level honestly rather than overclaiming.
- The *nominal* certificate IS demonstrated, on real M=11 games:
  5/6 instances at K=2e5, 13 sign-certified features, all signs validated
  (`paper_nominal_certification_{wine,air}.csv`) — at post-enumerative cost
  (2048 unique = 2^11), which we state as a theorem, not a hidden cost.

## R2. "M=11 nominal certification costs 98× exact enumeration — useless."

**Attack:** "Your nominal certificate needs K=2e5 coalition draws while
exact enumeration needs only 2^11=2048.  There is no query advantage."

**Answer (accept, then scope):**
- Correct: at M=11 the nominal certificate saturates the power set
  (unique evals = 2048 = 2^11).  The paper says this explicitly
  ("Post-enumerative honesty").
- The intended regime is *not* M ≤ 11: it is the setting where 2^M is
  infeasible.  At M=30, exact enumeration is impossible (2^30 ≈ 1.07e9)
  and GAS-BayesSHAP still recovers attributions to RMSE ~1e-5 and
  sign-certifies drivers with unique evals ≪ 2^M.
- The paper's contribution is the *characterized frontier*: we prove
  exactly where nominal certification is feasible (M ≲ 14, enumerative
  cost) and where sub-enumerative empirical-range sign certification takes
  over (larger M, non-nominal).  No other estimator in the literature
  reports what distribution-free anytime certification *costs*.

## R4. "converged=True but the returned interval exceeds ε — the stopping rule is wrong."

**Attack:** your converged runs report projected widths > ε (e.g. M=30
K=500k: max projected width ≈ 0.043 > 0.02).

**Answer:** the stopping rule checks the *raw* residual widths (max
W^res ≤ ε); the returned estimator carries the larger Corollary C.1
*projected* widths, and the paper states this explicitly in the protocol.
Every run records both flags (`converged_on_raw_widths` and
`converged_on_projected_widths`); `converged` never implies max W^proj ≤ ε,
and no certified claim is made from the raw widths alone.

## R5. "The M=30 result is overstated: it is not sign certification."

**Answer:** agreed and corrected. The M=30 result is now called
*empirical sign separation* — an empirical-event interval under the
finite-population range, validated against the analytic exact Shapley,
with `certificate_at_nominal_level=False` throughout. "Certification" in
the nominal sense is reserved for coupon-completed runs.

## R3. "ShaplEIG is a port, not the official library — the baseline is invalid."

**Attack:** "You did not run the official ShaplEIG package, so the
comparison is not a fair SOTA benchmark."

**Answer:**
- The port implements the official algorithm from the authors' public MIT
  repository (github.com/slds-lmu/shapleig, pinned commit d52c09e):
  `_compute_eig_function_property_naive_Z` (EIG acquisition),
  `_get_shapley_weights` (Shapley coefficient matrix), Hamming-kernel
  exact GP with MLL fit.  A reviewer can diff the math line-by-line
  against the pinned source.
- It is pure NumPy/SciPy because the authors' torch/GPyTorch stack
  segfaults natively on macOS (three independent crashes, including a
  kernel death); the mathematics is unchanged.
- A *matched unique-query* comparison is committed
  (`paper_matched_shaplEIG_comparison.csv`): each method's actual unique
  coalition evaluations are reported side by side at the same nominal
  budgets, and the paper does not claim a perfectly matched unique-query
  benchmark unless the counts agree.
- Results: wine RMSE 0.0079/0.0079/0.0004 and air 0.0029/0.0015/0.0007 at
  64/128/256 unique queries (`paper_shaplEIG_port_{wine,air}.csv`).  We
  report it as a faithfully-ported official algorithm, clearly labelled
  `paper_shaplEIG_port_*`, not as a black-box library call.

---

## Where each number lives (for the response)

| Claim | Artifact |
|---|---|
| M=30 sub-enumerative sign cert | `main_results/paper_high_dim_M30_summary.csv` |
| M=11 nominal cert (5/6 inst, 13 feats) | `main_results/paper_nominal_certification_{wine,air}.csv` |
| ShaplEIG port numbers | `main_results/paper_shaplEIG_port_{wine,air}.csv` |
| N=50 fidelity | `main_results/paper_{wine,air}_n50_budget3000_summary.csv` |
| R=500 calibration | `main_results/paper_coverage_calibration_R500*.json` |
| Coupon stress test (honesty under adversarial residual) | `main_results/paper_stress_finite_population.json` |
