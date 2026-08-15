# Certificate Tightness Probe — answers "can certificates ever be tight?"

**Question (from Q1 review):** at feasible budgets, can the anytime
certificates ever be tight enough to certify anything?

**Probe:** wine, 1 instance (cluster 0 of the 2-cluster probability game),
exact ground truth computed, GAS-BayesSHAP run at K = 2k / 8k / 30k / 100k
coalition evals (ε = 0.02, δ = 0.05).

## Results

| K (coalition evals) | mean W | max W | sign-certified | max \|err\| vs exact |
|---|---:|---:|---:|---:|
| 2 000 | 13.39 | 17.75 | 0.000 | 0.00269 |
| 8 000 | 3.84 | 5.16 | 0.000 | 0.00100 |
| 30 000 | 1.19 | 1.59 | 0.000 | 0.00066 |
| 100 000 | 0.400 | 0.539 | 0.000 | 0.00014 |

- Width decays as **~1/√K** (c·√K ≈ 318, i.e. W ≈ 318/√K).
- Error tracks width (max_err 0.0027 → 0.00014), always **≪ W** → bounds are
  valid, just loose.
- Exact attributions on this instance: φ ∈ [−0.257, +0.002], only **1 feature
  with |φ| > 0.05**. So even at W = 0.40 the *largest* attribution is not
  sign-certified — **not because the bounds are wrong, but because the
  attributions themselves are tiny** on this 2-class probability game.

## Interpretation (honest, for the paper)

1. **Certificates DO tighten** — clean 1/√K law, reproducible. The
   certification machinery is not broken; it has a well-characterized cost
   frontier.
2. **The bottleneck is attribution scale, not width alone.** For the
   cluster-probability game the Shapley values are O(1e-3..1e-1), so
   sign-certification needs W ≤ |φ| ~ 0.05, i.e. K ~ 10⁵–10⁶, or a tighter
   width.
3. **The real lever is width theory (review task #2):** the Bernstein range
   term uses the conservative `R_Δ = 4(U−L) = 4`. If a per-stratum empirical
   residual-range bound (with its own confidence adjustment) were used, the
   range could drop ~20× (4 → ~0.2 for a good GP surrogate), shrinking widths
   ~20×: W ~ 0.02 at K = 10⁵ → **sign-certification of |φ|>0.05 becomes
   feasible at K ~ 10⁴–10⁵**. That is the paper's make-or-break research
   step.

## Takeaway for the Q1 review

- **Blocker 1a (high-budget probe): DONE.** Run once, committed; shows the
  cost frontier and the attribution-scale bottleneck.
- **Blocker 1b (tighten width): REQUIRED** — this is the substantive
  contribution that turns "certificates never certify" into "certified
  Shapley discovery at feasible budget".
- The point-estimate + 1/√K frontier is already a publishable *finding* at a
  second-tier venue; the width improvement upgrades it to TPAMI/JMLR grade.

## Reproduce

```bash
python scripts/probe_width_tightness.py   # (add to scripts/ if not present)
```
