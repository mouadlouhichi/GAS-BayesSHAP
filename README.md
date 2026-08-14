# GAS-BayesSHAP

**Gaussian-Adaptive Stratified Bayesian Shapley Estimation (v11.0)**

Bounded-linear Bayesian control variates + Neyman-stratified anytime
empirical-Bernstein certification for distribution-free Shapley estimation
with certified confidence widths.

> Authoritative specification: [`specs/GAS_BayesSHAP_Implementation_Spec (4).md`](specs/GAS_BayesSHAP_Implementation_Spec%20(4).md)

---

## What it does

GAS-BayesSHAP estimates Shapley values of a black-box set function
`v(S) = E[f(x_S, Z_{¬S})]` over `M` features using two cooperating modules:

1. **Module A — Active bounded-linear GP control variate**
   Learns a GP surrogate on actively acquired coalitions and shrinks it
   linearly so that `m_b(S) ∈ [L, U]` for **all** `2^M` coalitions without
   nonlinear clipping. Its attribution `φ(m_b) = λ·K_{φ,D}·α` is computed in
   closed form (zero sampling variance) via the exact analytical
   hypergeometric cross-covariance (Lemma D, O(M²)) and the exact prior
   Shapley covariance (Lemma E, O(M³)).

2. **Module B — Neyman-stratified residual certifier**
   Measures the residual game `r_D(S) = v(S) − m_b(S)` with add-one /
   remove-one marginals (Lemma F), identifies the extreme singleton strata
   exactly (Lemma G), allocates samples with the coupled adjacent-stratum
   Neyman program (Theorem A), and certifies the unified estimator with an
   **anytime empirical-Bernstein confidence sequence** (Theorem B).

3. **Posterior-diagonal efficiency projection (Theorem C / Corollary C.1)**
   Projects the raw estimator onto `Σφ = v(N) − v(∅)` weighted by posterior
   variance and returns certified post-projection widths plus
   sign-certified feature importance (Definition 1).

## Key properties

- `φ̂_i^raw = φ_i(m_b) + φ̂_i(r_D)` — the core decoupled estimator.
- Anytime stopping: `τ = inf{n : max_i W_i^res(n_i) ≤ ε}` with
  `P(∃n, ∃i: |φ̂_i^raw − φ_i| > W_i^res) ≤ δ`.
- Exact query accounting: `num_coalition_evals` (true oracle calls) and
  `num_model_evals` (forward passes; B per hybrid coalition, 0 for ∅, 1 for N).
- Strict Stage-2 budget guard: `max_budget` bounds individual coalition
  evaluations of the adaptive loop; it is never exceeded.
- Reproducible: seeded RNG, frozen background, full environment/git/RNG
  provenance, deterministic repeated runs.
- Resumable: atomic checkpoints at every stage; `--resume` restores GP state,
  residual observations, Neyman state, RNG state, query counters and
  iteration — without rerunning Stage 1 or repeating cached oracle queries.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .[dev]
```

Requires Python ≥ 3.9, NumPy, SciPy, PyYAML (scikit-learn for the silhouette
game; jupyter stack for the notebook).

## Quick start

```bash
# mathematical validation (Lemma D / Lemma E vs brute force)
python scripts/validate_math.py

# exact reference (small M)
python scripts/run_exact.py --M 5

# main run
python scripts/run_bayesshap.py --config configs/default.yaml --M 5 --epsilon 0.6

# resume
python scripts/run_bayesshap.py --config configs/default.yaml --M 5 --resume

# status dashboard
python scripts/run_bayesshap.py --run-id <RUN_ID> --status

# benchmark: exact vs Monte-Carlo vs GP-only vs GAS-BayesSHAP
python scripts/benchmark.py --M 5 --budget 200

# everything
python scripts/run_all.py --M 5

# notebook (17 sections + status dashboard)
jupyter nbconvert --to notebook --execute notebooks/run_all.ipynb
```

## Tests

```bash
python -m pytest tests/ -q          # 173 tests across 6 suites
```

| Suite | Coverage |
|---|---|
| `tests/mathematical` | Lemma D (M=1 sanity + M=2..6), Lemma E (M=2..6 incl. M=2 off-diagonal), Shapley weights, brute-force self-consistency |
| `tests/numerical` | Sherman–Morrison vs direct inversion, near-duplicate Schur fallback, stable combinatorics, GP posterior |
| `tests/statistical` | Tiers 3–5, 9, 10: null-player containment, coverage calibration (30 trials), inflation tightness, M=2 exactness, additive recovery, anytime widths, missing strata + completeness flags |
| `tests/protocol` | Query accounting (this-call vs run-total vs end-to-end), budget guard, cache correctness + compatibility + model-identity hardening, oracle determinism + output-bounds enforcement, deterministic repeated runs, width vector, config/game presets, structured logging, evidence-based compliance audit |
| `tests/integration` | Full 10-tier suite, parity with the spec's inline reference, pipeline/schema/status, domain games (membership, contrastive, archetype, silhouette + degenerate-label guard, group-lag M=11 exact) |
| `tests/resume` | Checkpoint/resume equivalence, **cumulative Stage-2 budget across resume**, checkpoint payload-integrity verification + fallback, corrupted/incompatible checkpoint rejection, failure injection (crash during GP/Stage-2/adaptive, NaN) |

## Audit-driven guarantees (beyond the v11.0 spec)

* **Cumulative resume budget** — `max_budget` is a *run-level* Stage-2
  allowance; a resumed run deducts pre-crash work instead of restarting with a
  fresh budget (`stage2_attempted_total` persisted in every checkpoint).
* **Checkpoint integrity** — every checkpoint records payload + NPZ/JSON
  SHA-256 hashes in `checkpoint_manifest.json`; `load()` verifies them and
  rejects corruption/tampering (`CheckpointIntegrityError`), falling back to
  the previous valid checkpoint.
* **Honest compliance audit** — `spec_compliance.json` statuses are
  `IMPLEMENTED / TESTED / VALIDATED / MISSING`; the math items execute the
  brute-force validation at audit time (evidence strings recorded).  The
  report additionally consumes real test artifacts when provided:
  `compliance_from_pytest(passed, failed, commit)` embeds a pytest summary
  and the current commit into the evidence, and
  `write_results(pytest_evidence={...})` persists it.  It is never a bare
  self-declaration.
* **Output-bounds contract** — oracle outputs are checked against the declared
  `[L, U]` on every evaluation (incl. non-finite rejection) and raise
  `OutputBoundViolation`; a certificate can never be labeled rigorous for a
  model that violates its own range.
* **Model identity** — without an explicit `model_tag`, the oracle identity
  includes a source-derived artifact hash (module + qualname + source), so
  distinct lambdas (all named `<lambda>`) are never cache/checkpoint
  compatible; pass `model_artifact_hash` for fitted-parameter digests.
* **Complete/partial estimates** — results report
  `point_estimate_complete` and `missing_cells_by_feature`; features with
  unobserved strata are flagged, never silently presented as complete.
* **Query accounting** — results report per-call *and* run-total
  coalition/model evaluations plus `baseline_model_evals` (the `B` E_base
  passes excluded from per-call deltas) and `num_model_evals_end_to_end`.

## Results

Every run writes `results/runs/<run_id>/`:

```
manifest.json  config.yaml  provenance.json  environment.json
spec_compliance.json  summary.json  summary.md  reproducibility_report.md
logs/  checkpoints/  oracle/  gp/  residual/  neyman/  certification/
benchmarks/  tables/  figures/  cache_manifest.json
```

## Repository layout

```
configs/          YAML configuration (default / games / certification / experiments)
gas_bayesshap/    package: core, game, kernels, gp, residual, certification,
                  acquisition, numerics, checkpointing, logging, cache,
                  benchmarking, utils, reference (spec parity oracle)
scripts/          validate_math, run_exact, run_bayesshap, benchmark, run_all
tests/            mathematical, numerical, statistical, protocol, integration, resume
notebooks/        run_all.ipynb (executed)
specs/            authoritative v11.0 specs + analysis
```

## Compliance

`spec_compliance.json` (written into every results dir) audits the
architecture, mathematics (Lemmas D/E/F/G, Theorems A/B/C, Corollary C.1),
numerics, certification and engineering requirements of the specification
(spec section 53).  The verification tests cover the complete 10-tier suite
(spec section 6) plus the additional required tests (spec section 44).

## Domain-model pipeline status

The repository implements the **complete algorithm engine** (bounded-linear
control variate, Lemmas D–G, Theorems A–C, Corollary C.1, certification,
checkpointing, logging, caching) plus the **domain-game wrappers**
(membership, contrastive, archetype, silhouette, group-lag).  However, the
**CLI scripts currently run synthetic stand-in models** (logistic/linear toy
games) — the real Wine-quality / Beijing air-quality clustering pipelines
(PCA → K-Means → LightGBM surrogate, DEC + temporal transformer) and their
data loaders are **not bundled**.  The CLI prints a notice whenever
`--dataset wine|beijing_*` is used; script results are for engine validation,
not for the paper's experimental claims.

## Caveats / engineering decisions (documented)

* `h_lb`/`h_ub` are **safe kernel-induced bounds** (conservative, from the
  kernel range `[σ₀²ρ^M, σ₀²]` with α sign-splitting), not exact extrema —
  they guarantee `m_b(S) ∈ [L, U]` for every representable coalition.
* The coupled Neyman allocation is **approximately optimal**: the convex
  program optimizes expected sample counts; realized counts are random.
* Heuristic-bounds runs (`output_bounds: null`) are always flagged
  (`range_bound_is_heuristic: true`) and never claim rigor.
* The CLI scripts use **synthetic stand-in models** (see "Domain-model
  pipeline status"); engine results are for validation, not paper experiments.

## Caveats / engineering decisions (documented)

- **Cache semantics**: coalition-cache hits return stored values with `+0`
  query cost; the Stage-2 budget guard accounts *attempted* evaluations so
  the loop always terminates and matches the no-cache reference round-for-round.
- **Heuristic bounds**: without explicit `output_bounds`, the Remark-2.2
  fallback bounds are used and flagged
  (`range_bound_is_heuristic=True`, never rigorous).
- **Near-duplicate guard**: the spec's `schur < η²` rejection rule is kept
  verbatim (parity with the reference); exact duplicates may still pass at
  `schur ≈ 2η²`, exactly as in the reference engine.
- **Resume of a completed run** returns the stored final result (a
  `final_stage` checkpoint exists); resuming after a crash continues from the
  smallest valid checkpoint.
