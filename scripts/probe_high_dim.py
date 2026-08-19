#!/usr/bin/env python
"""Sub-enumerative high-dimensional certification probe (audit Q1).

The M=11 nominal-certification runs were POST-ENUMERATIVE: they cached the
entire 2^11 = 2048 power set (unique coalition evals = 2048), so the
certificate closed only after full enumeration.  This probe answers the
audit's decisive question at a dimension where 2^M is infeasible:

  * M = 30 (2^30 ~ 1.07e9) -- exact enumeration impossible;
  * sparse synthetic game with CLOSED-FORM exact Shapley values
    (additive + sparse pairwise interactions), so RMSE and sign
    validation remain checkable;
  * GAS-BayesSHAP, finite-population range, at increasing budgets;
  * reports, per run:
      - unique coalition evals (cache misses) vs 2^M  (sub-enumerative?);
      - attempted Stage-2 draws (budget accounting);
      - status / converged / certificate_at_nominal_level / realised
        level / delta1 (coupon budget);
      - sign-certified count vs the analytic exact, signs validated;
      - RMSE vs analytic, widths, evals, wall time.

Honest expectation (to be confirmed by the run): unique evals << 2^30
(sub-enumerative point estimation + empirical-range sign certification of
the driver features), while certificate_at_nominal_level stays False
because the coupon-collector budget over C(M-1,s) pairs at M=30 cannot
close at feasible K -- the rigorous nominal certificate remains bounded by
near-enumeration (the characterised cost frontier).

Usage:
    python scripts/probe_high_dim.py --M 30 --budgets 20000,50000,100000
Outputs -> results/paper_experiments/high_dim_M{M}_{summary,instances}.csv
           + main_results/paper_high_dim_M{M}_{summary,instances}.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle

OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def make_sparse_game(M: int, seed: int = 1301):
    """Sparse additive + pairwise game with closed-form Shapley.

    v(S) = 0.5 + sum_{i in S} w_i + sum_{i<j, i,j in S} w_ij,
    with 6 driver features (|w| larger, signed) and 15 sparse pairwise
    interactions among them.  Exact Shapley: phi_i = w_i + (1/2) sum_j w_ij.
    v(S) in [0.05, 0.95] so output_bounds=(0.0, 1.0) is safe.
    """
    rng = np.random.RandomState(seed)
    w = np.full(M, 0.006)
    drivers = rng.choice(M, size=6, replace=False)
    signs = np.array([1, 1, -1, -1, 1, -1])
    w[drivers] = 0.035 * signs
    pairs = {}
    for a in range(5):
        for b in range(a + 1, 6):
            i, j = drivers[a], drivers[b]
            pairs[frozenset((i, j))] = 0.004 * (1.0 if (a + b) % 2 == 0 else -1.0)
    # exact Shapley
    phi_exact = w.copy()
    for (i, j), wij in pairs.items():
        phi_exact[list({i, j})] += 0.5 * wij
    max_v = 0.5 + float(np.abs(w).sum()) + float(sum(abs(v) for v in pairs.values()))
    assert max_v <= 0.95, f"max_v {max_v} > 0.95"

    def g_c(x):
        x = np.asarray(x, dtype=float)
        val = 0.5 + float(np.dot(x, w))
        for (i, j), wij in pairs.items():
            i, j = list({i, j})
            val += wij * x[i] * x[j]
        return float(val)

    return g_c, phi_exact, w


def make_parity_game(M: int, seed: int = 1301):
    """XOR game over a 4-feature subset (high-order interaction, no low-order
    structure): v(S) = 0.5 + 0.25 * (-1)^{|S \cap D|} with |D|=4.  Exact
    Shapley is zero for every feature (symmetric), so any nonzero estimated
    attribution is pure error -- a deliberately unfavourable game for the
    smooth GP control variate.  Used to test the residual estimator and the
    spec-range interval under misspecification."""
    rng = np.random.RandomState(seed)
    D = rng.choice(M, size=4, replace=False)
    phi_exact = np.zeros(M)

    def g_c(x):
        x = np.asarray(x, dtype=float)
        s = int(np.sum(x[D]))
        return 0.5 + 0.25 * (-1.0) ** s

    return g_c, phi_exact, D


def run_config(M, K, eps, mode, seed=1301, game="sparse"):
    if game == "parity":
        g_c, phi_exact, _ = make_parity_game(M, seed)
    else:
        g_c, phi_exact, w = make_sparse_game(M, seed)
    x0 = np.ones(M)
    bg = np.zeros((1, M))   # all-zero rows are identical, so B=1 defines
                            # v(S) = f(1_S) exactly (no redundant forwards)

    t0 = time.time()
    eng = GASBayesSHAP(g_c, bg, output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(seed),
                       config={"checkpoint_enabled": False, "cache_enabled": True,
                               "persist_cache": False, "log_level": "NONE",
                               "range_mode": mode})
    r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=K,
                    n_pilot=3, n_active_steps=10)
    dt = time.time() - t0

    W = np.asarray(r["certified_projected_widths"])
    phi = np.asarray(r["shapley_values"])
    sc = np.abs(phi) > W
    sc_ok = int(np.all(np.sign(phi[sc]) == np.sign(phi_exact[sc]))) if sc.any() else 1
    margin = float(np.min(np.abs(phi_exact[sc]) - W[sc])) if sc.any() else float("nan")
    unique = int(r["num_coalition_evals_this_call"])
    attempted = int(r.get("extra", {}).get("stage2_attempted_total", 0)) \
        if isinstance(r.get("extra"), dict) else int(r.get("stage2_attempted_total", 0))
    if attempted == 0:
        attempted = int(r["num_coalition_evals_this_call"])

    return {
        "M": M, "K": K, "range_mode": mode,
        "status": r["status"], "converged": bool(r["converged"]),
        "certificate_is_rigorous": bool(r.get("certificate_is_rigorous", False)),
        "certificate_at_nominal_level": bool(r.get("certificate_at_nominal_level", False)),
        "realised_coverage_level": r.get("finite_population_coverage_level"),
        "delta1_coupon": r.get("finite_population_delta1"),
        "unique_coalition_evals": unique,
        "attempted_stage2_draws": attempted,
        "pow2_M": int(2 ** M),
        "unique_vs_2M_ratio": float(unique) / float(2 ** M),
        "rmse_vs_exact": float(np.sqrt(np.mean((phi - phi_exact) ** 2))),
        "sim_cov": float(np.all(np.abs(phi - phi_exact) <= W)),
        "n_sign_certified": int(sc.sum()),
        "signs_match_exact": sc_ok,
        "min_certified_margin": margin,
        "mean_width": float(np.mean(W)),
        "max_width": float(np.max(W)),
        "model_evals": int(r["num_model_evals_this_call"]),
        "elapsed_s": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--budgets", default="20000,50000,100000")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--mode", default="finite_population",
                    choices=["finite_population", "spec", "empirical_max"])
    ap.add_argument("--game", default="sparse", choices=["sparse", "parity"])
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    rows = []
    for K in budgets:
        row = run_config(args.M, K, args.eps, args.mode, game=args.game)
        row["game"] = args.game
        rows.append(row)
        print(f"  M={row['M']} K={K:>7}: {row['status']:16s} "
              f"conv={row['converged']} at_nom={row['certificate_at_nominal_level']} "
              f"level={row['realised_coverage_level']} "
              f"unique={row['unique_coalition_evals']} "
              f"unique/2^M={row['unique_vs_2M_ratio']:.2e} "
              f"sign_cert={row['n_sign_certified']} signs_ok={row['signs_match_exact']} "
              f"rmse={row['rmse_vs_exact']:.2e} W={row['mean_width']:.4f} ({row['elapsed_s']:.0f}s)")

    import pandas as pd
    d = pd.DataFrame(rows)
    tag = f"high_dim_M{args.M}"
    # per-mode files so fp and spec runs never overwrite each other; MERGE
    # with any existing rows so running notebook sections in any order
    # accumulates the full grid (fix: each section used to overwrite).
    fname = f"{tag}_{args.game}_{args.mode}_summary.csv"
    oldf = OUT / fname
    if oldf.exists():
        d = pd.concat([pd.read_csv(oldf), d], ignore_index=True)
        d = d.drop_duplicates(subset=["game", "K"], keep="last").sort_values("K")
    d.to_csv(oldf, index=False)
    import shutil
    shutil.copy2(oldf, MAIN / f"paper_{fname}")

    # combined file (concatenate existing per-mode files)
    import glob
    parts = [pd.read_csv(f) for f in sorted(OUT.glob(f"{tag}_*_summary.csv"))]
    if parts:
        comb = pd.concat(parts, ignore_index=True)
        gk = [c for c in ("game", "range_mode", "K") if c in comb.columns]
        comb = comb.drop_duplicates(subset=gk, keep="last")
        comb = comb.sort_values(["range_mode", "K"]).reset_index(drop=True)
        comb.to_csv(OUT / f"{tag}_summary.csv", index=False)
        shutil.copy2(OUT / f"{tag}_summary.csv", MAIN / f"paper_{tag}_summary.csv")

    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_{fname} + paper_{tag}_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
