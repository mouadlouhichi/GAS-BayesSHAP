#!/usr/bin/env python
"""Matched unique-query comparison: GAS-BayesSHAP vs the official ShaplEIG port.

The earlier summary compared ShaplEIG at 64/128/256 UNIQUE queries against
GAS at nominal K=128/256/512 (~393/480/565 actual unique evals) — not
actually matched.  This script runs GAS at nominal budgets K in {64,128,256}
and records its ACTUAL unique coalition evaluations, pairing each GAS run
with the ShaplEIG run at the same nominal budget, and reports both actual
unique-query counts side by side.  Nothing is claimed as 'matched' unless
the unique counts are close; the CSV reports the truth.

Usage:
    python scripts/run_matched_shaplEIG.py --n 2 --budgets 64,128,256
Outputs -> results/paper_experiments/matched_shaplEIG_comparison.csv
           + main_results/paper_matched_shaplEIG_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse

from run_paper_experiments import (
    load_wine, load_air_station, build_surrogate, make_proba_fn,
    WINE_FEATURES, FEATURES as AIR_FEATURES,
)
from run_official_shaplEIG import run_official_shaplEIG, SHAPLEIG_SOURCE

OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def run_gas(fn, bg, x0, M, K, seed):
    eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(seed),
                       config={"checkpoint_enabled": False, "cache_enabled": True,
                               "persist_cache": False, "log_level": "NONE",
                               "range_mode": "spec"})
    r = eng.explain(x0, epsilon=0.05, delta=0.05, max_budget=K,
                    n_pilot=3, n_active_steps=10)
    return (np.asarray(r["shapley_values"]),
            int(r["num_coalition_evals_this_call"]),
            int(r.get("stage2_attempted_total", r["num_coalition_evals_this_call"])))


def run_one(ds, X, feats, nc, i, K, seed):
    m, X_te, _, _ = build_surrogate(X, nc, feats)
    cid = i % nc
    fn = make_proba_fn(m, feats, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=seed + i).values

    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                             model_tag=f"matched-{ds}-{i}-exact")
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    def game_fn(S_mask):
        o = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                            model_tag=f"matched-{ds}-{i}")
        return o.evaluate(x0, S_mask)

    # GAS at nominal K
    t0 = time.time()
    phi_g, ug, att = run_gas(fn, bg, x0, X.shape[1], K, seed + i)
    t_gas = time.time() - t0

    # ShaplEIG port at the same nominal budget (unique queries capped at K)
    t0 = time.time()
    phi_s, us, t_s = run_official_shaplEIG(game_fn, X.shape[1], K, seed=seed + i)
    t_s = time.time() - t0

    return {
        "dataset": ds, "instance": i, "nominal_K": K,
        "gas_rmse": rmse(phi_g, phi_exact),
        "shaplEIG_rmse": rmse(phi_s, phi_exact),
        "gas_unique_evals": ug,
        "shaplEIG_unique_queries": us,
        "gas_attempted_draws": att,
        "gas_wall_s": round(t_gas, 1),
        "shaplEIG_wall_s": round(t_s, 1),
        "matched_unique": bool(abs(ug - us) <= 0.35 * max(ug, us)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--budgets", default="64,128,256")
    ap.add_argument("--dataset", choices=["wine", "air", "both"], default="both")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in (["wine", "air"] if args.dataset == "both" else [args.dataset]):
        X, feats, nc = (load_wine(), WINE_FEATURES, 2) if ds == "wine" \
            else (load_air_station(n_clusters=4), AIR_FEATURES, 4)
        X = X[0]
        for i in range(args.n):
            for K in budgets:
                row = run_one(ds, X, feats, nc, i, K, seed=1301)
                rows.append(row)
                print(f"  {ds} inst {i} K={K}: gas_rmse={row['gas_rmse']:.5f} "
                      f"(unique={row['gas_unique_evals']}) vs "
                      f"shaplEIG_rmse={row['shaplEIG_rmse']:.5f} "
                      f"(unique={row['shaplEIG_unique_queries']}) "
                      f"matched={row['matched_unique']}")

    d = pd.DataFrame(rows)
    d["source"] = SHAPLEIG_SOURCE
    d.to_csv(OUT / "matched_shaplEIG_comparison.csv", index=False)
    import shutil
    shutil.copy2(OUT / "matched_shaplEIG_comparison.csv",
                 MAIN / "paper_matched_shaplEIG_comparison.csv")

    print(f"\nSummary (mean over instances):")
    g = d.groupby(["dataset", "nominal_K"]).agg(
        gas_rmse=("gas_rmse", "mean"),
        gas_unique=("gas_unique_evals", "mean"),
        shaplEIG_rmse=("shaplEIG_rmse", "mean"),
        shaplEIG_unique=("shaplEIG_unique_queries", "mean"),
    ).round(5)
    print(g.to_string())
    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_matched_shaplEIG_comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
