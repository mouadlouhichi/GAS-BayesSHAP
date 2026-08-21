#!/usr/bin/env python
"""Unique-query-capped GAS vs ShaplEIG port comparison (audit item 2).

The earlier matched comparison showed GAS uses 4-7x more UNIQUE evals than
ShaplEIG at equal nominal budgets (Stage-1 GP design + exact singleton
init), so the methods were NOT unique-matched.  This script closes that:

  * GAS is hard-capped at U unique coalition evaluations by running with
    the oracle cache DISABLED (every attempted draw is a unique
    evaluation) and budgeting Stage-2 as (U - Stage-1 evals), so the
    Stage-1 overhead is INSIDE the cap: total unique evals ~= U.
  * Measured fixed Stage-1+init+pilot cost at M=11 is ~371 unique evals
    (active-GP candidate evaluations dominate), so caps below ~400 cannot
    be honoured -- `cap_honored` is reported per row and such rows are
    excluded from the summary.  Meaningful sub-enumerative caps at M=11
    are therefore {512, 1024} (both << 2^11 = 2048).
  * The ShaplEIG port uses exactly U unique queries.
  * Both are evaluated against the same exact ground truth; the CSV
    reports the ACTUAL unique evals of each, so the comparison is
    genuinely unique-query-matched when both ~= U.

Usage:
    python scripts/run_unique_capped_shaplEIG.py --n 10 --caps 64,128,256
Outputs -> results/paper_experiments/unique_capped_shaplEIG.csv
           + main_results/paper_unique_capped_shaplEIG.csv
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

GAS_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False,
              "persist_cache": False, "log_level": "NONE",
              "range_mode": "spec"}


def run_gas_capped(fn, bg, x0, M, cap, seed):
    """GAS with a hard cap on UNIQUE coalition evals (cache disabled).

    Stage-1 evals are measured first on a throwaway engine (deterministic
    design), then Stage-2 is budgeted to cap - stage1 so the TOTAL unique
    evals of the actual run stay within the cap (Stage-1 overhead inside
    the cap, as the audit requires).
    """
    eng0 = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                        rng=np.random.RandomState(seed), config=dict(GAS_CONFIG))
    s1 = eng0.explain_stage1_only(x0, n_active_steps=10)
    stage1_evals = int(s1["num_coalition_evals"])
    budget2 = max(0, cap - stage1_evals)

    eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(seed), config=dict(GAS_CONFIG))
    r = eng.explain(x0, epsilon=0.05, delta=0.05, max_budget=budget2,
                    n_pilot=3, n_active_steps=10)
    unique = int(r["num_coalition_evals_this_call"])  # cache off -> all unique
    return np.asarray(r["shapley_values"]), unique, int(r.get("stage2_attempted_total", 0))


def run_one(ds, X, feats, nc, i, cap, seed):
    m, X_te, _, _ = build_surrogate(X, nc, feats)
    cid = i % nc
    fn = make_proba_fn(m, feats, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=seed + i).values

    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                             model_tag=f"uc-{ds}-{i}-exact")
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    def game_fn(S_mask):
        o = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                            model_tag=f"uc-{ds}-{i}")
        return o.evaluate(x0, S_mask)

    t0 = time.time()
    phi_g, ug, att = run_gas_capped(fn, bg, x0, X.shape[1], cap, seed + i)
    t_gas = time.time() - t0

    t0 = time.time()
    phi_s, us, t_s = run_official_shaplEIG(game_fn, X.shape[1], cap, seed=seed + i)
    t_s = time.time() - t0

    return {
        "dataset": ds, "instance": i, "unique_cap": cap,
        "gas_rmse": rmse(phi_g, phi_exact),
        "shaplEIG_rmse": rmse(phi_s, phi_exact),
        "gas_unique_evals": ug,
        "shaplEIG_unique_queries": us,
        "gas_attempted_draws": att,
        "gas_wall_s": round(t_gas, 1),
        "shaplEIG_wall_s": round(t_s, 1),
        "cap_honored": bool(ug <= cap * 1.05 + 2),
        "unique_matched": bool(abs(ug - us) <= 0.35 * max(ug, us)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--caps", default="512,1024")
    ap.add_argument("--dataset", choices=["wine", "air", "both"], default="both")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in (["wine", "air"] if args.dataset == "both" else [args.dataset]):
        X, feats, nc = (load_wine(), WINE_FEATURES, 2) if ds == "wine" \
            else (load_air_station(n_clusters=4), AIR_FEATURES, 4)
        X = X[0]
        for i in range(args.n):
            for cap in caps:
                row = run_one(ds, X, feats, nc, i, cap, seed=1301)
                rows.append(row)
                print(f"  {ds} inst {i} cap={cap}: gas={row['gas_rmse']:.5f} "
                      f"(unique={row['gas_unique_evals']}) vs "
                      f"shaplEIG={row['shaplEIG_rmse']:.5f} "
                      f"(unique={row['shaplEIG_unique_queries']}) "
                      f"matched={row['unique_matched']}")

    d = pd.DataFrame(rows)
    d["source"] = SHAPLEIG_SOURCE
    d.to_csv(OUT / "unique_capped_shaplEIG.csv", index=False)
    import shutil
    shutil.copy2(OUT / "unique_capped_shaplEIG.csv",
                 MAIN / "paper_unique_capped_shaplEIG.csv")

    hon = d[d.cap_honored]
    print("\n=== mean over instances (cap-honored rows only) ===")
    if hon.empty:
        print("WARNING: no rows honoured the unique cap (GAS fixed Stage-1 cost"
              " ~371 at M=11 exceeds caps below ~400).  Use caps 512,1024.")
    else:
        g = hon.groupby(["dataset", "unique_cap"]).agg(
            gas_rmse=("gas_rmse", "mean"),
            gas_unique=("gas_unique_evals", "mean"),
            shaplEIG_rmse=("shaplEIG_rmse", "mean"),
            shaplEIG_unique=("shaplEIG_unique_queries", "mean"),
            matched_frac=("unique_matched", "mean"),
        ).round(5)
        print(g.to_string())
    print(f"\ncap_honored rows: {int(hon.shape[0])}/{len(d)}")
    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_unique_capped_shaplEIG.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
