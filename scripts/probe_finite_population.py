#!/usr/bin/env python
"""Finite-population empirical-range probe (Theorem E demonstration).

Produces two paper artifacts:

1. ``results/paper_experiments/coverage_calibration_finite_population.json``
   (+ copy in ``main_results/``): R=500 coverage validation of the
   ``finite_population`` range mode on the M=3 calibration game, with the
   realised coupon-collector budget delta1 and coverage level.

2. ``results/paper_experiments/wine_range_modes_N{K}.csv`` (+ copy): matched
   real-data comparison (wine, K instances, eps=0.05, budget=3000) of the
   spec range vs the finite-population range: RMSE, mean certified width,
   realised delta1, realised coverage level, sign-certified fraction.

Usage:
    python scripts/probe_finite_population.py --n 5 --trials 500
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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse, coverage_report
from gas_bayesshap.game.domain_games import membership_game

WINE_FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
RNG = np.random.RandomState(1301)
OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def load_wine():
    import warnings
    path = ROOT / "data" / "winequality-white.csv"
    df = pd.read_csv(path, sep=";") if path.exists() else None
    if df is None or list(df.columns[:2]) != ["fixed acidity", "volatile acidity"]:
        raise RuntimeError("wine data missing — place winequality-white.csv in data/")
    X = df[WINE_FEATURES].values
    y = (df["quality"] >= 7).astype(int).values
    return X, y


def run_calibration(trials: int, mode: str) -> dict:
    from gas_bayesshap.game.brute_force import brute_force_shapley
    from gas_bayesshap.benchmarking.metrics import coverage_report

    def model_cal(x):
        return float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])

    phi_true = brute_force_shapley(
        lambda S: model_cal(np.asarray(S, dtype=float)), 3)
    cfg = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}
    phis, widths, costs, d1s, levels = [], [], [], [], []
    for trial in range(trials):
        eng = GASBayesSHAP(model_cal, np.zeros((3, 3)), output_bounds=(-2.0, 5.0),
                           rng=np.random.RandomState(7000 + trial),
                           config={**cfg, "range_mode": mode})
        r = eng.explain(np.ones(3), epsilon=1.5, delta=0.05, max_budget=300, n_pilot=3)
        phis.append(r["shapley_values"])
        widths.append(r["certified_projected_widths"])
        costs.append(r["num_coalition_evals"])
        if mode == "finite_population":
            d1s.append(r.get("finite_population_delta1") or 0.0)
            levels.append(r.get("finite_population_coverage_level") or 1.0)
    rep = coverage_report(phis, widths, phi_true)
    rep["oracle_query_cost_mean"] = float(np.mean(costs))
    rep["range_mode"] = mode
    if mode == "finite_population":
        rep["finite_population_delta1_mean"] = float(np.mean(d1s))
        rep["realised_coverage_level_mean"] = float(np.mean(levels))
    return rep


def run_wine_instance(X, y, idx, mode, eps, budget):
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1301)
    m = lgb.LGBMClassifier(n_estimators=64, max_depth=4, verbose=-1,
                           random_state=1301).fit(Xtr, ytr)
    fn = lambda x: float(m.predict_proba(pd.DataFrame(x.reshape(1, -1),
                                                      columns=WINE_FEATURES))[0, 1])
    x0 = Xte[idx]
    bg = Xte[np.random.RandomState(1301 + idx).choice(len(Xte), 32, replace=False)]
    oracle, spec = membership_game(fn, bg)
    vals = exact_game_values(oracle, x0, spec.M)
    phi_exact = exact_shapley_from_values(vals, spec.M)
    eng = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(1301 + idx),
                       config={"checkpoint_enabled": False, "cache_enabled": True,
                               "persist_cache": False, "log_level": "NONE",
                               "range_mode": mode})
    r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=budget, n_pilot=3,
                    n_active_steps=10)
    W = np.asarray(r["certified_projected_widths"])
    phi = np.asarray(r["shapley_values"])
    return {
        "instance": idx, "mode": mode,
        "rmse": rmse(phi, phi_exact),
        "sim_cov": float(np.all(np.abs(phi - phi_exact) <= W)),
        "mean_width": float(np.mean(W)),
        "sign_cert": float(np.mean(np.abs(phi) > W)),
        "gas_evals": r["num_coalition_evals_this_call"],
        "status": r["status"],
        "R_eff": r["R_delta_res_effective"],
        "delta1": r.get("finite_population_delta1"),
        "level": r.get("finite_population_coverage_level"),
        "heuristic": bool(r["range_bound_is_heuristic"]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="wine instances per mode")
    p.add_argument("--trials", type=int, default=500, help="calibration trials")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--budget", type=int, default=3000)
    args = p.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    # ---- 1. R=500 calibration, both modes -------------------------------- #
    cal = {}
    for mode in ("spec", "finite_population"):
        cal[mode] = run_calibration(args.trials, mode)
        print(f"[calibration {mode}] coverage={cal[mode]['empirical_coverage']:.3f} "
              f"width={cal[mode]['mean_width']:.3f} "
              f"finite_width={cal[mode]['finite_width_rate']:.3f}")
        if mode == "finite_population":
            print(f"  delta1_mean={cal[mode]['finite_population_delta1_mean']:.4f} "
                  f"level_mean={cal[mode]['realised_coverage_level_mean']:.4f}")
    payload = {"n_trials": args.trials, "M": 3, "epsilon": 1.5, "delta": 0.05,
               "max_budget": 300, "modes": cal}
    for dest in (OUT, MAIN):
        (dest / "paper_coverage_calibration_R500_finite_population.json").write_text(
            json.dumps(payload, indent=1))
    print(f"wrote paper_coverage_calibration_R500_finite_population.json "
          f"({time.time()-t0:.0f}s)")

    # ---- 2. wine matched-budget comparison -------------------------------- #
    X, y = load_wine()
    rows = []
    for mode in ("spec", "finite_population"):
        for idx in range(args.n):
            rows.append(run_wine_instance(X, y, idx, mode, args.eps, args.budget))
            print(f"[wine {mode} inst {idx}] "
                  f"rmse={rows[-1]['rmse']:.5f} W={rows[-1]['mean_width']:.3f} "
                  f"delta1={rows[-1]['delta1']} level={rows[-1]['level']}")
    d = pd.DataFrame(rows)
    fn = f"wine_range_modes_n{args.n}.csv"
    d.to_csv(OUT / fn, index=False)
    summ = d.groupby("mode").agg(
        rmse_mean=("rmse", "mean"), mean_width=("mean_width", "mean"),
        sim_cov=("sim_cov", "mean"), sign_cert=("sign_cert", "mean"),
        delta1_mean=("delta1", "mean"), level_mean=("level", "mean"),
        R_eff_mean=("R_eff", "mean"), gas_evals_mean=("gas_evals", "mean"),
    ).reset_index()
    summ.to_csv(OUT / f"wine_range_modes_n{args.n}_summary.csv", index=False)
    for f in (fn, f"wine_range_modes_n{args.n}_summary.csv"):
        import shutil
        shutil.copy2(OUT / f, MAIN / f"paper_{f}")
    print(summ.to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
