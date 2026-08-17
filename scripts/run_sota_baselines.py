#!/usr/bin/env python
"""Reference-baseline ablation (review task #4 / #6) on real data.

Produces `paper_reference_baselines_ablation.csv`: RMSE of GAS-BayesSHAP
vs an OddSHAP-style *transform* reference (exact Shapley of the log-odds
game, computed once per instance -- NOT budget-matched) and a ShaplEIG-style
GP-quadrature reference (GP posterior-mean Shapley, non-certified, design
size scaled with K, deduplicated).  Both are method-style reimplementations
and are NOT official reproductions; see the comparability caveats in
paper_results_summary.md.

Honest labelling (see module docstring in
gas_bayesshap/benchmarking/sota_baselines.py): official OddSHAP / ShaplEIG
code is not public in this environment; these are faithful *method-style*
reimplementations from their published descriptions, reported as
non-certified reference points — NOT official reproductions.

Usage:
    python scripts/run_sota_baselines.py --n 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse
from gas_bayesshap.benchmarking.sota_baselines import (
    odd_shapley_exact,
    gp_quadrature_shapley,
)

WINE_FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
AIR_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
                "RAIN", "WSPM"]
RNG = np.random.RandomState(1301)
OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def load_wine():
    df = pd.read_csv(ROOT / "data" / "winequality-white.csv", sep=";")
    return df[WINE_FEATURES].dropna().reset_index(drop=True), WINE_FEATURES


def load_air():
    df = pd.read_csv(ROOT / "data" / "Beijing_MultiSite_AirQuality.csv")
    st = df["station"].value_counts().index[0]
    X = df[df["station"] == st][AIR_FEATURES].dropna().reset_index(drop=True)
    if len(X) > 40000:
        X = X.iloc[:40000].reset_index(drop=True)
    return X, AIR_FEATURES


def build_surrogate(X, n_clusters, feat_names):
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=1301, n_init=10).fit(Xs)
    lab = km.labels_
    X_tr, X_te, y_tr, y_te = train_test_split(X, lab, test_size=0.3, random_state=1301,
                                              stratify=lab)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=n_clusters, random_state=1301,
                           n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m, X_te, lab


def make_proba_fn(m, feat_names, cluster_id):
    def model_fn(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(m.predict_proba(pd.DataFrame(x, columns=feat_names))[0, cluster_id])
    return model_fn


def run_comparison(name, X, feat_names, n_clusters, N, budgets):
    m, X_te, _ = build_surrogate(X, n_clusters, feat_names)
    rows = []
    for i in range(N):
        x0 = X_te.iloc[i].values
        cid = i % n_clusters
        fn = make_proba_fn(m, feat_names, cid)
        bg = X.sample(64, random_state=1301 + i).values
        oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0))
        phi_exact = exact_shapley_from_values(
            exact_game_values(oracle, x0, X.shape[1]), X.shape[1])
        # OddSHAP-style: exact log-odds Shapley (needs 2^M evals)
        phi_odd = odd_shapley_exact(oracle, x0, X.shape[1])
        # ---- OddSHAP-style: exact log-odds Shapley (needs 2^M evals) ---- #
        # NOTE: this is a *transform* reference (log-odds game), computed once
        # per instance (not budget-matched) -- see the comparability caveat in
        # the module docstring and paper_results_summary.md.
        phi_odd = odd_shapley_exact(oracle, x0, X.shape[1])

        # ---- ShaplEIG-style: GP posterior-mean Shapley on a K-scaled design ---- #
        # FIX (audit): one rng per instance, advanced per draw -- previously
        # RandomState(7+i) was re-seeded inside the loop, so all design masks
        # were IDENTICAL (256 copies of one coalition).  Design size now scales
        # with K and is deduplicated; the GP is fitted on the unique coalitions.
        rng_design = np.random.RandomState(7 + i)
        for K in budgets:
            design_size = max(64, min(K, 1024))
            design_raw = [rng_design.randint(0, 2, X.shape[1]).astype(bool)
                          for _ in range(design_size)]
            design = np.unique(np.asarray(design_raw, dtype=bool), axis=0)
            # unique rows may be fewer than design_size; fit on what is unique
            y_design = np.array([oracle.evaluate(x0, S) for S in design])
            phi_gp, _ = gp_quadrature_shapley(oracle, x0, X.shape[1],
                                              list(design), y_design)

            eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                               rng=np.random.RandomState(1301 + i),
                               config={"checkpoint_enabled": False, "cache_enabled": True,
                                       "persist_cache": False, "log_level": "NONE"})
            r = eng.explain(x0, epsilon=0.05, delta=0.05, max_budget=K,
                            n_pilot=3, n_active_steps=10)
            phi_gas = np.asarray(r["shapley_values"])
            rows.append({
                "dataset": name, "instance": i, "K": K,
                "gas_rmse": rmse(phi_gas, phi_exact),
                "odd_logodds_rmse": rmse(phi_odd, phi_exact),
                "gp_quadrature_rmse": rmse(phi_gp, phi_exact),
                "gas_coalition_evals": r["num_coalition_evals_this_call"],
                "gas_model_evals": r.get("num_model_evals_this_call", 0),
                "gp_design_size": int(design_size),
                "gp_unique_coalitions": int(len(design)),
                "game": "membership",
            })
        print(f"  {name} inst {i}: " + " ".join(
            f"K{r2['K']}: gas={r2['gas_rmse']:.5f} odd={r2['odd_logodds_rmse']:.5f} "
            f"gp={r2['gp_quadrature_rmse']:.5f}" for r2 in rows[-len(budgets):]))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    budgets = [256, 1024, 2048]
    Xw, _ = load_wine()
    Xa, _ = load_air()
    dw = run_comparison("wine", Xw, WINE_FEATURES, 2, args.n, budgets)
    da = run_comparison("air", Xa, AIR_FEATURES, 4, args.n, budgets)
    d = pd.concat([dw, da], ignore_index=True)
    # Renamed per audit: NOT a matched-budget comparison -- OddSHAP-style is a
    # transform reference, GP-quadrature is a fixed-cost surrogate reference.
    d.to_csv(OUT / "reference_baselines_ablation.csv", index=False)
    import shutil
    shutil.copy2(OUT / "reference_baselines_ablation.csv",
                 MAIN / "paper_reference_baselines_ablation.csv")

    summ = d.groupby(["dataset", "K"]).agg(
        gas_rmse_mean=("gas_rmse", "mean"),
        odd_logodds_rmse_mean=("odd_logodds_rmse", "mean"),
        gp_quadrature_rmse_mean=("gp_quadrature_rmse", "mean"),
        gas_coalition_evals_mean=("gas_coalition_evals", "mean"),
    ).reset_index()
    print("\n=== Summary (mean RMSE over N=%d instances) ===" % args.n)
    print(summ.to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
