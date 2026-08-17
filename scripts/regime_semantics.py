#!/usr/bin/env python
"""Air-quality regime naming + RQ3 semantic preservation (review task #7).

1. Names the k=4 K-Means clusters using their pollutant/meteorological
   profiles (photochemical / winter smog / stagnant inversion / clean air)
   per the paper's domain narrative.
2. RQ3: for each named regime, computes the Spearman rank correlation
   between GAS-BayesSHAP |attribution| rankings and the expected driver
   rankings from the 2025 domain findings (photochemical -> O3,TEMP;
   winter smog -> CO,SO2,PM10; stagnant -> low WSPM, PRES; clean -> WSPM).

Usage:
    python scripts/regime_semantics.py [--n 10] [--clusters 4]
Outputs -> results/paper_experiments/regime_semantics.csv + main_results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse

FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Expected driver rankings per regime (2025 paper findings) - higher = more important
EXPECTED_DRIVERS = {
    "photochemical": {"O3": 5, "TEMP": 4, "PM2.5": 3, "NO2": 2, "WSPM": 1, "SO2": 1, "CO": 1, "PM10": 1, "PRES": 0, "DEWP": 0, "RAIN": 0},
    "winter_smog":   {"CO": 5, "SO2": 5, "PM10": 4, "PM2.5": 4, "PRES": 2, "WSPM": 1, "NO2": 1, "O3": 0, "TEMP": 0, "DEWP": 0, "RAIN": 0},
    "stagnant":      {"WSPM": 5, "PRES": 4, "PM2.5": 3, "PM10": 3, "DEWP": 2, "CO": 1, "SO2": 1, "NO2": 1, "O3": 0, "TEMP": 0, "RAIN": 0},
    "clean_air":     {"WSPM": 4, "O3": 3, "NO2": 2, "SO2": 2, "CO": 1, "PM2.5": 1, "PM10": 1, "TEMP": 0, "PRES": 0, "DEWP": 0, "RAIN": 0},
}


def name_regime(means: pd.Series, X: pd.DataFrame) -> str:
    """Name a cluster by its standardized feature profile."""
    lo, hi = X.quantile(0.25), X.quantile(0.75)
    if means["O3"] > hi["O3"] and means["TEMP"] > X["TEMP"].mean():
        return "photochemical"
    if means["CO"] > hi["CO"] or means["SO2"] > hi["SO2"]:
        return "winter_smog"
    if means["WSPM"] < lo["WSPM"] and means["PRES"] > X["PRES"].mean():
        return "stagnant"
    return "clean_air"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "Beijing_MultiSite_AirQuality.csv")
    st = df["station"].value_counts().index[0]
    X = df[df["station"] == st][FEATURES].dropna().reset_index(drop=True)
    if len(X) > 40000:
        X = X.sample(40000, random_state=1301).reset_index(drop=True)

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.clusters, random_state=1301, n_init=10).fit(Xs)
    lab = km.labels_

    # name regimes from inverse-scaled centers
    centers = pd.DataFrame(
        StandardScaler().fit_transform(X).mean(0)[None, :] * 0 +  # placeholder
        StandardScaler().fit_transform(X).mean(0), columns=FEATURES)
    centers_raw = pd.DataFrame(
        StandardScaler().fit(X).inverse_transform(km.cluster_centers_), columns=FEATURES)
    regime_names = [name_regime(centers_raw.iloc[c], X) for c in range(args.clusters)]
    # audit P1-8: two clusters can map to the same name (e.g. two clean-air
    # subregimes); suffix duplicates so the distinction is explicit.
    seen = {}
    for c in range(args.clusters):
        n = regime_names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            regime_names[c] = f"{n}_{seen[n]}"
    print("named regimes:", dict(enumerate(regime_names)))

    # surrogate
    X_tr, X_te, y_tr, _ = train_test_split(X, lab, test_size=0.3, random_state=1301, stratify=lab)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=args.clusters, random_state=1301,
                           n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced")
    m.fit(X_tr, y_tr)

    rows = []
    for i in range(args.n):
        cid = i % args.clusters
        rname = regime_names[cid]

        def fn(x):
            x = np.asarray(x, float).reshape(1, -1)
            return float(m.predict_proba(pd.DataFrame(x, columns=FEATURES))[0, cid])

        x0 = X_te.iloc[i].values
        bg = X.sample(32, random_state=1301 + i).values
        oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0))
        phi_exact = exact_shapley_from_values(exact_game_values(oracle, x0, X.shape[1]), X.shape[1])
        eng = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(1301 + i),
                           config={"checkpoint_enabled": False, "cache_enabled": True,
                                   "persist_cache": False, "log_level": "NONE"})
        r = eng.explain(x0, epsilon=args.eps, delta=0.05, max_budget=args.budget,
                        n_pilot=3, n_active_steps=10)
        phi = np.asarray(r["shapley_values"])
        # RQ3: rank correlation of GAS |phi| vs expected drivers.
        # Suffixed duplicate names (clean_air_2) share the base regime's
        # expected-driver reference: strip the _N suffix for the lookup.
        rref = rname.rsplit("_", 1)[0] if rname.rsplit("_", 1)[-1].isdigit() else rname
        exp = np.array([EXPECTED_DRIVERS[rref][f] for f in FEATURES])
        abs_phi = np.abs(phi)
        rho, _ = spearmanr(abs_phi, exp) if exp.std() > 0 else (np.nan, 1.0)
        rows.append({
            "instance": i, "cluster": cid, "regime": rname,
            "rmse_vs_exact": rmse(phi, phi_exact),
            "spearman_driver_corr": float(rho),
            "top3_gas": ",".join([FEATURES[j] for j in np.argsort(-abs_phi)[:3]]),
        })
        print(f"  inst {i} [{rname:14s}] rmse={rows[-1]['rmse_vs_exact']:.5f} "
              f"rho={rows[-1]['spearman_driver_corr']:.3f} top3={rows[-1]['top3_gas']}")

    d = pd.DataFrame(rows)
    out = ROOT / "results" / "paper_experiments"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "regime_semantics.csv", index=False)
    summ = d.groupby("regime").agg(
        rmse_mean=("rmse_vs_exact", "mean"),
        spearman_mean=("spearman_driver_corr", "mean"),
        n=("instance", "count")).reset_index()
    summ.to_csv(out / "regime_semantics_summary.csv", index=False)
    import shutil
    (ROOT / "main_results").mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "regime_semantics.csv", ROOT / "main_results" / "paper_regime_semantics.csv")
    shutil.copy2(out / "regime_semantics_summary.csv", ROOT / "main_results" / "paper_regime_semantics_summary.csv")
    print(summ.to_string(index=False))
    return d


if __name__ == "__main__":
    main()
