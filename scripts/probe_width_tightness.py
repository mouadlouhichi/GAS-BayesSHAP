#!/usr/bin/env python
"""Certificate-tightness probe (answers Q1-review Blocker 1a).

Runs GAS-BayesSHAP on ONE wine instance at increasing coalition budgets and
reports width decay (1/sqrt(K)), sign-certified fraction, and error vs exact.

Usage:
    python scripts/probe_width_tightness.py [--budgets 2000,8000,30000,100000]
Outputs -> results/paper_experiments/width_probe.csv + main_results/
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

FEAT = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="2000,8000,30000,100000",
                    help="comma-separated max_budget values")
    ap.add_argument("--seed", type=int, default=1301)
    ap.add_argument("--range-mode", default="spec",
                    choices=["spec", "finite_population", "empirical_max"])
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    seed = args.seed
    rm = args.range_mode

    df = pd.read_csv(ROOT / "data" / "winequality-white.csv", sep=";")
    X = df[FEAT].dropna().reset_index(drop=True)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(Xs)
    lab = km.labels_
    X_tr, X_te, _, _ = train_test_split(X, lab, test_size=0.3, random_state=seed, stratify=lab)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=2, random_state=seed,
                           n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced")
    m.fit(X_tr, lab[X_tr.index])

    def fn(x):
        x = np.asarray(x, float).reshape(1, -1)
        return float(m.predict_proba(pd.DataFrame(x, columns=FEAT))[0, 0])

    x0 = X_te.iloc[0].values
    bg = X.sample(64, random_state=seed).values

    o = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0))
    phi_exact = exact_shapley_from_values(exact_game_values(o, x0, X.shape[1]), X.shape[1])
    print("exact phi: min=%.4f max=%.4f | |phi|>0.05: %d"
          % (phi_exact.min(), phi_exact.max(), int((np.abs(phi_exact) > 0.05).sum())))

    rows = []
    for K in budgets:
        t0 = time.time()
        eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0), rng=np.random.RandomState(seed),
                           config={"checkpoint_enabled": False, "cache_enabled": True,
                                   "persist_cache": False, "log_level": "NONE",
                                   "range_mode": rm})
        r = eng.explain(x0, epsilon=0.02, delta=0.05, max_budget=K, n_pilot=3, n_active_steps=10)
        W = np.asarray(r["certified_projected_widths"])
        phi = np.asarray(r["shapley_values"])
        err = float(np.max(np.abs(phi - phi_exact)))
        sc = np.abs(phi) > W
        # validation against exact ground truth: certified sign must equal the
        # exact sign and |phi_exact| must exceed the certified width (no
        # boundary/zero-attribution cases certified).
        sc_ok = int(np.all(np.sign(phi[sc]) == np.sign(phi_exact[sc]))) if sc.any() else 1
        sc_min_margin = float(np.min(np.abs(phi_exact[sc]) - W[sc])) if sc.any() else float("nan")
        rows.append({
            "budget": K, "status": r["status"], "converged": r["converged"],
            "range_mode": rm,
            "mean_width": float(W.mean()), "max_width": float(W.max()),
            "sign_certified_fraction": float(np.mean(sc)),
            "n_sign_certified": int(sc.sum()),
            "signs_match_exact": sc_ok,
            "min_exact_margin": sc_min_margin,
            "max_err_vs_exact": err, "elapsed_s": round(time.time() - t0, 1),
            "delta1": r.get("finite_population_delta1"),
            "realised_level": r.get("finite_population_coverage_level"),
            "at_nominal_level": r.get("finite_population_at_level_delta"),
            "certificate_is_rigorous": bool(r.get("certificate_is_rigorous", False)),
        })
        extra = "" if rm == "spec" else (
            f" delta1={rows[-1]['delta1']:.3f} level={rows[-1]['realised_level']:.3f} "
            f"at_nominal={rows[-1]['at_nominal_level']}")
        print(f"K={K:>7}: {r['status']:15s} meanW={W.mean():.3f} sign_cert="
              f"{np.mean(sc):.3f} ({int(sc.sum())} feats) signs_ok={sc_ok} "
              f"max_err={err:.5f} ({rows[-1]['elapsed_s']:.0f}s){extra}")

    out_dir = ROOT / "results" / "paper_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    sfx = "" if rm == "spec" else f"_{rm}"
    d.to_csv(out_dir / f"width_probe{sfx}.csv", index=False)
    # copy to main_results
    (ROOT / "main_results").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out_dir / f"width_probe{sfx}.csv",
                 ROOT / "main_results" / f"paper_width_probe{sfx}.csv")
    print(f"saved results/paper_experiments/width_probe{sfx}.csv "
          f"+ main_results/paper_width_probe{sfx}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
