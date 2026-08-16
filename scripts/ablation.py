#!/usr/bin/env python
"""4-tier ablation study (review task #5 / paper Table 2).

Isolates the contribution of each engine component on real data:

  Tier 1  Uniform MC        : uniform stratum sampling, raw marginals, no GP
  Tier 2  Neyman MC         : coupled Neyman-allocated raw marginals, no GP
  Tier 3  GP-only           : bounded-linear surrogate attribution, no residual
  Tier 4  Full GAS-BayesSHAP: GP control variate + Neyman residual certification

All tiers are evaluated at a MATCHED coalition budget K against exact Shapley
(2^11 = 2048 coalitions), same instances, same background, same seed.

Usage:
    python scripts/ablation.py --dataset wine --K 1000 --n 10
    python scripts/ablation.py --dataset air --K 1000 --n 10 --clusters 4
Outputs -> results/paper_experiments/ablation_<dataset>.csv + main_results/
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
from gas_bayesshap.residual.neyman import solve_coupled_neyman_allocation

WINE_FEAT = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
             "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
             "pH", "sulphates", "alcohol"]
AIR_FEAT = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def load(name, clusters):
    if name == "wine":
        df = pd.read_csv(ROOT / "data" / "winequality-white.csv", sep=";")
        X = df[WINE_FEAT].dropna().reset_index(drop=True)
        feats = WINE_FEAT
        nc = 2
    else:
        df = pd.read_csv(ROOT / "data" / "Beijing_MultiSite_AirQuality.csv")
        st = df["station"].value_counts().index[0]
        X = df[df["station"] == st][AIR_FEAT].dropna().reset_index(drop=True)
        if len(X) > 40000:
            X = X.sample(40000, random_state=1301).reset_index(drop=True)
        feats = AIR_FEAT
        nc = clusters
    return X, feats, nc


def build_surrogate(X, nc):
    Xs = StandardScaler().fit_transform(X)
    lab = KMeans(n_clusters=nc, random_state=1301, n_init=10).fit_predict(Xs)
    X_tr, X_te, y_tr, _ = train_test_split(X, lab, test_size=0.3, random_state=1301, stratify=lab)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=nc, random_state=1301,
                           n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced")
    m.fit(X_tr, y_tr)
    return m, X_te


def proba_fn(m, feats, cid):
    def fn(x):
        x = np.asarray(x, float).reshape(1, -1)
        return float(m.predict_proba(pd.DataFrame(x, columns=feats))[0, cid])
    return fn


def tier1_uniform_mc(oracle, x0, M, K, rng):
    """Uniform stratum sampling of raw marginals; phi = (1/M) sum_s mean_s."""
    sums = np.zeros(M)
    evals = 0
    # split K evals into ~K/(M+1) rounds (1 base + M marginals per round)
    n_rounds = max(1, K // (M + 1))
    for _ in range(n_rounds):
        s = rng.randint(1, M - 1)  # interior stratum
        S = np.zeros(M, dtype=bool)
        S[rng.permutation(M)[:s]] = True
        vS = oracle.evaluate(x0, S)
        for i in range(M):
            if not S[i]:
                Su = S.copy(); Su[i] = True
                sums[i] += oracle.evaluate(x0, Su) - vS
                evals += 2
    return sums / n_rounds, evals


def tier2_neyman_mc(oracle, x0, M, K, sigma, rng):
    """Coupled Neyman-allocated raw marginals (no GP)."""
    sol = solve_coupled_neyman_allocation(sigma, M, K_cert=1.0)
    sums = np.zeros(M)
    n_rounds = max(1, K // (M + 1))
    for _ in range(n_rounds):
        if np.sum(sol.probabilities) > 0:
            s = rng.choice(M, p=sol.probabilities)
        else:
            s = M // 2
        S = np.zeros(M, dtype=bool)
        S[rng.permutation(M)[:s]] = True
        vS = oracle.evaluate(x0, S)
        for i in range(M):
            if not S[i]:
                Su = S.copy(); Su[i] = True
                sums[i] += oracle.evaluate(x0, Su) - vS
            elif s > 0:
                Sm = S.copy(); Sm[i] = False
                sums[i] += vS - oracle.evaluate(x0, Sm)
    return sums / n_rounds, n_rounds * (M + 1)


def run_ablation(name, K, n_inst, clusters):
    X, feats, nc = load(name, clusters)
    m, X_te = build_surrogate(X, nc)
    rows = []
    for i in range(n_inst):
        cid = i % nc
        fn = proba_fn(m, feats, cid)
        x0 = X_te.iloc[i].values
        bg = X.sample(32, random_state=1301 + i).values
        # SHARED cached oracle so exact enumeration populates the cache and all
        # tiers reuse those values (cache hits cost 0 model evals)
        from gas_bayesshap.cache.coalition_cache import CoalitionCache
        cache = CoalitionCache(config_hash="ablation", oracle_hash="o", background_hash="b")
        oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0), cache=cache,
                                 model_tag=f"abl-{name}-{i}")
        phi_exact = exact_shapley_from_values(exact_game_values(oracle, x0, X.shape[1]), X.shape[1])
        M = X.shape[1]
        rng = np.random.RandomState(1301 + i)
        cfg = {"checkpoint_enabled": False, "cache_enabled": True,
               "persist_cache": False, "log_level": "NONE"}

        # Tier 3: GP-only (no residual certification)
        eng3 = GASBayesSHAP(oracle=oracle, rng=rng, config=cfg)
        gp_only = eng3.explain_stage1_only(x0, n_active_steps=10)
        r_gp = rmse(np.asarray(gp_only["shapley_values"]), phi_exact)

        # Tier 4: full GAS (shared oracle -> cached exact values)
        eng4 = GASBayesSHAP(oracle=oracle, rng=rng, config=cfg)
        r4 = eng4.explain(x0, epsilon=0.05, delta=0.05, max_budget=K, n_pilot=3, n_active_steps=10)
        r_full = rmse(np.asarray(r4["shapley_values"]), phi_exact)

        # Tier 1: uniform MC (matched K, cached oracle)
        t1, ev1 = tier1_uniform_mc(oracle, x0, M, K, np.random.RandomState(1301 + i))
        # Tier 2: Neyman MC (matched K)
        sig = np.ones((M, M)) * 0.5
        t2, ev2 = tier2_neyman_mc(oracle, x0, M, K, sig, np.random.RandomState(1301 + i))

        W4 = np.asarray(r4["certified_projected_widths"])
        phi4 = np.asarray(r4["shapley_values"])
        err4 = np.abs(phi4 - phi_exact)
        rows.append({
            "instance": i, "cluster": cid,
            "tier1_uniform_rmse": rmse(t1, phi_exact), "tier1_evals": ev1,
            "tier2_neyman_rmse": rmse(t2, phi_exact), "tier2_evals": ev2,
            "tier3_gp_rmse": r_gp, "tier3_evals": gp_only["num_coalition_evals"],
            "tier4_full_rmse": r_full, "tier4_evals": r4["num_coalition_evals_this_call"],
            "tier4_mean_width": float(W4.mean()),
            "tier4_sim_cov": float(np.all(err4 <= W4)),
            "tier4_sign_cert": float(np.mean(np.abs(phi4) > W4)),
        })
        print(f"  inst {i}: uniform={rows[-1]['tier1_uniform_rmse']:.5f} "
              f"neyman={rows[-1]['tier2_neyman_rmse']:.5f} "
              f"gp={rows[-1]['tier3_gp_rmse']:.5f} full={rows[-1]['tier4_full_rmse']:.5f}")

    d = pd.DataFrame(rows)
    out = ROOT / "results" / "paper_experiments"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"ablation_{name}.csv", index=False)
    summ = {
        "dataset": name, "K": K, "n_instances": len(d),
        "tier1_uniform_rmse_mean": d.tier1_uniform_rmse.mean(),
        "tier2_neyman_rmse_mean": d.tier2_neyman_rmse.mean(),
        "tier3_gp_rmse_mean": d.tier3_gp_rmse.mean(),
        "tier4_full_rmse_mean": d.tier4_full_rmse.mean(),
        "tier4_evals_mean": d.tier4_evals.mean(),
        "tier4_mean_width": d.tier4_mean_width.mean(),
        "tier4_sim_cov_rate": d.tier4_sim_cov.mean(),
        "tier4_sign_cert_fraction": d.tier4_sign_cert.mean(),
    }
    pd.DataFrame([summ]).to_csv(out / f"ablation_{name}_summary.csv", index=False)
    import shutil
    (ROOT / "main_results").mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / f"ablation_{name}.csv", ROOT / "main_results" / f"paper_ablation_{name}.csv")
    shutil.copy2(out / f"ablation_{name}_summary.csv", ROOT / "main_results" / f"paper_ablation_{name}_summary.csv")
    print(f"[{name}] summary: {summ}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wine", "air"], default="wine")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=4)
    args = ap.parse_args()
    run_ablation(args.dataset, args.K, args.n, args.clusters)


if __name__ == "__main__":
    main()
