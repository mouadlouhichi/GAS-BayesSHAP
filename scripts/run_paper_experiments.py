#!/usr/bin/env python
"""Paper-grade GAS-BayesSHAP experiments on real data (RQ1/RQ2/RQ3/RQ5).

Addresses the audit of `main_results`:
- A: tight epsilon (default 0.05) with a large budget, so certificates are
      non-vacuous; widths and sign-certified fractions are reported honestly
      even when the budget exhausts before convergence.
- B: multi-instance aggregation (N instances, default 20): mean/std RMSE,
      simultaneous coverage rate, mean widths, sign-certified fraction,
      mean coalition evals.
- C: air uses n_clusters=4 (the paper's four regimes).
- D: matched-budget curves K in {128, 256, 512, 1024, 2048}: RMSE vs K for
      GAS-BayesSHAP, KernelSHAP, SamplingSHAP.
- E: Tier-B group-lag (M=66 -> 11) with pollutant macro names and macro
      simultaneous coverage.

Usage:
    python scripts/run_paper_experiments.py --n 20 --eps 0.05 --budget 3000
    python scripts/run_paper_experiments.py --only wine
    python scripts/run_paper_experiments.py --only air
    python scripts/run_paper_experiments.py --only curves

Outputs (CSV + PNG) -> results/paper_experiments/ and copied to main_results/.
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import shap

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse, mae, max_abs_error
from gas_bayesshap.benchmarking.monte_carlo import monte_carlo_shapley
from gas_bayesshap.game.domain_games import group_lag_game, build_group_lags

FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WINE_FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
RNG = np.random.RandomState(1301)

OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def out(*parts):
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Data + surrogate builders
# --------------------------------------------------------------------------- #
def load_wine():
    df = pd.read_csv(ROOT / "data" / "winequality-white.csv", sep=";")
    X = df[WINE_FEATURES].dropna().reset_index(drop=True)
    return X, WINE_FEATURES


def load_air_station(n_clusters=4, max_n=40000):
    df = pd.read_csv(ROOT / "data" / "Beijing_MultiSite_AirQuality.csv")
    st = df["station"].value_counts().index[0]
    X = df[df["station"] == st][FEATURES].dropna().reset_index(drop=True)
    if len(X) > max_n:
        X = X.sample(max_n, random_state=1301).reset_index(drop=True)
    return X, FEATURES


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
    acc = float(accuracy_score(y_te, m.predict(X_te)))
    f1 = float(f1_score(y_te, m.predict(X_te), average="macro"))
    return m, X_te, acc, f1


def make_proba_fn(m, feat_names, cluster_id):
    def model_fn(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(m.predict_proba(pd.DataFrame(x, columns=feat_names))[0, cluster_id])
    return model_fn


# --------------------------------------------------------------------------- #
# Per-instance evaluation
# --------------------------------------------------------------------------- #
def evaluate_instance(model_fn, X, x0, B, eps, budget, n_active=10, seed=1301,
                       range_mode="spec"):
    """Exact + GAS + KernelSHAP + SamplingSHAP for one instance."""
    bg = X.sample(B, random_state=seed).values
    oracle = CoalitionOracle(model_fn, bg, output_bounds=(0.0, 1.0),
                             model_tag=f"paper-{seed}")

    # exact ground truth
    values = exact_game_values(oracle, x0, X.shape[1])
    phi_exact = exact_shapley_from_values(values, X.shape[1])
    exact_evals = oracle.total_coalition_evals

    # GAS-BayesSHAP
    eng = GASBayesSHAP(model_fn, bg, output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(seed),
                       config={"checkpoint_enabled": False, "cache_enabled": True,
                               "persist_cache": False, "log_level": "NONE",
                               "range_mode": range_mode})
    r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=budget,
                    n_pilot=3, n_active_steps=n_active)
    phi_gas = np.asarray(r["shapley_values"])
    W_gas = np.asarray(r["certified_projected_widths"])
    gas_evals = r["num_coalition_evals_this_call"]
    err = np.abs(phi_gas - phi_exact)
    sim_cov = float(np.all(err <= W_gas))
    mar_cov = float(np.mean(err <= W_gas))
    sign_cert = float(np.mean(np.abs(phi_gas) > W_gas))

    # KernelSHAP (matched background)
    ke = shap.KernelExplainer(_proba_matrix_fn(model_fn), bg)
    kernel_phi = ke.shap_values(x0, nsamples=256)
    kernel_evals = 256

    # SamplingSHAP (Monte-Carlo)
    mc_oracle = CoalitionOracle(model_fn, bg, output_bounds=(0.0, 1.0),
                                model_tag=f"paper-mc-{seed}")
    n_rounds = budget // (X.shape[1] + 1)
    mc = monte_carlo_shapley(mc_oracle, x0, n_samples=n_rounds,
                             rng=np.random.RandomState(seed))
    mc_phi = np.asarray(mc["shapley_values"])
    mc_evals = mc["num_coalition_evals"]

    return {
        "phi_exact": phi_exact, "phi_gas": phi_gas, "W_gas": W_gas,
        "kernel_phi": np.asarray(kernel_phi), "mc_phi": mc_phi,
        "exact_evals": exact_evals, "gas_evals": gas_evals,
        "kernel_evals": kernel_evals, "mc_evals": mc_evals,
        "rmse_gas": rmse(phi_gas, phi_exact), "mae_gas": mae(phi_gas, phi_exact),
        "rmse_kernel": rmse(kernel_phi, phi_exact),
        "rmse_mc": rmse(mc_phi, phi_exact),
        "sim_cov": sim_cov, "mar_cov": mar_cov, "sign_cert": sign_cert,
        "mean_width": float(np.mean(W_gas)), "max_width": float(np.max(W_gas)),
        "status": r["status"], "converged": bool(r["converged"]),
    }


def _proba_matrix_fn(model_fn):
    def pm(Xm):
        return np.array([model_fn(r) for r in np.asarray(Xm)])
    return pm


# module-level placeholders (filled per dataset to keep closures simple)
m_ = None
feat_names_ = None


# --------------------------------------------------------------------------- #
# Dataset experiment
# --------------------------------------------------------------------------- #
def run_dataset(name, X, feat_names, n_clusters, eps, budget, N, range_mode="spec"):
    global m_, feat_names_
    m_, feat_names_ = build_surrogate(X, n_clusters, feat_names)[:2]
    m, X_te, acc, f1 = build_surrogate(X, n_clusters, feat_names)
    print(f"[{name}] surrogate acc={acc:.4f} macro-F1={f1:.4f} clusters={n_clusters}")

    rows = []
    for i in range(N):
        x0 = X_te.iloc[i].values
        cid = i % n_clusters  # rotate over clusters
        fn = make_proba_fn(m, feat_names, cid)
        try:
            res = evaluate_instance(fn, X, x0, B=64, eps=eps, budget=budget,
                                     seed=1301 + i, range_mode=range_mode)
            res.update({"instance": i, "cluster": cid})
            rows.append(res)
            print(f"  instance {i}: gas_rmse={res['rmse_gas']:.5f} "
                  f"sim_cov={res['sim_cov']} width={res['mean_width']:.2f} "
                  f"status={res['status']}")
        except Exception as e:
            print(f"  instance {i} FAILED: {type(e).__name__} {str(e)[:120]}")
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N} instances done")

    df = pd.DataFrame(rows)
    summary = {
        "dataset": name, "n_instances": len(df), "clusters": n_clusters,
        "eps": eps, "budget": budget, "range_mode": range_mode,
        "rmse_gas_mean": df["rmse_gas"].mean(), "rmse_gas_std": df["rmse_gas"].std(),
        "rmse_kernel_mean": df["rmse_kernel"].mean(),
        "rmse_mc_mean": df["rmse_mc"].mean(),
        "simultaneous_coverage_rate": df["sim_cov"].mean(),
        "marginal_coverage_rate": df["mar_cov"].mean(),
        "sign_certified_fraction": df["sign_cert"].mean(),
        "mean_width": df["mean_width"].mean(), "max_width_max": df["max_width"].max(),
        "gas_evals_mean": df["gas_evals"].mean(), "exact_evals": df["exact_evals"].iloc[0],
        "kernel_evals": df["kernel_evals"].iloc[0], "mc_evals_mean": df["mc_evals"].mean(),
        "converged_fraction": df["converged"].mean(),
        "status_counts": df["status"].value_counts().to_dict(),
    }
    sfx = "" if range_mode == "spec" else f"_range{range_mode}"
    df.to_csv(out(f"{name}_n{len(df)}_budget{budget}{sfx}_instances.csv"), index=False)
    pd.DataFrame([summary]).to_csv(out(f"{name}_n{len(df)}_budget{budget}{sfx}_summary.csv"), index=False)
    print(f"[{name}] summary: {json.dumps(summary, indent=2, default=str)}")
    return df, summary


# --------------------------------------------------------------------------- #
# Matched-budget curves
# --------------------------------------------------------------------------- #
def run_curves(name, X, feat_names, n_clusters, Ks, N=10, range_mode="spec"):
    global m_, feat_names_
    m_, feat_names_ = build_surrogate(X, n_clusters, feat_names)[:2]
    m, X_te, acc, f1 = build_surrogate(X, n_clusters, feat_names)
    rows = []
    for K in Ks:
        r_gas, r_ker, r_mc = [], [], []
        for i in range(N):
            x0 = X_te.iloc[i].values
            cid = i % n_clusters
            fn = make_proba_fn(m, feat_names, cid)
            bg = X.sample(64, random_state=1301 + i).values
            try:
                eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                                   rng=np.random.RandomState(1301 + i),
                                   config={"checkpoint_enabled": False, "cache_enabled": True,
                                           "persist_cache": False, "log_level": "NONE",
                                           "range_mode": range_mode})
                r = eng.explain(x0, epsilon=0.05, delta=0.05, max_budget=K,
                                n_pilot=3, n_active_steps=10)
                phi_exact = exact_shapley_from_values(
                    exact_game_values(CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0)), x0, X.shape[1]),
                    X.shape[1])
                r_gas.append(rmse(np.asarray(r["shapley_values"]), phi_exact))
                # KernelSHAP at K
                ke = shap.KernelExplainer(_proba_matrix_fn(fn), bg)
                kp = ke.shap_values(x0, nsamples=K)
                r_ker.append(rmse(np.asarray(kp), phi_exact))
                # SamplingSHAP at ~K coalition evals
                mc_oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0))
                mc = monte_carlo_shapley(mc_oracle, x0, n_samples=max(10, K // (X.shape[1] + 1)),
                                         rng=np.random.RandomState(1301 + i))
                r_mc.append(rmse(np.asarray(mc["shapley_values"]), phi_exact))
            except Exception as e:
                print(f"  K={K} inst {i} FAILED: {str(e)[:100]}")
        rows.append({"K": K,
                     "gas_rmse": float(np.mean(r_gas)) if r_gas else np.nan,
                     "kernel_rmse": float(np.mean(r_ker)) if r_ker else np.nan,
                     "mc_rmse": float(np.mean(r_mc)) if r_mc else np.nan})
        print(f"  K={K}: gas={rows[-1]['gas_rmse']:.5f} kernel={rows[-1]['kernel_rmse']:.5f} mc={rows[-1]['mc_rmse']:.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(out(f"{name}_matched_budget.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #
# Width-vs-budget diagnostic (GAS only; fast - no exact/KernelSHAP/MC)
# --------------------------------------------------------------------------- #
def run_widths(name, X, feat_names, n_clusters, budgets, N=5, eps=0.05,
                    range_mode="spec"):
    global m_, feat_names_
    m_, feat_names_ = build_surrogate(X, n_clusters, feat_names)[:2]
    m, X_te, acc, f1 = build_surrogate(X, n_clusters, feat_names)
    print(f"[{name}] widths-vs-budget (surrogate acc={acc:.4f})")
    rows = []
    for K in budgets:
        widths, sfs, conv, stats = [], [], [], {}
        for i in range(N):
            x0 = X_te.iloc[i].values
            cid = i % n_clusters
            fn = make_proba_fn(m, feat_names, cid)
            bg = X.sample(64, random_state=1301 + i).values
            try:
                eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                                   rng=np.random.RandomState(1301 + i),
                                   config={"checkpoint_enabled": False, "cache_enabled": True,
                                           "persist_cache": False, "log_level": "NONE",
                                           "range_mode": range_mode})
                r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=K,
                                n_pilot=3, n_active_steps=10)
                W = np.asarray(r["certified_projected_widths"])
                widths.append(float(np.mean(W)))
                sfs.append(float(np.mean(np.abs(np.asarray(r["shapley_values"])) > W)))
                conv.append(bool(r["converged"]))
                stats[r["status"]] = stats.get(r["status"], 0) + 1
            except Exception as e:
                print(f"  K={K} inst {i} FAILED: {str(e)[:100]}")
        rows.append({"budget": K,
                     "mean_width": float(np.mean(widths)) if widths else np.nan,
                     "sign_certified_fraction": float(np.mean(sfs)) if sfs else np.nan,
                     "converged_fraction": float(np.mean(conv)) if conv else np.nan,
                     "status_counts": stats})
        print(f"  K={K}: mean_width={rows[-1]['mean_width']:.2f} "
              f"sign_cert={rows[-1]['sign_certified_fraction']:.4f} "
              f"converged={rows[-1]['converged_fraction']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out(f"{name}_width_vs_budget.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #
# Tier B (group-lag)
# --------------------------------------------------------------------------- #
def run_tier_b(eps, budget, N=10, range_mode="spec"):
    df = pd.read_csv(ROOT / "data" / "Beijing_MultiSite_AirQuality.csv")
    st = df["station"].value_counts().index[0]
    X = df[df["station"] == st][FEATURES].dropna().reset_index(drop=True)
    if len(X) > 40000:
        # contiguous *chronological* window — never sample() before lagging,
        # which would destroy temporal adjacency (audit finding).
        X = X.iloc[:40000].reset_index(drop=True)

    LAGS = (0, 1, 3, 6, 12, 24)

    def make_lagged(d, lags):
        cols = {}
        for var in d.columns:
            for lag in lags:
                cols[f"{var}_t-{lag}"] = d[var].shift(lag)
        o = pd.DataFrame(cols)
        return o.loc[o.dropna().index].reset_index(drop=True)

    X_lag = make_lagged(X, LAGS)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=1301, n_init=10).fit(Xs)
    lab = km.labels_
    # Alignment fix: make_lagged drops exactly the first max(LAGS) rows
    # (all shifts are >= 0, NaN only at the start), so X_lag[i] corresponds
    # to original time index i + max(LAGS).  Labels must be shifted by the
    # same offset — never lab[:len(X_lag)] (a 24-hour label shift).
    lag_target = np.asarray(lab)[max(LAGS): max(LAGS) + len(X_lag)]
    X_tr, X_te, y_tr, y_te = train_test_split(X_lag, lag_target, test_size=0.3,
                                              random_state=1301, stratify=lag_target)
    m = lgb.LGBMClassifier(objective="multiclass", num_class=4, random_state=1301,
                           n_estimators=200, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced")
    m.fit(X_tr, y_tr)
    print(f"[tierB] lagged surrogate acc={accuracy_score(y_te, m.predict(X_te)):.4f}")

    macro_names = FEATURES  # pollutant names
    groups = build_group_lags(n_vars=11, lags=LAGS)
    rows = []
    for i in range(N):
        cid = i % 4
        def fn(x):
            x = np.asarray(x, float).reshape(1, -1)
            return float(m.predict_proba(pd.DataFrame(x, columns=list(X_lag.columns)))[0, cid])
        x0 = X_te.iloc[i].values
        bg = X_lag.sample(32, random_state=1301 + i).values
        try:
            oracle, spec = group_lag_game(fn, bg, n_vars=11, lags=LAGS, output_bounds=(0.0, 1.0))
            values = exact_game_values(oracle, x0, spec.M)
            phi_exact_g = exact_shapley_from_values(values, spec.M)
            eng = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(1301 + i),
                               config={"checkpoint_enabled": False, "cache_enabled": True,
                                       "persist_cache": False, "log_level": "NONE",
                                       "range_mode": range_mode})
            r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=budget,
                            n_pilot=3, n_active_steps=10)
            phi_g = np.asarray(r["shapley_values"])
            W_g = np.asarray(r["certified_projected_widths"])
            err = np.abs(phi_g - phi_exact_g)
            rows.append({
                "instance": i, "cluster": cid,
                "rmse": rmse(phi_g, phi_exact_g),
                "sim_cov": float(np.all(err <= W_g)),
                "mar_cov": float(np.mean(err <= W_g)),
                "sign_cert": float(np.mean(np.abs(phi_g) > W_g)),
                "mean_width": float(np.mean(W_g)),
                "gas_evals": r["num_coalition_evals_this_call"],
                "status": r["status"],
            })
        except Exception as e:
            print(f"  tierB inst {i} FAILED: {str(e)[:120]}")
    d = pd.DataFrame(rows)
    sfx = "" if range_mode == "spec" else f"_range{range_mode}"
    d.to_csv(out(f"air_tierB{sfx}_instances.csv"), index=False)
    pd.DataFrame([{
        "n_instances": len(d), "rmse_mean": d["rmse"].mean(),
        "simultaneous_coverage_rate": d["sim_cov"].mean(),
        "sign_certified_fraction": d["sign_cert"].mean(),
        "mean_width": d["mean_width"].mean(), "gas_evals_mean": d["gas_evals"].mean(),
        "range_mode": range_mode,
    }]).to_csv(out(f"air_tierB{sfx}_summary.csv"), index=False)
    print(f"[tierB] summary: rmse={d['rmse'].mean():.5f} sim_cov={d['sim_cov'].mean():.2f} "
          f"sign_cert={d['sign_cert'].mean():.3f} mean_width={d['mean_width'].mean():.2f}")
    return d


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Paper-grade GAS-BayesSHAP experiments")
    ap.add_argument("--n", type=int, default=20, help="instances per dataset")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--only", choices=["wine", "air", "curves", "widths", "tierb", "all"], default="all")
    ap.add_argument("--range-mode", choices=["spec", "finite_population", "empirical_max"],
                    default="spec", help="certificate range mode (default spec)")
    args = ap.parse_args()
    rm = args.range_mode

    t0 = time.time()
    if args.only in ("wine", "all"):
        Xw, _ = load_wine()
        run_dataset("wine", Xw, WINE_FEATURES, n_clusters=2, eps=args.eps,
                    budget=args.budget, N=args.n, range_mode=rm)
    if args.only in ("air", "all"):
        Xa, _ = load_air_station(n_clusters=4)
        run_dataset("air", Xa, FEATURES, n_clusters=4, eps=args.eps,
                    budget=args.budget, N=args.n, range_mode=rm)
    if args.only in ("curves", "all"):
        Xw, _ = load_wine()
        run_curves("wine", Xw, WINE_FEATURES, n_clusters=2, Ks=[128, 256, 512, 1024, 2048], N=8,
                   range_mode=rm)
        Xa, _ = load_air_station(n_clusters=4)
        run_curves("air", Xa, FEATURES, n_clusters=4, Ks=[128, 256, 512, 1024, 2048], N=8,
                   range_mode=rm)
    if args.only in ("widths", "all"):
        Xw, _ = load_wine()
        run_widths("wine", Xw, WINE_FEATURES, n_clusters=2,
                   budgets=[500, 1000, 2000, 4000, 8000], N=5, eps=args.eps, range_mode=rm)
        Xa, _ = load_air_station(n_clusters=4)
        run_widths("air", Xa, FEATURES, n_clusters=4,
                   budgets=[500, 1000, 2000, 4000, 8000], N=5, eps=args.eps, range_mode=rm)
    if args.only in ("tierb", "all"):
        # N is user-controlled (--n); previously capped at 10 instances.
        run_tier_b(eps=args.eps, budget=args.budget, N=args.n, range_mode=rm)

    # copy to main_results with prefixes
    MAIN.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in sorted(OUT.glob("*.csv")):
        shutil.copy2(f, MAIN / f"paper_{f.name}")
    print(f"\ndone in {time.time()-t0:.0f}s; results in results/paper_experiments/ "
          f"and main_results/paper_*.csv")


if __name__ == "__main__":
    main()
