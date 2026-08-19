#!/usr/bin/env python
"""Multi-instance NOMINAL certification probe (audit blockers 1+2).

The single biggest scientific caveat in the latest audit: on the standard
budget (K=3000) the finite-population intervals are width-tight but NOT
nominal 1-delta certificates (`fraction_at_nominal_level=0.0`), and no
feature is sign-certified.  The wine frontier probe already showed that at
K=200,000 the coupon thresholds close and the run returns status CERTIFIED
with `certificate_at_nominal_level=True` and a sign-certified feature.

This script generalises that to MULTIPLE real instances per dataset (wine
+ air) so the paper can claim: "at the characterised frontier cost, nominal
1-delta certification and sign-certified features are achieved on N real
instances, with every certified sign validated against exact ground truth."

For each instance:
  - exact Shapley ground truth (2^M enumeration);
  - GAS-BayesSHAP, range_mode=finite_population, max_budget=K (default 2e5);
  - records status / converged / certificate_is_rigorous /
    certificate_at_nominal_level / realised level / delta1 /
    sign-certified count / signs_match_exact / min certified margin /
    widths / evals.

Usage:
    python scripts/probe_nominal_certification.py --n 3 --budget 200000
Outputs -> results/paper_experiments/nominal_certification_{wine,air}.csv
           + main_results/paper_nominal_certification_{wine,air}.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # reuse loaders from the runner

import numpy as np
import pandas as pd

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse

from run_paper_experiments import (  # noqa: E402
    load_wine,
    load_air_station,
    build_surrogate,
    make_proba_fn,
    WINE_FEATURES,
    FEATURES as AIR_FEATURES,
)

OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"


def run_instance(name, X, feat_names, n_clusters, i, K, eps, m, X_te):
    cid = i % n_clusters
    fn = make_proba_fn(m, feat_names, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=1301 + i).values
    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0))
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    t0 = time.time()
    eng = GASBayesSHAP(fn, bg, output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(1301 + i),
                       config={"checkpoint_enabled": False, "cache_enabled": True,
                               "persist_cache": False, "log_level": "NONE",
                               "range_mode": "finite_population"})
    r = eng.explain(x0, epsilon=eps, delta=0.05, max_budget=K,
                    n_pilot=3, n_active_steps=10)
    dt = time.time() - t0
    W = np.asarray(r["certified_projected_widths"])
    phi = np.asarray(r["shapley_values"])
    sc = np.abs(phi) > W
    sc_ok = int(np.all(np.sign(phi[sc]) == np.sign(phi_exact[sc]))) if sc.any() else 1
    margin = float(np.min(np.abs(phi_exact[sc]) - W[sc])) if sc.any() else float("nan")
    unique = int(r["num_coalition_evals_this_call"])   # cache misses = unique coalitions
    attempted = int(r.get("extra", {}).get("stage2_attempted_total")
                    if isinstance(r.get("extra"), dict) else r.get("stage2_attempted_total", unique))
    if attempted == 0:
        attempted = unique
    cache_hit_rate = float(r.get("extra", {}).get("cache_hit_rate", 0.0)) \
        if isinstance(r.get("extra"), dict) else float(r.get("cache_hit_rate", 0.0))
    row = {
        "dataset": name, "instance": i, "cluster": cid, "K": K,
        "status": r["status"], "converged": bool(r["converged"]),
        "certificate_is_rigorous": bool(r.get("certificate_is_rigorous", False)),
        "certificate_at_nominal_level": bool(r.get("certificate_at_nominal_level", False)),
        "realised_coverage_level": r.get("finite_population_coverage_level"),
        "delta1_coupon": r.get("finite_population_delta1"),
        "rmse_vs_exact": rmse(phi, phi_exact),
        "sim_cov": float(np.all(np.abs(phi - phi_exact) <= W)),
        "n_sign_certified": int(sc.sum()),
        "signs_match_exact": sc_ok,
        "min_certified_margin": margin,
        "mean_width": float(np.mean(W)),
        "max_width": float(np.max(W)),
        "unique_coalition_evals": unique,
        "attempted_stage2_draws": attempted,
        "unique_fraction_of_power_set": float(unique) / float(2 ** X.shape[1]),
        "cache_hit_rate": cache_hit_rate,
        "coalition_evals": unique,  # alias kept for backward compat
        "elapsed_s": round(dt, 1),
    }
    print(f"  {name} inst {i}: {r['status']} conv={row['converged']} "
          f"at_nominal={row['certificate_at_nominal_level']} "
          f"level={row['realised_coverage_level']} "
          f"sign_cert={row['n_sign_certified']} signs_ok={sc_ok} "
          f"rmse={row['rmse_vs_exact']:.5f} ({dt:.0f}s)")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="instances per dataset")
    ap.add_argument("--budget", type=int, default=200000)
    ap.add_argument("--eps", type=float, default=0.02)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)
    all_rows = []

    Xw, _ = load_wine()
    print(f"[wine] {len(Xw)} rows")
    mw, Xte_w, _, _ = build_surrogate(Xw, 2, WINE_FEATURES)
    for i in range(args.n):
        all_rows.append(run_instance("wine", Xw, WINE_FEATURES, 2, i,
                                     args.budget, args.eps, mw, Xte_w))

    Xa, _ = load_air_station(n_clusters=4)
    print(f"[air] {len(Xa)} rows")
    ma, Xte_a, _, _ = build_surrogate(Xa, 4, AIR_FEATURES)
    for i in range(args.n):
        all_rows.append(run_instance("air", Xa, AIR_FEATURES, 4, i,
                                     args.budget, args.eps, ma, Xte_a))

    d = pd.DataFrame(all_rows)
    for ds in ("wine", "air"):
        sub = d[d.dataset == ds]
        sub.to_csv(OUT / f"nominal_certification_{ds}.csv", index=False)
        import shutil
        shutil.copy2(OUT / f"nominal_certification_{ds}.csv",
                     MAIN / f"paper_nominal_certification_{ds}.csv")

    # compact summary
    print("\n=== Nominal-certification summary ===")
    g = d.groupby("dataset").agg(
        n=("instance", "count"),
        n_at_nominal=("certificate_at_nominal_level", "sum"),
        n_converged=("converged", "sum"),
        n_with_sign_cert=("n_sign_certified", lambda s: int((s > 0).sum())),
        all_signs_validated=("signs_match_exact", "all"),
        mean_realised_level=("realised_coverage_level", "mean"),
        mean_width=("mean_width", "mean"),
        mean_evals=("coalition_evals", "mean"),
    )
    print(g.to_string())
    print(f"\ndone in {time.time()-t0:.0f}s; artifacts in main_results/paper_nominal_certification_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
