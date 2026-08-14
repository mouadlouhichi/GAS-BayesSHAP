#!/usr/bin/env python
"""Benchmark: exact vs Monte-Carlo vs GP-only vs full GAS-BayesSHAP
(spec section 46).

Usage:
    python scripts/benchmark.py --game membership --M 6 --budget 300 --trials 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.benchmarking.exact import run_exact_benchmark
from gas_bayesshap.benchmarking.metrics import benchmark_metrics
from gas_bayesshap.benchmarking.monte_carlo import monte_carlo_shapley
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.utils.serialization import write_json_atomic


def synthetic_model(M: int, seed: int):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)

    def model(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(M)))
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="GAS-BayesSHAP benchmark")
    p.add_argument("--game", default="membership")
    p.add_argument("--M", type=int, default=6)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    M, B = args.M, args.B
    model = synthetic_model(M, args.seed)
    rng = np.random.RandomState(args.seed)
    background = rng.randn(B, M)
    x = np.clip(rng.randn(M), -2, 2)

    print(f"benchmark: game={args.game} M={M} B={B} budget={args.budget} "
          f"epsilon={args.epsilon} trials={args.trials}")

    # 1. exact ground truth
    t0 = time.time()
    exact = run_exact_benchmark(model, background, x, output_bounds=(0.0, 1.0), M=M)
    phi_true = exact["shapley_values"]
    t_exact = time.time() - t0
    print(f"exact: phi={np.round(phi_true, 4)}  "
          f"queries={exact['num_coalition_evals']}  t={t_exact:.2f}s")

    rows = {}

    # 2. plain Monte Carlo
    oracle_mc = CoalitionOracle(model, background, output_bounds=(0.0, 1.0))
    mc = monte_carlo_shapley(oracle_mc, x, n_samples=args.budget // M)
    rows["monte_carlo"] = {
        **benchmark_metrics(mc["shapley_values"], phi_true),
        "num_coalition_evals": mc["num_coalition_evals"],
        "num_model_evals": mc["num_model_evals"],
    }
    print(f"monte_carlo: MAE={rows['monte_carlo']['MAE']:.4f} "
          f"RMSE={rows['monte_carlo']['RMSE']:.4f} "
          f"queries={mc['num_coalition_evals']}")

    # 3. GP-only (Module A)
    t0 = time.time()
    eng_gp = GASBayesSHAP(model, background, output_bounds=(0.0, 1.0),
                          rng=np.random.RandomState(args.seed),
                          config={"checkpoint_enabled": False, "cache_enabled": False,
                                  "log_level": "NONE"})
    gp_only = eng_gp.explain_stage1_only(x)
    rows["gp_only"] = {
        **benchmark_metrics(gp_only["shapley_values"], phi_true),
        "num_coalition_evals": gp_only["num_coalition_evals"],
        "num_model_evals": gp_only["num_model_evals"],
    }
    print(f"gp_only: MAE={rows['gp_only']['MAE']:.4f} RMSE={rows['gp_only']['RMSE']:.4f} "
          f"queries={gp_only['num_coalition_evals']} t={time.time()-t0:.2f}s")

    # 4. full GAS-BayesSHAP (several trials, report mean metrics + coverage)
    phis, widths = [], []
    t0 = time.time()
    for t in range(args.trials):
        eng = GASBayesSHAP(model, background, output_bounds=(0.0, 1.0),
                           rng=np.random.RandomState(args.seed * 1000 + t),
                           config={"checkpoint_enabled": False, "cache_enabled": False,
                                   "log_level": "NONE"})
        res = eng.explain(x, epsilon=args.epsilon, delta=0.05,
                          max_budget=args.budget, n_pilot=3, n_active_steps=10)
        phis.append(res["shapley_values"])
        widths.append(res["certified_projected_widths"])
    t_gas = (time.time() - t0) / args.trials

    from gas_bayesshap.benchmarking.metrics import coverage_report, mae, rmse, max_abs_error
    mean_phi = np.mean(np.vstack(phis), axis=0)
    cov = coverage_report(phis, widths, phi_true)
    rows["gas_bayesshap"] = {
        "MAE": mae(mean_phi, phi_true),
        "RMSE": rmse(mean_phi, phi_true),
        "max_error": max_abs_error(mean_phi, phi_true),
        "num_coalition_evals": int(res["num_coalition_evals"]),
        "num_model_evals": int(res["num_model_evals"]),
        "finite_width_rate": cov["finite_width_rate"],
        "empirical_coverage": cov["empirical_coverage"],
        "mean_width": cov["mean_width"],
    }
    print(f"gas_bayesshap: MAE={rows['gas_bayesshap']['MAE']:.4f} "
          f"queries={res['num_coalition_evals']} coverage={cov['empirical_coverage']:.2f} "
          f"t/run={t_gas:.2f}s")

    # query-efficiency factor vs exact
    rows["query_efficiency_vs_exact"] = {
        "exact_queries": exact["num_coalition_evals"],
        "gas_queries": rows["gas_bayesshap"]["num_coalition_evals"],
        "reduction_factor": exact["num_coalition_evals"] / max(1, rows["gas_bayesshap"]["num_coalition_evals"]),
    }
    print(f"query reduction vs exact: "
          f"{rows['query_efficiency_vs_exact']['reduction_factor']:.1f}x")

    if args.out:
        write_json_atomic(args.out, {"M": M, "budget": args.budget, "rows": rows})
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
