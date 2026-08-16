#!/usr/bin/env python
"""Coverage validation (spec section 45): repeated synthetic trials.

Usage:
    python scripts/coverage_validation.py --trials 50 --M 3 --epsilon 1.5 --max-budget 300

Reports: n_trials, finite-width rate, empirical coverage, coverage given
finite intervals, mean/median/max width, oracle-query cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.benchmarking.metrics import coverage_report

ENGINE_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}


def main() -> int:
    p = argparse.ArgumentParser(description="GAS-BayesSHAP coverage validation")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--epsilon", type=float, default=1.5)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--max-budget", type=int, default=300)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--range-mode", type=str, default="spec",
                   choices=["spec", "finite_population", "empirical_max"])
    args = p.parse_args()

    # exact ground truth for the calibration game (M=3)
    def model_cal(x):
        return float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])

    from gas_bayesshap.game.brute_force import brute_force_shapley
    phi_true = brute_force_shapley(
        lambda S: model_cal(np.asarray(S, dtype=float)), args.M
    )

    phis, widths, costs = [], [], []
    d1s, levels = [], []
    for trial in range(args.trials):
        eng = GASBayesSHAP(model_cal, np.zeros((3, args.M)),
                           output_bounds=(-2.0, 5.0),
                           rng=np.random.RandomState(args.seed0 + trial),
                           config={**ENGINE_CONFIG, "range_mode": args.range_mode})
        r = eng.explain(np.ones(args.M), epsilon=args.epsilon, delta=args.delta,
                        max_budget=args.max_budget)
        phis.append(r["shapley_values"])
        widths.append(r["certified_projected_widths"])
        costs.append(r["num_coalition_evals"])
        if args.range_mode == "finite_population":
            d1 = r.get("finite_population_delta1")
            d1s.append(d1 if d1 is not None else 0.0)
            lvl = r.get("finite_population_coverage_level")
            levels.append(lvl if lvl is not None else 1.0)

    rep = coverage_report(phis, widths, phi_true)
    print("=== Coverage validation ===")
    for k, v in rep.items():
        print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    print(f"  {'oracle_query_cost (mean)':24s}: {np.mean(costs):.1f}")
    print(f"  {'oracle_query_cost (max)':24s}: {np.max(costs)}")
    if args.range_mode == "finite_population":
        print(f"  {'finite_population_delta1 (mean)':24s}: {np.mean(d1s):.4f}")
        print(f"  {'realised coverage level (mean)':24s}: {np.mean(levels):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
