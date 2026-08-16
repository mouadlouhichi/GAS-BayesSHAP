#!/usr/bin/env python
"""Adversarial stress test of the finite-population certificate.

The impossibility remark in the paper (Remark on the observed-max range)
shows the *generic* empirical-max range is NOT distribution-free: a rare
extreme value can stay unobserved.  The finite-population mode charges
exactly this event to the coupon-collector budget
delta1 = sum_{i,s} (1 - 1/C(M-1,s))^n_{i,s}, and reports the realised
coverage level 1 - delta2 - delta1 plus the nominal-level flag.

This script stress-tests that accounting on synthetic games with a PLANTED
rare-extreme coalition:

  * M=3  (population C(2,1)=2 per interior cell): the coupon collector
          closes fast -> the certificate should reach the nominal level and
          empirical coverage should be >= 1 - delta (within MC slack).
  * M=6  (population C(5,2)=10 per cell): the extreme pair is rare; the
          certificate must honestly report realised level < nominal and set
          certificate_at_nominal_level=False while the coupon is open.

No coverage is claimed while the coupon is open — the point is that the
flag, the realised level and delta1 are all honest under an adversarial
residual structure.

Usage:
    python scripts/stress_finite_population.py --trials 200
Outputs -> results/paper_experiments/stress_finite_population.json + copy
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

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.brute_force import brute_force_shapley

ENGINE_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}


def game_m3(x):
    """M=3 calibration game + planted rare extreme on one coalition."""
    base = x[0] + 2.0 * x[1] - x[2] + x[0] * x[1]
    spike = 1.5 * float(np.all(x == np.array([1.0, 1.0, 0.0])))
    return base + spike


def game_m6(x):
    """M=6 additive game + planted rare extreme coalition of size 2."""
    w = np.array([0.6, -0.9, 0.4, -0.3, 0.8, -0.2])
    base = float(np.dot(x, w))
    spike = 2.0 * float(np.all(x == np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])))
    return base + spike


def run_stress(game, M, bounds, trials, max_budget, seed0, delta=0.05):
    cfg = {**ENGINE_CONFIG, "range_mode": "finite_population"}
    phi_true = brute_force_shapley(
        lambda S: game(np.asarray(S, dtype=float)), M)
    cover_all, cover_nominal = [], []
    d1s, levels, at_nom = [], [], []
    finite = 0
    for t in range(trials):
        eng = GASBayesSHAP(game, np.zeros((3, M)), output_bounds=bounds,
                           rng=np.random.RandomState(seed0 + t), config=cfg)
        r = eng.explain(np.ones(M), epsilon=2.5, delta=delta, max_budget=max_budget,
                        n_pilot=3)
        W = np.asarray(r["certified_projected_widths"])
        if not np.all(np.isfinite(W)):
            continue
        finite += 1
        phi = np.asarray(r["shapley_values"])
        covered = bool(np.all(np.abs(phi - phi_true) <= W))
        cover_all.append(covered)
        d1 = r.get("finite_population_delta1")
        d1 = d1 if d1 is not None else 0.0
        lvl = r.get("finite_population_coverage_level")
        lvl = lvl if lvl is not None else 1.0
        at = bool(r.get("finite_population_at_level_delta", False))
        d1s.append(d1); levels.append(lvl); at_nom.append(at)
        cover_nominal.append(covered if at else True)  # only require coverage when nominal
    n = len(cover_all)
    rep = {
        "M": M, "trials": trials, "finite_width_rate": finite / trials,
        "empirical_coverage_all": float(np.mean(cover_all)) if n else float("nan"),
        "empirical_coverage_given_nominal": float(np.mean(cover_nominal)) if n else float("nan"),
        "fraction_at_nominal_level": float(np.mean(at_nom)) if n else float("nan"),
        "mean_delta1": float(np.mean(d1s)) if d1s else float("nan"),
        "mean_realised_level": float(np.mean(levels)) if levels else float("nan"),
        "min_realised_level": float(np.min(levels)) if levels else float("nan"),
    }
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=9100)
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== finite-population adversarial stress test ({args.trials} trials) ===")
    reps = []
    for M, game, bounds, budget in ((3, game_m3, (-2.0, 6.0), 120),
                                    (6, game_m6, (-2.0, 5.0), 400)):
        rep = run_stress(game, M, bounds, args.trials, budget, args.seed0)
        reps.append(rep)
        print(f"M={M}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                    else f"{k}={v}" for k, v in rep.items()))
    payload = {"trials": args.trials, "delta": 0.05, "results": reps}
    for dest in ("results/paper_experiments", "main_results"):
        p = Path(ROOT) / dest
        p.mkdir(parents=True, exist_ok=True)
        (p / "stress_finite_population.json").write_text(json.dumps(payload, indent=1))
    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"results/paper_experiments/stress_finite_population.json + "
          f"main_results/paper_stress_finite_population.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
