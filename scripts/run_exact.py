#!/usr/bin/env python
"""Exact Shapley reference computation for small M (spec section 12 / 40).

Usage:
    python scripts/run_exact.py --game membership --M 5 [--seed 0] [--out results/runs/exact.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gas_bayesshap.benchmarking.exact import run_exact_benchmark
from gas_bayesshap.utils.serialization import write_json_atomic


def synthetic_model_factory(game: str, M: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = rng.randn(M)
    if game == "membership":
        def model(x):
            z = np.dot(x, weights) / max(1.0, np.sqrt(M))
            return 1.0 / (1.0 + np.exp(-z))  # in (0,1)
        bounds = (0.0, 1.0)
    elif game == "contrastive":
        w2 = rng.randn(M)

        def model(x):
            g1 = 1.0 / (1.0 + np.exp(-np.dot(x, weights) / np.sqrt(M)))
            g2 = 1.0 / (1.0 + np.exp(-np.dot(x, w2) / np.sqrt(M)))
            return float(g1 - g2)  # in (-1, 1)
        bounds = (-1.0, 1.0)
    else:
        def model(x):
            return float(np.dot(x, weights) / np.sqrt(M))
        bounds = None
    return model, bounds


def main() -> int:
    p = argparse.ArgumentParser(description="Exact Shapley reference (small M)")
    p.add_argument("--game", default="membership")
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.M > 6:
        print(f"refusing: M={args.M} > 6 makes 2^M enumeration impractical "
              f"(2^{args.M} = {2 ** args.M} coalitions)")
        return 1

    rng = np.random.RandomState(args.seed)
    model, bounds = synthetic_model_factory(args.game, args.M, args.seed)
    background = rng.randn(args.B, args.M)
    x = rng.randn(args.M)
    x = np.clip(x, -2, 2)

    res = run_exact_benchmark(model, background, x, output_bounds=bounds, M=args.M)
    print(f"game={args.game} M={args.M}")
    print("exact shapley:", np.round(res["shapley_values"], 6))
    print(f"delta_total          = {res['delta_total']:.6f}")
    print(f"sum(phi)             = {np.sum(res['shapley_values']):.6f}")
    print(f"efficiency_error     = {res['efficiency_error']:.3e}")
    print(f"coalition_evals      = {res['num_coalition_evals']} (2^{args.M} = {2 ** args.M})")

    if args.out:
        write_json_atomic(args.out, {
            "game": args.game, "M": args.M, "seed": args.seed,
            "shapley_values": res["shapley_values"].tolist(),
            "delta_total": res["delta_total"],
            "efficiency_error": res["efficiency_error"],
            "num_coalition_evals": res["num_coalition_evals"],
        })
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
