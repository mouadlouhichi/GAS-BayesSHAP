#!/usr/bin/env python
"""Mathematical validation: Lemma D & Lemma E vs independent brute force.

Usage:
    python scripts/validate_math.py [--M 6] [--sigma0 1.0] [--lengthscale 1.5]

Runs (spec test tiers T1/T2 plus the M=1 sanity check):
  * Lemma D  O(M^2)  == brute-force 2^M enumeration (M=4 and M=1)
  * Lemma E  O(M^3)  == brute-force 4^M double enumeration for M in 2..6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gas_bayesshap.game.brute_force import (
    brute_force_cross_covariance,
    brute_force_prior_covariance,
)
from gas_bayesshap.kernels.covariance import lemma_D_cross_cov, lemma_E_prior_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel


def check_lemma_D(kernel, M: int, atol: float = 1e-10) -> float:
    rng = np.random.RandomState(1234)
    worst = 0.0
    trials = [np.zeros(M, dtype=bool), np.ones(M, dtype=bool)]
    for _ in range(4):
        p = np.zeros(M, dtype=bool)
        p[rng.permutation(M)[: rng.randint(0, M + 1)]] = True
        trials.append(p)
    for S_j in trials:
        analytic = lemma_D_cross_cov(kernel, S_j, M)
        brute = brute_force_cross_covariance(kernel, S_j, M)
        d = float(np.max(np.abs(analytic - brute)))
        assert d <= atol, f"Lemma D mismatch (M={M}, |S|={S_j.sum()}): max|diff|={d:.3e}"
        worst = max(worst, d)
    return worst


def check_lemma_E(kernel, M: int, atol: float = 1e-10) -> float:
    analytic = lemma_E_prior_cov(kernel, M)
    brute = brute_force_prior_covariance(kernel, M)
    diff = float(np.max(np.abs(analytic - brute)))
    assert diff <= atol, f"Lemma E mismatch at M={M}: max|diff|={diff:.3e}"
    # explicit M=2 off-diagonal check (spec section 12)
    if M == 2:
        assert abs(analytic[0, 1] - brute[0, 1]) <= atol, "M=2 off-diagonal covariance mismatch"
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description="GAS-BayesSHAP mathematical validation")
    parser.add_argument("--max-M", type=int, default=6)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=1.5)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    print("=== GAS-BayesSHAP mathematical validation ===")
    # M=1 sanity check (Lemma D)
    k1 = ExponentialHammingKernel(sigma0=args.sigma0, lengthscale=args.lengthscale)
    d1 = check_lemma_D(k1, 1, args.atol)
    print(f"  ✓ Lemma D M=1 sanity: max|diff| = {d1:.3e}")

    for M in (4,):
        k = ExponentialHammingKernel(sigma0=args.sigma0, lengthscale=args.lengthscale)
        d = check_lemma_D(k, M, args.atol)
        print(f"  ✓ Lemma D M={M}: max|diff| = {d:.3e}")

    for M in range(2, args.max_M + 1):
        k = ExponentialHammingKernel(sigma0=args.sigma0, lengthscale=args.lengthscale)
        d = check_lemma_E(k, M, args.atol)
        print(f"  ✓ Lemma E M={M}: max|diff| = {d:.3e}")

    print("ALL MATHEMATICAL VALIDATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
