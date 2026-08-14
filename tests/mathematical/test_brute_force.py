"""Brute-force reference engine self-consistency."""

import numpy as np

from gas_bayesshap.game.brute_force import (
    brute_force_shapley,
    exact_game_values,
    exact_shapley_from_values,
)
from gas_bayesshap.game.oracle import CoalitionOracle


def test_brute_force_shapley_linear_game():
    M = 4
    w = np.array([1.0, 2.0, 3.0, -1.0])
    phi = brute_force_shapley(lambda S: float(np.dot(S, w)), M)
    assert np.allclose(phi, w, atol=1e-12)


def test_brute_force_efficiency():
    M = 4
    rng = np.random.RandomState(0)
    A = rng.randn(M, M)
    b = rng.randn(M)

    def v(S):
        x = np.asarray(S, dtype=float)
        return float(x @ A @ x + b @ x)

    phi = brute_force_shapley(v, M)
    delta = v(np.ones(M)) - v(np.zeros(M))
    assert abs(np.sum(phi) - delta) < 1e-9


def test_exact_values_through_oracle_counts_queries():
    M = 3
    w = np.array([1.0, -1.0, 0.5])

    def model(x):
        return float(np.dot(x, w))

    oracle = CoalitionOracle(model, np.zeros((2, M)))
    x = np.ones(M)
    values = exact_game_values(oracle, x, M)
    assert len(values) == 2 ** M
    phi = exact_shapley_from_values(values, M)
    assert np.allclose(phi, w, atol=1e-12)
    # exactly 2^M coalition evaluations (no duplicates, no cache)
    assert oracle.total_coalition_evals == 2 ** M
