"""SOTA-style baselines: OddSHAP (log-odds) + ShaplEIG (GP quadrature)."""

import numpy as np

from gas_bayesshap.benchmarking.sota_baselines import (
    gp_quadrature_shapley,
    odd_shapley_exact,
)
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.game.oracle import CoalitionOracle


def _make_game(M=5, B=8, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(M)
    bg = rng.randn(B, M)

    def model(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(M)))

    oracle = CoalitionOracle(model, bg, output_bounds=(0.0, 1.0))
    x = rng.randn(M)
    return oracle, x, w


def test_odd_shapley_exact_shape_and_different():
    oracle, x, _ = _make_game()
    M = x.shape[0]
    odd = odd_shapley_exact(oracle, x, M)
    assert odd.shape == (M,)
    # exact game values
    values = exact_game_values(oracle, x, M)
    phi_v = exact_shapley_from_values(values, M)
    # log-odds Shapley differs from probability Shapley (nonlinear transform)
    assert not np.allclose(odd, phi_v, atol=1e-4)


def test_odd_shapley_linear_game_matches():
    """For a linear-in-logit model the log-odds game is exact-linear, so
    log-odds Shapley equals the weight vector up to scale."""
    M = 5
    w = np.array([1.0, -2.0, 0.5, 3.0, -1.0])

    def model(x):
        z = np.dot(x, w)
        return 1.0 / (1.0 + np.exp(-z))

    oracle = CoalitionOracle(model, np.zeros((8, M)), output_bounds=(0.0, 1.0))
    x = np.ones(M)
    odd = odd_shapley_exact(oracle, x, M)
    # logit(v(S)) = w . (x_S mean) approx -> Shapley proportional to w
    rel = odd / w
    assert np.allclose(rel, rel[0], atol=0.15)  # proportional


def test_gp_quadrature_shapley_shape():
    oracle, x, _ = _make_game(M=5, B=8)
    M = x.shape[0]
    # design: all singleton-ish coalitions + empty + full
    design = [np.zeros(M, dtype=bool), np.ones(M, dtype=bool)]
    for i in range(M):
        m = np.zeros(M, dtype=bool); m[i] = True
        design.append(m)
    y = np.array([oracle.evaluate(x, m) for m in design])
    phi, std = gp_quadrature_shapley(oracle, x, M, design, y)
    assert phi.shape == (M,)
    assert std is not None and std.shape == (M,)
    assert np.all(std >= 0)


def test_gp_quadrature_recovers_additive():
    """GP fitted on enough coalitions should approximate the game; for a
    fully observed 2^M design it recovers the exact Shapley."""
    M = 4
    w = np.array([1.0, 2.0, -1.0, 0.5])

    def model(x):
        return float(np.dot(x, w))

    oracle = CoalitionOracle(model, np.zeros((4, M)), output_bounds=(-6.0, 6.0))
    x = np.ones(M)
    from gas_bayesshap.game.subsets import all_subsets
    design = list(all_subsets(M))
    y = np.array([oracle.evaluate(x, m) for m in design])
    phi, _ = gp_quadrature_shapley(oracle, x, M, design, y)
    values = exact_game_values(oracle, x, M)
    phi_exact = exact_shapley_from_values(values, M)
    assert np.allclose(phi, phi_exact, atol=0.3)
