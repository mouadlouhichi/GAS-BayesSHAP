"""Domain games (spec section 3): membership, contrastive, archetype,
silhouette and group-lag."""

import numpy as np
import pytest

from gas_bayesshap.game.domain_games import (
    R_DELTA_CONTRASTIVE,
    R_DELTA_MEMBERSHIP,
    archetype_game,
    build_group_lags,
    contrastive_game,
    group_lag_game,
    group_mask_to_feature_mask,
    membership_game,
    silhouette_game,
)
from gas_bayesshap.game.oracle import CoalitionOracle

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def _membership_model(seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(4)

    def g_c(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w)))
    return g_c


def test_membership_game_bounds():
    g_c = _membership_model()
    oracle, spec = membership_game(g_c, np.zeros((6, 4)))
    assert spec.output_bounds == (0.0, 1.0)
    assert spec.r_delta_res == R_DELTA_MEMBERSHIP == 4.0
    assert oracle.output_bounds == (0.0, 1.0)
    x = np.ones(4)
    v = oracle.evaluate(x, np.array([True, False, True, False]))
    assert 0.0 <= v <= 1.0


def test_contrastive_game_bounds():
    rng = np.random.RandomState(1)
    w1, w2 = rng.randn(4), rng.randn(4)

    def g_c(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w1)))

    def g_cp(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w2)))

    oracle, spec = contrastive_game(g_c, g_cp, np.zeros((6, 4)))
    assert spec.r_delta_res == R_DELTA_CONTRASTIVE == 8.0
    assert spec.output_bounds == (-1.0, 1.0)
    v = oracle.evaluate(np.ones(4), np.ones(4, dtype=bool))
    assert -1.0 <= v <= 1.0


def test_archetype_game():
    g_c = _membership_model(2)
    archetypes = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    oracle, spec = archetype_game(g_c, archetypes, np.zeros((4, 4)))
    assert spec.name == "archetype"
    assert spec.r_delta_res == 4.0
    v = oracle.evaluate(np.ones(4), np.array([True, True, False, False]))
    assert 0.0 <= v <= 1.0
    # query accounting: archetype hybrid costs |archetypes| * B model passes
    n0 = oracle.total_model_evals
    oracle.evaluate(np.ones(4), np.array([True, False, True, False]))
    assert oracle.total_model_evals - n0 == 2 * 4


def test_silhouette_game():
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.randn(20, 4) + np.array([3.0, 3.0, 0.0, 0.0]),
        rng.randn(20, 4) - np.array([3.0, 3.0, 0.0, 0.0]),
    ])
    oracle, spec = silhouette_game(X, n_clusters=2, random_state=0)
    assert spec.r_delta_res == R_DELTA_CONTRASTIVE  # 8.0
    v_empty = oracle.evaluate(None, np.zeros(4, dtype=bool))
    assert v_empty == 0.0  # convention v_sil(empty) = 0
    v_full = oracle.evaluate(None, np.ones(4, dtype=bool))
    assert -1.0 <= v_full <= 1.0


def test_group_lag_masks():
    groups = build_group_lags(n_vars=2, lags=(0, 1, 3))
    assert len(groups) == 2
    assert sum(len(g) for g in groups) == 6
    feat = group_mask_to_feature_mask(np.array([True, False]), groups, M=6)
    assert feat.tolist() == [True, True, True, False, False, False]
    feat2 = group_mask_to_feature_mask(np.array([False, True]), groups, M=6)
    assert feat2.tolist() == [False, False, False, True, True, True]


def test_group_lag_game_exact_ground_truth():
    """M_group=11 style game: exact ground truth at 2^11 via brute force."""
    n_vars, lags = 3, (0, 1)
    M_feat = n_vars * len(lags)
    rng = np.random.RandomState(0)
    w = rng.randn(M_feat)

    def model(x):
        return float(np.dot(x, w))

    background = rng.randn(8, M_feat)
    oracle, spec = group_lag_game(model, background, n_vars=n_vars, lags=lags,
                                  output_bounds=(-10.0, 10.0))
    assert spec.M == n_vars
    # exact ground truth via brute-force at macro level (2^3 = 8 coalitions)
    from gas_bayesshap.benchmarking.exact import exact_shapley_for_oracle
    x = np.ones(M_feat)
    exact = exact_shapley_for_oracle(oracle, x, spec.M)
    assert exact["num_coalition_evals"] == 2 ** spec.M
    assert exact["efficiency_error"] < 1e-9


def test_group_lag_11_groups_exact():
    """The spec's M_group=11 benchmark shape (2^11 = 2048 coalitions)."""
    n_vars, lags = 11, (0, 1, 3, 6, 12, 24)
    M_feat = n_vars * len(lags)
    assert M_feat == 66
    groups = build_group_lags(n_vars, lags)
    assert len(groups) == 11
    rng = np.random.RandomState(0)
    w = rng.randn(M_feat)

    def model(x):
        return float(np.dot(x, w) / np.sqrt(M_feat))

    oracle, spec = group_lag_game(model, rng.randn(8, M_feat), n_vars=n_vars, lags=lags,
                                  output_bounds=(-2.0, 2.0))
    assert spec.M == 11
    assert spec.extra["M_feat"] == 66
    x = np.ones(M_feat)
    from gas_bayesshap.benchmarking.exact import exact_shapley_for_oracle
    exact = exact_shapley_for_oracle(oracle, x, 11)
    assert exact["num_coalition_evals"] == 2048
    assert exact["efficiency_error"] < 1e-9
