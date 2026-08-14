"""Group-lag game input validation (audit #2 Medium)."""

import numpy as np
import pytest

from gas_bayesshap.game.domain_games import build_group_lags, group_lag_game


def test_group_lag_basic():
    groups = build_group_lags(n_vars=2, lags=(0, 1, 3))
    assert len(groups) == 2
    assert groups[0] == (0, 1, 2)
    assert groups[1] == (3, 4, 5)
    flat = [f for g in groups for f in g]
    assert len(flat) == len(set(flat))  # disjoint


def test_group_lag_rejects_empty_lags():
    with pytest.raises(ValueError, match="non-empty"):
        build_group_lags(n_vars=2, lags=())


def test_group_lag_rejects_negative_lags():
    with pytest.raises(ValueError, match="non-negative"):
        build_group_lags(n_vars=2, lags=(0, -1))


def test_group_lag_rejects_duplicate_lags():
    with pytest.raises(ValueError, match="distinct"):
        build_group_lags(n_vars=2, lags=(0, 1, 1))


def test_group_lag_rejects_bad_n_vars():
    with pytest.raises(ValueError, match="positive integer"):
        build_group_lags(n_vars=0, lags=(0, 1))


def test_group_lag_game_validates_feature_dim():
    rng = np.random.RandomState(0)

    def model(x):
        return float(np.sum(x))

    with pytest.raises(ValueError):
        group_lag_game(model, rng.randn(4, 5), n_vars=2, lags=(0, 1, 2),
                       output_bounds=(0.0, 10.0))  # 5 != 2*3
