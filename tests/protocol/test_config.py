"""Configuration validation and game presets (spec section 39)."""

import numpy as np
import pytest

from gas_bayesshap.utils.config import ConfigError, load_config, load_game_preset

REPO_CONFIGS = ["configs/default.yaml", "configs/certification.yaml"]


@pytest.mark.parametrize("path", REPO_CONFIGS)
def test_flat_configs_load(path):
    cfg = load_config(path)
    for key in ("sigma0", "lengthscale", "eta", "epsilon", "delta", "max_budget",
                "n_pilot", "n_active_steps", "certification_mode", "seed"):
        assert key in cfg
    assert 0 < cfg["epsilon"]
    assert 0 < cfg["delta"] < 1
    assert cfg["max_budget"] > 0


def test_invalid_config_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"delta": 1.5})
    with pytest.raises(ConfigError):
        load_config(overrides={"epsilon": -1})
    with pytest.raises(ConfigError):
        load_config(overrides={"eta": 0})
    with pytest.raises(ConfigError):
        load_config(overrides={"output_bounds": [2.0, 1.0]})


def test_preset_library_rejected_as_flat_config():
    with pytest.raises(ConfigError, match="preset library"):
        load_config("configs/games.yaml")
    with pytest.raises(ConfigError, match="preset library"):
        load_config("configs/experiments.yaml")


@pytest.mark.parametrize("game,expected_bounds,r_delta", [
    ("membership", (0.0, 1.0), 4.0),
    ("contrastive", (-1.0, 1.0), 8.0),
    ("archetype", (0.0, 1.0), 4.0),
    ("silhouette", (-1.0, 1.0), 8.0),
])
def test_game_presets(game, expected_bounds, r_delta):
    cfg = load_game_preset(game)
    assert cfg["domain_game"] == game
    assert tuple(cfg["output_bounds"]) == expected_bounds
    L, U = cfg["output_bounds"]
    assert 4.0 * (U - L) == r_delta


def test_unknown_game_preset_rejected():
    with pytest.raises(ConfigError, match="unknown game preset"):
        load_game_preset("not_a_game")
