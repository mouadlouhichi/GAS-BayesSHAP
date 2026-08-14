"""Run the full engine on domain-game oracles."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.domain_games import membership_game

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_membership_game_full_pipeline():
    rng = np.random.RandomState(0)
    w = rng.randn(5)
    background = rng.randn(8, 5)

    def g_c(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(5)))

    oracle, spec = membership_game(g_c, background)
    eng = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "domain_game": "membership"})
    res = eng.explain(np.ones(5), epsilon=0.4, delta=0.05, max_budget=150)
    assert res["domain_game"] == "membership"
    assert res["certificate_is_rigorous"] is True  # membership has exact bounds
    assert abs(np.sum(res["shapley_values"]) - res["delta_total"]) < 1e-9
    # values in a sensible range relative to delta_total
    assert np.all(np.isfinite(res["shapley_values"]))
