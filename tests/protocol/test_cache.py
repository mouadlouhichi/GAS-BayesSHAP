"""Coalition cache correctness and compatibility rejection (spec sections 33, 44)."""

import numpy as np
import pytest

from gas_bayesshap.cache.coalition_cache import CacheCompatibilityError, CoalitionCache
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.utils.hashing import config_hash

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": True,
    "persist_cache": False,
    "log_level": "NONE",
}


def test_cache_hit_does_not_count():
    model_calls = {"n": 0}

    def model(x):
        model_calls["n"] += 1
        return float(np.sum(x))

    cache = CoalitionCache(config_hash="c", oracle_hash="o", background_hash="b")
    oracle = CoalitionOracle(model, np.zeros((4, 3)), cache=cache,
                             config_hash="c")
    x = np.ones(3)
    S = np.array([True, False, True])
    v1 = oracle.evaluate(x, S)
    c1 = oracle.total_coalition_evals
    m1 = oracle.total_model_evals
    v2 = oracle.evaluate(x, S)  # cache hit
    assert v1 == v2
    assert oracle.total_coalition_evals == c1   # no new coalition eval
    assert oracle.total_model_evals == m1       # no new model eval
    assert oracle.cache_hits == 1
    assert cache.hit_rate() > 0


def test_cache_misses_counted():
    cache = CoalitionCache(config_hash="c", oracle_hash="o", background_hash="b")
    oracle = CoalitionOracle(lambda x: float(np.sum(x)), np.zeros((2, 2)),
                             cache=cache, config_hash="c")
    oracle.evaluate(np.ones(2), np.array([True, False]))
    assert oracle.total_coalition_evals == 1
    assert oracle.cache_misses == 1


def test_incompatible_cache_rejected_on_disk(tmp_path):
    """A persisted cache with different hashes must be rejected, not reused."""
    path = tmp_path / "cache.json"
    c1 = CoalitionCache(config_hash="cfg-A", oracle_hash="oracle-A",
                        background_hash="bg-A", persist_path=path)
    c1.put("k", 1.0)
    c1.persist()

    with pytest.raises(CacheCompatibilityError):
        CoalitionCache(config_hash="cfg-B", oracle_hash="oracle-A",
                       background_hash="bg-A", persist_path=path)
    with pytest.raises(CacheCompatibilityError):
        CoalitionCache(config_hash="cfg-A", oracle_hash="oracle-B",
                       background_hash="bg-A", persist_path=path)
    with pytest.raises(CacheCompatibilityError):
        CoalitionCache(config_hash="cfg-A", oracle_hash="oracle-A",
                       background_hash="bg-B", persist_path=path)

    # compatible reload works
    c2 = CoalitionCache(config_hash="cfg-A", oracle_hash="oracle-A",
                        background_hash="bg-A", persist_path=path)
    assert c2.get("k") == 1.0


def test_engine_cache_roundtrip():
    from gas_bayesshap import GASBayesSHAP

    def model(x):
        return float(np.sum(x))

    eng = GASBayesSHAP(model, np.zeros((3, 3)), output_bounds=(0.0, 3.0),
                       rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "persist_cache": False})
    eng.explain(np.ones(3), epsilon=1.0, delta=0.05, max_budget=30)
    assert eng._cache is not None
    assert len(eng._cache) > 0
