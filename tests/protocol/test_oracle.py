"""Deterministic oracle and interventional imputation semantics."""

import numpy as np

from gas_bayesshap.game.oracle import CoalitionOracle


def test_oracle_deterministic():
    rng = np.random.RandomState(0)
    M, B = 4, 5
    background = rng.randn(B, M)

    def model(x):
        return float(np.sin(x[0]) + x[1] - x[2] * x[3])

    oracle = CoalitionOracle(model, background)
    x = rng.randn(M)
    S = np.array([True, False, True, False])
    v1 = oracle.evaluate(x, S)
    v2 = oracle.evaluate(x, S)
    assert v1 == v2
    assert oracle.validate_determinism(x, S)


def test_interventional_imputation_hybrid():
    """Retained features come from x; excluded from background, independently."""
    B, M = 3, 3
    background = np.array([[10.0, 20.0, 30.0],
                           [11.0, 21.0, 31.0],
                           [12.0, 22.0, 32.0]])
    seen = {"rows": []}

    def model(row):
        seen["rows"].append(row.copy())
        return float(np.sum(row))

    oracle = CoalitionOracle(model, background)
    seen["rows"] = []  # reset: constructor already ran B background passes
    x = np.array([1.0, 2.0, 3.0])
    S = np.array([True, False, True])
    v = oracle.evaluate(x, S)
    assert len(seen["rows"]) == B
    for row in seen["rows"]:
        assert row[0] == 1.0 and row[2] == 3.0       # retained from x
        assert row[1] in (20.0, 21.0, 22.0)           # excluded from background
    assert v == np.mean([r.sum() for r in seen["rows"]])


def test_e_base_is_background_mean():
    B, M = 4, 2
    background = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])

    def model(x):
        return float(x[0] + x[1])

    oracle = CoalitionOracle(model, background)
    assert abs(oracle.E_base - np.mean([3.0, 6.0, 9.0, 12.0])) < 1e-12
    v_empty = oracle.evaluate(np.zeros(M), np.zeros(M, dtype=bool))
    assert v_empty == oracle.E_base


def test_output_bounds_validation():
    with np.testing.assert_raises(ValueError):
        CoalitionOracle(lambda x: 0.0, np.zeros((2, 2)), output_bounds=(1.0, 1.0))
    with np.testing.assert_raises(ValueError):
        CoalitionOracle(lambda x: 0.0, np.zeros((2, 2)), output_bounds=(2.0, 1.0))
