"""Output-bounds contract enforcement (audit finding: output-bound contract)."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.numerics.validation import OutputBoundViolation

ENGINE_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}


def test_violating_model_raises():
    """A model whose output leaves the declared [L, U] must raise, never be
    silently accepted (the certification contract depends on the range)."""
    M = 3

    def model(x):  # returns 3.0 at x = ones(3) but bounds are (0, 2)
        return float(np.sum(x))

    oracle = CoalitionOracle(model, np.zeros((2, M)), output_bounds=(0.0, 2.0))
    with pytest.raises(OutputBoundViolation):
        oracle.evaluate(np.ones(M), np.ones(M, dtype=bool))

    # and through the full engine
    eng = GASBayesSHAP(model, np.zeros((2, M)), output_bounds=(0.0, 2.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    with pytest.raises(OutputBoundViolation):
        eng.explain(np.ones(M), epsilon=1.0, max_budget=30)


def test_nan_output_raises():
    M = 2

    def model(x):
        return float("nan")

    oracle = CoalitionOracle(model, np.zeros((2, M)), output_bounds=(0.0, 1.0))
    with pytest.raises(OutputBoundViolation):
        oracle.evaluate(np.ones(M), np.ones(M, dtype=bool))


def test_within_bounds_passes():
    M = 3
    oracle = CoalitionOracle(lambda x: float(np.sum(x)), np.zeros((2, M)),
                             output_bounds=(0.0, 3.0))
    v = oracle.evaluate(np.ones(M), np.ones(M, dtype=bool))
    assert v == 3.0


def test_boundary_value_ok_within_tolerance():
    M = 3
    oracle = CoalitionOracle(lambda x: float(np.sum(x)), np.zeros((2, M)),
                             output_bounds=(0.0, 3.0))
    v = oracle.evaluate(np.ones(M), np.ones(M, dtype=bool))
    assert abs(v - 3.0) < 1e-9  # exactly on the upper boundary is accepted
