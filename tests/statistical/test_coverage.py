"""Tier-4: empirical coverage calibration (spec sections 6, 45)."""

import numpy as np
import pytest

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.benchmarking.metrics import coverage_report

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}

PHI_EXACT = np.array([1.5, 2.5, -1.0])


def _model_cal(x):
    return float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])


def test_coverage_calibration_30_trials():
    """The supplied 30-trial regression test (coverage-given-finite >= 90%)."""
    R = 30
    finite_count = 0
    covered_count = 0
    for trial in range(R):
        e = GASBayesSHAP(_model_cal, np.zeros((3, 3)), output_bounds=(-2.0, 5.0),
                         rng=np.random.RandomState(trial), config=ENGINE_CONFIG)
        r = e.explain(np.ones(3), epsilon=1.5, delta=0.05, max_budget=300)
        is_finite = np.all(np.isfinite(r["certified_projected_widths"]))
        is_covered = np.all(np.abs(r["shapley_values"] - PHI_EXACT) <= r["certified_projected_widths"])
        if is_finite:
            finite_count += 1
            if is_covered:
                covered_count += 1
    cov_given_finite = covered_count / max(1, finite_count)
    assert finite_count > 0
    assert cov_given_finite >= 0.90


def test_coverage_report_configurable_trials():
    """Configurable number of repeated trials (spec section 45)."""
    R = 12
    phis, widths = [], []
    for trial in range(R):
        e = GASBayesSHAP(_model_cal, np.zeros((3, 3)), output_bounds=(-2.0, 5.0),
                         rng=np.random.RandomState(100 + trial), config=ENGINE_CONFIG)
        r = e.explain(np.ones(3), epsilon=1.5, delta=0.05, max_budget=250)
        phis.append(r["shapley_values"])
        widths.append(r["certified_projected_widths"])
    rep = coverage_report(phis, widths, PHI_EXACT)
    assert rep["n_trials"] == R
    assert rep["finite_width_rate"] > 0.5
    assert "empirical_coverage" in rep
    assert np.isfinite(rep["mean_width"])


def test_coverage_linear_game_wide_bounds():
    """With very wide bounds the intervals must contain the truth on a
    deterministic linear game for every trial."""
    def model(x):
        return float(np.dot(x, np.array([1.0, -2.0, 3.0])))

    truth = np.array([1.0, -2.0, 3.0])
    for trial in range(6):
        e = GASBayesSHAP(model, np.zeros((3, 3)), output_bounds=(-10.0, 10.0),
                         rng=np.random.RandomState(trial), config=ENGINE_CONFIG)
        r = e.explain(np.ones(3), epsilon=1.0, delta=0.05, max_budget=250)
        assert np.all(np.isfinite(r["certified_projected_widths"]))
        assert np.all(np.abs(r["shapley_values"] - truth) <= r["certified_projected_widths"])
