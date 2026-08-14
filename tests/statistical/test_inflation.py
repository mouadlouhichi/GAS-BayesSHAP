"""Tier-5: Corollary C.1 post-projection width inflation tightness."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.certification.projection import corollary_widths

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}


def test_inflation_ratio_small():
    M = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])

    def model(x):
        return float(np.dot(x, weights) + 0.5 * x[0] * x[1])

    eng = GASBayesSHAP(model, np.zeros((5, M)), output_bounds=(-5.0, 10.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=400)

    raw = np.asarray(res["raw_confidence_widths"], dtype=np.float64)
    proj = np.asarray(res["certified_projected_widths"], dtype=np.float64)
    finite_mask = np.isfinite(raw)
    if np.any(finite_mask):
        ratio = float(np.mean(proj[finite_mask] / raw[finite_mask]))
        # spec target: controlled ~1.5-2.5x (paper: ~1.5-2.0x)
        assert 1.0 < ratio < 3.0
        print(f"inflation ratio = {ratio:.2f}x")


def test_corollary_widths_consistent():
    W = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, 1.0, 0.5])
    Wp = corollary_widths(W, v)
    expected = W + v * (np.sum(W) / np.sum(v))
    assert np.allclose(Wp, expected)
    assert np.all(Wp >= W)  # projection can only widen


def test_infinite_width_propagates():
    W = np.array([1.0, np.inf, 3.0])
    v = np.array([1.0, 1.0, 1.0])
    Wp = corollary_widths(W, v)
    assert not np.all(np.isfinite(Wp))
    assert Wp[1] == np.inf
