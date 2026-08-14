"""Anytime confidence-sequence behaviour (spec sections 23-24, 44)."""

import numpy as np
import pytest

from gas_bayesshap.certification.bernstein import cell_width, residual_widths
from gas_bayesshap.certification.confidence_sequences import anytime_check
from gas_bayesshap.residual.strata import StratumStore


def test_cell_width_formula():
    n, sigma, M, delta, R = 10, 0.5, 5, 0.05, 4.0
    log_term = np.log((np.pi ** 2 * M ** 2 * n ** 2) / (3.0 * delta))
    expected = np.sqrt((2.0 * sigma ** 2 * log_term) / n) + (7.0 * R * log_term) / (3.0 * (n - 1))
    assert abs(cell_width(n, sigma, M, delta, R) - float(expected)) < 1e-12


def test_cell_width_edge_cases():
    # interior n < 2 -> inf
    assert np.isinf(cell_width(1, 0.5, 5, 0.05, 4.0))
    # extreme stratum contributes 0 once observed
    assert cell_width(3, 0.5, 5, 0.05, 4.0, interior=False) == 0.0
    # extreme stratum unobserved -> inf
    assert np.isinf(cell_width(0, 0.5, 5, 0.05, 4.0, interior=False))
    # widths shrink with n
    w10 = cell_width(10, 0.5, 5, 0.05, 4.0)
    w100 = cell_width(100, 0.5, 5, 0.05, 4.0)
    assert w100 < w10


def test_width_vector_extreme_zero_and_interior_positive():
    M = 4
    store = StratumStore(M)
    # fill extreme strata for every feature
    for i in range(M):
        store.append(i, 0, np.zeros(M, dtype=bool), "add_one", 0.1, 0)
        store.append(i, M - 1, np.ones(M, dtype=bool), "remove_one", 0.1, 0)
    sigma_res = np.zeros((M, M))
    widths = residual_widths(store, sigma_res, M, 0.05, 4.0)
    # interior cells empty -> inf
    assert np.all(np.isinf(widths))
    # fill interior cells
    for s in range(1, M - 1):
        for i in range(M):
            for _ in range(3):
                store.append(i, s, np.zeros(M, dtype=bool), "add_one", 0.1 * (i + 1), 0)
            sigma_res[s, i] = 0.1
    widths = residual_widths(store, sigma_res, M, 0.05, 4.0)
    assert np.all(np.isfinite(widths))
    # all widths positive and finite; extreme strata contribute 0
    assert np.all(widths > 0)


def test_anytime_check_full_vector():
    w = np.array([0.1, 0.05, 0.2])
    check = anytime_check(w, 0.15)
    assert not check.converged
    assert check.max_width == 0.2
    assert check.argmax_feature == 2
    assert np.isclose(check.mean_width, np.mean(w))
    assert np.isclose(check.median_width, np.median(w))
    assert check.all_finite

    check2 = anytime_check(np.array([0.1, 0.05, 0.14]), 0.15)
    assert check2.converged

    check3 = anytime_check(np.array([0.1, np.inf, 0.14]), 0.15)
    assert not check3.converged
    assert check3.max_width == np.inf
    assert not check3.all_finite


def test_width_formula_matches_spec_reference():
    """Cross-check the width formula against the spec reference engine."""
    from gas_bayesshap.reference.spec_v11_reference import safe_std as ref_safe_std
    M = 5
    store = StratumStore(M)
    rng = np.random.RandomState(0)
    for s in range(1, M - 1):
        for i in range(M):
            vals = rng.randn(4)
            for v in vals:
                store.append(i, s, np.zeros(M, dtype=bool), "add_one", float(v), 0)
            for i2 in range(M):
                pass
    # reference width computation on the same values
    ref_cells = {s: {i: store.values(i, s).tolist() for i in range(M)} for s in range(M)}
    delta = 0.05
    R = 4.0
    ref_widths = np.zeros(M)
    for i in range(M):
        W = 0.0
        for s in range(1, M - 1):
            vals = ref_cells[s][i]
            n = len(vals)
            sig = ref_safe_std(vals, 0.5)
            log_term = np.log((np.pi ** 2 * M ** 2 * n ** 2) / (3.0 * delta))
            W += (1.0 / M) * (np.sqrt(2 * sig ** 2 * log_term / n) + 7 * R * log_term / (3 * (n - 1)))
        ref_widths[i] = W
    sigma_res = np.array([[ref_safe_std(store.values(i, s).tolist(), 0.5) for i in range(M)] for s in range(M)])
    # ensure extreme cells are populated so widths are finite
    for i in range(M):
        store.append(i, 0, np.zeros(M, dtype=bool), "add_one", 0.0, 0)
        store.append(i, M - 1, np.ones(M, dtype=bool), "remove_one", 0.0, 0)
    my_widths = residual_widths(store, sigma_res, M, delta, R)
    assert np.allclose(my_widths, ref_widths, atol=1e-12)
