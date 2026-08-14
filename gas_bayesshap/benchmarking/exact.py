"""Exact Shapley reference runner (small M)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..game.brute_force import exact_game_values, exact_shapley_from_values


def exact_shapley_for_oracle(oracle, x: np.ndarray, M: int) -> Dict[str, Any]:
    """Exact Shapley values via full 2^M enumeration through the oracle."""
    values = exact_game_values(oracle, x, M)
    phi = exact_shapley_from_values(values, M)
    coal_evals = oracle.total_coalition_evals
    # efficiency check: sum(phi) == v(N) - v(empty)
    v_full = values[(1 << M) - 1]
    v_empty = values[0]
    eff = float(np.sum(phi))
    return {
        "shapley_values": phi,
        "values": values,
        "num_coalition_evals": coal_evals,
        "delta_total": v_full - v_empty,
        "efficiency_error": abs(eff - (v_full - v_empty)),
    }


def run_exact_benchmark(model_fn, background, x, output_bounds=None, M: Optional[int] = None) -> Dict[str, Any]:
    """Convenience: build a plain oracle and compute exact Shapley values."""
    from ..game.oracle import CoalitionOracle
    M = M if M is not None else int(np.atleast_2d(background).shape[1])
    oracle = CoalitionOracle(model_fn=model_fn, background=background, output_bounds=output_bounds)
    return exact_shapley_for_oracle(oracle, x, M)
