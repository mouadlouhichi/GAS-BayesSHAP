"""Anytime stratified empirical-Bernstein confidence sequences (Theorem B).

Spec section 2.7 / 23.  For frozen bounded linear surrogate with bounded
output range ``[L, U]`` (``R_delta_res = 4*(U-L)``), interior strata
``s in {1..M-2}`` require ``n_{i,s} >= 2``; extreme singleton strata
``s = 0, M-1`` contribute **0** width (Lemma G).

.. math::

    W_i^{res}(\\mathbf{n}_i) = \\frac{1}{M} \\sum_{s=1}^{M-2}
    \\left(
    \\sqrt{\\frac{2 (\\hat\\sigma^r_{i,s})^2
        \\log\\left(\\frac{\\pi^2 M^2 n_{i,s}^2}{3\\delta}\\right)}{n_{i,s}}}
    + \\frac{7 R_\\Delta^{\\text{res}}
        \\log\\left(\\frac{\\pi^2 M^2 n_{i,s}^2}{3\\delta}\\right)}
        {3(n_{i,s}-1)}
    \\right)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..residual.strata import StratumStore


def cell_width(
    n: int,
    sigma_hat: float,
    M: int,
    delta: float,
    R_delta_res: float,
    interior: bool = True,
) -> float:
    """Width contribution of one cell.

    Extreme strata (``interior=False``) contribute ``0.0`` once observed.
    Interior strata with ``n < 2`` return ``inf`` (invalid cell).
    """
    if not interior:
        return 0.0 if n >= 1 else float("inf")
    if n < 2:
        return float("inf")
    log_term = math.log((math.pi ** 2 * M ** 2 * n ** 2) / (3.0 * delta))
    w = math.sqrt((2.0 * (sigma_hat ** 2) * log_term) / n) + (
        7.0 * R_delta_res * log_term
    ) / (3.0 * (n - 1))
    return float(w)


def residual_widths(
    store: StratumStore,
    sigma_res: np.ndarray,
    M: int,
    delta: float,
    R_delta_res: float,
) -> np.ndarray:
    """Full width vector ``W_1..W_M`` (exact Theorem-B arithmetic).

    For every feature the loop covers **all** strata; extreme strata
    (``s=0``, ``s=M-1``) require ``n >= 1`` and contribute 0; interior
    strata require ``n >= 2`` (otherwise ``W_i = inf``).
    """
    widths = np.zeros(M, dtype=np.float64)
    for i in range(M):
        W_i = 0.0
        all_cells_valid = True
        for s in range(M):
            n_is = store.count(i, s)
            if s == 0 or s == M - 1:
                if n_is < 1:
                    all_cells_valid = False
                    W_i = float("inf")
                    break
                continue
            if n_is < 2:
                all_cells_valid = False
                W_i = float("inf")
                break
            sig_curr = sigma_res[s, i]
            log_term = math.log((math.pi ** 2 * M ** 2 * n_is ** 2) / (3.0 * delta))
            w_s = math.sqrt((2.0 * (sig_curr ** 2) * log_term) / n_is) + (
                7.0 * R_delta_res * log_term
            ) / (3.0 * (n_is - 1))
            W_i += (1.0 / M) * w_s
        widths[i] = W_i if all_cells_valid else float("inf")
    return widths
