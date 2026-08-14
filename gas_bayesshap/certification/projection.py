"""Posterior-diagonal uncertainty-weighted efficiency projection (Theorem C
& Corollary C.1, spec sections 2.8 / 28-29).

.. math::

    \\phi_i^* = \\hat\\phi_i^{\\text{raw}} + v_i
        \\frac{\\Delta_{\\text{total}} - \\sum_j \\hat\\phi_j^{\\text{raw}}}
        {\\sum_j v_j}

    W_i^{\\text{proj}} = W_i^{\\text{res}} + \\frac{v_i}{\\sum_j v_j}
        \\sum_j W_j^{\\text{res}}

    \\text{sign-certified}(i) \\iff |\\phi_i^*| > W_i^{\\text{proj}}
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..numerics.validation import NumericalFailure, assert_finite


def project_efficiency(
    phi_raw: np.ndarray,
    delta_total: float,
    posterior_variances: np.ndarray,
) -> np.ndarray:
    """Theorem C: diagonal uncertainty-weighted projection onto efficiency."""
    phi_raw = np.asarray(phi_raw, dtype=np.float64)
    v = np.asarray(posterior_variances, dtype=np.float64)
    assert_finite(phi_raw, "phi_raw")
    assert_finite(v, "posterior variances")
    residual = delta_total - float(np.sum(phi_raw))
    sum_v = float(np.sum(v))
    if sum_v <= 0:
        raise NumericalFailure("sum of posterior variances is not positive")
    return phi_raw + v * (residual / sum_v)


def corollary_widths(
    W_res: np.ndarray,
    posterior_variances: np.ndarray,
) -> np.ndarray:
    """Corollary C.1: post-projection certified widths."""
    W = np.asarray(W_res, dtype=np.float64)
    v = np.asarray(posterior_variances, dtype=np.float64)
    if np.all(np.isfinite(W)):
        inflation = v * (float(np.sum(W)) / float(np.sum(v)))
        return W + inflation
    return W.copy()


def sign_certified(phi_star: np.ndarray, W_proj: np.ndarray) -> np.ndarray:
    """Definition 1: ``|phi_i*| > W_i^proj`` (strictly excludes zero)."""
    return np.abs(phi_star) > W_proj


def projection_summary(phi_raw, phi_star, W_res, W_proj, v) -> dict:
    return {
        "raw_estimates": np.asarray(phi_raw).tolist(),
        "projected_estimates": np.asarray(phi_star).tolist(),
        "raw_widths": np.asarray(W_res).tolist(),
        "projected_widths": np.asarray(W_proj).tolist(),
        "inflation_ratio": (np.asarray(W_proj) / np.asarray(W_res)).tolist(),
        "sign_certified": sign_certified(phi_star, W_proj).tolist(),
    }
