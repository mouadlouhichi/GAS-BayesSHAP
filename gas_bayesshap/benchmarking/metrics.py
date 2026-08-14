"""Benchmark metrics: errors, coverage, widths, query counts (spec section 46)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def mae(phi_hat: np.ndarray, phi_true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(phi_hat) - np.asarray(phi_true))))


def rmse(phi_hat: np.ndarray, phi_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(phi_hat) - np.asarray(phi_true)) ** 2)))


def max_abs_error(phi_hat: np.ndarray, phi_true: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(phi_hat) - np.asarray(phi_true))))


def benchmark_metrics(phi_hat: np.ndarray, phi_true: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(phi_hat, phi_true),
        "RMSE": rmse(phi_hat, phi_true),
        "max_error": max_abs_error(phi_hat, phi_true),
    }


def coverage_report(
    phi_hats: Sequence[np.ndarray],
    widths: Sequence[np.ndarray],
    phi_true: np.ndarray,
) -> Dict[str, float]:
    """Empirical coverage over repeated trials (spec section 45)."""
    n = len(phi_hats)
    finite = 0
    covered = 0
    width_stats = []
    for ph, w in zip(phi_hats, widths):
        w = np.asarray(w, dtype=np.float64)
        if np.all(np.isfinite(w)):
            finite += 1
            width_stats.append(np.mean(w))
            if np.all(np.abs(ph - phi_true) <= w):
                covered += 1
    return {
        "n_trials": int(n),
        "finite_width_rate": finite / n if n else 0.0,
        "empirical_coverage": covered / n if n else 0.0,
        "coverage_given_finite": covered / max(1, finite),
        "mean_width": float(np.mean(width_stats)) if width_stats else float("nan"),
        "median_width": float(np.median(width_stats)) if width_stats else float("nan"),
        "max_width": float(np.max(width_stats)) if width_stats else float("nan"),
    }
