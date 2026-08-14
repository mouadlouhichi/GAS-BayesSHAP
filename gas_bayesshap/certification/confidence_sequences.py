"""Anytime stopping check (spec section 24).

At every iteration the **complete** width vector is computed and the
stop rule :math:`\\tau = \\inf\\{\\mathbf{n}: \\max_i W_i^{\\text{res}} \\le
\\epsilon\\}` is evaluated.  No feature is dropped early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AnytimeCheck:
    width_vector: np.ndarray
    max_width: float
    mean_width: float
    median_width: float
    argmax_feature: Optional[int]
    converged: bool
    epsilon: float
    all_finite: bool

    def to_dict(self) -> dict:
        return {
            "width_vector": self.width_vector.tolist(),
            "max_width": self.max_width,
            "mean_width": self.mean_width,
            "median_width": self.median_width,
            "argmax_feature": self.argmax_feature,
            "converged": self.converged,
            "epsilon": self.epsilon,
            "all_finite": self.all_finite,
        }


def anytime_check(width_vector: np.ndarray, epsilon: float) -> AnytimeCheck:
    """Evaluate the anytime stopping rule on the full width vector."""
    w = np.asarray(width_vector, dtype=np.float64)
    finite = np.isfinite(w)
    max_width = float(np.max(w)) if w.size else float("inf")
    mean_width = float(np.mean(w[finite])) if np.any(finite) else float("inf")
    median_width = float(np.median(w[finite])) if np.any(finite) else float("inf")
    argmax = int(np.argmax(w)) if w.size else None
    converged = bool(w.size > 0 and max_width <= epsilon)
    return AnytimeCheck(
        width_vector=w,
        max_width=max_width,
        mean_width=mean_width,
        median_width=median_width,
        argmax_feature=argmax,
        converged=converged,
        epsilon=float(epsilon),
        all_finite=bool(np.all(finite)),
    )
