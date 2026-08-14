"""GP engine: Sherman-Morrison updates, bounded linear surrogate, posterior."""

from .control_variate import BoundedLinearSurrogate, fit_bounded_surrogate
from .posterior import gp_posterior, gp_predict
from .updates import rank1_inverse_update

__all__ = [
    "BoundedLinearSurrogate",
    "fit_bounded_surrogate",
    "gp_posterior",
    "gp_predict",
    "rank1_inverse_update",
]
