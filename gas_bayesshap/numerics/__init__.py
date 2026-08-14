"""Numerical foundations: stable combinatorics, linear algebra, validation."""

from .linear_algebra import (
    check_symmetric,
    is_psd,
    is_symmetric,
    posterior_covariance,
    posterior_variances,
    solve_inverse,
    solve_system,
    sym,
)
from .stable_combinatorics import (
    comb,
    comb_exact,
    comb_log,
    delta_weight,
    factorial,
    shapley_weight,
)
from .validation import NumericalFailure, assert_finite, assert_shape, safe_std

__all__ = [
    "check_symmetric",
    "is_psd",
    "is_symmetric",
    "posterior_covariance",
    "posterior_variances",
    "solve_inverse",
    "solve_system",
    "sym",
    "comb",
    "comb_exact",
    "comb_log",
    "delta_weight",
    "factorial",
    "shapley_weight",
    "NumericalFailure",
    "assert_finite",
    "assert_shape",
    "safe_std",
]
