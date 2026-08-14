"""Stratified residual certification machinery (Module B)."""

from .bernstein import cell_width, residual_widths
from .confidence_sequences import AnytimeCheck, anytime_check
from .projection import corollary_widths, project_efficiency, sign_certified

__all__ = [
    "cell_width",
    "residual_widths",
    "AnytimeCheck",
    "anytime_check",
    "corollary_widths",
    "project_efficiency",
    "sign_certified",
]
