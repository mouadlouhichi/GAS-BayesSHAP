"""Benchmarking: exact, Monte-Carlo, GP-only and full-pipeline comparisons."""

from .exact import run_exact_benchmark
from .metrics import (
    benchmark_metrics,
    coverage_report,
    mae,
    max_abs_error,
    rmse,
)
from .monte_carlo import monte_carlo_shapley

__all__ = [
    "run_exact_benchmark",
    "benchmark_metrics",
    "coverage_report",
    "mae",
    "max_abs_error",
    "rmse",
    "monte_carlo_shapley",
]
