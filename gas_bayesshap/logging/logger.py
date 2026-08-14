"""Standard-library logging setup for ``run.log`` and ``errors.log``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..utils.serialization import ensure_dir


def setup_logger(
    log_dir: os.PathLike,
    name: str = "gas_bayesshap",
    level: str = "INFO",
) -> logging.Logger:
    """Configure a logger writing to ``<log_dir>/run.log`` and ``errors.log``."""
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    run_handler = logging.FileHandler(str(log_dir / "run.log"), encoding="utf-8")
    run_handler.setFormatter(fmt)
    run_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(run_handler)

    err_handler = logging.FileHandler(str(log_dir / "errors.log"), encoding="utf-8")
    err_handler.setFormatter(fmt)
    err_handler.setLevel(logging.WARNING)
    logger.addHandler(err_handler)
    return logger
