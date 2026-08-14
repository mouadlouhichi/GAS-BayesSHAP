"""Structured logging layer (spec section 37)."""

from .events import EventLogger
from .logger import setup_logger

__all__ = ["EventLogger", "setup_logger"]
