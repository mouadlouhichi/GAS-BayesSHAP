"""Isolated per-test run/checkpoint directories."""

import pytest


@pytest.fixture
def run_dirs(tmp_path):
    """Return fresh results/checkpoints dirs for each test."""
    return {
        "results_dir": str(tmp_path / "results"),
        "checkpoints_dir": str(tmp_path / "checkpoints"),
    }
