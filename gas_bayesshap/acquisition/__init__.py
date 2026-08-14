"""Active acquisition (Module A): candidate pool + attribution-aware score."""

from .candidate_pool import CandidatePool, candidate_pool, default_pool_size
from .scoring import acquisition_score

__all__ = [
    "CandidatePool",
    "candidate_pool",
    "default_pool_size",
    "acquisition_score",
]
