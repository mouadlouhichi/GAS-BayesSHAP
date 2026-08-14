"""Stratified residual storage (spec section 15).

Maintains :math:`\\mathcal{D}_{\\text{cert}}(i, s)` for every
:math:`s = 0..M-1`, :math:`i = 0..M-1`.  Each record carries full metadata:

``feature, stratum, coalition, direction, residual_value, iteration, random_seed``

No observation is silently discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..numerics.validation import safe_std


@dataclass
class ResidualRecord:
    feature: int
    stratum: int
    coalition: list          # bool list (M,)
    direction: str           # "add_one" | "remove_one"
    residual_value: float
    iteration: int
    random_seed: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


class StratumStore:
    """Per-cell residual observation store with metadata-rich records."""

    def __init__(self, M: int):
        self.M = int(M)
        self._cells: Dict[int, Dict[int, List[ResidualRecord]]] = {
            s: {i: [] for i in range(self.M)} for s in range(self.M)
        }
        self.n_records = 0

    # ------------------------------------------------------------------ #
    def append(
        self,
        feature: int,
        stratum: int,
        coalition: np.ndarray,
        direction: str,
        value: float,
        iteration: int,
        random_seed: Optional[int] = None,
    ) -> None:
        if not (0 <= stratum < self.M and 0 <= feature < self.M):
            raise IndexError(f"cell ({stratum}, {feature}) out of range for M={self.M}")
        rec = ResidualRecord(
            feature=int(feature),
            stratum=int(stratum),
            coalition=np.asarray(coalition, dtype=bool).tolist(),
            direction=str(direction),
            residual_value=float(value),
            iteration=int(iteration),
            random_seed=random_seed,
        )
        self._cells[stratum][feature].append(rec)
        self.n_records += 1

    # ------------------------------------------------------------------ #
    def records(self, feature: int, stratum: int) -> List[ResidualRecord]:
        return self._cells[stratum][feature]

    def count(self, feature: int, stratum: int) -> int:
        return len(self._cells[stratum][feature])

    def values(self, feature: int, stratum: int) -> np.ndarray:
        return np.array([r.residual_value for r in self._cells[stratum][feature]], dtype=np.float64)

    def mean(self, feature: int, stratum: int) -> float:
        vals = self.values(feature, stratum)
        return float(np.mean(vals)) if len(vals) else 0.0

    def std(self, feature: int, stratum: int, default: float = 0.5) -> float:
        return safe_std(self.values(feature, stratum), default)

    def counts_matrix(self) -> np.ndarray:
        """(M, M) matrix ``counts[s, i]``."""
        return np.array(
            [[self.count(i, s) for i in range(self.M)] for s in range(self.M)],
            dtype=np.int64,
        )

    def means_matrix(self) -> np.ndarray:
        return np.array(
            [[self.mean(i, s) for i in range(self.M)] for s in range(self.M)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    def missing_cells(self) -> List[tuple]:
        """Cells with zero observations (explicit; never silently ignored)."""
        return [(s, i) for s in range(self.M) for i in range(self.M) if self.count(i, s) == 0]

    def summary(self) -> dict:
        counts = self.counts_matrix()
        return {
            "n_records": self.n_records,
            "counts": counts.tolist(),
            "missing_cells": self.missing_cells(),
        }

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "M": self.M,
            "n_records": self.n_records,
            "cells": {
                s: {i: [r.to_dict() for r in recs] for i, recs in cell.items()}
                for s, cell in self._cells.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StratumStore":
        store = cls(int(data["M"]))
        store.n_records = int(data.get("n_records", 0))
        for s_str, cell in data["cells"].items():
            s = int(s_str)
            for i_str, recs in cell.items():
                i = int(i_str)
                for r in recs:
                    store._cells[s][i].append(ResidualRecord(**r))
        return store
