"""Result schema, scientific status model and result-directory writer.

Spec sections 30, 48, 49, 50: explicit statuses, the minimum result schema,
and the ``results/runs/<run_id>/`` layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.hashing import json_sha256
from ..utils.serialization import ensure_dir, jsonable, write_json_atomic, write_text_atomic


class ResultStatus:
    """Scientific status model (spec section 50)."""

    VALID = "VALID"
    CERTIFIED = "CERTIFIED"
    NOT_CERTIFIED = "NOT_CERTIFIED"
    CONVERGED = "CONVERGED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MISSING_STRATA = "MISSING_STRATA"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    HEURISTIC_BOUNDS = "HEURISTIC_BOUNDS"
    FAILED = "FAILED"

    ALL = {
        VALID, CERTIFIED, NOT_CERTIFIED, CONVERGED, BUDGET_EXHAUSTED,
        MISSING_STRATA, NUMERICAL_FAILURE, HEURISTIC_BOUNDS, FAILED,
    }


@dataclass
class RunResults:
    """Full result object with the spec-minimum schema (section 49)."""

    shapley_values: np.ndarray
    surrogate_shapley: np.ndarray
    residual_shapley: np.ndarray
    raw_confidence_widths: np.ndarray
    certified_projected_widths: np.ndarray
    posterior_std: np.ndarray

    num_coalition_evals: int
    num_model_evals: int
    num_gp_predictions: int
    num_residual_samples: int
    num_sampling_rounds: int

    converged: bool
    certificate_is_rigorous: bool
    range_bound_is_heuristic: bool
    uncertified_features: List[int]
    sign_certified_features: List[int]

    run_id: str
    M: int
    domain_game: str
    config_hash: str
    oracle_hash: str
    background_hash: str
    git_commit: str

    status: str = ResultStatus.NOT_CERTIFIED
    status_detail: str = ""
    converged_early: bool = False  # spec reference key (identical to converged)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        def arr(v):
            return v.tolist() if include_arrays else None

        return {
            "shapley_values": arr(self.shapley_values),
            "surrogate_shapley": arr(self.surrogate_shapley),
            "residual_shapley": arr(self.residual_shapley),
            "raw_confidence_widths": arr(self.raw_confidence_widths),
            "certified_projected_widths": arr(self.certified_projected_widths),
            "posterior_std": arr(self.posterior_std),
            "num_coalition_evals": int(self.num_coalition_evals),
            "num_model_evals": int(self.num_model_evals),
            "num_gp_predictions": int(self.num_gp_predictions),
            "num_residual_samples": int(self.num_residual_samples),
            "num_sampling_rounds": int(self.num_sampling_rounds),
            "converged": bool(self.converged),
            "converged_early": bool(self.converged_early),
            "certificate_is_rigorous": bool(self.certificate_is_rigorous),
            "range_bound_is_heuristic": bool(self.range_bound_is_heuristic),
            "uncertified_features": list(self.uncertified_features),
            "sign_certified_features": list(self.sign_certified_features),
            "run_id": self.run_id,
            "M": int(self.M),
            "domain_game": self.domain_game,
            "config_hash": self.config_hash,
            "oracle_hash": self.oracle_hash,
            "background_hash": self.background_hash,
            "git_commit": self.git_commit,
            "status": self.status,
            "status_detail": self.status_detail,
            **self.extra,
        }

    def result_hash(self) -> str:
        return json_sha256(self.to_dict(include_arrays=True))

    def summary_text(self) -> str:
        lines = [
            f"GAS-BayesSHAP run {self.run_id}",
            f"  status               : {self.status}",
            f"  M                    : {self.M}",
            f"  domain game          : {self.domain_game}",
            f"  converged            : {self.converged}",
            f"  certificate rigorous : {self.certificate_is_rigorous}",
            f"  heuristic bounds     : {self.range_bound_is_heuristic}",
            f"  coalition evals      : {self.num_coalition_evals}",
            f"  model evals          : {self.num_model_evals}",
            f"  GP predictions       : {self.num_gp_predictions}",
            f"  residual samples     : {self.num_residual_samples}",
            f"  sampling rounds      : {self.num_sampling_rounds}",
            "  shapley values       : "
            + ", ".join(f"{v:.6f}" for v in self.shapley_values),
            "  raw widths           : "
            + ", ".join(f"{w:.6f}" if np.isfinite(w) else "inf" for w in self.raw_confidence_widths),
            "  proj widths          : "
            + ", ".join(f"{w:.6f}" if np.isfinite(w) else "inf" for w in self.certified_projected_widths),
            "  sign certified       : " + str(self.sign_certified_features),
            "  uncertified          : " + str(self.uncertified_features),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def write_run_results(
    results: RunResults,
    results_dir: os.PathLike,
    config: Dict[str, Any],
    environment: Dict[str, Any],
    provenance: Dict[str, Any],
    spec_compliance: Dict[str, Any],
) -> Path:
    """Write the full ``results/runs/<run_id>/`` layout (spec section 48)."""
    run_dir = ensure_dir(Path(results_dir) / results.run_id)
    for sub in (
        "logs", "checkpoints", "oracle", "gp", "residual",
        "neyman", "certification", "benchmarks", "tables", "figures",
    ):
        ensure_dir(run_dir / sub)

    write_json_atomic(run_dir / "manifest.json", results.to_dict(), sort_keys=True)
    write_json_atomic(run_dir / "config.yaml", config, sort_keys=True)  # JSON-ified copy
    write_json_atomic(run_dir / "provenance.json", provenance, sort_keys=True)
    write_json_atomic(run_dir / "environment.json", environment, sort_keys=True)
    write_json_atomic(run_dir / "spec_compliance.json", spec_compliance, sort_keys=True)
    write_json_atomic(run_dir / "summary.json", results.to_dict(), sort_keys=True)
    write_text_atomic(run_dir / "summary.md", results.summary_text())
    write_text_atomic(
        run_dir / "reproducibility_report.md",
        _reproducibility_markdown(environment, provenance, results),
    )
    return run_dir


def _reproducibility_markdown(env: Dict[str, Any], provenance: Dict[str, Any], results: RunResults) -> str:
    git = env.get("git", {})
    lines = [
        "# Reproducibility Report",
        "",
        f"- run_id: `{results.run_id}`",
        f"- git commit: `{git.get('commit', '')}` (dirty: {git.get('dirty', '?')})",
        f"- python: {env.get('python_version', '')}",
        f"- packages: {env.get('packages', {})}",
        f"- os: {env.get('os', '')}",
        f"- cpu: {env.get('cpu', '')}",
        f"- config_hash: `{results.config_hash}`",
        f"- oracle_hash: `{results.oracle_hash}`",
        f"- background_hash: `{results.background_hash}`",
        f"- result_hash: `{results.result_hash()}`",
        f"- status: `{results.status}`",
        "",
        "A repeated run with identical recorded state reproduces results within",
        "declared numerical tolerance (seeded RNG, frozen background, frozen oracle).",
        "",
    ]
    return "\n".join(lines)
