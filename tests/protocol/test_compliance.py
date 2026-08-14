"""Compliance audit is evidence-based, not self-declarative
(audit finding: Medium 1)."""

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.core.compliance import (
    IMPLEMENTED,
    MISSING,
    TESTED,
    VALIDATED,
    run_compliance_audit,
)

ENGINE_CONFIG = {"checkpoint_enabled": False, "cache_enabled": False, "log_level": "NONE"}


def _run_engine():
    rng = np.random.RandomState(0)
    M = 4
    w = rng.randn(M)

    def model(x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(M)))

    eng = GASBayesSHAP(model, rng.randn(5, M), output_bounds=(0.0, 1.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    eng.explain(np.ones(M), epsilon=1.0, delta=0.05, max_budget=60, n_pilot=2, n_active_steps=5)
    return eng


def test_audit_runs_real_math_validation():
    """Lemma D/E items must be VALIDATED (brute force executed), not merely
    'IMPLEMENTED'."""
    eng = _run_engine()
    audit = run_compliance_audit(eng)
    by_id = {i["id"]: i for i in audit["items"]}
    assert by_id["MATH_2"]["status"] == VALIDATED
    assert "max|diff|" in by_id["MATH_2"]["evidence"]
    assert by_id["MATH_3"]["status"] == VALIDATED
    assert audit["overall"] == "COMPLIANT"


def test_audit_has_no_missing_for_small_m():
    eng = _run_engine()
    audit = run_compliance_audit(eng)
    missing = [i for i in audit["items"] if i["status"] == MISSING]
    assert missing == [], f"missing items: {missing}"
    # dynamic checks actually executed
    statuses = {i["status"] for i in audit["items"]}
    assert TESTED in statuses or VALIDATED in statuses
    assert audit["audit_mode"] == "dynamic"


def test_audit_accepts_external_test_evidence():
    eng = _run_engine()
    evidence = {
        "math_validated": "pytest tests/mathematical passed (26 tests, exit 0)",
        "query_accounting": "pytest tests/protocol/test_accounting.py passed",
    }
    audit = run_compliance_audit(eng, evidence=evidence)
    assert audit["overall"] == "COMPLIANT"
    assert audit["evidence"]["math_validated"] == evidence["math_validated"]
    by_id = {i["id"]: i for i in audit["items"]}
    assert by_id["MATH_2"]["status"] == VALIDATED


def test_spec_compliance_json_is_evidence_based():
    """write_results() must produce an audit whose math items carry evidence."""
    import json
    import tempfile
    from pathlib import Path
    rng = np.random.RandomState(0)
    M = 3
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0), rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "results_dir": str(tempfile.mkdtemp() + "/res")})
    eng.explain(np.ones(M), epsilon=1.0, max_budget=40)
    run_dir = eng.write_results()
    audit = json.loads((Path(run_dir) / "spec_compliance.json").read_text())
    assert audit["audit_mode"] == "dynamic"
    math2 = [i for i in audit["items"] if i["id"] == "MATH_2"][0]
    assert math2["status"] == VALIDATED
    assert math2["evidence"]
