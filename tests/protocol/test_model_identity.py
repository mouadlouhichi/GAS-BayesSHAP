"""Model-identity hardening (audit finding: potential cache identity risk).

Distinct lambda models must never be treated as the same model: the oracle
identity includes a source-derived artifact hash when no explicit model_tag
is given, and caller-supplied artifact hashes are part of the identity.
"""

import numpy as np

from gas_bayesshap.game.oracle import CoalitionOracle


def test_distinct_lambdas_get_distinct_identity():
    """Two lambdas with the same name '<lambda>' but different bodies must
    hash differently (no explicit model_tag)."""
    a = CoalitionOracle(lambda x: float(np.sum(x)), np.zeros((2, 3)))
    b = CoalitionOracle(lambda x: float(np.sum(x) * 2), np.zeros((2, 3)))
    assert a.oracle_h != b.oracle_h
    assert a.model_artifact_hash is not None
    assert a.model_artifact_hash != b.model_artifact_hash


def test_same_function_object_same_identity():
    def m(x):
        return float(np.sum(x))

    a = CoalitionOracle(m, np.zeros((2, 3)))
    b = CoalitionOracle(m, np.zeros((2, 3)))
    assert a.oracle_h == b.oracle_h  # identical function -> identical identity


def test_same_source_different_qualname_differs():
    """The hardening: two functions with identical bodies but different names
    are distinct models and must hash differently (default identity)."""

    def m1(x):
        return float(np.sum(x))

    def m2(x):
        return float(np.sum(x))

    a = CoalitionOracle(m1, np.zeros((2, 3)))
    b = CoalitionOracle(m2, np.zeros((2, 3)))
    assert a.oracle_h != b.oracle_h


def test_explicit_artifact_hash_overrides():
    def m(x):
        return float(np.sum(x))

    a = CoalitionOracle(m, np.zeros((2, 3)), model_artifact_hash="h1")
    b = CoalitionOracle(m, np.zeros((2, 3)), model_artifact_hash="h2")
    assert a.oracle_h != b.oracle_h


def test_explicit_tag_pins_identity():
    """With an explicit model_tag, the caller declares identity (used by
    checkpoint/cache compatibility across crash and resume)."""
    a = CoalitionOracle(lambda x: float(np.sum(x)), np.zeros((2, 3)), model_tag="model-a")
    b = CoalitionOracle(lambda x: float(np.sum(x) * 2), np.zeros((2, 3)), model_tag="model-a")
    assert a.oracle_h == b.oracle_h  # tag is authoritative
