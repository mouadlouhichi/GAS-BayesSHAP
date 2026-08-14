"""End-to-end pipeline: result schema, efficiency, statuses, results writer."""

import json

import numpy as np

from gas_bayesshap import GASBayesSHAP, ResultStatus

ENGINE_CONFIG = {
    "checkpoint_enabled": False,
    "cache_enabled": False,
    "log_level": "NONE",
}

REQUIRED_KEYS = [
    "shapley_values", "surrogate_shapley", "residual_shapley",
    "raw_confidence_widths", "certified_projected_widths", "posterior_std",
    "num_coalition_evals", "num_model_evals",
    "num_gp_predictions", "num_residual_samples", "num_sampling_rounds",
    "converged", "certificate_is_rigorous", "range_bound_is_heuristic",
    "uncertified_features", "sign_certified_features",
    "run_id", "M", "domain_game", "config_hash", "oracle_hash",
    "background_hash", "git_commit",
    # audit extensions: per-call vs run-total accounting + completeness
    "num_coalition_evals_this_call", "num_coalition_evals_run_total",
    "num_model_evals_this_call", "num_model_evals_run_total",
    "baseline_model_evals", "num_model_evals_end_to_end",
    "point_estimate_complete", "missing_cells_by_feature",
]


def test_result_schema_section49():
    M = 4
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((3, M)),
                       output_bounds=(0.0, 4.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=0.5, delta=0.05, max_budget=120)
    for key in REQUIRED_KEYS:
        assert key in res, f"missing result key: {key}"
    assert res["M"] == M
    assert res["run_id"] == eng.run_id
    assert res["config_hash"] == eng.config_hash
    assert res["oracle_hash"] == eng.oracle.oracle_h
    assert res["background_hash"] == eng.oracle.background_h


def test_efficiency_projection_exact():
    """sum(phi*) == delta_total exactly (machine precision)."""
    M = 4
    rng = np.random.RandomState(3)
    w = rng.randn(M)
    bg = rng.randn(4, M)

    def model(x):
        return float(np.dot(x, w))

    eng = GASBayesSHAP(model, bg, output_bounds=(-4.0, 4.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(M), epsilon=0.5, delta=0.05, max_budget=120)
    delta = res["delta_total"]
    assert abs(np.sum(res["shapley_values"]) - delta) < 1e-9


def test_projection_formula_direct():
    from gas_bayesshap.certification.projection import project_efficiency
    phi_raw = np.array([1.0, 2.0, 3.0])
    v = np.array([0.5, 0.5, 1.0])
    delta = 7.5
    out = project_efficiency(phi_raw, delta, v)
    expected = phi_raw + v * ((delta - np.sum(phi_raw)) / np.sum(v))
    assert np.allclose(out, expected)
    assert abs(np.sum(out) - delta) < 1e-12


def test_statuses_and_rigor():
    # rigorous + converged -> CERTIFIED
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, 2)),
                       output_bounds=(0.0, 2.0),
                       rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res = eng.explain(np.ones(2), epsilon=1.0, max_budget=20)
    assert res["status"] == ResultStatus.CERTIFIED
    assert res["certificate_is_rigorous"] is True

    # heuristic bounds -> never rigorous, status flagged HEURISTIC_BOUNDS
    eng2 = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, 2)),
                        rng=np.random.RandomState(0), config=ENGINE_CONFIG)
    res2 = eng2.explain(np.ones(2), epsilon=1.0, max_budget=20)
    assert res2["range_bound_is_heuristic"] is True
    assert res2["certificate_is_rigorous"] is False


def test_results_writer(tmp_path):
    M = 3
    eng = GASBayesSHAP(lambda x: float(np.sum(x)), np.zeros((2, M)),
                       output_bounds=(0.0, 3.0),
                       rng=np.random.RandomState(0),
                       config={**ENGINE_CONFIG, "results_dir": str(tmp_path / "results")})
    res = eng.explain(np.ones(M), epsilon=0.5, delta=0.05, max_budget=60)
    run_dir = eng.write_results()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "spec_compliance.json").exists()
    assert (run_dir / "reproducibility_report.md").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == eng.run_id


def test_sign_certified_features_definition():
    from gas_bayesshap.certification.projection import sign_certified
    phi = np.array([1.0, -2.0, 0.1, 0.0])
    W = np.array([0.5, 0.5, 0.2, 0.1])
    sc = sign_certified(phi, W)
    assert sc.tolist() == [True, True, False, False]
