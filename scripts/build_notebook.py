#!/usr/bin/env python
"""Build notebooks/run_all.ipynb (17 sections + status dashboard).

The notebook imports the package and orchestrates the real implementation;
it does not duplicate any scientific algorithm.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def cell(source, cell_type="code", outputs=None):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = outputs if outputs is not None else []
    return c

def md(text):
    return cell(text, "markdown")

cells = []
A = cells.append

A(md("# GAS-BayesSHAP — run_all.ipynb\n\n"
     "Gaussian-Adaptive Stratified Bayesian Shapley Estimation (v11.0): bounded-linear "
     "Bayesian control variates + Neyman-stratified anytime empirical-Bernstein certification. "
     "This notebook orchestrates the real package implementation "
     "(`gas_bayesshap`); it duplicates **no** scientific algorithm."))

A(md("## 1. Environment"))
A(cell("""import sys, time, json, warnings
import numpy as np
sys.path.insert(0, "..")
import gas_bayesshap
from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.domain_games import membership_game, group_lag_game
from gas_bayesshap.game.brute_force import brute_force_shapley
from gas_bayesshap.utils.config import load_config
print("GAS-BayesSHAP", gas_bayesshap.__version__)
print("numpy", np.__version__)"""))

A(md("## 2. Configuration"))
A(cell("""CONFIG = dict(load_config("../configs/default.yaml"))
CONFIG.update({
    "domain_game": "membership",
    "output_bounds": (0.0, 1.0),
    "epsilon": 0.45,
    "max_budget": 300,
    "n_active_steps": 10,
    "n_pilot": 3,
    "checkpoint_enabled": False,   # notebook run: no checkpoint dirs
    "cache_enabled": False,
    "log_level": "NONE",
    "results_dir": "../results/notebook",
    "checkpoints_dir": "../checkpoints/notebook",
})
print(json.dumps(CONFIG, indent=2, default=str)[:600])"""))

A(md("## 3. Mathematical validation (Lemma D & E vs brute force)"))
A(cell("""from gas_bayesshap.kernels.covariance import lemma_D_cross_cov, lemma_E_prior_cov
from gas_bayesshap.kernels.hamming import ExponentialHammingKernel
from gas_bayesshap.game.brute_force import brute_force_cross_covariance, brute_force_prior_covariance

kernel = ExponentialHammingKernel(1.0, 1.5)
for m in [2, 3, 4]:
    dE = float(np.max(np.abs(lemma_E_prior_cov(kernel, m) - brute_force_prior_covariance(kernel, m))))
    print(f"Lemma E M={m}: max|diff| = {dE:.2e}")
S = np.array([True, False, True, False])
dD = float(np.max(np.abs(lemma_D_cross_cov(kernel, S, 4) - brute_force_cross_covariance(kernel, S, 4))))
print(f"Lemma D M=4: max|diff| = {dD:.2e}")"""))

A(md("## 4. Domain game (membership)"))
A(cell("""rng = np.random.RandomState(0)
M = 5
w = rng.randn(M)
background = rng.randn(8, M)

def g_c(x):
    return 1.0 / (1.0 + np.exp(-np.dot(x, w) / np.sqrt(M)))

oracle, spec = membership_game(g_c, background)
print("game:", spec.name, "| bounds:", spec.output_bounds, "| R_delta_res =", spec.r_delta_res)
x = np.ones(M)"""))

A(md("## 5. Background (frozen, interventional imputation)"))
A(cell("""print("background shape:", oracle.background.shape)
print("background_hash:", oracle.background_h)
print("E_base = E[f(Z)] =", round(oracle.E_base, 6))
S = np.array([True, False, True, False, True])
print("v(S) =", round(oracle.evaluate(x, S), 6), "| oracle deterministic:", oracle.validate_determinism(x, S))"""))

A(md("## 6. Exact reference (brute force, small M)"))
A(cell("""from gas_bayesshap.benchmarking.exact import exact_shapley_for_oracle
exact = exact_shapley_for_oracle(oracle, x, M)
print("exact phi:", np.round(exact["shapley_values"], 6))
print("efficiency error:", f"{exact['efficiency_error']:.2e}")"""))

A(md("## 7. GP stage (Module A: active bounded-linear surrogate)"))
A(cell("""eng = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(0), config=CONFIG)
gp = eng.explain_stage1_only(x, n_active_steps=10)
print("surrogate phi(m_b):", np.round(gp["surrogate_shapley"], 6))
print("GP observations:", eng._surrogate.D_coalitions.shape[0])
print("posterior std:", np.round(gp["posterior_std"], 8))"""))

A(md("## 8. Bounded surrogate"))
A(cell("""s = eng._surrogate
print("h_lb =", round(s.h_lb, 6), "| h_ub =", round(s.h_ub, 6))
print("lambda =", round(s.scale, 6), "| c =", round(s.shift, 6))
from gas_bayesshap.gp.posterior import validate_surrogate_boundedness
ok = validate_surrogate_boundedness(s.D_coalitions, s.alpha, eng.kernel, s.scale, s.shift, M, 0.0, 1.0)
print("m_b in [0, 1] over all 2^M coalitions:", ok)"""))

A(md("## 9. Surrogate Shapley"))
A(cell("""print("phi(m_b) = lambda * K_phi,D * alpha")
print(np.round(gp["surrogate_shapley"], 6))"""))

A(md("## 10. Residual pilot (Lemma G + interior pilot)"))
A(cell("""store, sigma_res, neyman, start_iter = eng._stage2_enter(x, n_pil=3, resumed_state=None, checkpoint=False)
print("residual records:", store.n_records)
print("interior counts (s=1..M-2):", store.counts_matrix()[1:-1, :].sum(axis=1))
print("neyman probabilities:", np.round(neyman.probabilities, 4))"""))

A(md("## 11. Neyman allocation (coupled adjacent-stratum program)"))
A(cell("""print("objective:", round(neyman.objective_value, 6), "| success:", neyman.success)
print("allocation counts (K_cert=100):", neyman.counts)"""))

A(md("## 12. Adaptive certification (anytime empirical-Bernstein)"))
A(cell("""t0 = time.time()
result = eng.explain(x, epsilon=CONFIG["epsilon"], delta=0.05,
                     max_budget=CONFIG["max_budget"], n_pilot=3, n_active_steps=10)
print(f"status={result['status']}  converged={result['converged']}  "
      f"rounds={result['num_sampling_rounds']}  t={time.time()-t0:.2f}s")"""))

A(md("## 13. Efficiency projection"))
A(cell("""print("sum(phi*) =", round(float(np.sum(result["shapley_values"])), 9),
      "| delta_total =", round(result["delta_total"], 9))
print("surrogate:", np.round(result["surrogate_shapley"], 5))
print("residual :", np.round(result["residual_shapley"], 5))
print("final    :", np.round(result["shapley_values"], 5))"""))

A(md("## 14. Exact-vs-estimated comparison"))
A(cell("""err = np.abs(np.asarray(result["shapley_values"]) - exact["shapley_values"])
print("MAE vs exact:", round(float(np.mean(err)), 6))
print("RMSE vs exact:", round(float(np.sqrt(np.mean(err**2))), 6))
covered = np.all(np.abs(np.asarray(result["shapley_values"]) - exact["shapley_values"])
                 <= np.asarray(result["certified_projected_widths"]))
print("exact contained in certified intervals:", covered)"""))

A(md("## 15. Runtime / query benchmark"))
A(cell("""print(f"coalition evals : {result['num_coalition_evals']}")
print(f"model evals     : {result['num_model_evals']}")
print(f"GP predictions  : {result['num_gp_predictions']}")
print(f"residual samples: {result['num_residual_samples']}")
print(f"sampling rounds : {result['num_sampling_rounds']}")"""))

A(md("## 16. Checkpoint / resume demonstration"))
A(cell("""import tempfile, os
td = tempfile.mkdtemp()
cfg_r = dict(CONFIG)
cfg_r.update({"checkpoint_enabled": True, "cache_enabled": True, "persist_cache": True,
              "results_dir": os.path.join(td, "results"), "checkpoints_dir": os.path.join(td, "ck"),
              "run_id": "nb-resume", "log_level": "NONE"})
r1 = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(0), config=cfg_r)
res1 = r1.explain(x, epsilon=CONFIG["epsilon"], delta=0.05, max_budget=300,
                  n_pilot=3, n_active_steps=10, resume=False)
r2 = GASBayesSHAP(oracle=oracle, rng=np.random.RandomState(0), config=cfg_r)
res2 = r2.explain(x, epsilon=CONFIG["epsilon"], delta=0.05, max_budget=300,
                  n_pilot=3, n_active_steps=10, resume=True)
print("resume == fresh:", np.allclose(res1["shapley_values"], res2["shapley_values"], atol=1e-9))
print("latest checkpoint:", r2._checkpoint_manager.manifest.latest()["stage"])"""))

A(md("## 17. Final report"))
A(cell("""print(json.dumps({k: result[k] for k in [
    "status", "converged", "certificate_is_rigorous", "range_bound_is_heuristic",
    "num_coalition_evals", "num_model_evals", "sign_certified_features"]}, indent=2))
print("shapley_values      :", np.round(result["shapley_values"], 6).tolist())
print("raw widths          :", np.round(result["raw_confidence_widths"], 6).tolist())
print("projected widths    :", np.round(result["certified_projected_widths"], 6).tolist())
print("\\nNotebook sections 1-17 executed against the real package.")"""))

A(md("## Status dashboard"))
A(cell("""def status_dashboard(engine):
    s = engine.status()
    keys = ["run_id", "M", "domain_game", "current_stage", "iteration",
            "gp_observations", "residual_observations", "num_coalition_evals",
            "num_model_evals", "sampling_rounds", "cache_hit_rate",
            "latest_checkpoint"]
    for k in keys:
        print(f"  {k:24s}: {s.get(k)}")

print("engine.status():")
status_dashboard(eng)"""))

A(md("---\n*GAS-BayesSHAP v11.0 — implementation spec `specs/GAS_BayesSHAP_Implementation_Spec (4).md`.*"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "run_all.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(cells)} cells")
