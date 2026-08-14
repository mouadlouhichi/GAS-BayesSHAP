"""GAS-BayesSHAP core estimator (Module A + Module B + projection).

This is the authoritative orchestrator implementing the v11.0 spec:

* **Module A** (active bounded-linear GP control variate):
  seeds -> active A-optimal acquisition -> bounded linear surrogate
  ``m_b(S) = c + lambda h(S)`` -> analytical ``phi(m_b) = lambda K_phi,D alpha``
  -> ``lambda^2`` scaled posterior covariance.
* **Module B** (Neyman-stratified residual certifier):
  Lemma-G deterministic extreme strata -> add-one/remove-one residual sampling
  -> coupled adjacent-stratum Neyman allocation -> anytime
  empirical-Bernstein confidence sequences with strict budget guard.
* **Projection**: posterior-diagonal uncertainty-weighted efficiency projection
  (Theorem C) + post-projection certificate (Corollary C.1) + sign certification
  (Definition 1).

The control flow and arithmetic reproduce the spec's certified inline
reference implementation exactly (verified 10/10 on the spec test suite);
checkpointing, resuming, caching, structured logging and the results layer
are additive engineering (spec sections 33-52).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .._version import __version__ as _engine_version
from ..acquisition.scoring import acquisition_score
from ..cache.coalition_cache import CoalitionCache
from ..certification.bernstein import residual_widths
from ..certification.confidence_sequences import anytime_check
from ..certification.projection import (
    corollary_widths,
    project_efficiency,
    sign_certified,
)
from ..checkpointing.compatibility import CheckpointCompatibilityError
from ..checkpointing.manager import CheckpointManager
from ..game.oracle import CoalitionOracle
from ..game.subsets import candidate_pool, random_subset, seed_coalitions
from ..gp.control_variate import (
    BoundedLinearSurrogate,
    heuristic_output_bounds,
)
from ..gp.updates import rank1_inverse_update_detailed
from ..kernels.covariance import lemma_D_cross_cov, lemma_D_cross_cov_matrix, lemma_E_prior_cov
from ..kernels.hamming import ExponentialHammingKernel
from ..logging.events import EventLogger
from ..numerics.validation import NumericalFailure, assert_finite, safe_std
from ..residual.estimator import raw_unified_estimator, residual_shapley
from ..residual.neyman import NeymanSolution, solve_coupled_neyman_allocation
from ..residual.sampling import add_one_residual, remove_one_residual
from ..residual.strata import StratumStore
from ..utils.config import load_default_config
from ..utils.hashing import config_hash as _config_hash
from ..utils.hashing import input_hash
from ..utils.reproducibility import environment_manifest, git_commit_and_dirty
from ..utils.rng_state import dict_to_rng_state, rng_state_to_dict
from ..utils.serialization import ensure_dir
from .results import ResultStatus, RunResults, write_run_results
from .state import RunState

# --------------------------------------------------------------------------- #
# Stage names (spec section 3)
# --------------------------------------------------------------------------- #
STAGE_PREFLIGHT = "PREFLIGHT"
STAGE_ORACLE_VALIDATION = "ORACLE_VALIDATION"
STAGE_MATH_VALIDATION = "MATHEMATICAL_VALIDATION"
STAGE_GP_INIT = "GP_INITIALIZATION"
STAGE_ACTIVE_GP = "ACTIVE_GP"
STAGE_BOUNDED_SURROGATE = "BOUNDED_SURROGATE"
STAGE_SURROGATE_SHAPLEY = "SURROGATE_SHAPLEY"
STAGE_RESIDUAL_PILOT = "RESIDUAL_PILOT"
STAGE_NEYMAN = "NEYMAN_ALLOCATION"
STAGE_ADAPTIVE = "ADAPTIVE_CERTIFICATION"
STAGE_PROJECTION = "EFFICIENCY_PROJECTION"
STAGE_FINAL = "FINAL_RESULT"
STAGE_BENCHMARK = "BENCHMARK"
STAGE_REPORT = "REPORT"
STAGE_COMPLIANCE = "COMPLIANCE_AUDIT"

ALL_STAGES = [
    STAGE_PREFLIGHT, STAGE_ORACLE_VALIDATION, STAGE_MATH_VALIDATION,
    STAGE_GP_INIT, STAGE_ACTIVE_GP, STAGE_BOUNDED_SURROGATE,
    STAGE_SURROGATE_SHAPLEY, STAGE_RESIDUAL_PILOT, STAGE_NEYMAN,
    STAGE_ADAPTIVE, STAGE_PROJECTION, STAGE_FINAL, STAGE_BENCHMARK,
    STAGE_REPORT, STAGE_COMPLIANCE,
]


class GASBayesSHAP:
    """Gaussian-Adaptive Stratified Bayesian Shapley Estimator (v11.0).

    Parameters
    ----------
    model_fn:
        ``model_fn(x) -> float``.  Either ``model_fn``/``background`` or an
        explicit ``oracle`` must be provided.
    background:
        ``(B, M)`` empirical background (frozen).
    output_bounds:
        Known global output range ``(L, U)``; ``None`` triggers the
        Remark-2.2 heuristic bounds (flagged, never rigorous).
    sigma0, lengthscale, eta:
        Kernel amplitude, lengthscale (:math:`\\rho = e^{-1/\\ell}`), jitter.
    oracle:
        Pre-built :class:`~gas_bayesshap.game.oracle.CoalitionOracle`.
    rng:
        ``numpy.random.RandomState``; defaults to a fresh one seeded from
        ``config['seed']``.
    config:
        Optional configuration dict (merged over defaults).
    run_id:
        Run identifier used for results/checkpoints/logs.
    """

    def __init__(
        self,
        model_fn=None,
        background=None,
        output_bounds: Optional[tuple] = None,
        sigma0: float = 1.0,
        lengthscale: float = 1.5,
        eta: float = 1e-4,
        oracle: Optional[CoalitionOracle] = None,
        rng: Optional[np.random.RandomState] = None,
        config: Optional[dict] = None,
        run_id: Optional[str] = None,
        model_tag: Optional[str] = None,
        logger=None,
    ):
        self.config = dict(load_default_config())
        if config:
            self.config.update(config)
        if output_bounds is not None:
            self.config["output_bounds"] = tuple(float(v) for v in output_bounds)

        self.sigma0 = float(sigma0)
        self.lengthscale = float(lengthscale)
        self.eta = float(eta)
        self.rho = float(np.exp(-1.0 / lengthscale))

        self.config_hash = _config_hash(self.config)
        self.rng = rng if rng is not None else np.random.RandomState(int(self.config.get("seed", 42)))
        self.logger = logger
        self.run_id = run_id or str(self.config.get("run_id") or f"run-{self.config_hash[:8]}")
        self.domain_game = str(self.config.get("domain_game", "membership"))

        # --- oracle ------------------------------------------------------ #
        self._event_logger: Optional[EventLogger] = None
        if oracle is not None:
            self.oracle = oracle
            self._event_logger = getattr(self.oracle, "logger", None)
            # adopt the oracle's declared output bounds for rigor when the
            # config does not override them
            if self.config.get("output_bounds") is None and oracle.output_bounds is not None:
                self.config["output_bounds"] = tuple(float(v) for v in oracle.output_bounds)
                self.config_hash = _config_hash(self.config)
        else:
            if model_fn is None or background is None:
                raise ValueError("either (model_fn, background) or oracle must be provided")
            self.oracle = CoalitionOracle(
                model_fn=model_fn,
                background=background,
                output_bounds=self.config.get("output_bounds"),
                model_tag=model_tag,
                cache=None,  # attached in _prepare_run
                config_hash=self.config_hash,
                logger=None,
            )
        self.M = int(self.oracle.M)
        self.B = int(self.oracle.B)

        # --- engine state ------------------------------------------------ #
        self.state = RunState(run_id=self.run_id)
        self.kernel = ExponentialHammingKernel(sigma0=self.sigma0, lengthscale=self.lengthscale)
        self.K_phi_phi = lemma_E_prior_cov(self.kernel, self.M)

        self._surrogate: Optional[BoundedLinearSurrogate] = None
        self._gp_predictions_counter: List[int] = []
        self._results: Optional[RunResults] = None
        self._residual_store: Optional[StratumStore] = None

        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._cache: Optional[CoalitionCache] = None
        self._results_dir: Optional[Path] = None
        self._prepared = False

    # ======================================================================= #
    # Infrastructure
    # ======================================================================= #
    def _prepare_run(self, checkpoint: bool = True) -> None:
        if self._prepared:
            return
        cfg = self.config
        results_dir = Path(cfg.get("results_dir", "results/runs"))
        ckpt_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))

        if self._event_logger is None and cfg.get("log_level", "INFO") != "NONE":
            run_log_dir = ensure_dir(results_dir / self.run_id / "logs")
            self._event_logger = EventLogger(self.run_id, run_log_dir, counters={
                "num_coalition_evals": self.oracle.total_coalition_evals,
                "num_model_evals": self.oracle.total_model_evals,
                "iteration": 0,
            })
            # stdlib logger -> run.log / errors.log (spec section 37);
            # unique per-instance name so cached loggers never collide
            from ..logging.logger import setup_logger
            self._run_logger = setup_logger(
                run_log_dir, name=f"gas_bayesshap.{self.run_id}.{id(self)}",
                level=cfg.get("log_level", "INFO"),
            )
            # wire the oracle's event logger (oracle_calls.jsonl)
            self.oracle.logger = self._event_logger
        else:
            self._run_logger = None

        if cfg.get("cache_enabled", True) and self._cache is None:
            cache_path = None
            if cfg.get("persist_cache", True):
                cache_path = results_dir / self.run_id / "cache_manifest.json"
            self._cache = CoalitionCache(
                config_hash=self.config_hash,
                oracle_hash=self.oracle.oracle_h,
                background_hash=self.oracle.background_h,
                persist_path=cache_path,
                enabled=True,
            )
            self.oracle.cache = self._cache
            self.oracle.config_hash = self.config_hash

        if checkpoint and cfg.get("checkpoint_enabled", True) and self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(
                run_id=self.run_id,
                directory=ckpt_dir,
                config_hash=self.config_hash,
                oracle_hash=self.oracle.oracle_h,
                background_hash=self.oracle.background_h,
                M=self.M,
                engine_version=_engine_version,
                logger=self._event_logger,
            )
        self._results_dir = results_dir
        self._prepared = True

    def _log_event(
        self,
        stage: str,
        event: str,
        status: str = "",
        topic: str = "events",
        **fields,
    ) -> None:
        """Structured event logging: JSONL topic files + run.log mirror.

        ``topic`` selects the per-topic JSONL file (spec section 37):
        oracle_calls / gp_updates / acquisition / residual_sampling / neyman /
        certification / checkpoints / events.  Events are also mirrored to the
        stdlib ``run.log`` and WARNING+ to ``errors.log``.
        """
        if self._event_logger is not None:
            self._event_logger.set_stage(stage)
            self._event_logger.set_counters(
                num_coalition_evals=self.oracle.total_coalition_evals,
                num_model_evals=self.oracle.total_model_evals,
                iteration=self.state.iteration,
            )
            self._event_logger.event(topic, event=event, status=status, **fields)
        if self._run_logger is not None:
            level = logging.WARNING if status in ("WARNING", "ERROR", "MISSING_STRATA") else logging.INFO
            self._run_logger.log(
                level,
                "[%s] %s%s%s",
                stage,
                event,
                f" | {status}" if status else "",
                f" | {json.dumps(fields, default=str)[:300]}" if fields else "",
            )

    # ======================================================================= #
    # Public API
    # ======================================================================= #
    def explain(
        self,
        x: np.ndarray,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        max_budget: Optional[int] = None,
        n_pilot: Optional[int] = None,
        n_active_steps: Optional[int] = None,
        max_rounds: Optional[int] = None,
        resume: bool = False,
        checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full GAS-BayesSHAP dual estimation for instance ``x``.

        Returns the spec result dictionary (12 reference keys + section-49
        extension keys).  ``max_budget`` bounds the individual coalition
        evaluations spent in the Stage-2 adaptive loop (spec section 32);
        ``max_rounds`` optionally caps the number of Stage-2 sampling
        iterations (run-control knobs that do not alter the config hash).
        """
        cfg = self.config
        eps = float(epsilon if epsilon is not None else cfg["epsilon"])
        dlt = float(delta if delta is not None else cfg["delta"])
        budget = int(max_budget if max_budget is not None else cfg["max_budget"])
        n_pil = int(n_pilot if n_pilot is not None else cfg["n_pilot"])
        n_act = int(n_active_steps if n_active_steps is not None else cfg["n_active_steps"])
        rounds_cap = (
            int(max_rounds) if max_rounds is not None
            else (int(cfg["max_rounds"]) if cfg.get("max_rounds") is not None else None)
        )

        self._prepare_run(checkpoint=checkpoint)
        x = np.asarray(x, dtype=np.float64)
        self.state.stage = STAGE_PREFLIGHT
        self.state.iteration = 0

        x_h = input_hash(x)
        evals_start_coal = self.oracle.total_coalition_evals
        evals_start_model = self.oracle.total_model_evals

        # --- PREFLIGHT --------------------------------------------------- #
        v_N = self.oracle.evaluate(x, np.ones(self.M, dtype=bool))
        delta_total = v_N - self.oracle.E_base
        self.state.extra["delta_total"] = float(delta_total)
        self.state.extra["x_hash"] = x_h

        if self.config.get("output_bounds") is not None:
            L, U = self.config["output_bounds"]
            R_delta_res = 4.0 * (U - L)
            heuristic_bounds = False
        else:
            L, U = heuristic_output_bounds(self.oracle.E_base, v_N, delta_total)
            R_delta_res = 4.0 * (U - L)
            heuristic_bounds = True
        self.state.extra.update({"L": float(L), "U": float(U), "R_delta_res": float(R_delta_res)})

        self._log_event(STAGE_PREFLIGHT, "preflight", "OK",
                        delta_total=delta_total, L=L, U=U, R_delta_res=R_delta_res,
                        epsilon=eps, delta=dlt, max_budget=budget)

        # --- OPTIONAL ORACLE_VALIDATION ---------------------------------- #
        if cfg.get("oracle_validation", False):
            self.state.stage = STAGE_ORACLE_VALIDATION
            self._validate_oracle(x)

        # --- OPTIONAL MATHEMATICAL_VALIDATION ---------------------------- #
        if cfg.get("mathematical_validation", False) and self.M <= 6:
            self.state.stage = STAGE_MATH_VALIDATION
            self._validate_math()

        # --- RESUME ------------------------------------------------------ #
        resumed_state: Optional[Dict[str, Any]] = None
        if resume and self._checkpoint_manager is not None:
            resumed_state = self._resume_checkpoint(x_h)
            if resumed_state is not None and resumed_state.get("done"):
                return resumed_state["result"]

        # On resume the counters were restored from the checkpoint: re-baseline
        # the per-call deltas so only *new* evaluations are reported.
        if resumed_state is not None:
            evals_start_coal = self.oracle.total_coalition_evals
            evals_start_model = self.oracle.total_model_evals

        # --- STAGE 1 (skipped on resume: surrogate restored from checkpoint) - #
        if resumed_state is None:
            self._stage1_active_gp(x, n_act, checkpoint=checkpoint)

        # --- STAGE 2 ----------------------------------------------------- #
        store, sigma_res, neyman, start_iter = self._stage2_enter(
            x, n_pil, resumed_state, checkpoint=checkpoint
        )
        widths, converged, budget_exhausted, missing_strata = self._stage2_adaptive(
            x, store, sigma_res, neyman, start_iter,
            eps, dlt, budget, checkpoint=checkpoint, max_rounds=rounds_cap,
        )

        # --- STAGE 3 ----------------------------------------------------- #
        result = self._stage3_assemble(
            x, store, sigma_res, widths, converged, budget_exhausted,
            missing_strata, heuristic_bounds, delta_total,
            evals_start_coal, evals_start_model, eps, dlt, x_h,
            checkpoint=checkpoint,
        )
        self.state.stage = STAGE_FINAL
        self._log_event(STAGE_FINAL, "final_result", result.get("status"))
        if self._cache is not None:
            self._cache.persist()
        return result

    # ======================================================================= #
    # STAGE 1
    # ======================================================================= #
    def _stage1_active_gp(self, x, n_act: int, checkpoint: bool = True) -> Dict[str, Any]:
        """Seed design -> active A-optimal acquisition -> frozen surrogate."""
        M = self.M
        cfg = self.config
        rng = self.rng
        eta_sq = self.eta ** 2
        L = float(self.state.extra.get("L", 0.0))
        U = float(self.state.extra.get("U", 0.0))

        self.state.stage = STAGE_GP_INIT
        D_gp_coalitions: List[np.ndarray] = []
        D_gp_y: List[float] = []
        inv_K_DD = np.empty((0, 0), dtype=np.float64)
        K_phi_D = np.empty((M, 0), dtype=np.float64)

        def _try_add(S: np.ndarray, v_val: float) -> bool:
            """Rank-1 update; near-duplicates are skipped (Schur guard)."""
            nonlocal inv_K_DD, K_phi_D
            k_vec = np.array([self.kernel.k(S, S_obs) for S_obs in D_gp_coalitions], dtype=np.float64)
            upd = rank1_inverse_update_detailed(inv_K_DD, k_vec, self.kernel.k_self(), eta_sq)
            if not upd.ok:
                self._log_event(STAGE_ACTIVE_GP, "near_duplicate", "SKIPPED",
                                topic="gp_updates",
                                coalition=bool_mask_to_int(S), schur=upd.schur,
                                threshold=upd.threshold, action=upd.action)
                return False
            inv_K_DD = upd.inv_K
            K_phi_D = np.hstack([K_phi_D, lemma_D_cross_cov(self.kernel, S, M).reshape(-1, 1)])
            D_gp_coalitions.append(S.copy())
            D_gp_y.append(v_val - self.oracle.E_base)
            return True

        # --- seeds -------------------------------------------------------- #
        seeds = seed_coalitions(rng, M)
        for S_seed in seeds:
            v_val = self.oracle.evaluate(x, S_seed)
            ok = _try_add(S_seed, v_val)
            self._log_event(STAGE_GP_INIT, "seed_evaluated", "OK" if ok else "SKIPPED",
                            topic="gp_updates", coalition=bool_mask_to_int(S_seed))
        self._log_event(STAGE_GP_INIT, "seeds_complete", "OK",
                        topic="gp_updates",
                        n_seeds=len(seeds), n_accepted=len(D_gp_coalitions))

        # --- active acquisition -------------------------------------------- #
        self.state.stage = STAGE_ACTIVE_GP
        pool_size = int(cfg["pool_size"]) if cfg.get("pool_size") else max(32, 2 * M)
        for step in range(int(n_act)):
            best_score = -1.0
            best_S = None
            for p in candidate_pool(rng, M, pool_size):
                score = acquisition_score(p, D_gp_coalitions, inv_K_DD, K_phi_D, self.kernel, self.eta)
                if score > best_score:
                    best_score = score
                    best_S = p
            if best_S is not None:
                v_val = self.oracle.evaluate(x, best_S)
                ok = _try_add(best_S, v_val)
                self._log_event(STAGE_ACTIVE_GP, "acquisition",
                                "OK" if ok else "SKIPPED",
                                topic="acquisition",
                                coalition=bool_mask_to_int(best_S), score=best_score,
                                n_gp=len(D_gp_coalitions))

        # --- freeze bounded linear surrogate -------------------------------- #
        self.state.stage = STAGE_BOUNDED_SURROGATE
        y_gp_vec = np.array(D_gp_y, dtype=np.float64)
        alpha = inv_K_DD @ y_gp_vec
        assert_finite(alpha, "alpha")
        D_matrix = np.array(D_gp_coalitions, dtype=bool)
        K_phi_D_final = lemma_D_cross_cov_matrix(self.kernel, D_matrix, M)

        pos_alpha_sum = float(np.sum(alpha[alpha > 0]))
        neg_alpha_sum = float(np.sum(alpha[alpha < 0]))
        h_lb = self.kernel.sigma0_sq * ((self.kernel.rho ** M) * pos_alpha_sum + neg_alpha_sum)
        h_ub = self.kernel.sigma0_sq * (pos_alpha_sum + (self.kernel.rho ** M) * neg_alpha_sum)
        if h_ub > h_lb:
            scale = min(1.0, (U - L) / (h_ub - h_lb))
        else:
            scale = 1.0
        shift = L - scale * h_lb

        self._surrogate = BoundedLinearSurrogate(
            D_coalitions=D_matrix, D_y=y_gp_vec, inv_K_DD=inv_K_DD, K_phi_D=K_phi_D_final,
            alpha=alpha, h_lb=float(h_lb), h_ub=float(h_ub),
            scale=float(scale), shift=float(shift), M=M, eta=self.eta,
            _K_phi_phi=self.K_phi_phi,
        )

        # --- surrogate Shapley + posterior covariance ------------------------ #
        self.state.stage = STAGE_SURROGATE_SHAPLEY
        phi_m_D = self._surrogate.surrogate_shapley(self.kernel)
        phi_cov_h = self.K_phi_phi - (K_phi_D_final @ inv_K_DD @ K_phi_D_final.T)
        phi_cov_h = 0.5 * (phi_cov_h + phi_cov_h.T)
        phi_cov_mb = (scale ** 2) * phi_cov_h
        posterior_variances = np.maximum(np.diag(phi_cov_mb), 1e-10)

        self._log_event(STAGE_BOUNDED_SURROGATE, "bounded_surrogate", "OK",
                        topic="gp_updates",
                        h_lb=h_lb, h_ub=h_ub, lambda_=scale, c=shift,
                        n_gp=len(D_gp_coalitions))
        self._log_event(STAGE_SURROGATE_SHAPLEY, "surrogate_shapley", "OK",
                        topic="gp_updates",
                        phi_m_D=phi_m_D.tolist(),
                        posterior_std=np.sqrt(posterior_variances).tolist())

        # boundedness sweep (validation; small M only)
        if cfg.get("validate_boundedness", False) and M <= int(cfg.get("boundedness_sweep_max_M", 12)):
            from ..gp.posterior import validate_surrogate_boundedness
            ok_b = validate_surrogate_boundedness(
                D_matrix, alpha, self.kernel, scale, shift, M, L, U, tol=1e-10
            )
            if not ok_b:
                raise NumericalFailure("bounded linear surrogate violated [L, U] on some coalition")
            self._log_event(STAGE_BOUNDED_SURROGATE, "boundedness_sweep", "OK")

        # ---- checkpoint: gp_stage ------------------------------------------- #
        if checkpoint and self._checkpoint_manager is not None:
            self._checkpoint_gp(float(self.state.extra.get("delta_total", 0.0)))

        self.state.extra["phi_m_D"] = phi_m_D
        self.state.extra["posterior_variances"] = posterior_variances
        return {
            "phi_m_D": phi_m_D,
            "posterior_variances": posterior_variances,
        }

    # ======================================================================= #
    # STAGE 2
    # ======================================================================= #
    def _stage2_enter(self, x, n_pil: int, resumed_state: Optional[Dict], checkpoint: bool = True):
        """Lemma-G extreme init + interior pilot + initial Neyman allocation."""
        M = self.M
        rng = self.rng
        self.state.stage = STAGE_RESIDUAL_PILOT

        if resumed_state is not None and resumed_state.get("store") is not None:
            store = resumed_state["store"]
            sigma_res = resumed_state["sigma_res"]
            neyman = resumed_state["neyman"]
            start_iter = int(resumed_state.get("iteration", 0))
            self._residual_store = store
            self._log_event(STAGE_RESIDUAL_PILOT, "resume_restored", "OK",
                            iteration=start_iter, n_records=store.n_records)
            return store, sigma_res, neyman, start_iter

        store = StratumStore(M)
        sigma_res = np.zeros((M, M), dtype=np.float64)

        # ---- Lemma G: deterministic extreme strata (s=0 and s=M-1) ---------- #
        v_empty = self.oracle.evaluate(x, np.zeros(M, dtype=bool))
        m_empty = self._predict(np.zeros(M, dtype=bool))
        for i in range(M):
            S_i = np.zeros(M, dtype=bool)
            S_i[i] = True
            v_Si = self.oracle.evaluate(x, S_i)
            m_Si = self._predict(S_i)
            # reference: (v(S_i) - v(empty)) - (m(S_i) - m(empty)) -> stratum 0
            store.append(
                feature=i, stratum=0, coalition=S_i, direction="add_one",
                value=add_one_residual(v_empty, v_Si, m_empty, m_Si),
                iteration=0, random_seed=None,
            )
            sigma_res[0, i] = 0.0  # exact singleton stratum variance

        v_full = self.oracle.evaluate(x, np.ones(M, dtype=bool))
        m_full = self._predict(np.ones(M, dtype=bool))
        for missing_i in range(M):
            S_no_i = np.ones(M, dtype=bool)
            S_no_i[missing_i] = False
            v_Sno_i = self.oracle.evaluate(x, S_no_i)
            m_Sno_i = self._predict(S_no_i)
            # reference: (v(full) - v(full\{i})) - (m(full) - m(full\{i})) -> stratum M-1
            store.append(
                feature=missing_i, stratum=M - 1, coalition=S_no_i, direction="remove_one",
                value=remove_one_residual(v_full, v_Sno_i, m_full, m_Sno_i),
                iteration=0, random_seed=None,
            )
            sigma_res[M - 1, missing_i] = 0.0

        self._log_event(STAGE_RESIDUAL_PILOT, "extreme_strata", "OK",
                        topic="residual_sampling", n_calls=2 * M + 2)

        # ---- pilot: interior strata ------------------------------------------- #
        for s in range(1, max(1, M - 1)):
            for _ in range(int(n_pil)):
                S = random_subset(rng, M, s)
                v_S = self.oracle.evaluate(x, S)
                m_S = self._predict(S)
                for i in range(M):
                    if not S[i]:
                        S_u = S.copy()
                        S_u[i] = True
                        v_Su = self.oracle.evaluate(x, S_u)
                        m_Su = self._predict(S_u)
                        store.append(
                            feature=i, stratum=s, coalition=S_u, direction="add_one",
                            value=add_one_residual(v_S, v_Su, m_S, m_Su),
                            iteration=0, random_seed=None,
                        )
                    elif s > 0:
                        S_m = S.copy()
                        S_m[i] = False
                        v_Sm = self.oracle.evaluate(x, S_m)
                        m_Sm = self._predict(S_m)
                        store.append(
                            feature=i, stratum=s - 1, coalition=S_m, direction="remove_one",
                            value=remove_one_residual(v_S, v_Sm, m_S, m_Sm),
                            iteration=0, random_seed=None,
                        )
        self._log_event(STAGE_RESIDUAL_PILOT, "pilot_complete", "OK",
                        topic="residual_sampling", n_records=store.n_records)

        # ---- sigma_res + initial Neyman ---------------------------------------- #
        for s in range(1, max(1, M - 1)):
            for i in range(M):
                sigma_res[s, i] = safe_std(store.values(i, s), 0.5)

        self.state.stage = STAGE_NEYMAN
        neyman = solve_coupled_neyman_allocation(sigma_res, M, K_cert=1.0)
        self._log_event(STAGE_NEYMAN, "neyman_allocated", "OK",
                        topic="neyman",
                        probabilities=neyman.probabilities.tolist(),
                        objective=neyman.objective_value,
                        optimizer_status=neyman.status, message=neyman.message,
                        fallback_used=neyman.fallback_used)

        self._residual_store = store

        # ---- checkpoint: residual_stage iteration 0 ----------------------------- #
        if checkpoint and self._checkpoint_manager is not None:
            self._checkpoint_residual(store, sigma_res, neyman, iteration=0, widths=None)

        return store, sigma_res, neyman, 0

    # ----------------------------------------------------------------------- #
    def _stage2_adaptive(
        self, x, store: StratumStore, sigma_res, neyman: NeymanSolution,
        start_iter: int, epsilon: float, delta: float, max_budget: int,
        checkpoint: bool = True, max_rounds: Optional[int] = None,
    ):
        """Anytime adaptive residual sampling loop with strict budget guard."""
        M = self.M
        rng = self.rng
        cfg = self.config
        self.state.stage = STAGE_ADAPTIVE
        self.state.iteration = start_iter

        R_delta_res = float(self.state.extra.get("R_delta_res", 4.0))
        refresh_interval = (
            int(cfg["neyman_refresh_interval"])
            if cfg.get("neyman_refresh_interval")
            else max(1, 5 * M)
        )
        if max_rounds is None and cfg.get("max_rounds") is not None:
            max_rounds = int(cfg["max_rounds"])
        checkpoint_every = max(1, int(cfg.get("checkpoint_every", 1)))

        raw_widths = np.full(M, np.inf)
        # Cumulative Stage-2 attempted-evaluation counter (across resume):
        # `max_budget` is a *run-level* allowance, so a resumed run deducts the
        # Stage-2 work completed before the checkpoint instead of restarting
        # with a fresh budget (audit finding: High 1).
        if not hasattr(self, "_stage2_attempted_total"):
            self._stage2_attempted_total = 0
        iter_count = int(start_iter)
        converged = False
        budget_exhausted = False

        while True:
            # --- complete width vector ---------------------------------------- #
            raw_widths = residual_widths(store, sigma_res, M, delta, R_delta_res)
            check = anytime_check(raw_widths, epsilon)
            self._log_event(STAGE_ADAPTIVE, "width_check", "INFO",
                            topic="certification",
                            widths=raw_widths.tolist(), max_width=check.max_width,
                            mean_width=check.mean_width, median_width=check.median_width,
                            argmax=check.argmax_feature, converged=check.converged,
                            n_records=store.n_records)

            if check.converged:
                converged = True
                self.state.iteration = iter_count
                self._checkpoint_certification(store, sigma_res, neyman, raw_widths, True, checkpoint)
                break

            # --- strict budget guard ------------------------------------------- #
            # `max_budget` bounds the individual coalition evaluations of the
            # Stage-2 adaptive loop (spec section 32).  Each round would cost
            # `1 + M` coalition evaluations (base subset + one marginal per
            # player).  The counter is *cumulative across resume*, so a resumed
            # run respects the same total allowance as an uninterrupted one.
            # When the coalition cache is enabled, repeated draws are cache
            # hits (0 true evals), so we account *attempted* evaluations —
            # identical round accounting to the no-cache reference, and the
            # loop always terminates.
            round_cost_upper = 1 + M
            if self._stage2_attempted_total + round_cost_upper > max_budget:
                budget_exhausted = True
                self._log_event(STAGE_ADAPTIVE, "budget_exhausted", "BUDGET_EXHAUSTED",
                                topic="certification",
                                current_stage2_evals=self._stage2_attempted_total,
                                round_cost_upper=round_cost_upper, max_budget=max_budget)
                # leave a residual checkpoint so resume continues the adaptive loop
                self._checkpoint_residual(store, sigma_res, neyman, iteration=iter_count, widths=raw_widths)
                break

            if max_rounds is not None and iter_count >= int(max_rounds):
                budget_exhausted = True
                self._log_event(STAGE_ADAPTIVE, "max_rounds_reached", "BUDGET_EXHAUSTED",
                                topic="certification",
                                max_rounds=int(max_rounds))
                self._checkpoint_residual(store, sigma_res, neyman, iteration=iter_count, widths=raw_widths)
                break

            # --- dynamic Neyman reallocation -------------------------------------- #
            if iter_count > 0 and iter_count % refresh_interval == 0:
                prev_probs = neyman.probabilities.copy()
                for s in range(1, max(1, M - 1)):
                    for i in range(M):
                        sigma_res[s, i] = safe_std(store.values(i, s), 0.5)
                neyman = solve_coupled_neyman_allocation(sigma_res, M, K_cert=1.0)
                self._log_event(STAGE_NEYMAN, "neyman_refresh", "OK",
                                topic="neyman",
                                iteration=iter_count,
                                previous_probabilities=prev_probs.tolist(),
                                updated_probabilities=neyman.probabilities.tolist(),
                                objective=neyman.objective_value,
                                optimizer_status=neyman.status, message=neyman.message,
                                counts=neyman.counts.tolist(),
                                n_coalition_evals=self.oracle.total_coalition_evals,
                                n_records=store.n_records)

            # --- draw & evaluate one coalition path -------------------------------- #
            if np.sum(neyman.probabilities) > 0:
                s_target = int(rng.choice(M, p=neyman.probabilities))
            else:
                s_target = 1
            S_new = random_subset(rng, M, s_target)

            v_S = self.oracle.evaluate(x, S_new)
            m_S = self._predict(S_new)
            # record seed label: derived from the run RNG state WITHOUT consuming
            # randomness (consuming it here would break reference parity)
            round_seed = _rng_state_label(self.rng)
            for i in range(M):
                if not S_new[i]:
                    S_u = S_new.copy()
                    S_u[i] = True
                    v_Su = self.oracle.evaluate(x, S_u)
                    m_Su = self._predict(S_u)
                    store.append(
                        feature=i, stratum=s_target, coalition=S_u, direction="add_one",
                        value=add_one_residual(v_S, v_Su, m_S, m_Su),
                        iteration=iter_count, random_seed=round_seed,
                    )
                    if s_target != 0 and s_target != M - 1:
                        sigma_res[s_target, i] = safe_std(store.values(i, s_target), 0.5)
                elif s_target > 0:
                    S_m = S_new.copy()
                    S_m[i] = False
                    v_Sm = self.oracle.evaluate(x, S_m)
                    m_Sm = self._predict(S_m)
                    store.append(
                        feature=i, stratum=s_target - 1, coalition=S_m, direction="remove_one",
                        value=remove_one_residual(v_S, v_Sm, m_S, m_Sm),
                        iteration=iter_count, random_seed=round_seed,
                    )
                    if (s_target - 1) != 0 and (s_target - 1) != M - 1:
                        sigma_res[s_target - 1, i] = safe_std(store.values(i, s_target - 1), 0.5)

            self._log_event(STAGE_ADAPTIVE, "residual_round", "OK", iteration=iter_count,
                            topic="residual_sampling",
                            s_target=s_target, coalition=bool_mask_to_int(S_new),
                            n_records=store.n_records)

            iter_count += 1
            self._stage2_attempted_total += round_cost_upper
            self.state.iteration = iter_count
            if checkpoint and self._checkpoint_manager is not None and iter_count % checkpoint_every == 0:
                # checkpoint records state AFTER round iter_count-1, labelled with
                # the next iteration to run -> resume continues from iter_count.
                self._checkpoint_residual(store, sigma_res, neyman, iteration=iter_count, widths=raw_widths)

        missing_strata = not bool(np.all(np.isfinite(raw_widths)))
        if missing_strata:
            self._log_event(STAGE_ADAPTIVE, "missing_strata", "MISSING_STRATA",
                            topic="certification",
                            widths=raw_widths.tolist(),
                            strict=cfg.get("certification_mode", "STRICT"))
        self.state.iteration = iter_count
        return raw_widths, converged, budget_exhausted, missing_strata

    def explain_stage1_only(
        self,
        x: np.ndarray,
        epsilon: Optional[float] = None,
        n_active_steps: Optional[int] = None,
        checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """Run Module A only (preflight + active GP + bounded surrogate).

        Returns the analytical surrogate attribution and posterior quantities
        (used by ``scripts/run_bayesshap.py --until-stage gp`` and the
        GP-only benchmark baseline).
        """
        cfg = self.config
        self._prepare_run(checkpoint=checkpoint)
        x = np.asarray(x, dtype=np.float64)
        self.state.stage = STAGE_PREFLIGHT
        evals_start_coal = self.oracle.total_coalition_evals
        evals_start_model = self.oracle.total_model_evals

        v_N = self.oracle.evaluate(x, np.ones(self.M, dtype=bool))
        delta_total = v_N - self.oracle.E_base
        self.state.extra["delta_total"] = float(delta_total)
        self.state.extra["x_hash"] = input_hash(x)
        if self.config.get("output_bounds") is not None:
            L, U = self.config["output_bounds"]
        else:
            L, U = heuristic_output_bounds(self.oracle.E_base, v_N, delta_total)
        self.state.extra.update({"L": float(L), "U": float(U),
                                 "R_delta_res": float(4.0 * (U - L))})

        n_act = int(n_active_steps if n_active_steps is not None else cfg["n_active_steps"])
        gp_state = self._stage1_active_gp(x, n_act, checkpoint=checkpoint)
        phi_m_D = gp_state["phi_m_D"]
        posterior_variances = gp_state["posterior_variances"]
        self.state.stage = STAGE_FINAL
        return {
            "surrogate_shapley": phi_m_D,
            "posterior_std": np.sqrt(posterior_variances),
            "shapley_values": phi_m_D,
            "raw_confidence_widths": None,
            "certified_projected_widths": None,
            "num_coalition_evals": self.oracle.total_coalition_evals - evals_start_coal,
            "num_model_evals": self.oracle.total_model_evals - evals_start_model,
            "status": "GP_ONLY",
            "delta_total": float(delta_total),
            "surrogate_scale": float(self._surrogate.scale) if self._surrogate else 1.0,
        }

    # ======================================================================= #
    # STAGE 3
    # ======================================================================= #
    def _stage3_assemble(
        self, x, store, sigma_res, widths, converged, budget_exhausted,
        missing_strata, heuristic_bounds, delta_total,
        evals_start_coal, evals_start_model, epsilon, delta, x_h,
        checkpoint: bool = True,
    ) -> Dict[str, Any]:
        self.state.stage = STAGE_PROJECTION
        M = self.M

        phi_m_D = np.asarray(self.state.extra.get("phi_m_D"))
        posterior_variances = np.asarray(self.state.extra.get("posterior_variances"))

        phi_r_strat = residual_shapley(store, M)
        phi_raw = raw_unified_estimator(phi_m_D, phi_r_strat)
        phi_final = project_efficiency(phi_raw, float(delta_total), posterior_variances)
        certified_widths = corollary_widths(widths, posterior_variances)
        sign_cert = sign_certified(phi_final, certified_widths)

        # explicit diagnostic on numerical failure (spec sections 51-52):
        # NaN/Inf estimates must never be silently returned
        if not np.all(np.isfinite(phi_final)):
            bad = np.where(~np.isfinite(phi_final))[0].tolist()
            raise NumericalFailure(
                f"non-finite Shapley estimates at features {bad} — NaN/Inf residual "
                "samples detected (check model_fn determinism and output_bounds)"
            )

        # stratum-coverage completeness: a feature with missing cells has a
        # *partial* point estimate (unobserved cells contribute 0 to phi_res)
        # and must be flagged explicitly (audit finding: Medium 2).
        from ..residual.estimator import stratum_completeness
        point_complete, missing_cells = stratum_completeness(store, M)

        all_finite = bool(np.all(np.isfinite(widths)))
        call_coal = self.oracle.total_coalition_evals - evals_start_coal
        call_model = self.oracle.total_model_evals - evals_start_model
        n_gp_pred = len(self._gp_predictions_counter)
        n_res_samp = store.n_records

        rigorous = (not heuristic_bounds) and all_finite
        status, status_detail = _compose_status(
            converged=converged,
            rigorous=rigorous,
            heuristic=heuristic_bounds,
            budget_exhausted=budget_exhausted,
            missing_strata=missing_strata,
        )

        git = git_commit_and_dirty()
        results = RunResults(
            shapley_values=phi_final,
            surrogate_shapley=phi_m_D,
            residual_shapley=phi_r_strat,
            raw_confidence_widths=widths,
            certified_projected_widths=certified_widths,
            posterior_std=np.sqrt(posterior_variances),
            num_coalition_evals=int(call_coal),
            num_model_evals=int(call_model),
            num_gp_predictions=int(n_gp_pred),
            num_residual_samples=int(n_res_samp),
            num_sampling_rounds=int(self.state.iteration),
            converged=bool(converged),
            certificate_is_rigorous=bool(rigorous),
            range_bound_is_heuristic=bool(heuristic_bounds),
            uncertified_features=np.where(~np.isfinite(widths))[0].tolist(),
            sign_certified_features=np.where(sign_cert)[0].tolist(),
            run_id=self.run_id,
            M=int(M),
            domain_game=self.domain_game,
            config_hash=self.config_hash,
            oracle_hash=self.oracle.oracle_h,
            background_hash=self.oracle.background_h,
            git_commit=git.get("commit", ""),
            status=status,
            status_detail=status_detail,
            converged_early=bool(converged),
            extra={
                # --- per-call vs full-run query accounting (audit Medium 3) ---
                "num_coalition_evals_this_call": int(call_coal),
                "num_coalition_evals_run_total": int(self.oracle.total_coalition_evals),
                "num_model_evals_this_call": int(call_model),
                "num_model_evals_run_total": int(self.oracle.total_model_evals),
                # --- baseline model cost (audit Medium 4) ---
                "baseline_model_evals": int(self.B),
                "num_model_evals_end_to_end": int(self.oracle.total_model_evals),
                # --- point-estimate completeness (audit Medium 2) ---
                "point_estimate_complete": [bool(v) for v in point_complete],
                "missing_cells_by_feature": {
                    str(i): miss for i, miss in missing_cells.items()
                },
                "stage2_attempted_total": int(getattr(self, "_stage2_attempted_total", 0)),
                "epsilon": float(epsilon),
                "delta": float(delta),
                "max_width": float(np.max(widths)) if widths.size else float("inf"),
                "mean_width": float(np.mean(widths[np.isfinite(widths)])) if np.any(np.isfinite(widths)) else float("inf"),
                "median_width": float(np.median(widths[np.isfinite(widths)])) if np.any(np.isfinite(widths)) else float("inf"),
                "argmax_feature": int(np.argmax(widths)) if widths.size else None,
                "delta_total": float(delta_total),
                "surrogate_scale": float(self._surrogate.scale) if self._surrogate else 1.0,
                "surrogate_shift": float(self._surrogate.shift) if self._surrogate else 0.0,
                "n_gp_observations": int(len(self._surrogate.D_coalitions)) if self._surrogate is not None else 0,
                "range_bounds": [float(self.state.extra.get("L")), float(self.state.extra.get("U"))],
                "cache_hit_rate": self._cache.hit_rate() if self._cache is not None else 0.0,
            },
        )
        self._results = results
        self.state.stage = STAGE_FINAL

        if checkpoint and self._checkpoint_manager is not None:
            self._checkpoint_final(results)

        return results.to_dict()

    # ======================================================================= #
    # Prediction helper (counts GP predictions)
    # ======================================================================= #
    def _predict(self, S: np.ndarray) -> float:
        self._gp_predictions_counter.append(1)
        if self._surrogate is None:
            return 0.0
        return self._surrogate.predict(S, self.kernel)

    # ======================================================================= #
    # Optional validations
    # ======================================================================= #
    def _validate_oracle(self, x) -> None:
        for mask in (np.zeros(self.M, dtype=bool), np.ones(self.M, dtype=bool)):
            if not self.oracle.validate_determinism(x, mask):
                raise NumericalFailure("oracle is not deterministic")
        self._log_event(STAGE_ORACLE_VALIDATION, "oracle_deterministic", "OK")

    def _validate_math(self) -> None:
        from ..game.brute_force import (
            brute_force_cross_covariance,
            brute_force_prior_covariance,
        )
        M = self.M
        S_j = np.zeros(M, dtype=bool)
        S_j[0] = True
        K_a = lemma_D_cross_cov(self.kernel, S_j, M)
        K_b = brute_force_cross_covariance(self.kernel, S_j, M)
        if not np.allclose(K_a, K_b, atol=1e-10):
            raise NumericalFailure("Lemma D mismatch vs brute force")
        Pa = self.K_phi_phi
        Pb = brute_force_prior_covariance(self.kernel, M)
        if not np.allclose(Pa, Pb, atol=1e-10):
            raise NumericalFailure("Lemma E mismatch vs brute force")
        self._log_event(STAGE_MATH_VALIDATION, "math_validated", "OK", M=M)

    # ======================================================================= #
    # Checkpoint helpers
    # ======================================================================= #
    def _checkpoint_payload_base(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "oracle_hash": self.oracle.oracle_h,
            "background_hash": self.oracle.background_h,
            "M": self.M,
            "engine_version": _engine_version,
            "x_hash": self.state.extra.get("x_hash", ""),
            "num_coalition_evals": self.oracle.total_coalition_evals,
            "num_model_evals": self.oracle.total_model_evals,
            "num_gp_predictions": len(self._gp_predictions_counter),
            "rng_state": rng_state_to_dict(self.rng),
        }

    def _gp_state_payload(self, delta_total: Optional[float] = None) -> Dict[str, Any]:
        """GP-state arrays shared by every checkpoint stage (needed for resume)."""
        if self._surrogate is None:
            return {}
        return {
            "D_coalitions": self._surrogate.D_coalitions,
            "D_y": self._surrogate.D_y,
            "inv_K_DD": self._surrogate.inv_K_DD,
            "K_phi_D": self._surrogate.K_phi_D,
            "alpha": self._surrogate.alpha,
            "h_lb": self._surrogate.h_lb,
            "h_ub": self._surrogate.h_ub,
            "scale": self._surrogate.scale,
            "shift": self._surrogate.shift,
            "delta_total": float(
                delta_total if delta_total is not None
                else self.state.extra.get("delta_total", 0.0)
            ),
            "L": float(self.state.extra.get("L", 0.0)),
            "U": float(self.state.extra.get("U", 0.0)),
            "R_delta_res": float(self.state.extra.get("R_delta_res", 4.0)),
        }

    def _persist_cache_now(self) -> None:
        """Persist the coalition cache alongside checkpoints so a crash does
        not lose cached oracle values (resume stays consistent)."""
        if self._cache is not None:
            self._cache.persist()

    def _checkpoint_gp(self, delta_total: float) -> None:
        if self._surrogate is None or self._checkpoint_manager is None:
            return
        payload = self._checkpoint_payload_base()
        payload.update({"stage": "gp_stage", "iteration": 0})
        payload.update(self._gp_state_payload(delta_total))
        self._checkpoint_manager.save("gp_stage", 0, payload)
        self._persist_cache_now()

    def _checkpoint_residual(self, store, sigma_res, neyman, iteration: int, widths) -> None:
        if self._checkpoint_manager is None:
            return
        payload = self._checkpoint_payload_base()
        payload.update(self._gp_state_payload())
        payload.update(
            {
                "stage": "residual_stage",
                "iteration": int(iteration),
                "stage2_attempted_total": int(getattr(self, "_stage2_attempted_total", 0)),
                "store": store.to_dict(),
                "sigma_res": sigma_res,
                "neyman_probs": neyman.probabilities,
                "neyman_counts": neyman.counts,
                "neyman_objective": neyman.objective_value,
                "neyman_status": neyman.status,
                "neyman_message": neyman.message,
                "neyman_fallback": neyman.fallback_used,
                "widths": np.full(self.M, np.inf) if widths is None else np.asarray(widths),
            }
        )
        self._checkpoint_manager.save("residual_stage", int(iteration), payload)
        self._persist_cache_now()

    def _checkpoint_certification(self, store, sigma_res, neyman, widths, converged: bool, checkpoint: bool) -> None:
        if not checkpoint or self._checkpoint_manager is None:
            return
        payload = self._checkpoint_payload_base()
        payload.update(self._gp_state_payload())
        payload.update(
            {
                "stage": "certification_stage",
                "iteration": int(self.state.iteration),
                "stage2_attempted_total": int(getattr(self, "_stage2_attempted_total", 0)),
                "converged": bool(converged),
                "store": store.to_dict(),
                "sigma_res": sigma_res,
                "neyman_probs": neyman.probabilities,
                "widths": widths,
            }
        )
        self._checkpoint_manager.save("certification_stage", int(self.state.iteration), payload)
        self._persist_cache_now()

    def _checkpoint_final(self, results: RunResults) -> None:
        if self._checkpoint_manager is None:
            return
        payload = self._checkpoint_payload_base()
        payload.update(self._gp_state_payload())
        payload.update(
            {
                "stage": "final_stage",
                "iteration": int(self.state.iteration),
                "stage2_attempted_total": int(getattr(self, "_stage2_attempted_total", 0)),
                "results": results.to_dict(include_arrays=True),
                "store": (self._residual_store.to_dict() if self._residual_store is not None else None),
                "widths": results.raw_confidence_widths,
                "sigma_res": np.zeros((self.M, self.M)),  # placeholder; store holds truth
            }
        )
        self._checkpoint_manager.save("final_stage", int(self.state.iteration), payload)
        self._persist_cache_now()

    # ======================================================================= #
    # Resume
    # ======================================================================= #
    def _resume_checkpoint(self, x_h: str) -> Optional[Dict[str, Any]]:
        """Load the latest valid checkpoint; returns engine state or
        ``{'done': True, 'result': ...}`` for completed runs."""
        try:
            ckpt = self._checkpoint_manager.load_latest()
        except FileNotFoundError:
            self._log_event(STAGE_PREFLIGHT, "resume_no_checkpoint", "SKIPPED")
            return None
        if ckpt.get("x_hash", "") != x_h:
            raise CheckpointCompatibilityError(
                f"checkpoint x_hash {ckpt.get('x_hash')!r} != input {x_h!r}"
            )
        self._log_event(STAGE_PREFLIGHT, "resume_checkpoint", "OK",
                        checkpoint_stage=ckpt.get("stage"), iteration=ckpt.get("iteration"))

        self.oracle.total_coalition_evals = int(ckpt.get("num_coalition_evals", self.oracle.total_coalition_evals))
        self.oracle.total_model_evals = int(ckpt.get("num_model_evals", self.oracle.total_model_evals))
        self._gp_predictions_counter = [1] * int(ckpt.get("num_gp_predictions", 0))
        self._stage2_attempted_total = int(ckpt.get("stage2_attempted_total", 0))
        if ckpt.get("rng_state"):
            dict_to_rng_state(self.rng, ckpt["rng_state"])

        self._surrogate = BoundedLinearSurrogate(
            D_coalitions=np.asarray(ckpt["D_coalitions"], dtype=bool),
            D_y=np.asarray(ckpt["D_y"], dtype=np.float64),
            inv_K_DD=np.asarray(ckpt["inv_K_DD"], dtype=np.float64),
            K_phi_D=np.asarray(ckpt["K_phi_D"], dtype=np.float64),
            alpha=np.asarray(ckpt["alpha"], dtype=np.float64),
            h_lb=float(ckpt["h_lb"]), h_ub=float(ckpt["h_ub"]),
            scale=float(ckpt["scale"]), shift=float(ckpt["shift"]),
            M=self.M, eta=self.eta, _K_phi_phi=self.K_phi_phi,
        )
        phi_m_D = self._surrogate.surrogate_shapley(self.kernel)
        posterior_variances = self._surrogate.posterior_variances()
        self.state.extra["phi_m_D"] = phi_m_D
        self.state.extra["posterior_variances"] = posterior_variances
        self.state.extra["delta_total"] = float(ckpt.get("delta_total", 0.0))
        self.state.extra["L"] = float(ckpt.get("L", 0.0))
        self.state.extra["U"] = float(ckpt.get("U", 0.0))
        self.state.extra["R_delta_res"] = float(ckpt.get("R_delta_res", 4.0))

        stage = ckpt.get("stage")

        if stage in ("certification_stage", "final_stage"):
            if "results" in ckpt:
                results = _results_from_dict(ckpt["results"], self)
                self._results = results
                return {"done": True, "result": results.to_dict()}
            # certification_stage: reassemble from stored state
            store = StratumStore.from_dict(ckpt["store"])
            widths = np.asarray(ckpt["widths"], dtype=np.float64)
            converged = bool(ckpt.get("converged", False))
            phi_r = residual_shapley(store, self.M)
            phi_raw = raw_unified_estimator(phi_m_D, phi_r)
            delta_total = float(self.state.extra["delta_total"])
            phi_final = project_efficiency(phi_raw, delta_total, posterior_variances)
            certified = corollary_widths(widths, posterior_variances)
            sign_cert = sign_certified(phi_final, certified)
            rigorous = self.config.get("output_bounds") is not None and bool(np.all(np.isfinite(widths)))
            git = git_commit_and_dirty()
            results = RunResults(
                shapley_values=phi_final, surrogate_shapley=phi_m_D, residual_shapley=phi_r,
                raw_confidence_widths=widths, certified_projected_widths=certified,
                posterior_std=np.sqrt(posterior_variances),
                num_coalition_evals=int(ckpt.get("num_coalition_evals", 0)),
                num_model_evals=int(ckpt.get("num_model_evals", 0)),
                num_gp_predictions=int(ckpt.get("num_gp_predictions", 0)),
                num_residual_samples=store.n_records,
                num_sampling_rounds=int(ckpt.get("iteration", 0)),
                converged=converged,
                certificate_is_rigorous=bool(rigorous),
                range_bound_is_heuristic=self.config.get("output_bounds") is None,
                uncertified_features=np.where(~np.isfinite(widths))[0].tolist(),
                sign_certified_features=np.where(sign_cert)[0].tolist(),
                run_id=self.run_id, M=self.M, domain_game=self.domain_game,
                config_hash=self.config_hash, oracle_hash=self.oracle.oracle_h,
                background_hash=self.oracle.background_h, git_commit=git.get("commit", ""),
                status=ResultStatus.CERTIFIED if (converged and rigorous) else ResultStatus.NOT_CERTIFIED,
                converged_early=converged,
            )
            self._results = results
            return {"done": True, "result": results.to_dict()}

        if stage == "residual_stage":
            store = StratumStore.from_dict(ckpt["store"])
            sigma_res = np.asarray(ckpt["sigma_res"], dtype=np.float64)
            neyman = NeymanSolution(
                probabilities=np.asarray(ckpt["neyman_probs"], dtype=np.float64),
                counts=np.asarray(ckpt["neyman_counts"], dtype=np.int64),
                objective_value=float(ckpt.get("neyman_objective", 0.0)),
                status=int(ckpt.get("neyman_status", 0)),
                success=bool(ckpt.get("neyman_status", 0) == 0),
                message=str(ckpt.get("neyman_message", "")),
                K_cert=1.0,
                fallback_used=bool(ckpt.get("neyman_fallback", False)),
            )
            self._residual_store = store
            return {
                "store": store,
                "sigma_res": sigma_res,
                "neyman": neyman,
                "iteration": int(ckpt.get("iteration", 0)),
            }

        # gp_stage: rerun Stage 2 from scratch (oracle cache prevents recount)
        return {"store": None, "sigma_res": None, "neyman": None, "iteration": 0}

    # ======================================================================= #
    # Dashboard / introspection
    # ======================================================================= #
    def status(self) -> Dict[str, Any]:
        from ..utils.rng_state import rng_state_hash
        return {
            "run_id": self.run_id,
            "M": self.M,
            "domain_game": self.domain_game,
            "current_stage": self.state.stage,
            "iteration": self.state.iteration,
            "gp_observations": len(self._surrogate.D_coalitions) if self._surrogate is not None else 0,
            "residual_observations": self._residual_store.n_records if self._residual_store is not None else 0,
            "num_coalition_evals": self.oracle.total_coalition_evals,
            "num_model_evals": self.oracle.total_model_evals,
            "num_gp_predictions": len(self._gp_predictions_counter),
            "sampling_rounds": self.state.iteration,
            "cache_hit_rate": self._cache.hit_rate() if self._cache is not None else 0.0,
            "latest_checkpoint": self._checkpoint_manager.manifest.latest() if self._checkpoint_manager else None,
            "config_hash": self.config_hash,
            "oracle_hash": self.oracle.oracle_h,
            "background_hash": self.oracle.background_h,
            "rng_state_hash": rng_state_hash(self.rng),
        }

    def estimate_cost(self, max_budget: Optional[int] = None, n_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Rough query-cost estimate for the current M (before running)."""
        M = self.M
        budget = int(max_budget if max_budget is not None else self.config.get("max_budget", 1500))
        n_act = int(self.config.get("n_active_steps", 25))
        n_pil = int(self.config.get("n_pilot", 3))
        stage1 = 2 + (M - 1) + n_act
        stage2_fixed = 2 * M + 2 + (M - 2) * n_pil * (1 + M)
        rounds = n_rounds if n_rounds is not None else max(0, budget // (1 + M))
        return {
            "M": M,
            "stage1_coalition_evals_est": stage1,
            "stage2_fixed_coalition_evals": stage2_fixed,
            "adaptive_rounds_upper": rounds,
            "adaptive_coalition_evals_upper": rounds * (1 + M),
            "total_coalition_evals_upper": stage1 + stage2_fixed + rounds * (1 + M),
            "model_evals_per_coalition": self.B,
        }

    def results(self) -> Optional[RunResults]:
        return self._results

    def write_results(self, spec_compliance: Optional[dict] = None, pytest_evidence: Optional[dict] = None) -> Path:
        """Write the results/runs/<run_id>/ layout (spec section 48).

        ``pytest_evidence`` (e.g. ``{"passed": N, "failed": 0}``) is embedded
        into ``spec_compliance.json`` so the compliance report reflects actual
        test artifacts for the current commit.
        """
        if self._results is None:
            raise RuntimeError("no results yet — call explain() first")
        if spec_compliance is None:
            if pytest_evidence is not None:
                from .compliance import compliance_from_pytest
                compliance = compliance_from_pytest(
                    self, passed=int(pytest_evidence.get("passed", 0)),
                    failed=int(pytest_evidence.get("failed", 0)),
                )
            else:
                compliance = self.compliance_audit()
        env = environment_manifest()
        provenance = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "oracle_hash": self.oracle.oracle_h,
            "background_hash": self.oracle.background_h,
            "git_commit": git_commit_and_dirty().get("commit", ""),
            "config": {k: (None if v is None else v) for k, v in self.config.items()},
        }
        return write_run_results(
            self._results, self._results_dir or "results/runs", self.config, env, provenance, compliance
        )

    def compliance_audit(self) -> Dict[str, Any]:
        from .compliance import run_compliance_audit
        return run_compliance_audit(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bool_mask_to_int(mask: np.ndarray) -> int:
    bitmask = 0
    for bit, v in enumerate(np.asarray(mask, dtype=bool)):
        if v:
            bitmask |= 1 << bit
    return bitmask


def _rng_state_label(rng: np.random.RandomState) -> str:
    """Stable per-round seed label (no RNG consumption)."""
    from ..utils.rng_state import rng_state_hash
    return rng_state_hash(rng)[:16]


def _compose_status(converged, rigorous, heuristic, budget_exhausted, missing_strata):
    statuses = [ResultStatus.VALID]
    if heuristic:
        statuses.append(ResultStatus.HEURISTIC_BOUNDS)
    if converged and rigorous:
        statuses.append(ResultStatus.CERTIFIED)
    if converged:
        statuses.append(ResultStatus.CONVERGED)
    if budget_exhausted:
        statuses.append(ResultStatus.BUDGET_EXHAUSTED)
    if missing_strata:
        statuses.append(ResultStatus.MISSING_STRATA)
    if not (converged or budget_exhausted or missing_strata):
        statuses.append(ResultStatus.NOT_CERTIFIED)
    primary = (
        ResultStatus.CERTIFIED if ResultStatus.CERTIFIED in statuses
        else ResultStatus.BUDGET_EXHAUSTED if ResultStatus.BUDGET_EXHAUSTED in statuses
        else ResultStatus.MISSING_STRATA if ResultStatus.MISSING_STRATA in statuses
        else ResultStatus.NOT_CERTIFIED if ResultStatus.NOT_CERTIFIED in statuses
        else ResultStatus.VALID
    )
    return primary, "+".join(statuses)


def _results_from_dict(d: Dict[str, Any], engine: "GASBayesSHAP") -> RunResults:
    """Rehydrate a RunResults from a JSON dict (arrays stored as lists)."""
    def arr(name: str) -> np.ndarray:
        return np.asarray(d[name], dtype=np.float64)

    return RunResults(
        shapley_values=arr("shapley_values"),
        surrogate_shapley=arr("surrogate_shapley"),
        residual_shapley=arr("residual_shapley"),
        raw_confidence_widths=arr("raw_confidence_widths"),
        certified_projected_widths=arr("certified_projected_widths"),
        posterior_std=arr("posterior_std"),
        num_coalition_evals=int(d.get("num_coalition_evals", 0)),
        num_model_evals=int(d.get("num_model_evals", 0)),
        num_gp_predictions=int(d.get("num_gp_predictions", 0)),
        num_residual_samples=int(d.get("num_residual_samples", 0)),
        num_sampling_rounds=int(d.get("num_sampling_rounds", 0)),
        converged=bool(d.get("converged", False)),
        certificate_is_rigorous=bool(d.get("certificate_is_rigorous", False)),
        range_bound_is_heuristic=bool(d.get("range_bound_is_heuristic", False)),
        uncertified_features=list(d.get("uncertified_features", [])),
        sign_certified_features=list(d.get("sign_certified_features", [])),
        run_id=str(d.get("run_id", engine.run_id)),
        M=int(d.get("M", engine.M)),
        domain_game=str(d.get("domain_game", engine.domain_game)),
        config_hash=str(d.get("config_hash", engine.config_hash)),
        oracle_hash=str(d.get("oracle_hash", engine.oracle.oracle_h)),
        background_hash=str(d.get("background_hash", engine.oracle.background_h)),
        git_commit=str(d.get("git_commit", "")),
        status=str(d.get("status", ResultStatus.NOT_CERTIFIED)),
        status_detail=str(d.get("status_detail", "")),
        converged_early=bool(d.get("converged_early", False)),
        extra={k: v for k, v in d.items() if k not in _RESULT_KEYS},
    )


_RESULT_KEYS = {
    "shapley_values", "surrogate_shapley", "residual_shapley",
    "raw_confidence_widths", "certified_projected_widths", "posterior_std",
    "num_coalition_evals", "num_model_evals", "num_gp_predictions",
    "num_residual_samples", "num_sampling_rounds", "converged",
    "converged_early", "certificate_is_rigorous", "range_bound_is_heuristic",
    "uncertified_features", "sign_certified_features", "run_id", "M",
    "domain_game", "config_hash", "oracle_hash", "background_hash",
    "git_commit", "status", "status_detail",
}
