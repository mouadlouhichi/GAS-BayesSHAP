#!/usr/bin/env python
"""GAS-BayesSHAP main CLI runner (spec section 40).

Usage:
    python scripts/run_bayesshap.py --config configs/default.yaml [options]
    python scripts/run_bayesshap.py --config configs/default.yaml --resume
    python scripts/run_bayesshap.py --run-id RUN_ID --status

Supported options:
    --dataset, --game, --M, --epsilon, --delta, --max-budget, --max-rounds,
    --resume, --run-id, --from-stage, --until-stage, --seed, --out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.utils.config import ConfigError, load_config
from gas_bayesshap.utils.serialization import write_json_atomic


def synthetic_model_factory(game: str, M: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = rng.randn(M)
    if game == "membership":
        def model(x):
            z = np.dot(x, weights) / max(1.0, np.sqrt(M))
            return 1.0 / (1.0 + np.exp(-z))
        bounds = (0.0, 1.0)
    elif game == "contrastive":
        w2 = rng.randn(M)

        def model(x):
            g1 = 1.0 / (1.0 + np.exp(-np.dot(x, weights) / np.sqrt(M)))
            g2 = 1.0 / (1.0 + np.exp(-np.dot(x, w2) / np.sqrt(M)))
            return float(g1 - g2)
        bounds = (-1.0, 1.0)
    elif game == "archetype":
        def model(x):
            return 1.0 / (1.0 + np.exp(-np.dot(x, weights) / np.sqrt(M)))
        bounds = (0.0, 1.0)
    elif game == "group_lag":
        def model(x):
            # lagged AR-ish model over 2 * n_lags features
            z = np.dot(x, weights) / max(1.0, np.sqrt(len(x)))
            return float(z)
        bounds = None
    else:
        def model(x):
            return float(np.dot(x, weights) / np.sqrt(M))
        bounds = None
    return model, bounds


def build_oracle_and_engine(cfg, run_id):
    """Build the domain-game oracle and engine from configuration."""
    M = cfg.get("M") or 5
    seed = int(cfg.get("seed", 42))
    B = 8
    rng = np.random.RandomState(seed)

    if cfg.get("dataset") in ("wine", "beijing_static"):
        M = 11
    elif cfg.get("dataset") == "beijing_lagged":
        M = 11

    model, bounds = synthetic_model_factory(cfg.get("domain_game", "membership"), M, seed)
    if bounds is not None and cfg.get("output_bounds") is not None:
        bounds = tuple(cfg["output_bounds"])
    elif cfg.get("output_bounds") is not None:
        bounds = tuple(cfg["output_bounds"])

    engine = GASBayesSHAP(
        model_fn=model,
        background=rng.randn(B, M),
        output_bounds=bounds,
        rng=np.random.RandomState(seed),
        config={**cfg, "domain_game": cfg.get("domain_game", "membership")},
        run_id=run_id,
    )
    return engine


def main() -> int:
    p = argparse.ArgumentParser(description="GAS-BayesSHAP runner (v11.0)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--game", default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--max-budget", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--status", action="store_true",
                   help="print the live status dashboard for --run-id and exit")
    p.add_argument("--run-id", default=None)
    p.add_argument("--from-stage", default=None,
                   help="resume from a coarse stage: gp | residual | projection")
    p.add_argument("--until-stage", default=None,
                   help="stop after a coarse stage: preflight | gp | projection")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    overrides = {}
    for src, dst in [
        ("dataset", "dataset"), ("game", "domain_game"), ("M", "M"),
        ("epsilon", "epsilon"), ("delta", "delta"),
        ("max_budget", "max_budget"), ("max_rounds", "max_rounds"),
        ("seed", "seed"),
    ]:
        val = getattr(args, src)
        if val is not None:
            overrides[dst] = val

    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"run-{cfg['seed']}-{cfg['domain_game']}"
    engine = build_oracle_and_engine(cfg, run_id)

    if args.status:
        status = engine.status()
        # enrich with on-disk checkpoint manifest state
        from gas_bayesshap.checkpointing.manager import CheckpointManager
        mgr = CheckpointManager(
            run_id=run_id,
            directory=cfg["checkpoints_dir"],
            config_hash=engine.config_hash,
            oracle_hash=engine.oracle.oracle_h,
            background_hash=engine.oracle.background_h,
            M=engine.M,
            engine_version="11.0.0",
        )
        status["checkpoint_manifest"] = mgr.manifest_dict()
        status["checkpoints"] = mgr.list_checkpoints()
        print(json.dumps(status, indent=2, default=str))
        return 0

    # --- from-stage / until-stage coarse semantics ------------------------- #
    until = args.until_stage
    if until == "preflight":
        print("--until-stage preflight: nothing to run (preflight happens inside explain).")
        print(json.dumps(engine.estimate_cost(), indent=2))
        return 0
    if until == "gp":
        print("Running GP-only (Module A) ...")
        res = engine.explain_stage1_only(np.ones(engine.M))
        print(json.dumps({k: v for k, v in res.items()}, indent=2, default=str))
        return 0

    x = np.ones(engine.M)
    print(f"run_id={run_id}  M={engine.M}  game={cfg['domain_game']}  "
          f"resume={args.resume or args.from_stage is not None}")
    res = engine.explain(
        x,
        epsilon=cfg["epsilon"],
        delta=cfg["delta"],
        max_budget=cfg["max_budget"],
        n_pilot=cfg["n_pilot"],
        n_active_steps=cfg["n_active_steps"],
        resume=bool(args.resume or args.from_stage is not None),
        checkpoint=True,
    )

    print("\n=== RESULT ===")
    for k in ("status", "converged", "certificate_is_rigorous", "range_bound_is_heuristic",
              "num_coalition_evals", "num_model_evals", "num_gp_predictions",
              "num_residual_samples", "num_sampling_rounds"):
        print(f"  {k:28s}: {res[k]}")
    print("  shapley_values       :", np.round(res["shapley_values"], 6).tolist())
    print("  raw_confidence_widths:", np.round(res["raw_confidence_widths"], 6).tolist())
    print("  certified widths     :", np.round(res["certified_projected_widths"], 6).tolist())
    print("  sign_certified       :", res["sign_certified_features"])

    engine.write_results()
    print(f"results written to results/runs/{run_id}/")

    if args.out:
        write_json_atomic(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
