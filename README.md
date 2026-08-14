# GAS-BayesSHAP v11.0

A modular implementation of bounded-linear GP control variates plus Neyman-stratified residual certification for Shapley attribution.

## Install and run

```bash
python -m pip install -r requirements.txt
python scripts/validate_math.py
python scripts/run_bayesshap.py --config configs/default.yaml
```

Use `--resume` to restore the last atomically-written checkpoint and `--status` to inspect it. Results are written under `results/runs/<run_id>/`; checkpoints are stored under `checkpoints/<run_id>/`.

The run script uses a deterministic synthetic membership model as an executable example. Domain oracle classes are in `gas_bayesshap/game/domain_games.py`; production callers supply their deterministic model and fixed empirical background through `InterventionalOracle`.
