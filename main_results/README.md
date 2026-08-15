# main_results

Consolidated experiment outputs (paper figures/tables).

This folder is populated automatically by
[`scripts/run_all_notebooks.py`](../scripts/run_all_notebooks.py): after each
notebook executes successfully, the executed `.ipynb` and its generated
comparison tables/figures are copied here.

## Contents

| File | Source | Description |
|---|---|---|
| `AIR_QUALITY_GAS.ipynb` | `notebooks/` | Beijing air Tier A (M=11 exact) + Tier B (group-lag M=66→11) certified GAS-BayesSHAP |
| `SHAP_WINE_GAS.ipynb` | `notebooks/` | Wine Tier-A exact ground-truth + certified GAS-BayesSHAP |
| `Source_code_air.ipynb` | `notebooks/` | Air example baseline (PCA→KMeans→LightGBM→SHAP) |
| `run_all.ipynb` | `notebooks/` | Engine demo (sections 1–17) |
| `wine_comparison.csv` | `results/wine_tierA` | **Wine** RMSE / simultaneous+marginal coverage / query counts vs exact, KernelSHAP, SamplingSHAP |
| `air_comparison.csv` | `results/air_quality_tierA` | **Air Tier-A** same comparison |
| `air_group_lag_comparison.csv` | `results/air_quality_tierB` | Air Tier-B macro-player comparison |
| `air_macro_waterfall.png` | `results/air_quality_tierB` | Certified macro attribution waterfall with error bars |

> **Naming:** artifacts are prefixed by dataset (`wine_`, `air_`) so the wine
> and air `comparison.csv` never overwrite each other.

## Notes (important for the paper)

- Coverage columns report **simultaneous** coverage (the theorem's guarantee)
  and **marginal** coverage separately.
- `DATA_SOURCE` printed in each notebook indicates whether the run used real
  data (`cache:` / `url-*` / `dir-merge:`) or the synthetic fallback
  (`synthetic-fallback`) — always check before quoting numbers.
- The CSVs under `main_results/` are a snapshot from one executed run; the
  numbers inside each notebook's stored outputs are from that same run.
  Re-running regenerates them (and can shift values slightly since the
  surrogate/KMeans depend on the random seed).
- Generated per-run artifacts under `results/` are gitignored; this folder
  holds the copies you want to keep/commit for the paper.
