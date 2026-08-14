# main_results

Consolidated experiment outputs (paper figures/tables).

This folder is populated automatically by
[`scripts/run_all_notebooks.py`](../scripts/run_all_notebooks.py): after each
notebook executes successfully, the executed `.ipynb` and its generated
comparison tables/figures are copied here.

## Contents (after a successful `python scripts/run_all_notebooks.py`)

| File | Source | Description |
|---|---|---|
| `AIR_QUALITY_GAS.ipynb` | `notebooks/` | Beijing air Tier A (M=11 exact) + Tier B (group-lag M=66→11) certified GAS-BayesSHAP |
| `SHAP_WINE_GAS.ipynb` | `notebooks/` | Wine Tier-A exact ground-truth + certified GAS-BayesSHAP |
| `Source_code_air.ipynb` | `notebooks/` | Air example baseline (PCA→KMeans→LightGBM→SHAP) |
| `comparison.csv` | `results/wine_tierA` | Wine RMSE / simultaneous+marginal coverage / query counts vs exact, KernelSHAP, SamplingSHAP |
| `comparison.csv` | `results/air_quality_tierA` | Air Tier-A same comparison |
| `group_lag_comparison.csv` | `results/air_quality_tierB` | Air Tier-B macro-player comparison |
| `macro_waterfall.png` | `results/air_quality_tierB` | Certified macro attribution waterfall with error bars |

## Notes

- Coverage columns report **simultaneous** coverage (the theorem's guarantee)
  and **marginal** coverage separately.
- `DATA_SOURCE` printed in each notebook indicates whether the run used real
  data (`cache:` / `url-*` / `dir-merge:`) or the synthetic fallback
  (`synthetic-fallback`) — always check before quoting numbers.
- Generated per-run artifacts under `results/` are gitignored; this folder
  holds the copies you want to keep/commit for the paper.
