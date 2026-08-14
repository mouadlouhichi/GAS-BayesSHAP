#!/usr/bin/env python
"""Audit self-check: verify the experiment-layer fixes are present in the
committed builder sources (so a static audit cannot misread the branch).

Checks (mirror the recurring audit findings):
  air builder:  multi-site UCI ZIP concat, Tier-B lag valid_index alignment,
                KernelSHAP matched background, macro simultaneous coverage
  wine builder: simultaneous coverage, KernelSHAP matched background,
                TreeSHAP not in the same-game RMSE table

Usage:
    python scripts/check_experiment_audit.py
Exit 0 = all checks pass; exit 1 = a fix regressed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_show(rel: str) -> str:
    return subprocess.run(
        ["git", "show", f"HEAD:{rel}"], capture_output=True, text=True, cwd=ROOT
    ).stdout


def main() -> int:
    air = git_show("scripts/build_air_quality_gas_notebook.py")
    wine = git_show("scripts/build_wine_gas_notebook.py")

    checks = [
        # --- air builder ---
        ("air: multi-site UCI ZIP concatenates all CSVs",
         "csv_names = [n for n in z.namelist()" in air and "pd.concat(frames" in air),
        ("air: per-station column added from filename stem",
         'tmp["station"] = os.path.splitext(os.path.basename(nm))[0]' in air),
        ("air: Tier-B lag labels aligned via valid_index",
         "lag_target = np.asarray(regime_labels)[valid_index]" in air
         and "valid = out.dropna().index" in air),
        ("air: Tier-A simultaneous coverage (np.all)",
         "gas_simultaneous = float(np.all(err_gas <= W_proj))" in air),
        ("air: Tier-B macro simultaneous coverage (np.all)",
         "macro_sim = float(np.all(err_g <= W_proj_g))" in air),
        ("air: KernelSHAP uses matched full background",
         "KernelExplainer(proba_matrix, background)" in air),
        ("air: no marginal-only coverage remnant",
         "coverage=float(np.mean(np.abs(phi_gas - phi_exact) <= W_proj))" not in air),
        # --- wine builder ---
        ("wine: simultaneous coverage (np.all)",
         "gas_simultaneous = float(np.all(err_gas <= W_proj))" in wine),
        ("wine: KernelSHAP uses matched full background",
         "KernelExplainer(proba_matrix, background)" in wine),
        ("wine: no marginal-only coverage remnant",
         "coverage=float(np.mean(np.abs(phi_gas - phi_exact) <= W_proj))" not in wine),
        ("wine: TreeSHAP not in same-game RMSE table",
         '"TreeSHAP (logit)"' not in wine and "rmse(tree_phi" not in wine),
        ("wine: TreeSHAP reported separately as space-mismatch",
         "TreeSHAP (logit space, model-specific baseline, NOT same-game)" in wine),
        # --- data loading (Beijing per-station CSVs) ---
        ("air: scans data/ directly for PRSA_Data_*.csv",
         'fn.lower().startswith("prsa_data_")' in air
         and "scan_dirs = [DATA_DIR," in air),
        ("air: merges per-station CSVs with station column",
         'tmp["station"] = os.path.splitext(fn)[0]' in air
         and 'DATA_SOURCE = f"dir-merge:{subdir}' in air),
    ]

    failed = 0
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += (not ok)
    print("-" * 60)
    print(f"{len(checks) - failed}/{len(checks)} experiment-layer audit checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
