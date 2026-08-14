#!/usr/bin/env python
"""Run ALL experiment notebooks end-to-end (build + execute) in one command.

Usage:
    python scripts/run_all_notebooks.py                # all notebooks
    python scripts/run_all_notebooks.py --skip-source # skip the example notebook
    python scripts/run_all_notebooks.py --only wine   # only the wine notebooks

This rebuilds each notebook from its builder (so the source of truth is the
builder scripts) and executes it with the current kernel, saving outputs
back into the .ipynb.  Requires the data files (data/Beijing_MultiSite_*.csv,
data/winequality-white.csv or network access) and the experiment deps
(pandas, lightgbm, shap, matplotlib, altair).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

NOTEBOOKS = {
    "wine": [
        ("scripts/build_wine_gas_notebook.py", "notebooks/SHAP_WINE_GAS.ipynb"),
    ],
    "air": [
        ("scripts/build_air_quality_gas_notebook.py", "notebooks/AIR_QUALITY_GAS.ipynb"),
    ],
    "example": [
        (None, "notebooks/Source_code_air.ipynb"),  # no builder; execute as-is
    ],
    "engine": [
        ("scripts/build_notebook.py", "notebooks/run_all.ipynb"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + execute all experiment notebooks")
    ap.add_argument("--only", choices=list(NOTEBOOKS), default=None,
                    help="run only one group (wine | air | example | engine)")
    ap.add_argument("--skip-source", action="store_true",
                    help="skip the Source_code_air example notebook")
    ap.add_argument("--timeout", type=int, default=1500, help="per-cell timeout (s)")
    args = ap.parse_args()

    groups = [args.only] if args.only else list(NOTEBOOKS)
    if args.skip_source:
        groups = [g for g in groups if g != "example"]

    rc = 0
    for group in groups:
        for builder, nb_path in NOTEBOOKS[group]:
            nb_path = ROOT / nb_path
            if builder:
                import subprocess
                print(f"\n[build] {builder}")
                r = subprocess.run([sys.executable, str(ROOT / builder)], cwd=ROOT)
                if r.returncode != 0:
                    print(f"[FAIL] build {builder}")
                    rc = 1
                    continue
            print(f"[run ] {nb_path.name}")
            try:
                nb = nbformat.read(nb_path, as_version=4)
                client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                        resources={"metadata": {"path": str(nb_path.parent)}})
                client.execute()
                errs = [o for c in nb.cells for o in c.get("outputs", [])
                        if o.output_type == "error"]
                nbformat.write(nb, nb_path)
                if errs:
                    print(f"[FAIL] {nb_path.name}: {len(errs)} error outputs")
                    for c in nb.cells:
                        for o in c.get("outputs", []):
                            if o.output_type == "error":
                                print("   -", o.get("ename"), str(o.get("evalue", ""))[:150])
                    rc = 1
                else:
                    print(f"[OK  ] {nb_path.name}: executed with 0 errors")
            except Exception as exc:
                print(f"[FAIL] {nb_path.name}: {type(exc).__name__} {str(exc)[:200]}")
                rc = 1

    print("\n" + "=" * 60)
    print("ALL NOTEBOOKS DONE" if rc == 0 else "SOME NOTEBOOKS FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
