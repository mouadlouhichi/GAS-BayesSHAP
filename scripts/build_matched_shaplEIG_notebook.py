#!/usr/bin/env python
"""Build the 'matched ShaplEIG vs GAS' notebook (audit P1-7).

The earlier comparison paired ShaplEIG at 64/128/256 UNIQUE queries against
GAS at nominal K=128/256/512 (~393/480/565 actual unique evals) — not
matched.  This notebook runs `scripts/run_matched_shaplEIG.py`, which runs
BOTH methods at the same nominal budgets {64,128,256}, records each
method's ACTUAL unique coalition evaluations side by side, and flags
whether the counts are actually matched.  Nothing is claimed as matched
unless the counts agree.

Usage:
    python scripts/build_matched_shaplEIG_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "main_results" / "RUN_MATCHED_SHAPLEIG.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

cells.append(md(
    "# GAS-BayesSHAP — matched ShaplEIG vs GAS (unique-query comparison)\n"
    "\n"
    "Closes audit P1-7: the earlier summary compared ShaplEIG at 64/128/256 "
    "**unique** queries against GAS at nominal K=128/256/512 (~393/480/565 "
    "actual unique evals) — not actually matched.  This runs both methods at "
    "the **same nominal budgets** and reports each method's *actual unique* "
    "coalition evaluations side by side (`matched_unique` is True only when "
    "the counts agree within 35%).  Orchestrates "
    "`scripts/run_matched_shaplEIG.py` only."
))

cells.append(md("## 0. Environment & config"))

cells.append(code(
    "import sys, os, time, subprocess\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, \"..\")\n"
    "import gas_bayesshap\n"
    "\n"
    "ROOT = Path(\"..\").resolve()\n"
    "SCRIPTS = ROOT / \"scripts\"\n"
    "print(\"GAS-BayesSHAP\", gas_bayesshap.__version__)\n"
    "\n"
    "N_INST  = int(os.environ.get(\"N_INST\", \"2\"))       # instances per dataset\n"
    "BUDGETS = os.environ.get(\"BUDGETS\", \"64,128,256\")\n"
    "SKIP    = set(os.environ.get(\"GAS_SKIP\", \"\").split(\",\")) - {\"\"}\n"
    "\n"
    "def run(*args, tag=\"\", skip=False):\n"
    "    if skip:\n"
    "        print(f\"--- SKIPPED: {tag or ' '.join(args)}\"); return 0.0\n"
    "    cmd = [sys.executable, str(SCRIPTS / args[0]), *args[1:]]\n"
    "    t0 = time.time()\n"
    "    print(f\"\\n>>> {tag or ' '.join(args)}\")\n"
    "    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)\n"
    "    dt = time.time() - t0\n"
    "    if r.returncode != 0:\n"
    "        err = (r.stderr or r.stdout or '').strip().splitlines()\n"
    "        print(' | '.join(err[-6:]) if err else '<no output>')\n"
    "        raise RuntimeError(f\"FAILED ({r.returncode}): {' '.join(args)}\")\n"
    "    print(f\"<<< done in {dt/60:.1f} min\")\n"
    "    return dt\n"
    "\n"
    "print(f\"N_INST={N_INST} BUDGETS={BUDGETS} SKIP={sorted(SKIP)}\")"
))

cells.append(md("## A. Matched comparison (wine + air)"))

cells.append(md(
    "Runs GAS (spec range) and the ShaplEIG port at nominal K ∈ {64,128,256} "
    "on N_INST instances per dataset, recording actual unique coalition "
    "evals for both.  **~30–45 min on a laptop** (12 ShaplEIG runs, each "
    "refits a GP per acquisition round).  Smoke: `N_INST=1 BUDGETS=64`."
))

cells.append(code(
    'run("run_matched_shaplEIG.py", "--n", str(N_INST), "--budgets", BUDGETS,\n'
    '    tag=f"A. matched ShaplEIG vs GAS (N={N_INST}, K={BUDGETS})", skip=False)'
))

cells.append(md("## B. Summary table"))

cells.append(code(
    "import pandas as pd\n"
    "p = ROOT / \"main_results\" / \"paper_matched_shaplEIG_comparison.csv\"\n"
    "if not p.exists():\n"
    "    print(\"NOT FOUND — run section A first\")\n"
    "else:\n"
    "    d = pd.read_csv(p)\n"
    "    cols = [\"dataset\", \"instance\", \"nominal_K\", \"gas_rmse\",\n"
    "            \"shaplEIG_rmse\", \"gas_unique_evals\", \"shaplEIG_unique_queries\",\n"
    "            \"matched_unique\"]\n"
    "    print(d[cols].to_string(index=False))\n"
    "    g = d.groupby([\"dataset\", \"nominal_K\"]).agg(\n"
    "        gas_rmse=(\"gas_rmse\", \"mean\"), gas_unique=(\"gas_unique_evals\", \"mean\"),\n"
    "        shaplEIG_rmse=(\"shaplEIG_rmse\", \"mean\"),\n"
    "        shaplEIG_unique=(\"shaplEIG_unique_queries\", \"mean\"),\n"
    "    ).round(5)\n"
    "    print(\"\\n=== mean over instances ===\")\n"
    "    print(g.to_string())\n"
    "    print(\"\\nHONEST NOTE: rows with matched_unique=False are NOT a matched\"\n"
    "          \" unique-query comparison — report the actual counts, not a claim.\")"
))

cells.append(md(
    "## Notes\n"
    "- The ShaplEIG port is pure NumPy/SciPy (same math as slds-lmu/shapleig@\n"
    "  d52c09e; no torch — macOS crash-free).  GAS uses the spec range.\n"
    "- Commit `paper_matched_shaplEIG_comparison.csv` when done."
))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
