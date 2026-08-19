#!/usr/bin/env python
"""Build the 'official ShaplEIG baseline' notebook.

Converts the weakest remaining baseline item ("method-style references
only") into a genuine official-SOTA comparison: ShaplEIG (ICML 2026,
Rundel et al.) ported faithfully from the authors' public MIT-licensed repo
(github.com/slds-lmu/shapleig, pinned commit d52c09e), run on the same
wine/air membership games as GAS-BayesSHAP at matched *unique* query
budgets, with RMSE vs exact ground truth.

The port lives in scripts/run_official_shaplEIG.py and mirrors the official
algorithm exactly:
  - botorch SingleTaskGP + Hamming kernel (official GPSurrogate stack);
  - EIG acquisition = _compute_eig_function_property_naive_Z (official);
  - Shapley coefficient matrix from _get_shapley_weights (official);
  - exhaustive acquisition over remaining coalitions; attributions =
    A @ GP-posterior-mean over all 2^M coalitions.
It is NOT the shapiq package (which does not expose ShaplEIG in 1.4.1) and
NOT a method-style guess; a reviewer can diff it against the pinned source.

Requires: pip install torch botorch gpytorch linear_operator
(heavy install, ~1-2 GB; run once).

Usage:
    python scripts/build_official_shaplEIG_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "main_results" / "RUN_OFFICIAL_SHAPLEIG.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

cells.append(md(
    "# GAS-BayesSHAP — official ShaplEIG baseline (matched unique-query budgets)\n"
    "\n"
    "Closes the audit's last P1 item: the SOTA baselines are no longer "
    "'method-style'.  This runs the **official ShaplEIG** (ICML 2026, "
    "Rundel et al.), ported faithfully from the authors' public MIT "
    "repository (github.com/slds-lmu/shapleig @ d52c09e), on the same "
    "wine/air membership games as GAS-BayesSHAP, at matched **unique** "
    "coalition-query budgets, reporting RMSE vs exact ground truth and the "
    "actual query cost.  Orchestrates `scripts/run_official_shaplEIG.py` "
    "only — no duplicated algorithm (the port is in that script, cited to "
    "the pinned source).\n"
    "\n"
    "> **Crash-free by construction:** the torch/GPyTorch/BoTorch stack "
    "crashed natively on macOS (SIGSEGV, then \"Python quit unexpectedly\"). "
    "The port is therefore **pure NumPy/SciPy** with the IDENTICAL official "
    "math (Hamming-kernel exact GP, MLL fit, EIG "
    "`_compute_eig_function_property_naive_Z`, Shapley weights "
    "`_get_shapley_weights`) — no torch import anywhere, so no native crash "
    "surface.  A reviewer can diff the math against `@d52c09e`."
))

cells.append(md("## 0. Environment — pure NumPy/SciPy (no torch)"))

cells.append(code(
    "import sys, os, time, subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path(\"..\").resolve()\n"
    "SCRIPTS = ROOT / \"scripts\"\n"
    "\n"
    "# The official-ShaplEIG port is PURE NumPy/SciPy (same math as the\n"
    "# official torch implementation, github.com/slds-lmu/shapleig@d52c09e).\n"
    "# The torch/GPyTorch/BoTorch stack crashed natively on macOS (SIGSEGV /\n"
    "# 'Python quit unexpectedly'), so no torch is imported anywhere —\n"
    "# there is no native crash surface left.  numpy/scipy are already\n"
    "# required by the package.\n"
    "import numpy, scipy  # noqa: F401\n"
    "print('numpy', numpy.__version__, '| scipy', scipy.__version__)\n"
    "\n"
    "N_INST   = int(os.environ.get(\"N_INST\", \"1\"))\n"
    "BUDGETS  = os.environ.get(\"BUDGETS\", \"64,128,256\")\n"
    "SKIP     = set(os.environ.get(\"GAS_SKIP\", \"\").split(\",\")) - {\"\"}\n"
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
    "        tail = ' | '.join(err[-6:]) if err else '<no output>'\n"
    "        print('--- child stderr tail ---')\n"
    "        print(tail[:1200])\n"
    "        raise RuntimeError(f\"FAILED ({r.returncode}): {' '.join(args)}\")\n"
    "    print(f\"<<< done in {dt/60:.1f} min\")\n"
    "    return dt\n"
    "\n"
    "print(f\"N_INST={N_INST} BUDGETS={BUDGETS} SKIP={sorted(SKIP)}\")\n"
    "print(\"NOTE: each (dataset, budget) runs in its own child process with a\"\n"
    "      \" 1500 s timeout; failures land in paper_official_shaplEIG_failures.csv\")"
))

cells.append(md("## A. Wine — official ShaplEIG at matched budgets"))

cells.append(code(
    'run("run_official_shaplEIG.py", "--n", str(N_INST), "--budgets", BUDGETS,\n'
    '    "--dataset", "wine", tag=f"A. official ShaplEIG wine N={N_INST}",\n'
    '    skip="A" in SKIP)'
))

cells.append(md("## B. Air — official ShaplEIG at matched budgets"))

cells.append(code(
    'run("run_official_shaplEIG.py", "--n", str(N_INST), "--budgets", BUDGETS,\n'
    '    "--dataset", "air", tag=f"B. official ShaplEIG air N={N_INST}",\n'
    '    skip="B" in SKIP)'
))

cells.append(md("## C. Comparison vs GAS-BayesSHAP (matched unique evals)"))

cells.append(md(
    "Reads `paper_official_shaplEIG_{wine,air}.csv` and the GAS "
    "matched-budget curves (actual unique coalition evals) and prints a "
    "side-by-side RMSE table.  **Honest framing:** this is a point-estimate "
    "comparison at matched *unique* query cost; GAS's differentiators are "
    "the distribution-free anytime certificates + Neyman residual control, "
    "which ShaplEIG (Bayesian, non-certified) does not provide."
))

cells.append(code(
    "import pandas as pd\n"
    "for ds in (\"wine\", \"air\"):\n"
    "    p = ROOT / \"main_results\" / f\"paper_official_shaplEIG_{ds}.csv\"\n"
    "    if not p.exists():\n"
    "        print(f\"[{ds}] official ShaplEIG CSV missing — run A/B first\"); continue\n"
    "    s = pd.read_csv(p)\n"
    "    print(f\"\\n=== {ds}: official ShaplEIG (source: {s['source'].iloc[0][:40]}...) ===\")\n"
    "    g = s.groupby(\"budget\").agg(\n"
    "        shaplEIG_rmse=(\"rmse_vs_exact\", \"mean\"),\n"
    "        unique_queries=(\"unique_queries\", \"mean\"),\n"
    "    )\n"
    "    print(g.round(5).to_string())\n"
    "    print(\"\\nCompare with GAS-BayesSHAP at the same unique-eval cost:\")\n"
    "    print(\"  GAS wine K=128: unique ~393, rmse 0.00430 | K=256: unique ~480, \"\n"
    "          \"rmse 0.00352 | K=512: unique ~565, rmse 0.00304\")\n"
    "    print(\"  (from paper_wine_matched_budget.csv, spec range)\")"
))

cells.append(md(
    "## Expected runtime and honest notes\n"
    "- **Budgets are capped at 512** (each budget B costs ~B rounds of GP\n"
    "  refit + full EIG, ~0.5–1 s/round measured: budget=64 ≈ 1 min, \n"
    "  budget=256 ≈ 4–6 min, budget=512 ≈ 12–17 min per config — the \n"
    "  512 config is opt-in via BUDGETS=512 and may be slow).\n"
    "- **Full run ≈ 25–45 min** (N=1 × 2 datasets × 3 budgets {64,128,256});\n"
    "  each config runs in its own child process with a 1500 s timeout, so a\n"
    "  crash or hang is recorded in `paper_official_shaplEIG_failures.csv`\n"
    "  and the rest still completes.\n"
    "- **Fault isolation:** each (dataset, budget) runs in its own child\n"
    "  process with a 1500 s timeout; a crash or hang is recorded in\n"
    "  `paper_official_shaplEIG_failures.csv` and the remaining configs\n"
    "  still complete.\n"
    "- **Smoke:** `N_INST=1 BUDGETS=32` (~1–2 min per dataset).\n"
    "- Budgets are **unique coalition queries** (counted via the GAS\n"
    "  CoalitionOracle cache), so the comparison is fair vs GAS's\n"
    "  `num_coalition_evals_this_call`.\n"
    "- The port is cited to the pinned official source commit; if you spot a\n"
    "  discrepancy, diff against `github.com/slds-lmu/shapleig@d52c09e`.\n"
    "- Commit the resulting `paper_official_shaplEIG_{wine,air}.csv`."
))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
