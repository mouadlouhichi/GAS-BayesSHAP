#!/usr/bin/env python
"""Build the 'high-dimensional sub-enumerative certification' notebook.

Answers the latest audit's decisive question: does GAS-BayesSHAP certify
anything at a dimension where 2^M is infeasible, with unique coalition
evaluations << 2^M?  The M=11 nominal-certification runs were
post-enumerative (2048 unique = 2^11); this run moves to M=30 (2^30 ~ 1.07e9)
on a sparse synthetic game with CLOSED-FORM exact Shapley values.

Sections:
  A. M=30, finite-population, budgets {20k, 50k, 100k}  (fast grid)
  B. M=30, finite-population, budgets {500k, 1e6}       (sign-cert search)
  C. M=30, spec-range contrast at K=100k                (width ratio)
  D. Summary: unique-vs-2^M, sign-cert, at_nominal      (honest report)

Usage:
    python scripts/build_high_dim_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "main_results" / "RUN_HIGH_DIM.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

cells.append(md(
    "# GAS-BayesSHAP — high-dimensional sub-enumerative certification (M=30)\n"
    "\n"
    "The latest audit's decisive question: the M=11 nominal-certification "
    "runs were **post-enumerative** (2048 unique coalition evals = 2^11 — the "
    "full power set was cached before the certificate closed).  This notebook "
    "runs the same finite-population machinery at **M=30** where 2^30 ≈ 1.07e9 "
    "is infeasible, on a sparse synthetic game with **closed-form exact "
    "Shapley values** (so RMSE and sign validation stay checkable), and "
    "reports unique coalition evals vs 2^M, sign-certified features, and the "
    "nominal-certification status.  Orchestrates "
    "`scripts/probe_high_dim.py` only — no duplicated algorithm."
))

cells.append(md("## 0. Environment & config"))

cells.append(code(
    "import sys, os, time, json, subprocess\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, \"..\")\n"
    "import gas_bayesshap\n"
    "\n"
    "ROOT = Path(\"..\").resolve()\n"
    "SCRIPTS = ROOT / \"scripts\"\n"
    "print(\"GAS-BayesSHAP\", gas_bayesshap.__version__)\n"
    "\n"
    "M       = int(os.environ.get(\"HM_M\", \"30\"))\n"
    "GRID    = os.environ.get(\"HM_GRID\", \"20000,50000,100000\")\n"
    "HIGHK   = os.environ.get(\"HM_HIGHK\", \"500000,1000000\")\n"
    "SKIP    = set(os.environ.get(\"GAS_SKIP\", \"\").split(\",\")) - {\"\"}\n"
    "\n"
    "def run(*args, tag=\"\", skip=False):\n"
    "    if skip:\n"
    "        print(f\"--- SKIPPED: {tag or ' '.join(args)}\"); return 0.0\n"
    "    cmd = [sys.executable, str(SCRIPTS / args[0]), *args[1:]]\n"
    "    t0 = time.time()\n"
    "    print(f\"\\n>>> {tag or ' '.join(args)}\")\n"
    "    r = subprocess.run(cmd, cwd=ROOT)\n"
    "    dt = time.time() - t0\n"
    "    if r.returncode != 0:\n"
    "        raise RuntimeError(f\"FAILED ({r.returncode}): {' '.join(args)}\")\n"
    "    print(f\"<<< done in {dt/60:.1f} min\")\n"
    "    return dt\n"
    "\n"
    "print(f\"M={M} GRID={GRID} HIGHK={HIGHK} SKIP={sorted(SKIP)}\")"
))

cells.append(md("## A. M=30 grid — finite-population, K ∈ {20k, 50k, 100k}"))

cells.append(md(
    "Fast grid (~3 min).  Expected: unique coalition evals ~2e4–9e4 ≪ 2^30 "
    "(ratio ~1e-4), point RMSE ~1e-5, widths shrinking as ~1/√K; "
    "`certificate_at_nominal_level=False` (coupon budget over C(29,s) pairs "
    "cannot close at these K)."
))

cells.append(code(
    'run("probe_high_dim.py", "--M", str(M), "--budgets", GRID,\n'
    '    "--mode", "finite_population", tag=f"A. M={M} fp grid K={GRID}",\n'
    '    skip="A" in SKIP)'
))

cells.append(md("## B. M=30 sign-cert search — K ∈ {500k, 1e6}"))

cells.append(md(
    "The widths at K=1e5 (~0.09) are above the driver attributions "
    "(|φ|≈0.033–0.037); the width law predicts sign certification of the "
    "driver features somewhere in K ≈ 5e5–1e6.  **~15–25 min.**  If it "
    "appears, it is an *empirical-event* sign certification (the certified "
    "interval is not a nominal 1−δ certificate — the coupon stays open); "
    "the notebook reports that distinction explicitly."
))

cells.append(code(
    'run("probe_high_dim.py", "--M", str(M), "--budgets", HIGHK,\n'
    '    "--mode", "finite_population", tag=f"B. M={M} fp high-K {HIGHK}",\n'
    '    skip="B" in SKIP)'
))

cells.append(md("## C. Spec-range contrast at K=100k"))

cells.append(md(
    "One spec-range row for the width ratio (spec ≈ 20× fp at the same "
    "budget), confirming the empirical range is what makes high-dim sign "
    "certification feasible at all.  ~2 min."
))

cells.append(code(
    'run("probe_high_dim.py", "--M", str(M), "--budgets", "100000",\n'
    '    "--mode", "spec", tag=f"C. M={M} spec contrast K=100k",\n'
    '    skip="C" in SKIP)'
))

cells.append(md("## D. Summary — sub-enumerative? sign-cert? nominal?"))

cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "p = ROOT / \"main_results\" / f\"paper_high_dim_M{M}_summary.csv\"\n"
    "if not p.exists():\n"
    "    print(f\"NOT FOUND: {p.name} — run sections A–C first\")\n"
    "else:\n"
    "    d = pd.read_csv(p)\n"
    "    cols = [\"K\", \"range_mode\", \"status\", \"converged\",\n"
    "            \"certificate_at_nominal_level\", \"realised_coverage_level\",\n"
    "            \"unique_coalition_evals\", \"unique_vs_2M_ratio\",\n"
    "            \"n_sign_certified\", \"signs_match_exact\", \"rmse_vs_exact\",\n"
    "            \"mean_width\"]\n"
    "    print(d[cols].to_string(index=False))\n"
    "    fp = d[d.range_mode == \"finite_population\"]\n"
    "    print(\"\\nHONEST REPORT:\")\n"
    "    print(f\"  unique/2^M ratio: {fp['unique_vs_2M_ratio'].min():.2e} .. \"\n"
    "          f\"{fp['unique_vs_2M_ratio'].max():.2e}  (sub-enumerative if << 1)\")\n"
    "    print(f\"  max unique: {int(fp['unique_coalition_evals'].max())} vs 2^M = {2**M}\")\n"
    "    print(f\"  sign-certified runs: {int((fp['n_sign_certified'] > 0).sum())} \"\n"
    "          f\"| signs validated: {bool(fp['signs_match_exact'].all())}\")\n"
    "    print(f\"  at_nominal_level anywhere: {bool(fp['certificate_at_nominal_level'].any())} \"\n"
    "          \"(coupon wall at M=30 — expected False; the rigorous nominal \"\n"
    "          \"certificate is near-enumerative, stated honestly in the paper)\")"
))

cells.append(md(
    "## Expected runtime and honest notes\n"
    "- **Full run ≈ 25–35 min** (A ~3 min, B ~15–25 min, C ~2 min).\n"
    "- **Smoke:** `GAS_SKIP=B,C HM_GRID=5000` (~30 s).\n"
    "- The M=11 nominal-certification runs remain **post-enumerative** "
    "(2048 unique = 2^11) — this notebook does NOT change that; it "
    "establishes what happens when 2^M is infeasible.\n"
    "- The M=30 result will likely show: sub-enumerative point-estimate "
    "fidelity (unique ≪ 2^30), empirical-range sign certification of "
    "drivers at high K, but `certificate_at_nominal_level=False` — the "
    "coupon-collector budget over C(M-1,s) pairs is the honest scaling "
    "wall.  Report it exactly as it lands.\n"
    "- Commit the resulting `paper_high_dim_M{M}_summary.csv`."
))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
