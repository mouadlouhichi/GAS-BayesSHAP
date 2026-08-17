#!/usr/bin/env python
"""Build the 'audit-fix reruns' notebook.

Reruns the experiments invalidated or made incomplete by the latest deep
audit (commit af2d62e):

  A. SOTA-style baselines with the FIXED GP design (audit P0-1): the old
     script re-seeded RandomState(7+i) inside the loop, so all design masks
     were identical; the fixed script uses one rng per instance, advances
     per draw, dedupes, scales design with K, and renames the outputs
     (`paper_reference_baselines_ablation.csv`).
  B. N=50 finite-population wine+air with full certificate diagnostics
     (audit P0-4): per-instance + summary now include delta1_coupon,
     reported_coverage_level, coupon_threshold_satisfied,
     certificate_at_nominal_level, certificate_is_rigorous.
  C. Matched-budget curves with wall-clock seconds per method (audit P0-3).
  D. Regime semantics with duplicate-name suffixing (audit P1-8:
     clean_air -> clean_air_2) and optional N>=20 per regime.
  E. Finite-population Tier-B (audit P1-9).
  F. One-command reproduction manifest (audit P1-10).

Every section is env-overridable and skippable (GAS_SKIP=A,B,C,D,E,F).

Usage:
    python scripts/build_audit_fixes_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "main_results" / "RUN_AUDIT_FIXES.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

cells.append(md(
    "# GAS-BayesSHAP — audit-fix reruns\n"
    "\n"
    "Reruns the experiments that the latest deep audit (commit `af2d62e`) "
    "invalidated or found incomplete.  Orchestrates the real CLI scripts "
    "only; duplicates no scientific algorithm.  Each section states what "
    "changed and why the rerun is needed."
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
    "SOTA_N   = int(os.environ.get(\"SOTA_N\", \"20\"))\n"
    "FP_N     = int(os.environ.get(\"FP_N\", \"50\"))\n"
    "REG_N    = int(os.environ.get(\"REG_N\", \"20\"))\n"
    "EPS      = float(os.environ.get(\"GAS_EPS\", \"0.05\"))\n"
    "BUDGET   = int(os.environ.get(\"GAS_BUDGET\", \"3000\"))\n"
    "SKIP     = set(os.environ.get(\"GAS_SKIP\", \"\").split(\",\")) - {\"\"}\n"
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
    "print(f\"SOTA_N={SOTA_N} FP_N={FP_N} REG_N={REG_N} SKIP={sorted(SKIP)}\")"
))

cells.append(md("## A. SOTA-style baselines — FIXED GP design (audit P0-1)"))

cells.append(md(
    "**Bug:** `run_sota_baselines.py` re-seeded `RandomState(7 + i)` inside "
    "the design loop, so all 256 design masks were identical (the committed "
    "`paper_sota_baselines_comparison.csv` shows a constant ShaplEIG-style "
    "RMSE across K for each instance).  **Fix:** one rng per instance "
    "advanced per draw, design deduplicated and sized from K, outputs "
    "renamed to `paper_reference_baselines_ablation.csv` (OddSHAP-style = "
    "log-odds transform reference, GP-quadrature = surrogate reference; "
    "**not** a matched-budget comparison)."
))

cells.append(code(
    'run("run_sota_baselines.py", "--n", str(SOTA_N),\n'
    '    tag=f"A. reference baselines (fixed design) N={SOTA_N}", skip="A" in SKIP)'
))

cells.append(code(
    'import pandas as pd\n'
    'p = ROOT / "main_results" / "paper_reference_baselines_ablation.csv"\n'
    'if p.exists():\n'
    '    d = pd.read_csv(p)\n'
    '    print(d.groupby(["dataset", "K"]).agg(\n'
    '        gas=("gas_rmse", "mean"),\n'
    '        odd_logodds=("odd_logodds_rmse", "mean"),\n'
    '        gp_quadrature=("gp_quadrature_rmse", "mean"),\n'
    '        gp_unique=("gp_unique_coalitions", "min"),\n'
    '    ).round(5).to_string())\n'
    '    # sanity: GP design must now contain distinct coalitions\n'
    '    print("\\nmin unique coalitions across runs:", int(d["gp_unique_coalitions"].min()))\n'
    '    assert d["gp_unique_coalitions"].min() > 1, "design still degenerate!"'
))

cells.append(md("## B. N=50 finite-population with certificate diagnostics (audit P0-4)"))

cells.append(md(
    "The committed fp N=50 CSVs predate the diagnostics fields.  The runner "
    "now records `delta1_coupon`, `reported_coverage_level`, "
    "`coupon_threshold_satisfied`, `certificate_at_nominal_level`, "
    "`certificate_is_rigorous` per instance and aggregates them in the "
    "summary, so `simultaneous_coverage_rate=1.0` is interpretable as an "
    "empirical frequency alongside the formal certificate status.  "
    "~2.5–3 h for both datasets."
))

cells.append(code(
    'run("run_paper_experiments.py", "--only", "wine", "--n", str(FP_N),\n'
    '    "--eps", str(EPS), "--budget", str(BUDGET), "--range-mode", "finite_population",\n'
    '    tag=f"B1. wine fp N={FP_N} + diagnostics", skip="B" in SKIP)'
))

cells.append(code(
    'run("run_paper_experiments.py", "--only", "air", "--n", str(FP_N),\n'
    '    "--eps", str(EPS), "--budget", str(BUDGET), "--range-mode", "finite_population",\n'
    '    tag=f"B2. air fp N={FP_N} + diagnostics", skip="B" in SKIP)'
))

cells.append(md("## C. Matched-budget curves with wall-clock (audit P0-3)"))

cells.append(md(
    "The committed curves already carry `*_evals_actual` columns; the runner "
    "now also records per-method wall-clock seconds (`gas_wall_s`, "
    "`kernel_wall_s`, `mc_wall_s`) so the nominal-vs-actual gap is fully "
    "quantified.  ~1 h for both datasets."
))

cells.append(code(
    'run("run_paper_experiments.py", "--only", "curves", "--range-mode", "spec",\n'
    '    tag="C. instrumented curves + wall-clock", skip="C" in SKIP)'
))

cells.append(md("## D. Regime semantics — duplicate-name suffixing (audit P1-8)"))

cells.append(md(
    "`name_regime` mapped two k-means clusters to `clean_air`, silently "
    "collapsing distinct subregimes.  The script now suffixes duplicates "
    "(`clean_air_2`) and prints the full mapping.  Run with N=20 (default) "
    "or N=60 for ≥20 instances per named regime."
))

cells.append(code(
    'run("regime_semantics.py", "--n", str(REG_N), "--clusters", "4",\n'
    '    "--eps", str(EPS), "--budget", str(BUDGET),\n'
    '    tag=f"D. regime semantics N={REG_N} (suffixed names)", skip="D" in SKIP)'
))

cells.append(md("## E. Finite-population Tier-B (audit P1-9)"))

cells.append(md(
    "Tier-B (group-lag, M=66 → 11 macros) has only been run with the spec "
    "range.  This runs it with `range_mode=finite_population` to check "
    "whether group-level certification improves at the same budget."
))

cells.append(code(
    'run("run_paper_experiments.py", "--only", "tierb", "--n", "20",\n'
    '    "--eps", str(EPS), "--budget", str(BUDGET), "--range-mode", "finite_population",\n'
    '    tag="E. Tier-B finite-population N=20", skip="E" in SKIP)'
))

cells.append(md("## F. Reproduction manifest (audit P1-10)"))

cells.append(code(
    'run("reproduce_paper.py", "--manifest",\n'
    '    tag="F. write reproduce_manifest.json", skip="F" in SKIP)'
))

cells.append(code(
    'm = json.loads((ROOT / "main_results" / "reproduce_manifest.json").read_text())\n'
    'print("commit:", m["commit"], "| dirty:", m["git_dirty"],\n'
    '      "| artifacts hashed:", m["n_artifacts"])\n'
    'print("\\nTo regenerate everything from scratch:\\n"\n'
    '      "  python scripts/reproduce_paper.py --all")'
))

cells.append(md(
    "## Expected runtime (laptop) and notes\n"
    "- **Full run ≈ 5–6 h:** A ≈ 40–60 min, B ≈ 2.5–3 h, C ≈ 1 h,\n"
    "  D ≈ 15 min, E ≈ 35 min, F ≈ 0.\n"
    "- **Smoke run (≈ 12 min):** `GAS_SKIP=B,C,E SOTA_N=2 REG_N=4`\n"
    "- **What each section can/cannot claim is stated in its markdown cell.**\n"
    "  In particular: A replaces the invalid ShaplEIG artifact entirely;\n"
    "  B does not change any RMSE/coverage numbers (same seeds), it adds\n"
    "  certificate diagnostics; C adds wall-clock to the existing curves."
))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
