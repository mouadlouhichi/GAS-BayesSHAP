#!/usr/bin/env python
"""Build the 'audit P1 items' notebook (items 2-4 of the latest audit).

  A. Unique-query-capped GAS vs ShaplEIG comparison (audit item 2)
     `run_unique_capped_shaplEIG.py --n 10 --caps 512,1024`
     GAS is hard-capped at U unique evals (cache off; Stage-2 budget =
     U - Stage-1).  Honest note: GAS's fixed Stage-1 cost at M=11 is
     ~371 unique evals, so caps {64,128,256} cannot be honoured;
     {512,1024} are the meaningful sub-enumerative caps (both < 2^11).
  B. Regime semantics with >=20 instances per named regime (item 3)
     `regime_semantics.py --per-regime 20 --clusters 4`
     n = 20*4 = 80 instances -> 20 per cluster/regime name.
  C. Additional hard high-dim games (item 4): threshold (2-of-4) and
     unanimity (4-way AND) at M=30, spec range
     `probe_high_dim.py --game threshold|unanimity --budgets 50000,100000`

Usage:
    python scripts/build_audit_p1_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "main_results" / "RUN_AUDIT_P1_ITEMS.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

cells.append(md(
    "# GAS-BayesSHAP — audit P1 items (unique-capped ShaplEIG, regimes ≥20/regime, hard high-dim games)\n"
    "\n"
    "Closes the three open P1 items from the latest audit.  Orchestrates the "
    "real CLI scripts only — no duplicated algorithm."
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
    "N_INST  = int(os.environ.get(\"N_INST\", \"10\"))\n"
    "CAPS    = os.environ.get(\"CAPS\", \"512,1024\")\n"
    "PER_REG = int(os.environ.get(\"PER_REG\", \"20\"))\n"
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
    "print(f\"N_INST={N_INST} CAPS={CAPS} PER_REG={PER_REG} SKIP={sorted(SKIP)}\")"
))

cells.append(md("## A. Unique-query-capped GAS vs ShaplEIG (audit item 2)"))

cells.append(md(
    "`run_unique_capped_shaplEIG.py --n 10 --caps 512,1024` runs GAS (cache "
    "disabled → every draw is a unique evaluation; Stage-2 budget = cap − "
    "Stage-1) and the ShaplEIG port at the same caps, recording each "
    "method's ACTUAL unique evals.  **Honest caveat:** GAS's fixed "
    "Stage-1+init+pilot cost at M=11 is ~371 unique evals, so caps "
    "{64,128,256} cannot be honoured (`cap_honored=False`); {512,1024} are "
    "the meaningful sub-enumerative caps (both ≪ 2^11=2048).  "
    "**~1–1.5 h on a laptop** (60 ShaplEIG runs, each refits a GP per "
    "round).  Smoke: `N_INST=1 CAPS=512`."
))

cells.append(code(
    'run("run_unique_capped_shaplEIG.py", "--n", str(N_INST), "--caps", CAPS,\n'
    '    tag=f"A. unique-capped GAS vs ShaplEIG (N={N_INST}, caps={CAPS})", skip=False)'
))

cells.append(code(
    "import pandas as pd\n"
    "p = ROOT / \"main_results\" / \"paper_unique_capped_shaplEIG.csv\"\n"
    "if p.exists():\n"
    "    d = pd.read_csv(p)\n"
    "    print(d[[\"dataset\", \"instance\", \"unique_cap\", \"gas_rmse\", \"shaplEIG_rmse\",\n"
    "             \"gas_unique_evals\", \"shaplEIG_unique_queries\", \"cap_honored\",\n"
    "             \"unique_matched\"]].to_string(index=False))\n"
    "    hon = d[d.cap_honored]\n"
    "    print(\"\\n=== cap-honored summary ===\")\n"
    "    if not hon.empty:\n"
    "        print(hon.groupby([\"dataset\", \"unique_cap\"]).agg(\n"
    "            gas_rmse=(\"gas_rmse\", \"mean\"),\n"
    "            gas_unique=(\"gas_unique_evals\", \"mean\"),\n"
    "            shaplEIG_rmse=(\"shaplEIG_rmse\", \"mean\"),\n"
    "            shaplEIG_unique=(\"shaplEIG_unique_queries\", \"mean\"),\n"
    "        ).round(5).to_string())\n"
    "    print(f\"\\ncap_honored: {int(hon.shape[0])}/{len(d)} rows\")"
))

cells.append(md("## B. Regime semantics — ≥20 instances per named regime (item 3)"))

cells.append(md(
    "`regime_semantics.py --per-regime 20 --clusters 4` runs n = 80 "
    "instances (20 per cluster → 20 per named regime, incl. the suffixed "
    "clean-air subregimes).  **~15–25 min.**"
))

cells.append(code(
    'run("regime_semantics.py", "--per-regime", str(PER_REG), "--clusters", "4",\n'
    '    "--eps", "0.05", "--budget", "3000",\n'
    '    tag=f"B. regime semantics per-regime N={PER_REG}", skip=\"B\" in SKIP)'
))

cells.append(code(
    "import pandas as pd\n"
    "p = ROOT / \"main_results\" / \"paper_regime_semantics_summary.csv\"\n"
    "if p.exists():\n"
    "    d = pd.read_csv(p)\n"
    "    print(d.to_string(index=False))\n"
    "    print(\"\\nmin per-regime N:\", int(d[\"n\"].min()), \"(audit: >= 20)\")"
))

cells.append(md("## C. Hard high-dim games: threshold + unanimity (item 4)"))

cells.append(md(
    "`probe_high_dim.py --game threshold|unanimity --budgets 50000,100000` "
    "at M=30, spec range.  Both games have closed-form exact Shapley values "
    "(validated vs brute force to machine precision at M≤12): threshold "
    "2-of-4 (φ=0.0625 on 4 drivers), unanimity 4-way AND (φ=0.125 on 4 "
    "drivers).  Expect: GP control variate degrades vs the sparse game, the "
    "spec-range interval stays rigorous and certifies nothing falsely.  "
    "**~2 min each.**"
))

cells.append(code(
    'run("probe_high_dim.py", "--M", "30", "--budgets", "50000,100000",\n'
    '    "--game", "threshold", "--mode", "spec",\n'
    '    tag="C1. M=30 threshold game (spec)", skip="C" in SKIP)'
))

cells.append(code(
    'run("probe_high_dim.py", "--M", "30", "--budgets", "50000,100000",\n'
    '    "--game", "unanimity", "--mode", "spec",\n'
    '    tag="C2. M=30 unanimity game (spec)", skip="C" in SKIP)'
))

cells.append(code(
    "import pandas as pd\n"
    "for game in (\"threshold\", \"unanimity\"):\n"
    "    p = ROOT / \"main_results\" / f\"paper_high_dim_M30_{game}_spec_summary.csv\"\n"
    "    if p.exists():\n"
    "        d = pd.read_csv(p)\n"
    "        print(f\"\\n=== {game} ===\")\n"
    "        print(d[[\"K\", \"status\", \"unique_coalition_evals\", \"unique_vs_2M_ratio\",\n"
    "                 \"n_sign_certified\", \"rmse_vs_exact\", \"mean_width\"]].to_string(index=False))\n"
    "        print(f\"  certificate_is_rigorous: {bool(d['certificate_is_rigorous'].all())} \"\n"
    "              f\"(must be True: spec interval valid even under misspecification)\")"
))

cells.append(md(
    "## Expected runtime and honest notes\n"
    "- **Full run ≈ 1.5–2.5 h** (A ~1–1.5 h, B ~15–25 min, C ~5 min).\n"
    "- **Smoke:** `N_INST=1 CAPS=512 PER_REG=4 GAS_SKIP=C` (~5 min).\n"
    "- A: rows with `cap_honored=False` (cap below GAS's ~371 fixed cost)\n"
    "  are excluded from the summary; the CSV keeps them for honesty.\n"
    "- Commit the resulting CSVs: `paper_unique_capped_shaplEIG.csv`,\n"
    "  `paper_regime_semantics{,_summary}.csv`,\n"
    "  `paper_high_dim_M30_{threshold,unanimity}_spec_summary.csv`."
))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
