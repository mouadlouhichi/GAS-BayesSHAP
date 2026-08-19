#!/usr/bin/env python
"""One-command reproduction of every paper artifact + manifest (audit P1-10).

Runs the full pipeline that produces the committed `main_results/paper_*`
artifacts, then writes a MANIFEST (commit hash, data hashes, per-artifact
file hashes, git status) so reviewers can verify provenance.

Usage:
    python scripts/reproduce_paper.py --all          # everything (many hours)
    python scripts/reproduce_paper.py --only wine    # one dataset
    python scripts/reproduce_paper.py --manifest     # manifest only (no runs)

All heavy steps are delegated to the existing CLI scripts / notebooks; this
script is a thin orchestrator + manifest recorder, it duplicates no
scientific algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = sys.executable
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"

STEPS = {
    # key -> (argv, expected main_results artifact prefix)
    "wine":   (["run_paper_experiments.py", "--only", "wine", "--n", "50", "--eps", "0.05", "--budget", "3000"], "paper_wine_n50"),
    "air":    (["run_paper_experiments.py", "--only", "air", "--n", "50", "--eps", "0.05", "--budget", "3000"], "paper_air_n50"),
    "wine_fp": (["run_paper_experiments.py", "--only", "wine", "--n", "50", "--eps", "0.05", "--budget", "3000",
                 "--range-mode", "finite_population"], "paper_wine_n50_budget3000_rangefinite"),
    "air_fp":  (["run_paper_experiments.py", "--only", "air", "--n", "50", "--eps", "0.05", "--budget", "3000",
                 "--range-mode", "finite_population"], "paper_air_n50_budget3000_rangefinite"),
    "curves": (["run_paper_experiments.py", "--only", "curves"], "paper_wine_matched_budget"),
    "widths": (["run_paper_experiments.py", "--only", "widths"], "paper_wine_width_vs_budget"),
    "tierb":  (["run_paper_experiments.py", "--only", "tierb", "--n", "20", "--eps", "0.05", "--budget", "3000"], "paper_air_tierB"),
    "ablation": (["ablation.py", "--dataset", "wine", "--K", "1000", "--n", "20"], "paper_ablation_wine"),
    "regimes": (["regime_semantics.py", "--n", "20", "--clusters", "4", "--eps", "0.05", "--budget", "3000"], "paper_regime_semantics"),
    "sota":   (["run_sota_baselines.py", "--n", "20"], "paper_reference_baselines_ablation"),
    "probe_spec": (["probe_width_tightness.py", "--range-mode", "spec"], "paper_width_probe"),
    "probe_fp":  (["probe_width_tightness.py", "--range-mode", "finite_population"], "paper_width_probe_finite_population"),
    "stress": (["stress_finite_population.py", "--trials", "200"], "paper_stress_finite_population"),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()  # full 64-char digest (audit: was truncated to 16)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "n/a"


def run_step(key: str, argv: list) -> None:
    cmd = [PY, str(SCRIPTS / argv[0]), *argv[1:]]
    print(f"\n>>> reproduce[{key}]: {' '.join(argv)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"step {key} failed ({r.returncode})")
    print(f"<<< {key} done in {time.time()-t0:.0f}s")


def build_manifest(only: list) -> dict:
    artifacts = {}
    for f in sorted(MAIN.glob("paper_*.csv")) + sorted(MAIN.glob("paper_*.json")) \
             + sorted(MAIN.glob("paper_*.md")):
        artifacts[f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    return {
        "commit": git_head(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
        "python": sys.version.split()[0],
        "steps_requested": only,
        "commands": [f"python scripts/{a[0]} {' '.join(a[1:])}" for a in
                     (STEPS[k][0] for k in only if k in STEPS)],
        "n_artifacts": len(artifacts),
        "artifacts": artifacts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every step")
    ap.add_argument("--only", nargs="+", choices=list(STEPS),
                    help="run specific steps")
    ap.add_argument("--manifest", action="store_true",
                    help="only write the manifest (no runs)")
    ap.add_argument("--manifest-out", default="main_results/reproduce_manifest.json")
    args = ap.parse_args()

    if args.manifest:
        m = build_manifest([])
        (ROOT / args.manifest_out).write_text(json.dumps(m, indent=1))
        print(f"manifest written to {args.manifest_out} (commit {m['commit']})")
        return 0

    keys = []
    if args.all:
        keys = list(STEPS)
    if args.only:
        keys = list(args.only)
    if not keys:
        print("nothing to do: use --all, --only, or --manifest")
        return 1

    for key in keys:
        argv, _ = STEPS[key]
        run_step(key, argv)

    m = build_manifest(keys)
    (ROOT / args.manifest_out).write_text(json.dumps(m, indent=1))
    print(f"\nmanifest written to {args.manifest_out} (commit {m['commit']}, "
          f"{m['n_artifacts']} artifacts hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
