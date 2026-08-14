#!/usr/bin/env python
"""Orchestrate the full pipeline: validation -> exact -> bayesshap -> benchmark
(spec section 3 stages BENCHMARK / REPORT / COMPLIANCE_AUDIT).

Usage:
    python scripts/run_all.py [--config configs/default.yaml] [--M 5] [--budget 400]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: list) -> int:
    print(f"\n$ {' '.join(map(str, cmd))}")
    return subprocess.call([str(c) for c in cmd], cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="GAS-BayesSHAP full pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    steps = [
        ([sys.executable, "scripts/validate_math.py"], "MATHEMATICAL_VALIDATION"),
        ([sys.executable, "scripts/run_exact.py", "--M", args.M], "EXACT_REFERENCE"),
        ([sys.executable, "scripts/run_bayesshap.py", "--config", args.config,
          "--M", args.M, "--epsilon", args.epsilon, "--max-budget", args.budget],
         "GAS_BAYESHAP_RUN"),
    ]
    if not args.skip_benchmark:
        steps.append(
            ([sys.executable, "scripts/benchmark.py", "--M", args.M,
              "--budget", args.budget, "--epsilon", args.epsilon,
              "--trials", args.trials], "BENCHMARK")
        )

    failed = []
    for cmd, label in steps:
        rc = run(cmd)
        status = "EXECUTED" if rc == 0 else "FAILED"
        print(f"[{label}] {status} (exit {rc})")
        if rc != 0:
            failed.append(label)
    print("\n" + "=" * 60)
    if failed:
        print("FAILED stages:", failed)
        return 1
    print("ALL PIPELINE STAGES EXECUTED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
