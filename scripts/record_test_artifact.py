#!/usr/bin/env python
"""Run the test suite + math validation and record a TEST_ARTIFACT.json.

This produces the "actual recorded pytest artifact for the current commit"
that reviewers keep asking for:

    results/TEST_ARTIFACT.json
    {
      "commit": "8a821f2dd440...",
      "dirty": false,
      "python": "...",
      "packages": {...},
      "pytest": {"passed": N, "failed": N, "duration_s": ...},
      "math_validation": {"passed": true, "max_lemma_diff": ...},
      "timestamp": "..."
    }

Usage:
    python scripts/record_test_artifact.py [--out results/TEST_ARTIFACT.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gas_bayesshap.utils.reproducibility import environment_manifest  # noqa: E402
from gas_bayesshap.utils.serialization import write_json_atomic  # noqa: E402


def run(cmd: list) -> tuple:
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                          capture_output=True, text=True)
    return proc, time.time() - t0


def parse_pytest_summary(text: str) -> dict:
    """Parse the trailing 'N passed, M failed' summary from pytest -q output."""
    import re
    passed = failed = 0
    for m in re.finditer(r"(\d+)\s+passed", text):
        passed = max(passed, int(m.group(1)))
    for m in re.finditer(r"(\d+)\s+failed", text):
        failed = max(failed, int(m.group(1)))
    for m in re.finditer(r"(\d+)\s+error", text):
        failed = max(failed, int(m.group(1)))
    return {"passed": passed, "failed": failed}


def main() -> int:
    p = argparse.ArgumentParser(description="Record a pytest + math-validation artifact")
    p.add_argument("--out", default=str(ROOT / "results" / "TEST_ARTIFACT.json"))
    p.add_argument("--pytest-args", default="tests/ -q", help="pytest invocation args")
    args = p.parse_args()

    env = environment_manifest(str(ROOT))

    proc, dur = run([sys.executable, "-m", "pytest", *args.pytest_args.split()])
    summary = parse_pytest_summary(proc.stdout or proc.stderr)
    summary["duration_s"] = round(dur, 2)

    math_proc, _ = run([sys.executable, "scripts/validate_math.py"])
    math_ok = math_proc.returncode == 0

    artifact = {
        "commit": env["git"]["commit"],
        "dirty": env["git"]["dirty"],
        "python": env["python_version"].split()[0],
        "packages": env["packages"],
        "pytest": summary,
        "math_validation": {"passed": bool(math_ok),
                            "note": "scripts/validate_math.py exit 0" if math_ok else math_proc.stderr[-300:]},
        "timestamp": env["timestamp"],
    }
    write_json_atomic(args.out, artifact, sort_keys=True)
    print(json.dumps(artifact, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0 if summary["failed"] == 0 and math_ok else 1


if __name__ == "__main__":
    sys.exit(main())
