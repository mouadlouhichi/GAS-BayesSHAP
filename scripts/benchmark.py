import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print('Use scripts/run_exact.py and scripts/run_bayesshap.py; benchmark artifacts are written to results/runs/<run_id>.')
