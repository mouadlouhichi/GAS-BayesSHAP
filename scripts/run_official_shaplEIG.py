#!/usr/bin/env python
"""Official ShaplEIG baseline on GAS-BayesSHAP games (faithful port, hardened).

The previous `gp_quadrature_rmse` baseline in
`paper_reference_baselines_ablation.csv` was a *method-style* GP surrogate.
The audits asked for the **official** ShaplEIG (ICML 2026, Rundel et al.).
The authors' public repo (github.com/slds-lmu/shapleig, MIT) was pinned at
commit d52c09e and its core algorithm ported here:

  * GP surrogate over the binary coalition space: gpytorch ExactGP with a
    Hamming kernel + ScaleKernel + Gaussian likelihood, hyperparameters by
    marginal-likelihood maximisation (matches the official GPSurrogate
    HammingKernelConfig/MLMConfig family).
  * Acquisition = expected information gain about the Shapley property,
    exactly as `_compute_eig_function_property_naive_Z` in the official
    source:  EIG(Z) = log var_yz - log(var_yz - correction), with
    correction = diag( (A K_fz)^T (A K_fz A^T)^{-1} (A K_fz) )  (variance
    reduction of the property A f(Z) after conditioning).  Computed with
    plain torch.linalg (no gpytorch.inv_quad / linear_operator lazy path).
  * A = the Shapley coefficient matrix from the official
    `_get_shapley_weights` (w_in[k] = 1/(C(p-1,k-1) p)), i.e. phi = A @ v.
  * Exhaustive acquisition over all remaining coalitions; final attributions
    = A @ (GP posterior mean over all 2^M coalitions).

Robustness (fixes a SIGSEGV observed on macOS with the torch/linear_operator
lazy path):
  * `torch.set_num_threads(1)` / `set_num_interop_threads(1)` at import --
    OpenMP threading is the usual segfault trigger on macOS.
  * EIG computed with explicit torch.linalg.solve (no native lazy ops).
  * GP fitted with botorch's fit_gpytorch_mll (official fitter).
  * Per-config subprocess isolation: each (dataset, budget) runs in a child
    process; a native crash in one config is recorded as a failure row and
    the remaining configs still complete.

Usage:
    python scripts/run_official_shaplEIG.py --n 2 --budgets 64,256,512
    python scripts/run_official_shaplEIG.py --single wine 256   # one config
Outputs -> results/paper_experiments/official_shaplEIG_{wine,air}.csv
           + main_results/paper_official_shaplEIG_{wine,air}.csv
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# --- thread pinning FIRST (mitigates macOS OpenMP segfaults) --------------- #
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import numpy as np  # noqa: E402

try:
    import gpytorch  # noqa: F401
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import Kernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAVE_STACK = True
except ImportError as e:  # pragma: no cover
    HAVE_STACK = False
    _IMPORT_ERR = e

from gas_bayesshap import GASBayesSHAP  # noqa: E402
from gas_bayesshap.game.oracle import CoalitionOracle  # noqa: E402
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values  # noqa: E402
from gas_bayesshap.benchmarking.metrics import rmse  # noqa: E402

from run_paper_experiments import (  # noqa: E402
    load_wine, load_air_station, build_surrogate, make_proba_fn,
    WINE_FEATURES, FEATURES as AIR_FEATURES,
)

OUT = ROOT / "results" / "paper_experiments"
MAIN = ROOT / "main_results"

SHAPLEIG_SOURCE = "slds-lmu/shapleig@d52c09e (MIT) — ported from src/xac/acquisition_functions/acquisition_functions.py:_compute_eig_function_property_naive_Z and src/xac/applications/applications.py:_get_shapley_weights"


# --------------------------------------------------------------------------- #
# Hamming kernel (matches the official HammingKernelConfig surrogate)
# --------------------------------------------------------------------------- #
class HammingKernel(Kernel):
    """k(x, x') = exp(- d_H(x, x') / lengthscale)."""

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype,
                              device=x1.device)
        d = (x1.unsqueeze(-2) != x2.unsqueeze(-3)).sum(dim=-1).to(x1.dtype)
        return torch.exp(-d / self.lengthscale)


def _shapley_A(p: int, Z: torch.Tensor) -> torch.Tensor:
    """Official Shapley coefficient matrix: phi = A @ v(Z).

    Matches `_get_shapley_weights`: w_in[k] = 1/(C(p-1,k-1) p) applied to
    the marginal v(S u {i}) - v(S) for each feature i.
    Z: (2^p, p) binary rows.
    """
    n = Z.shape[0]
    A = torch.zeros(p, n, dtype=Z.dtype)
    for i in range(p):
        for r in range(n):
            S = Z[r]
            k = int(S.sum().item())          # |S|
            if S[i]:
                continue                     # out-coalition rows (i notin S)
            w = 1.0 / (math.comb(p - 1, k) * p)
            Sp = S.clone(); Sp[i] = True
            idx_in = int((Z == Sp).all(dim=1).nonzero()[0].item())
            A[i, idx_in] += w
            A[i, r] -= w
    return A


def _eig_naive_Z(surrogate, Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Port of _compute_eig_function_property_naive_Z (official).

    EIG(Z) = log var_yz - log(var_yz - correction), where var_yz is the
    posterior predictive variance (with observation noise) at Z and the
    correction is the variance reduction of the property A f(Z) from
    conditioning on the training data.  Computed with plain torch.linalg
    (no gpytorch.inv_quad / linear_operator lazy path -- the segfault
    source on macOS).
    """
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        post_f = surrogate.posterior(Z, observation_noise=False).mvn
        K_fz = post_f.covariance_matrix            # (n_Z, n_Z) dense
        # var_yz = posterior predictive variance WITH observation noise
        #          = posterior var + likelihood noise variance (one posterior)
        covar_yz_diag = post_f.variance + surrogate.likelihood.noise_covar.noise.squeeze()
        transformed = K_fz.matmul(A.T)             # (n_Z, p)
        quad = A.matmul(transformed)               # (p, p) = A K_fz A^T
        # correction_i = t_i^T quad^{-1} t_i  for each column t_i of (A K)^T
        rhs = transformed.transpose(-2, -1)        # (p, n_Z)
        jitter = 1e-8 * torch.eye(quad.shape[0], dtype=quad.dtype, device=quad.device)
        sol = torch.linalg.solve(quad + jitter, rhs)   # (p, n_Z)
        correction = (rhs * sol).sum(dim=0)        # (n_Z,)
        EIG = torch.log(covar_yz_diag) - torch.log(
            (covar_yz_diag - correction).clamp_min(1e-12))
        if EIG.ndim == 2:
            EIG = EIG.mean(dim=0)
        return EIG


class _GPSurrogate:
    """Botorch SingleTaskGP with Hamming kernel (official stack)."""

    def __init__(self, train_x, train_y):
        from botorch.models import SingleTaskGP
        kernel = ScaleKernel(HammingKernel())
        self.model = SingleTaskGP(train_x, train_y, covar_module=kernel)
        self.likelihood = self.model.likelihood

    def fit(self, steps: int = 40):
        from botorch.fit import fit_gpytorch_mll
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.model.train(); self.likelihood.train()
        with gpytorch.settings.fast_computations(False):
            fit_gpytorch_mll(mll)
        self.model.eval(); self.likelihood.eval()

    def posterior(self, Z, observation_noise=True):
        return self.model.posterior(Z, observation_noise=observation_noise)


def run_official_shaplEIG(game_fn, x0, M, budget, seed=1301, max_rounds=None):
    """Run the official ShaplEIG loop; return (phi, unique_queries, wall_s)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z = torch.tensor(
        [list(map(int, format(i, f"0{M}b"))) for i in range(2 ** M)],
        dtype=torch.float64,
    )
    A = _shapley_A(M, Z)

    archive_x = []
    archive_y = []
    queried = set()
    t0 = time.time()

    for init_idx in (0, 2 ** M - 1):
        S = Z[init_idx].numpy().astype(bool)
        archive_x.append(init_idx)
        archive_y.append(game_fn(S))
        queried.add(init_idx)

    rounds = 0
    max_rounds = max_rounds or (budget - len(queried))
    while len(queried) < budget and rounds < max_rounds:
        tx = Z[torch.tensor(archive_x)]
        ty = torch.tensor(archive_y, dtype=torch.float64).unsqueeze(-1)
        if len(tx) < 2:
            break
        model = _GPSurrogate(tx, ty)
        model.fit(steps=40)

        scores = _eig_naive_Z(model, Z, A)
        cand = torch.argsort(scores, descending=True)
        chosen = None
        for c in cand.tolist():
            if c not in queried:
                chosen = c
                break
        if chosen is None:
            break
        S = Z[chosen].numpy().astype(bool)
        archive_x.append(chosen)
        archive_y.append(game_fn(S))
        queried.add(chosen)
        rounds += 1

    tx = Z[torch.tensor(archive_x)]
    ty = torch.tensor(archive_y, dtype=torch.float64).unsqueeze(-1)
    model = _GPSurrogate(tx, ty)
    model.fit(steps=50)
    with torch.no_grad():
        mu = model.posterior(Z, observation_noise=False).mvn.mean
    phi = (A @ mu).numpy()
    wall = time.time() - t0
    return phi, len(queried), wall


def _game_fn(fn, bg, x0):
    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                             model_tag="shaplEIG")

    def game_fn(S_mask):
        return oracle.evaluate(x0, S_mask)

    return game_fn, oracle


def run_instance(name, X, feat_names, n_clusters, i, budget, seed=1301):
    m, X_te, _, _ = build_surrogate(X, n_clusters, feat_names)
    cid = i % n_clusters
    fn = make_proba_fn(m, feat_names, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=seed + i).values

    game_fn, oracle = _game_fn(fn, bg, x0)
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    phi, unique, wall = run_official_shaplEIG(game_fn, x0, X.shape[1], budget,
                                              seed=seed + i)
    return {
        "dataset": name, "instance": i, "budget": budget,
        "rmse_vs_exact": rmse(phi, phi_exact),
        "unique_queries": unique,
        "source": SHAPLEIG_SOURCE,
        "elapsed_s": round(wall, 1),
    }


def run_single(ds: str, X, feats, nc, i: int, B: int) -> dict:
    try:
        return run_instance(ds, X, feats, nc, i, B)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"dataset": ds, "instance": i, "budget": B,
                "rmse_vs_exact": float("nan"), "unique_queries": 0,
                "source": f"ERROR: {type(e).__name__}: {str(e)[:200]}",
                "elapsed_s": -1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--budgets", default="64,128,256")
    ap.add_argument("--dataset", choices=["wine", "air", "both"], default="both")
    ap.add_argument("--single", nargs=2, metavar=("DATASET", "BUDGET"),
                    help="run ONE (dataset, budget) config in-process (isolation mode)")
    ap.add_argument("--inst", type=int, default=0,
                    help="instance index for --single mode (default 0)")
    ap.add_argument("--timeout", type=int, default=1500,
                    help="per-config wall-clock timeout in seconds (default 1500)")
    args = ap.parse_args()

    if not HAVE_STACK:
        print(f"official ShaplEIG stack unavailable: {_IMPORT_ERR}")
        print("install with: pip install torch botorch gpytorch linear_operator")
        return 1

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- isolation mode: single config (called by the parent subprocess) -- #
    if args.single:
        ds, B = args.single[0], int(args.single[1])
        if ds == "wine":
            X, _ = load_wine(); feats, nc = WINE_FEATURES, 2
        else:
            X, _ = load_air_station(n_clusters=4); feats, nc = AIR_FEATURES, 4
        row = run_single(ds, X, feats, nc, args.inst, B)
        print(f"SINGLE {ds} inst={args.inst} budget={B}: rmse={row['rmse_vs_exact']:.5f} "
              f"unique={row['unique_queries']} ({row['elapsed_s']}s)")
        pd.DataFrame([row]).to_csv(OUT / f"official_shaplEIG_{ds}_b{B}.csv", index=False)
        return 0 if row["unique_queries"] > 0 else 1

    # ---- parent mode: launch each (dataset, instance, budget) isolated ---- #
    budgets = [int(b) for b in args.budgets.split(",")]
    datasets = ["wine", "air"] if args.dataset == "both" else [args.dataset]
    # M=11 -> 2^11 = 2048 unique coalitions; budgets must stay well below
    # that (each budget B costs ~B GP refits + B full-EIG rounds, ~1-2 s/round)
    MAX_B = 512
    budgets = [min(b, MAX_B) for b in budgets]
    print(f"budgets: {budgets}  (capped at {MAX_B}; ~1-2 s/round, "
          f"so budget=256 ~= 5-8 min/config)")

    all_rows = []
    failures = []
    for ds in datasets:
        for i in range(args.n):
            for B in budgets:
                # child process per config: a native crash (-11 etc.) or a
                # timeout is contained and reported; the rest still runs.
                cmd = [sys.executable, __file__, "--single", ds, str(B),
                       "--inst", str(i)]
                try:
                    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                       timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    failures.append({"dataset": ds, "instance": i, "budget": B,
                                     "returncode": "TIMEOUT",
                                     "stderr_tail": f"exceeded {args.timeout}s"})
                    print(f"  {ds} inst {i} budget={B}: TIMEOUT "
                          f"(>{args.timeout}s) — try a smaller budget")
                    continue
                f = OUT / f"official_shaplEIG_{ds}_b{B}.csv"
                if r.returncode == 0 and f.exists():
                    row = pd.read_csv(f).iloc[0].to_dict()
                    row["instance"] = i
                    all_rows.append(row)
                    print(f"  {ds} inst {i} budget={B}: "
                          f"rmse={row['rmse_vs_exact']:.5f} "
                          f"unique={row['unique_queries']} "
                          f"({row['elapsed_s']:.0f}s)")
                else:
                    err = (r.stderr or r.stdout or "").strip().splitlines()
                    tail = err[-3:] if err else ["<no output>"]
                    failures.append({"dataset": ds, "instance": i, "budget": B,
                                     "returncode": r.returncode,
                                     "stderr_tail": " | ".join(tail)[:300]})
                    print(f"  {ds} inst {i} budget={B}: FAILED rc={r.returncode} "
                          f"({tail[-1] if tail else ''})")

        if all_rows:
            d = pd.DataFrame([x for x in all_rows if x["dataset"] == ds])
            d.to_csv(OUT / f"official_shaplEIG_{ds}.csv", index=False)
            import shutil
            shutil.copy2(OUT / f"official_shaplEIG_{ds}.csv",
                         MAIN / f"paper_official_shaplEIG_{ds}.csv")

    if failures:
        pd.DataFrame(failures).to_csv(OUT / "official_shaplEIG_failures.csv", index=False)
        import shutil
        shutil.copy2(OUT / "official_shaplEIG_failures.csv",
                     MAIN / "paper_official_shaplEIG_failures.csv")
        print(f"\nWARNING: {len(failures)} config(s) crashed (see "
              f"paper_official_shaplEIG_failures.csv); if rc == -11 (SIGSEGV), "
              f"retry with torch.set_num_threads(1) already active, or pin "
              f"compatible versions: pip install 'torch==2.2.2' "
              f"'gpytorch==1.11' 'linear_operator==0.5.1' (Python <=3.11).")

    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_official_shaplEIG_{{wine,air}}.csv")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
