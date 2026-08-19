#!/usr/bin/env python
"""Official ShaplEIG baseline — pure NumPy/SciPy port (crash-free).

WHY THIS VERSION
----------------
The torch/GPyTorch/BoTorch stack crashed natively on macOS across three
attempts (SIGSEGV rc=-11, then rc=2, then the kernel itself died with
"Python quit unexpectedly").  This is a known macOS incompatibility of
torch+linear_operator (OpenMP / lazy-op paths), NOT a problem with the
ShaplEIG algorithm.  This file therefore ports the OFFICIAL algorithm
(slds-lmu/shapleig, pinned commit d52c09e, MIT) on pure NumPy/SciPy with
the IDENTICAL mathematics:

  * exact GP regression with the Hamming kernel + Gaussian likelihood,
    hyperparameters by marginal-log-likelihood maximisation (L-BFGS-B,
    multiple restarts) — matches the official GPSurrogate
    (HammingKernelConfig + MLMConfig);
  * acquisition = expected information gain about the Shapley property,
    exactly `_compute_eig_function_property_naive_Z`:
        EIG(Z) = log var_yz - log(var_yz - correction),
        correction = diag( (A K_fz)^T (A K_fz A^T)^{-1} (A K_fz) ),
    with K_fz = posterior covariance of f at Z (no observation noise) and
    var_yz = posterior variance + likelihood noise;
  * A = Shapley coefficient matrix from `_get_shapley_weights`
    (w_in[k] = 1/(C(p-1,k-1) p)), i.e. phi = A @ v;
  * exhaustive acquisition over all remaining coalitions; final
    attributions = A @ (GP posterior mean over all 2^M coalitions).

Only the linear-algebra backend differs (NumPy/SciPy instead of
torch/GPyTorch) — a reviewer can diff the math against the pinned source.

ROBUSTNESS
----------
  * no torch/gpytorch/botorch import anywhere → no native crash surface;
  * budgets capped at 512 (each budget B costs ~B GP-refit + full-EIG
    rounds, ~0.5-1 s/round);
  * per-config subprocess isolation + wall-clock timeout: a crash or hang
    in one config is recorded and the rest of the grid still completes.

Usage:
    python scripts/run_official_shaplEIG.py --n 1 --budgets 64,128,256
    python scripts/run_official_shaplEIG.py --single wine 256 --inst 0
Outputs -> results/paper_experiments/shaplEIG_port_{wine,air}.csv
           + main_results/paper_shaplEIG_port_{wine,air}.csv
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

import numpy as np  # noqa: E402
from scipy.linalg import cholesky, cho_solve, solve_triangular  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

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

SHAPLEIG_SOURCE = ("slds-lmu/shapleig@d52c09e (MIT) — pure-NumPy/SciPy port of the "
                   "official algorithm (_compute_eig_function_property_naive_Z + "
                   "_get_shapley_weights); same math, no torch (macOS crash-free)")

MAX_BUDGET = 512


# --------------------------------------------------------------------------- #
# Hamming-kernel exact GP (matches the official HammingKernelConfig surrogate)
# --------------------------------------------------------------------------- #
class HammingGP:
    """Exact GP with k(x,x') = outscale * exp(- d_H(x,x') / lengthscale)."""

    def __init__(self, tx: np.ndarray, ty: np.ndarray):
        self.tx = np.asarray(tx, dtype=float)
        self.ty = np.asarray(ty, dtype=float)
        self.logp = np.array([np.log(1.5), np.log(1.0), np.log(0.05)])  # l, out, noise
        self._L = None
        self._alpha = None

    def _cov(self, X, Y, outscale, lengthscale):
        d = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=-1)
        return outscale * np.exp(-d / lengthscale)

    def _neg_mll(self, logp):
        l, o, s = np.exp(logp)
        n = self.tx.shape[0]
        K = self._cov(self.tx, self.tx, o, l) + (s + 1e-8) * np.eye(n)
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = cho_solve((L, True), self.ty)
        mll = -0.5 * float(self.ty @ alpha) - float(np.log(np.diag(L)).sum()) \
            - 0.5 * n * np.log(2.0 * np.pi)
        return -mll

    def fit(self, restarts: int = 3, maxiter: int = 120) -> None:
        best = None
        rng = np.random.RandomState(0)
        inits = [self.logp.copy()]
        for _ in range(restarts - 1):
            inits.append(self.logp + rng.uniform(-1.0, 1.0, size=3))
        for p0 in inits:
            res = minimize(self._neg_mll, p0, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self.logp = best.x
        l, o, s = np.exp(self.logp)
        n = self.tx.shape[0]
        K = self._cov(self.tx, self.tx, o, l) + (s + 1e-8) * np.eye(n)
        self._L = cholesky(K, lower=True)
        self._alpha = cho_solve((self._L, True), self.ty)

    def posterior(self, Z: np.ndarray, observation_noise: bool = False):
        """Posterior mean + full covariance at Z (numpy port of forward_lazy_covar)."""
        l, o, s = np.exp(self.logp)
        Kxx = self._cov(self.tx, self.tx, o, l) + (s + 1e-8) * np.eye(self.tx.shape[0])
        Kzz = self._cov(Z, Z, o, l)
        Kzx = self._cov(Z, self.tx, o, l)
        # W = Kxx^{-1} Kxz  via cholesky
        W = cho_solve((self._L, True), Kzx.T)          # (n_train, n_z)
        K_fz = Kzz - Kzx @ W                            # posterior cov of f at Z
        mean = Kzx @ self._alpha
        if observation_noise:
            K_fz = K_fz + (s + 1e-8) * np.eye(Z.shape[0])
        return mean, K_fz


def _shapley_A(p: int, Z: np.ndarray) -> np.ndarray:
    """Official Shapley coefficient matrix: phi = A @ v(Z)."""
    n = Z.shape[0]
    A = np.zeros((p, n))
    for i in range(p):
        for r in range(n):
            S = Z[r]
            k = int(S.sum())
            if S[i]:
                continue
            w = 1.0 / (math.comb(p - 1, k) * p)
            Sp = S.copy(); Sp[i] = 1.0
            idx_in = int(np.where((Z == Sp).all(axis=1))[0][0])
            A[i, idx_in] += w
            A[i, r] -= w
    return A


def _eig_naive_Z(model: HammingGP, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Port of _compute_eig_function_property_naive_Z (official), in numpy."""
    _, K_fz = model.posterior(Z, observation_noise=False)
    _, K_yz = model.posterior(Z, observation_noise=True)
    covar_yz_diag = np.diag(K_yz)
    transformed = K_fz @ A.T                      # (n_Z, p)
    quad = A @ transformed                        # (p, p)
    sol = np.linalg.solve(quad + 1e-10 * np.eye(quad.shape[0]), transformed.T)  # (p, n_Z)
    correction = np.sum(transformed.T * sol, axis=0)  # (n_Z,)
    eig = np.log(covar_yz_diag) - np.log(np.maximum(covar_yz_diag - correction, 1e-12))
    return eig


def run_official_shaplEIG(game_fn, M, budget, seed=1301):
    """Run the official ShaplEIG loop; return (phi, unique_queries, wall_s)."""
    rng = np.random.RandomState(seed)
    Z = np.array([list(map(int, format(i, f"0{M}b"))) for i in range(2 ** M)],
                 dtype=float)
    A = _shapley_A(M, Z)

    archive_x = []
    archive_y = []
    queried = set()
    t0 = time.time()

    for init_idx in (0, 2 ** M - 1):
        S = Z[init_idx].astype(bool)
        archive_x.append(init_idx)
        archive_y.append(game_fn(S))
        queried.add(init_idx)

    rounds = 0
    while len(queried) < budget:
        tx = Z[archive_x]
        ty = np.array(archive_y, dtype=float)
        if len(tx) < 2:
            break
        model = HammingGP(tx, ty)
        model.fit(restarts=2, maxiter=80)

        scores = _eig_naive_Z(model, Z, A)
        cand = np.argsort(-scores)
        chosen = None
        for c in cand.tolist():
            if c not in queried:
                chosen = c
                break
        if chosen is None:
            break
        S = Z[chosen].astype(bool)
        archive_x.append(chosen)
        archive_y.append(game_fn(S))
        queried.add(chosen)
        rounds += 1
        if rounds > budget:
            break

    tx = Z[archive_x]
    ty = np.array(archive_y, dtype=float)
    model = HammingGP(tx, ty)
    model.fit(restarts=3, maxiter=150)
    mu, _ = model.posterior(Z, observation_noise=False)
    phi = A @ mu
    wall = time.time() - t0
    return phi, len(queried), wall


def run_instance(name, X, feat_names, n_clusters, i, budget, seed=1301):
    m, X_te, _, _ = build_surrogate(X, n_clusters, feat_names)
    cid = i % n_clusters
    fn = make_proba_fn(m, feat_names, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=seed + i).values

    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                             model_tag=f"shaplEIG-{name}-{i}-exact")
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    def game_fn(S_mask):
        o = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                            model_tag=f"shaplEIG-{name}-{i}")
        return o.evaluate(x0, S_mask)

    phi, unique, wall = run_official_shaplEIG(game_fn, X.shape[1], budget,
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

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.single:
        ds, B = args.single[0], int(args.single[1])
        if ds == "wine":
            X, _ = load_wine(); feats, nc = WINE_FEATURES, 2
        else:
            X, _ = load_air_station(n_clusters=4); feats, nc = AIR_FEATURES, 4
        row = run_single(ds, X, feats, nc, args.inst, B)
        print(f"SINGLE {ds} inst={args.inst} budget={B}: rmse={row['rmse_vs_exact']:.5f} "
              f"unique={row['unique_queries']} ({row['elapsed_s']}s)")
        pd.DataFrame([row]).to_csv(OUT / f"shaplEIG_port_{ds}_b{B}.csv", index=False)
        return 0 if row["unique_queries"] > 0 else 1

    budgets = [min(int(b), MAX_BUDGET) for b in args.budgets.split(",")]
    datasets = ["wine", "air"] if args.dataset == "both" else [args.dataset]
    print(f"budgets: {budgets}  (capped at {MAX_BUDGET}; ~0.5-1 s/round)")

    all_rows = []
    failures = []
    for ds in datasets:
        for i in range(args.n):
            for B in budgets:
                cmd = [sys.executable, __file__, "--single", ds, str(B),
                       "--inst", str(i)]
                try:
                    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                       timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    failures.append({"dataset": ds, "instance": i, "budget": B,
                                     "returncode": "TIMEOUT",
                                     "stderr_tail": f"exceeded {args.timeout}s"})
                    print(f"  {ds} inst {i} budget={B}: TIMEOUT (>{args.timeout}s)")
                    continue
                f = OUT / f"shaplEIG_port_{ds}_b{B}.csv"
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
                    print(f"  {ds} inst {i} budget={B}: FAILED rc={r.returncode}")

        if all_rows:
            d = pd.DataFrame([x for x in all_rows if x["dataset"] == ds])
            d.to_csv(OUT / f"shaplEIG_port_{ds}.csv", index=False)
            import shutil
            shutil.copy2(OUT / f"shaplEIG_port_{ds}.csv",
                         MAIN / f"paper_shaplEIG_port_{ds}.csv")

    if failures:
        pd.DataFrame(failures).to_csv(OUT / "shaplEIG_port_failures.csv", index=False)
        import shutil
        shutil.copy2(OUT / "shaplEIG_port_failures.csv",
                     MAIN / "paper_shaplEIG_port_failures.csv")
        print(f"\nWARNING: {len(failures)} config(s) failed (see "
              f"paper_shaplEIG_port_failures.csv)")

    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_shaplEIG_port_{{wine,air}}.csv")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
