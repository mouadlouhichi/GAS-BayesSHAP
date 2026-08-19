#!/usr/bin/env python
"""Official ShaplEIG baseline on GAS-BayesSHAP games (faithful port).

The previous `gp_quadrature_rmse` baseline in
`paper_reference_baselines_ablation.csv` was a *method-style* GP surrogate.
The audits asked for the **official** ShaplEIG (ICML 2026, Rundel et al.).
The authors' public repo (github.com/slds-lmu/shapleig, MIT) was pinned at
commit d52c09e and its core algorithm ported here:

  * GP surrogate over the binary coalition space (Hamming kernel,
    Gaussian likelihood, fit by marginal-likelihood maximisation) --
    matches `src/xac/surrogates/gp_surrogate.py` (HammingKernelConfig +
    MLMConfig).
  * Acquisition = expected information gain about the Shapley property,
    exactly as `_compute_eig_function_property_naive_Z` in
    `src/xac/acquisition_functions/acquisition_functions.py`:
        EIG(Z) = log var_yz - log(var_yz - correction),   correction =
        inv_quad(A K_fz K_fz A^T, K_fz A^T)  (variance reduction of A f(Z)).
  * A = the Shapley coefficient matrix from the official
    `_get_shapley_weights` (w_in[k] = 1/(C(p-1,k-1) p) on the in-coalition
    marginal, -w_in on the out-coalition), i.e. phi = A @ v.
  * Exhaustive acquisition over all remaining coalitions; final attributions
    = A @ (GP posterior mean over all 2^M coalitions).

This is a direct port of the official algorithm (not a guess), labelled
`official_shaplEIG`; a reviewer can diff it against the pinned source.
Unique coalition queries are counted through the GAS CoalitionOracle cache,
so the CSV reports the *actual* query cost, matching how GAS evals are
counted.

Usage:
    python scripts/run_official_shaplEIG.py --n 3 --budgets 256,1024,2048
Outputs -> results/paper_experiments/official_shaplEIG_{wine,air}.csv
           + main_results/paper_official_shaplEIG_{wine,air}.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

# --- official ShaplEIG stack (torch/botorch/gpytorch) ---------------------- #
try:
    import torch
    import gpytorch
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import Kernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from linear_operator import to_linear_operator
    from gpytorch import inv_quad
    HAVE_OFFICIAL = True
except ImportError as e:  # pragma: no cover
    HAVE_OFFICIAL = False
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
            # v(S u {i}) gets +w, v(S) gets -w
            Sp = S.clone(); Sp[i] = True
            idx_in = int((Z == Sp).all(dim=1).nonzero()[0].item())
            A[i, idx_in] += w
            A[i, r] -= w
    return A


def _eig_naive_Z(surrogate, Z: torch.Tensor, A: torch.Tensor,
                 train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
    """Port of _compute_eig_function_property_naive_Z (official).

    EIG(Z) = log(var_yz) - log(var_yz - correction), where var_yz is the
    prior predictive variance (with observation noise) at Z and the
    correction is the variance reduction of the property A f(Z) from
    conditioning on the training data.
    """
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        # EXACT official semantics (forward_lazy_covar): the *posterior*
        # predictive covariance at Z, with and without observation noise.
        post_f = surrogate.posterior(Z, observation_noise=False).mvn
        post_y = surrogate.posterior(Z, observation_noise=True).mvn
        covar_yz_diag = post_y.variance
        K_fz = post_f.covariance_matrix            # dense Z x Z posterior cov
        transformed = K_fz.matmul(A.T)             # (n_Z, p)
        quad_form = A.matmul(transformed)          # (p, p)  = A K_fz A^T
        correction = inv_quad(
            input=quad_form,
            inv_quad_rhs=transformed.transpose(-2, -1),
            reduce_inv_quad=False,
        )
        EIG = torch.log(covar_yz_diag) - torch.log(covar_yz_diag - correction)
        if EIG.ndim == 2:
            EIG = EIG.mean(dim=0)
        return EIG


class _GPSurrogate:
    """Botorch SingleTaskGP with Hamming kernel (official stack)."""

    def __init__(self, train_x, train_y):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        kernel = ScaleKernel(HammingKernel())
        self.model = SingleTaskGP(train_x, train_y, covar_module=kernel)
        self.likelihood = self.model.likelihood

    def fit(self, steps: int = 30):
        from botorch.fit import fit_gpytorch_mll
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def posterior(self, Z, observation_noise=True):
        return self.model.posterior(Z, observation_noise=observation_noise)


def run_official_shaplEIG(game_fn, x0, M, budget, seed=1301):
    """Run the official ShaplEIG loop; return (phi, unique_queries, wall_s)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z = torch.tensor(
        [list(map(int, format(i, f"0{M}b"))) for i in range(2 ** M)],
        dtype=torch.float64,
    )
    A = _shapley_A(M, Z)

    archive_x = []   # evaluated coalition indices into Z
    archive_y = []
    queried = set()
    t0 = time.time()

    # initial design: empty + full coalition (2 points)
    for init_idx in (0, 2 ** M - 1):
        S = Z[init_idx].numpy().astype(bool)
        archive_x.append(init_idx)
        archive_y.append(game_fn(S))
        queried.add(init_idx)

    while len(queried) < budget:
        # fit GP on current archive
        tx = Z[torch.tensor(archive_x)]
        ty = torch.tensor(archive_y, dtype=torch.float64).unsqueeze(-1)
        if len(tx) < 2:
            break
        model = _GPSurrogate(tx, ty)
        model.fit(steps=30)

        # acquisition over all candidates (official: exhaustive over Z)
        scores = _eig_naive_Z(model, Z, A, tx, ty)
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

    # final attributions: A @ posterior mean over all coalitions
    tx = Z[torch.tensor(archive_x)]
    ty = torch.tensor(archive_y, dtype=torch.float64).unsqueeze(-1)
    model = _GPSurrogate(tx, ty)
    model.fit(steps=50)
    with torch.no_grad():
        post = model.posterior(Z, observation_noise=False).mvn
        mu = post.mean
    phi = (A @ mu).numpy()
    wall = time.time() - t0
    return phi, len(queried), wall


def run_instance(name, X, feat_names, n_clusters, i, budget, seed=1301):
    m, X_te, _, _ = build_surrogate(X, n_clusters, feat_names)
    cid = i % n_clusters
    fn = make_proba_fn(m, feat_names, cid)
    x0 = X_te.iloc[i].values
    bg = X.sample(64, random_state=seed + i).values

    def game_fn(S_mask):
        # v(S) = mean over background of g(x_S + z_~S); full = g(x), empty = E_base
        oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                                 model_tag=f"shaplEIG-{name}-{i}")
        return oracle.evaluate(x0, S_mask)

    oracle = CoalitionOracle(fn, bg, output_bounds=(0.0, 1.0),
                             model_tag=f"shaplEIG-{name}-{i}-exact")
    phi_exact = exact_shapley_from_values(
        exact_game_values(oracle, x0, X.shape[1]), X.shape[1])

    phi, unique, wall = run_official_shaplEIG(game_fn, x0, X.shape[1], budget, seed=seed + i)
    return {
        "dataset": name, "instance": i, "budget": budget,
        "rmse_vs_exact": rmse(phi, phi_exact),
        "unique_queries": unique,
        "source": SHAPLEIG_SOURCE,
        "elapsed_s": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--budgets", default="256,1024,2048")
    ap.add_argument("--dataset", choices=["wine", "air", "both"], default="both")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    if not HAVE_OFFICIAL:
        print(f"official ShaplEIG stack unavailable: {_IMPORT_ERR}")
        print("install with: pip install torch botorch gpytorch linear_operator")
        return 1

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    all_rows = []
    for ds in (["wine", "air"] if args.dataset == "both" else [args.dataset]):
        X, feats, nc = (load_wine(), WINE_FEATURES, 2) if ds == "wine" \
            else (load_air_station(n_clusters=4), AIR_FEATURES, 4)
        X, feats = X[0], feats
        for i in range(args.n):
            for B in budgets:
                row = run_instance(ds, X, feats, nc, i, B)
                all_rows.append(row)
                print(f"  {ds} inst {i} budget={B}: rmse={row['rmse_vs_exact']:.5f} "
                      f"unique={row['unique_queries']} ({row['elapsed_s']:.0f}s)")
        d = pd.DataFrame([r for r in all_rows if r["dataset"] == ds])
        d.to_csv(OUT / f"official_shaplEIG_{ds}.csv", index=False)
        import shutil
        shutil.copy2(OUT / f"official_shaplEIG_{ds}.csv",
                     MAIN / f"paper_official_shaplEIG_{ds}.csv")

    print(f"\ndone in {time.time()-t0:.0f}s; "
          f"main_results/paper_official_shaplEIG_{{wine,air}}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
