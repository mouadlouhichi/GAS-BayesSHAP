"""SOTA baseline estimators (review task #6).

Official OddSHAP (ICML 2026) and ShaplEIG (ICML 2026) code is not public in
this environment, so we implement faithful **method-style** baselines from
their published descriptions and flag them as such in the results:

OddSHAP-style
    Shapley values of the **log-odds (odd-ratio) game** — the standard
    odd-ratio transform used by the odd-SHAP line of work for probability
    outputs: v_odd(S) = logit( E[g_c] ) with the caveat that we compute it
    on the coalition mean then take logit, i.e. the "raw-odd" convention.
    For M=11 we compute it EXACTLY (2^M enumeration) so it is an exact
    reference for that transform.

ShaplEIG-style
    GP-based Shapley estimation: fit a GP (exponential Hamming kernel) to a
    subset of coalition values, predict the posterior mean over all 2^M
    coalitions, and return the Shapley values of the posterior mean.  This
    matches the "active GP for Shapley" idea; posterior std is reported as
    its (Bayesian, non-frequentist) uncertainty.

Both are clearly **non-certified** baselines in the comparison tables.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from ..game.brute_force import exact_shapley_from_values
from ..kernels.covariance import lemma_D_cross_cov, lemma_E_prior_cov
from ..kernels.hamming import ExponentialHammingKernel
from ..gp.control_variate import fit_bounded_surrogate
from ..gp.posterior import gp_predict


# --------------------------------------------------------------------------- #
# OddSHAP-style: log-odds game (exact for M <= ~14)
# --------------------------------------------------------------------------- #
def odd_shapley_exact(
    oracle,
    x: np.ndarray,
    M: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Exact Shapley of the log-odds game v_odd(S) = logit(v(S)).

    ``v(S) = E[g_c]`` from the oracle; logit is taken with clipping to
    ``(eps, 1-eps)``.  Returns the exact Shapley values of the odd-ratio
    game (the OddSHAP-style transform).
    """
    from ..game.brute_force import exact_game_values
    values = exact_game_values(oracle, x, M)
    odd = {}
    for mask, v in values.items():
        vc = min(max(v, eps), 1.0 - eps)
        odd[mask] = float(logit(vc))
    return exact_shapley_from_values(odd, M)


# --------------------------------------------------------------------------- #
# ShaplEIG-style: GP posterior-mean Shapley (Bayesian, non-certified)
# --------------------------------------------------------------------------- #
def gp_quadrature_shapley(
    oracle,
    x: np.ndarray,
    M: int,
    design,
    y_design: np.ndarray,
    sigma0: float = 1.0,
    lengthscale: float = 1.5,
    eta: float = 1e-4,
    compute_std: bool = True,
):
    """ShaplEIG-style GP Shapley estimator.

    Fits a GP on ``design`` coalition values, predicts the posterior mean on
    all 2^M coalitions, and returns the Shapley values of the posterior mean
    (computed analytically via Lemma D: phi(m) = K_phi,D alpha).

    Returns
    -------
    phi_mean : (M,) Shapley of the GP posterior mean
    phi_std  : (M,) posterior std of each attribution (Bayesian, non-rigorous)
    """
    kernel = ExponentialHammingKernel(sigma0=sigma0, lengthscale=lengthscale)
    sur = fit_bounded_surrogate(list(design), list(y_design), kernel, eta=eta)
    phi_mean = sur.surrogate_shapley(kernel)
    if not compute_std:
        return phi_mean, None
    # posterior covariance of the attribution vector (lambda=1: no shrinkage)
    cov = sur.posterior_covariance()
    # note: sur.scale is 1.0 for a raw fit (no output bounds applied here)
    phi_std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return phi_mean, phi_std


def gp_quadrature_shapley_from_values(
    values_all: dict,
    design_masks: list,
    sigma0: float = 1.0,
    lengthscale: float = 1.5,
    eta: float = 1e-4,
    M: int = 11,
):
    """Variant that takes precomputed coalition values (design subset)."""
    design = [np.array([(mask >> b) & 1 for b in range(M)], dtype=bool) for mask in design_masks]
    y = np.array([values_all[m] for m in design_masks], dtype=np.float64)
    return gp_quadrature_shapley(None, None, M, design, y, sigma0, lengthscale, eta)


# --------------------------------------------------------------------------- #
# Comparison helper: run all baselines + GAS on one instance
# --------------------------------------------------------------------------- #
def baseline_table_entry(name, phi, phi_exact, evals, model_evals):
    from .metrics import mae, max_abs_error, rmse
    return {
        "method": name,
        "rmse": rmse(phi, phi_exact),
        "mae": mae(phi, phi_exact),
        "max_err": max_abs_error(phi, phi_exact),
        "coalition_evals": evals,
        "model_evals": model_evals,
        "certified": False,
    }
