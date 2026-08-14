"""
GAS-BayesSHAP: Complete Inline Reference Implementation (v11.0 Gold-Standard)

This module is a **verbatim copy of Section 4 of the authoritative spec**
``specs/GAS_BayesSHAP_Implementation_Spec (4).md`` (with the single driver
typo ``run_all_tests`` -> ``test_all`` fixed so the embedded 10-tier suite
actually runs).  It is kept strictly separate from the production package and
used only as a **parity oracle** in tests: identical seeds must produce
identical results and identical query accounting.
"""

import numpy as np
from scipy.special import comb
from scipy.optimize import minimize
from math import factorial
from typing import Callable, Dict, Any, List, Optional, Tuple


def safe_std(arr: List[float], default: float = 0.5) -> float:
    """Computes sample standard deviation while preserving true zero variance."""
    if len(arr) <= 1:
        return default
    val = float(np.std(arr, ddof=1))
    if not np.isfinite(val):
        return default
    return max(val, 0.0)


class GASBayesSHAP:
    """
    Gaussian-Adaptive Stratified Bayesian Shapley Estimator (GAS-BayesSHAP v11.0)
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], float],
        background: np.ndarray,
        output_bounds: Optional[Tuple[float, float]] = None,
        sigma0: float = 1.0,
        lengthscale: float = 1.5,
        eta: float = 1e-4
    ):
        self.model_fn = model_fn
        self.background = np.atleast_2d(background)
        self.B, self.M = self.background.shape
        self.output_bounds = output_bounds
        self.sigma0 = float(sigma0)
        self.rho = float(np.exp(-1.0 / lengthscale))
        self.eta = float(eta)

        # Cumulative query meters across all calls
        self.total_coalition_evals = 0
        self.total_model_evals = 0

        # Precompute baseline expectation E[f(X)]
        bg_preds = [self.model_fn(self.background[b]) for b in range(self.B)]
        self.total_model_evals += self.B
        self.E_base = float(np.mean(bg_preds))

        # Precompute exact prior Shapley covariance matrix K_phi_phi (Lemma E)
        self.K_phi_phi = self._compute_exact_K_phi_phi()
        self._alpha: Optional[np.ndarray] = None
        self._D_matrix: Optional[np.ndarray] = None
        self._surrogate_scale: float = 1.0
        self._surrogate_shift: float = 0.0

    def _eval_coalition(self, x: np.ndarray, S_mask: np.ndarray) -> float:
        """Evaluates interventional expectation v(S) with explicit query accounting."""
        self.total_coalition_evals += 1
        if np.all(S_mask):
            self.total_model_evals += 1
            return float(self.model_fn(x))
        if not np.any(S_mask):
            return float(self.E_base)

        X_hybrid = np.tile(x, (self.B, 1))
        X_hybrid[:, ~S_mask] = self.background[:, ~S_mask]

        preds = [self.model_fn(X_hybrid[b]) for b in range(self.B)]
        self.total_model_evals += self.B
        return float(np.mean(preds))

    def _kernel_val(self, S1: np.ndarray, S2: np.ndarray) -> float:
        """Exponential Hamming Kernel k(S1, S2)."""
        d_H = int(np.sum(S1 != S2))
        return (self.sigma0 ** 2) * (self.rho ** d_H)

    def _closed_form_cross_cov(self, S_j: np.ndarray) -> np.ndarray:
        """
        Lemma D: Exact Closed-Form Cross-Covariance Vector A_i[k(·, S_j)] in O(M^2) time.
        Exploits symmetry: all i in S_j share V_in, and all i not in S_j share V_out.
        """
        M = self.M
        r = int(np.sum(S_j))
        K_phi_j = np.zeros(M, dtype=np.float64)

        def eval_cross_scalar(r_no_i: int, sign: float) -> float:
            total_sum = 0.0
            for s in range(M):
                denom = comb(M - 1, s)
                if denom == 0:
                    continue
                l_min = max(0, s - (M - 1 - r_no_i))
                l_max = min(s, r_no_i)
                overlap_sum = 0.0
                for l in range(l_min, l_max + 1):
                    num = comb(r_no_i, l) * comb(M - 1 - r_no_i, s - l)
                    power = r_no_i + s - 2 * l
                    overlap_sum += (num / denom) * (self.rho ** power)
                total_sum += overlap_sum
            return ((self.sigma0 ** 2) * (1.0 - self.rho) / M) * sign * total_sum

        V_in = eval_cross_scalar(r - 1, +1.0) if r > 0 else 0.0
        V_out = eval_cross_scalar(r, -1.0) if r < M else 0.0

        K_phi_j[S_j] = V_in
        K_phi_j[~S_j] = V_out
        return K_phi_j

    def _compute_exact_K_phi_phi(self) -> np.ndarray:
        """Lemma E: Evaluates exact prior Shapley covariance K_phi_phi using raw pair counts."""
        M = self.M

        def w(s):
            return factorial(s) * factorial(M - 1 - s) / factorial(M)

        # 1. Diagonal variance V_diag
        v_diag_sum = 0.0
        for s in range(M):
            for t in range(M):
                denom_t = comb(M - 1, t)
                if denom_t == 0:
                    continue
                l_min = max(0, s + t - (M - 1))
                l_max = min(s, t)
                for l in range(l_min, l_max + 1):
                    num = comb(s, l) * comb(M - 1 - s, t - l)
                    v_diag_sum += (num / denom_t) * (self.rho ** (s + t - 2 * l))
        v_diag = (2.0 * (self.sigma0 ** 2) * (1.0 - self.rho) / (M ** 2)) * v_diag_sum

        # 2. Off-diagonal covariance V_off (RAW pair counts with Δw factors)
        v_off = 0.0
        if M > 1:
            dw = [w(s) - w(s + 1) for s in range(M - 1)]
            for s in range(M - 1):
                for t in range(M - 1):
                    l_min = max(0, s + t - (M - 2))
                    l_max = min(s, t)
                    for l in range(l_min, l_max + 1):
                        count = comb(M - 2, s) * comb(s, l) * comb(M - 2 - s, t - l)
                        v_off += dw[s] * dw[t] * count * (self.rho ** (s + t - 2 * l))
            v_off *= (self.sigma0 ** 2) * ((1.0 - self.rho) ** 2)

        mat = (v_diag - v_off) * np.eye(M) + v_off * np.ones((M, M))
        return 0.5 * (mat + mat.T)

    def _rank1_inverse_update(self, inv_K: np.ndarray, k_vec: np.ndarray, k_self: float) -> Tuple[np.ndarray, bool]:
        """Sherman-Morrison block-inversion update in O(D^2) time with success flag."""
        D = inv_K.shape[0]
        if D == 0:
            return np.array([[1.0 / (k_self + self.eta ** 2)]], dtype=np.float64), True

        v = inv_K @ k_vec
        schur = (k_self + self.eta ** 2) - float(k_vec.T @ v)
        if schur < (self.eta ** 2):
            return inv_K, False  # Near duplicate, skip update to prevent numerical blowup

        schur_inv = 1.0 / max(schur, 1e-10)
        top_left = inv_K + (schur_inv * np.outer(v, v))
        top_right = (-schur_inv * v).reshape(-1, 1)
        bottom_left = (-schur_inv * v).reshape(1, -1)
        bottom_right = np.array([[schur_inv]], dtype=np.float64)

        return np.block([[top_left, top_right], [bottom_left, bottom_right]]), True

    def _predict_gp_fast(self, S: np.ndarray) -> float:
        """
        Fast Vectorized O(DM) prediction of Bounded Linear Shrinkage Surrogate m_b(S).
        Preserves m_b(S) in [L, U] globally without nonlinear clipping!
        """
        if self._D_matrix is None or self._alpha is None or len(self._D_matrix) == 0:
            return 0.0
        d_H = np.sum(self._D_matrix != S[None, :], axis=1)
        k_vec = (self.sigma0 ** 2) * (self.rho ** d_H)
        h_val = float(k_vec @ self._alpha)
        return self._surrogate_shift + self._surrogate_scale * h_val

    def _solve_coupled_neyman_allocation(self, sigma_res: np.ndarray) -> np.ndarray:
        """
        Theorem 1: Solves the coupled adjacent-stratum allocation convex program:
        min sum_{s=1}^{M-2} A_s / [(M-s) K_s + (s+1) K_{s+1}]  s.t. sum_{q=1}^{M-1} K_q = 1, K_q >= 0.
        """
        M = self.M
        if M <= 2:
            return np.zeros(M)

        A = np.zeros(M)
        for s in range(1, M - 1):  # interior strata s = 1, ..., M-2
            A[s] = np.sum(sigma_res[s, :] ** 2) + 1e-8

        def objective(K_dec):
            # K_dec corresponds to draw sizes q = 1, ..., M-1
            K = np.zeros(M)
            K[1:M] = K_dec
            val = 0.0
            for s in range(1, M - 1):
                d_s = (M - s) * K[s] + (s + 1) * K[s + 1]
                val += A[s] / max(d_s, 1e-12)
            return val / M

        # Decision vector of length M-1: K_1, ..., K_{M-1}
        K0 = np.full(M - 1, 1.0 / (M - 1))
        bnds = [(0.0, 1.0) for _ in range(M - 1)]
        cons = ({'type': 'eq', 'fun': lambda k: np.sum(k) - 1.0})

        res = minimize(objective, K0, bounds=bnds, constraints=cons, method='SLSQP')
        probs = np.zeros(M)
        if res.success and np.sum(res.x) > 0:
            probs[1:M] = np.maximum(res.x, 0.0)
            probs[1:M] /= np.sum(probs[1:M])
        else:
            probs[1:M] = 1.0 / (M - 1)

        return probs

    def explain(
        self,
        x: np.ndarray,
        epsilon: float = 0.02,
        delta: float = 0.05,
        max_budget: int = 1500,
        n_pilot: int = 3,
        n_active_steps: int = 25
    ) -> Dict[str, Any]:
        """
        Executes GAS-BayesSHAP dual estimation:
        Active GP Control Variate + Stratified Anytime Supermartingale Certification.
        `max_budget` sets the maximum query budget for Stage 2 adaptive sampling.
        """
        evals_start_coal = self.total_coalition_evals
        evals_start_model = self.total_model_evals

        x = np.asarray(x, dtype=np.float64)
        v_N = self._eval_coalition(x, np.ones(self.M, dtype=bool))
        delta_total = v_N - self.E_base

        # Bounded range constant
        if self.output_bounds is not None:
            L, U = self.output_bounds
            R_Delta_res = 4.0 * (U - L)
        else:
            L = min(self.E_base, v_N) - abs(delta_total)
            U = max(self.E_base, v_N) + abs(delta_total)
            R_Delta_res = 4.0 * (U - L)

        # =====================================================================
        # Stage 1: Active GP Learning (Module A: Dataset D_gp)
        # =====================================================================
        D_gp_coalitions: List[np.ndarray] = []
        D_gp_y: List[float] = []
        inv_K_DD = np.empty((0, 0), dtype=np.float64)
        K_phi_D = np.empty((self.M, 0), dtype=np.float64)

        seed_subsets = [np.zeros(self.M, dtype=bool), np.ones(self.M, dtype=bool)]
        for s in range(1, self.M):
            p = np.zeros(self.M, dtype=bool)
            p[np.random.permutation(self.M)[:s]] = True
            seed_subsets.append(p)

        for S_seed in seed_subsets:
            v_val = self._eval_coalition(x, S_seed)
            k_vec = np.array([self._kernel_val(S_seed, S_obs) for S_obs in D_gp_coalitions], dtype=np.float64)
            k_self = self._kernel_val(S_seed, S_seed)
            new_inv, ok = self._rank1_inverse_update(inv_K_DD, k_vec, k_self)
            if ok:
                inv_K_DD = new_inv
                K_phi_D = np.hstack([K_phi_D, self._closed_form_cross_cov(S_seed).reshape(-1, 1)])
                D_gp_coalitions.append(S_seed)
                D_gp_y.append(v_val - self.E_base)

        pool_size = max(32, 2 * self.M)
        for _ in range(n_active_steps):
            best_score = -1.0
            best_S = None
            for _ in range(pool_size):
                s_sz = np.random.randint(0, self.M + 1)
                p = np.zeros(self.M, dtype=bool)
                p[np.random.permutation(self.M)[:s_sz]] = True

                k_cand = np.array([self._kernel_val(p, S_obs) for S_obs in D_gp_coalitions], dtype=np.float64)
                k_self = self._kernel_val(p, p)
                v_post_var = k_self - float(k_cand.T @ inv_K_DD @ k_cand)
                cov_phi = self._closed_form_cross_cov(p) - (K_phi_D @ inv_K_DD @ k_cand)
                score = float(np.sum(cov_phi ** 2)) / (max(v_post_var, 1e-8) + self.eta ** 2)

                if score > best_score:
                    best_score = score
                    best_S = p

            if best_S is not None:
                v_val = self._eval_coalition(x, best_S)
                k_vec = np.array([self._kernel_val(best_S, S_obs) for S_obs in D_gp_coalitions], dtype=np.float64)
                k_self = self._kernel_val(best_S, best_S)
                new_inv, ok = self._rank1_inverse_update(inv_K_DD, k_vec, k_self)
                if ok:
                    inv_K_DD = new_inv
                    K_phi_D = np.hstack([K_phi_D, self._closed_form_cross_cov(best_S).reshape(-1, 1)])
                    D_gp_coalitions.append(best_S)
                    D_gp_y.append(v_val - self.E_base)

        # Compute Conservative Global Bounds and Bounded Linear Surrogate Parameters
        y_gp_vec = np.array(D_gp_y, dtype=np.float64)
        self._alpha = inv_K_DD @ y_gp_vec
        self._D_matrix = np.array(D_gp_coalitions, dtype=bool)

        pos_alpha_sum = float(np.sum(self._alpha[self._alpha > 0]))
        neg_alpha_sum = float(np.sum(self._alpha[self._alpha < 0]))
        h_lb = (self.sigma0 ** 2) * ((self.rho ** self.M) * pos_alpha_sum + neg_alpha_sum)
        h_ub = (self.sigma0 ** 2) * (pos_alpha_sum + (self.rho ** self.M) * neg_alpha_sum)

        if h_ub > h_lb:
            self._surrogate_scale = min(1.0, (U - L) / (h_ub - h_lb))
        else:
            self._surrogate_scale = 1.0
        self._surrogate_shift = L - self._surrogate_scale * h_lb

        # Exact Analytical Surrogate Shapley Attribution & Lambda^2 Scaled Posterior Covariance
        phi_m_D = self._surrogate_scale * (K_phi_D @ self._alpha)
        phi_cov_h = self.K_phi_phi - (K_phi_D @ inv_K_DD @ K_phi_D.T)
        phi_cov_h = 0.5 * (phi_cov_h + phi_cov_h.T)

        phi_cov_mb = (self._surrogate_scale ** 2) * phi_cov_h
        posterior_variances = np.maximum(np.diag(phi_cov_mb), 1e-10)

        # =====================================================================
        # Stage 2: Neyman Stratified Residual Certification (Module B: D_cert)
        # =====================================================================
        strata_residuals = {s: {i: [] for i in range(self.M)} for s in range(self.M)}
        sigma_res = np.zeros((self.M, self.M), dtype=np.float64)

        # 1. Deterministic Extreme-Stratum Identification (Lemma G)
        v_empty = self._eval_coalition(x, np.zeros(self.M, dtype=bool))
        m_empty = self._predict_gp_fast(np.zeros(self.M, dtype=bool))
        for i in range(self.M):
            S_i = np.zeros(self.M, dtype=bool); S_i[i] = True
            v_Si = self._eval_coalition(x, S_i)
            m_Si = self._predict_gp_fast(S_i)
            strata_residuals[0][i].append((v_Si - v_empty) - (m_Si - m_empty))
            sigma_res[0, i] = 0.0

        v_full = self._eval_coalition(x, np.ones(self.M, dtype=bool))
        m_full = self._predict_gp_fast(np.ones(self.M, dtype=bool))
        for missing_i in range(self.M):
            S_no_i = np.ones(self.M, dtype=bool); S_no_i[missing_i] = False
            v_Sno_i = self._eval_coalition(x, S_no_i)
            m_Sno_i = self._predict_gp_fast(S_no_i)
            strata_residuals[self.M - 1][missing_i].append((v_full - v_Sno_i) - (m_full - m_Sno_i))
            sigma_res[self.M - 1, missing_i] = 0.0

        # 2. Pilot for interior strata s in [1, M-2]
        for s in range(1, max(1, self.M - 1)):
            for _ in range(n_pilot):
                perm = np.random.permutation(self.M)
                S = np.zeros(self.M, dtype=bool)
                S[perm[:s]] = True

                v_S = self._eval_coalition(x, S)
                m_S = self._predict_gp_fast(S)

                for i in range(self.M):
                    if not S[i]:
                        S_u = S.copy(); S_u[i] = True
                        v_Su = self._eval_coalition(x, S_u)
                        m_Su = self._predict_gp_fast(S_u)
                        strata_residuals[s][i].append((v_Su - v_S) - (m_Su - m_S))
                    else:
                        if s > 0:
                            S_m = S.copy(); S_m[i] = False
                            v_Sm = self._eval_coalition(x, S_m)
                            m_Sm = self._predict_gp_fast(S_m)
                            strata_residuals[s - 1][i].append((v_S - v_Sm) - (m_S - m_Sm))

        # Dynamic Coupled Neyman Allocation for draw sizes q in [1, M-1] (Theorem 1)
        for s in range(1, max(1, self.M - 1)):
            for i in range(self.M):
                sigma_res[s, i] = safe_std(strata_residuals[s][i], 0.5)

        neyman_probs = self._solve_coupled_neyman_allocation(sigma_res)
        iter_count = 0

        # Adaptive Residual Sampling Loop with Anytime Stopping
        raw_widths = np.full(self.M, np.inf)
        stage2_evals_start = self.total_coalition_evals

        while True:
            # 1. Check Stratified Anytime Empirical Bernstein Bounds (Lemma G)
            for i in range(self.M):
                W_i_res = 0.0
                all_cells_valid = True
                for s in range(self.M):
                    n_is = len(strata_residuals[s][i])
                    if s == 0 or s == self.M - 1:
                        if n_is < 1:
                            all_cells_valid = False
                            W_i_res = np.inf
                            break
                        continue

                    if n_is < 2:
                        all_cells_valid = False
                        W_i_res = np.inf
                        break

                    sig_curr = sigma_res[s, i]
                    log_term = np.log((np.pi ** 2 * (self.M ** 2) * (n_is ** 2)) / (3.0 * delta))
                    w_s = np.sqrt((2.0 * (sig_curr ** 2) * log_term) / n_is) + (7.0 * R_Delta_res * log_term) / (3.0 * (n_is - 1))
                    W_i_res += (1.0 / self.M) * w_s

                raw_widths[i] = W_i_res if all_cells_valid else np.inf

            if np.max(raw_widths) <= epsilon:
                break

            # Strict Stage-2 budget check before executing next round
            current_stage2_evals = self.total_coalition_evals - stage2_evals_start
            round_cost_upper = 1 + self.M
            if current_stage2_evals + round_cost_upper > max_budget:
                break

            # 2. Dynamic Neyman Reallocation every 5M evaluations
            if iter_count > 0 and iter_count % (5 * self.M) == 0:
                for s in range(1, max(1, self.M - 1)):
                    for i in range(self.M):
                        sigma_res[s, i] = safe_std(strata_residuals[s][i], 0.5)
                neyman_probs = self._solve_coupled_neyman_allocation(sigma_res)

            if np.sum(neyman_probs) > 0:
                s_target = int(np.random.choice(self.M, p=neyman_probs))
            else:
                s_target = 1

            perm = np.random.permutation(self.M)
            S_new = np.zeros(self.M, dtype=bool)
            S_new[perm[:s_target]] = True

            v_S = self._eval_coalition(x, S_new)
            m_S = self._predict_gp_fast(S_new)

            for i in range(self.M):
                if not S_new[i]:
                    S_u = S_new.copy(); S_u[i] = True
                    v_Su = self._eval_coalition(x, S_u)
                    m_Su = self._predict_gp_fast(S_u)
                    strata_residuals[s_target][i].append((v_Su - v_S) - (m_Su - m_S))
                    if s_target != 0 and s_target != self.M - 1:
                        sigma_res[s_target, i] = safe_std(strata_residuals[s_target][i], 0.5)
                else:
                    if s_target > 0:
                        S_m = S_new.copy(); S_m[i] = False
                        v_Sm = self._eval_coalition(x, S_m)
                        m_Sm = self._predict_gp_fast(S_m)
                        strata_residuals[s_target - 1][i].append((v_S - v_Sm) - (m_S - m_Sm))
                        if (s_target - 1) != 0 and (s_target - 1) != self.M - 1:
                            sigma_res[s_target - 1, i] = safe_std(strata_residuals[s_target - 1][i], 0.5)

            iter_count += 1

        # =====================================================================
        # Stage 3: Unified Assembly & MAP Efficiency Projection
        # =====================================================================
        phi_r_strat = np.zeros(self.M, dtype=np.float64)
        for i in range(self.M):
            stratum_sum = 0.0
            for s in range(self.M):
                if len(strata_residuals[s][i]) > 0:
                    stratum_sum += np.mean(strata_residuals[s][i])
            phi_r_strat[i] = stratum_sum / self.M

        phi_raw = phi_m_D + phi_r_strat

        residual = delta_total - float(np.sum(phi_raw))
        sum_post_vars = float(np.sum(posterior_variances))
        phi_final = phi_raw + posterior_variances * (residual / sum_post_vars)

        # Post-Projection Certified Bounds (Corollary C.1)
        if np.all(np.isfinite(raw_widths)):
            proj_inflation = posterior_variances * (float(np.sum(raw_widths)) / sum_post_vars)
            certified_widths = raw_widths + proj_inflation
        else:
            certified_widths = raw_widths

        call_coal_evals = self.total_coalition_evals - evals_start_coal
        call_model_evals = self.total_model_evals - evals_start_model

        return {
            "shapley_values": phi_final,
            "surrogate_shapley": phi_m_D,
            "residual_shapley": phi_r_strat,
            "raw_confidence_widths": raw_widths,
            "certified_projected_widths": certified_widths,
            "posterior_std": np.sqrt(posterior_variances),
            "num_coalition_evals": call_coal_evals,
            "num_model_evals": call_model_evals,
            "converged_early": (np.max(raw_widths) <= epsilon),
            "certificate_is_rigorous": (self.output_bounds is not None and np.all(np.isfinite(certified_widths))),
            "range_bound_is_heuristic": (self.output_bounds is None),
            "uncertified_features": np.where(~np.isfinite(raw_widths))[0].tolist()
        }


def test_all():
    print("Executing GAS-BayesSHAP Complete 10-Tier Verification Suite v11.0...")

    # Test 1: Lemma D Sign & Exact Enumeration (O(M^2) Algorithm)
    M = 4
    sigma0, lengthscale = 1.0, 1.5
    rho = np.exp(-1.0 / lengthscale)
    S_j = np.array([True, False, True, False])
    engine = GASBayesSHAP(lambda x: 0.0, np.zeros((1, M)), sigma0=sigma0, lengthscale=lengthscale)
    K_phi_analytic = engine._closed_form_cross_cov(S_j)

    K_phi_brute = np.zeros(M)
    for i in range(M):
        for mask in range(1 << M):
            S = np.array([(mask >> bit) & 1 for bit in range(M)], dtype=bool)
            if S[i]:
                continue
            s = int(np.sum(S))
            w_s = (factorial(s) * factorial(M - s - 1)) / factorial(M)
            Su = S.copy()
            Su[i] = True
            d_u = np.sum(Su != S_j)
            d_s = np.sum(S != S_j)
            K_phi_brute[i] += w_s * (sigma0 ** 2) * (rho ** d_u - rho ** d_s)

    assert np.allclose(K_phi_analytic, K_phi_brute, atol=1e-10)
    print("✓ Test 1 Passed: Lemma D O(M^2) Cross-Covariance exact matches brute force.")

    # Test 2: Lemma E Raw Pair Counts across M in [2, 3, 4, 5, 6]
    for m in [2, 3, 4, 5, 6]:
        eng = GASBayesSHAP(lambda x: 0.0, np.zeros((1, m)), sigma0=sigma0, lengthscale=lengthscale)
        K_analytic = eng.K_phi_phi
        K_brute = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                for m1 in range(1 << m):
                    S = np.array([(m1 >> b) & 1 for b in range(m)], dtype=bool)
                    if S[i]:
                        continue
                    s = int(np.sum(S))
                    w_s = factorial(s) * factorial(m - 1 - s) / factorial(m)
                    for m2 in range(1 << m):
                        T = np.array([(m2 >> b) & 1 for b in range(m)], dtype=bool)
                        if T[j]:
                            continue
                        t = int(np.sum(T))
                        w_t = factorial(t) * factorial(m - 1 - t) / factorial(m)
                        Su, Tu = S.copy(), T.copy()
                        Su[i] = True
                        Tu[j] = True
                        k11 = eng._kernel_val(Su, Tu)
                        k10 = eng._kernel_val(Su, T)
                        k01 = eng._kernel_val(S, Tu)
                        k00 = eng._kernel_val(S, T)
                        K_brute[i, j] += w_s * w_t * (k11 - k10 - k01 + k00)
        diff = np.max(np.abs(K_analytic - K_brute))
        assert diff < 1e-10, f"Lemma E failed at M={m}!"
    print("✓ Test 2 Passed: Lemma E Exact Raw Pair Counts validated on M in [2, 3, 4, 5, 6].")

    # Test 3: Null Player Certified Containment Check
    M_test3 = 5
    weights = np.array([1.5, -2.0, 0.5, 3.0, 0.0])
    model = lambda x: float(np.dot(x, weights) + 0.5 * x[0] * x[1])  # noqa: E731
    bg = np.zeros((5, M_test3))
    eng = GASBayesSHAP(model, bg, output_bounds=(-5.0, 10.0))
    res = eng.explain(np.ones(M_test3), epsilon=1.0, delta=0.05, max_budget=400)

    phi_null = res['shapley_values'][4]
    w_null = res['certified_projected_widths'][4]
    assert abs(phi_null - 0.0) <= w_null
    print(f"✓ Test 3 Passed: Null Player True Value (0.0) is strictly contained in [{phi_null - w_null:.3f}, {phi_null + w_null:.3f}].")

    # Test 4: Coverage Calibration on Ground Truth (R=30 trials)
    M_test4 = 3
    phi_exact = np.array([1.5, 2.5, -1.0])
    model_cal = lambda x: float(x[0] + 2.0 * x[1] - x[2] + x[0] * x[1])  # noqa: E731
    bg_cal = np.zeros((3, M_test4))

    finite_count = 0
    covered_count = 0
    R = 30

    for trial in range(R):
        np.random.seed(trial)
        e = GASBayesSHAP(model_cal, bg_cal, output_bounds=(-2.0, 5.0))
        r = e.explain(np.ones(M_test4), epsilon=1.5, delta=0.05, max_budget=300)

        is_finite = np.all(np.isfinite(r['certified_projected_widths']))
        is_covered = np.all(np.abs(r['shapley_values'] - phi_exact) <= r['certified_projected_widths'])

        if is_finite:
            finite_count += 1
            if is_covered:
                covered_count += 1

    finite_rate = finite_count / R
    cov_rate_given_finite = covered_count / max(1, finite_count)
    assert cov_rate_given_finite >= 0.90
    print(f"✓ Test 4 Passed: Finite Width Rate = {finite_rate*100:.1f}%, Coverage Given Finite = {cov_rate_given_finite*100:.1f}% (≥ 90%).")

    # Test 5: Corollary C.1 Tightness & Inflation Analysis
    finite_mask = np.isfinite(res['raw_confidence_widths'])
    if np.any(finite_mask):
        inflation_ratio = np.mean(res['certified_projected_widths'][finite_mask] / res['raw_confidence_widths'][finite_mask])
        print(f"✓ Test 5 Passed: Post-Projection Width Inflation Factor = {inflation_ratio:.2f}x (Controlled & Tight).")

    # Test 6: Query Counter Delta Isolation and Stage-2 Budget Guard
    res_call1 = eng.explain(np.ones(eng.M), epsilon=1.0, delta=0.05, max_budget=200)
    res_call2 = eng.explain(np.ones(eng.M) * 2, epsilon=1.0, delta=0.05, max_budget=200)
    assert res_call1['num_coalition_evals'] > 0
    assert res_call2['num_coalition_evals'] > 0
    print("✓ Test 6 Passed: Isolated Per-Explanation Query Accounting and Strict Stage-2 Budget Guard verified.")

    # Test 7: Surrogate Global Boundedness across all 2^M subsets
    M7 = 4
    L7, U7 = 0.0, 1.0
    eng7 = GASBayesSHAP(lambda x: float(np.mean(x)), np.zeros((4, M7)), output_bounds=(L7, U7))
    eng7.explain(np.ones(M7), max_budget=50)
    all_bounded = True
    for mask in range(1 << M7):
        S = np.array([(mask >> b) & 1 for b in range(M7)], dtype=bool)
        mb_val = eng7._predict_gp_fast(S)
        if mb_val < L7 - 1e-10 or mb_val > U7 + 1e-10:
            all_bounded = False
    assert all_bounded
    print("✓ Test 7 Passed: Surrogate m_b(S) strictly bounded in [L, U] across all 2^M subsets.")

    # Test 8: Zero Allocation on Extreme Strata in Stage 2
    neyman_p = eng7._solve_coupled_neyman_allocation(np.ones((M7, M7)))
    assert neyman_p[0] == 0.0
    print("✓ Test 8 Passed: Extreme stratum s=0 strictly allocated 0 probability in Stage 2.")

    # Test 9: M=2 Exact Certification Property
    M9 = 2
    eng9 = GASBayesSHAP(lambda x: float(2 * x[0] + 3 * x[1]), np.zeros((2, M9)), output_bounds=(0.0, 5.0))
    res9 = eng9.explain(np.ones(M9), max_budget=20)
    assert np.all(res9['certified_projected_widths'] == 0.0)
    print("✓ Test 9 Passed: M=2 exact certification produces strictly 0 width.")

    # Test 10: Surrogate Linearity & Exact Additive Fit
    M10 = 3
    w10 = np.array([1.0, 2.0, 3.0])
    eng10 = GASBayesSHAP(lambda x: float(np.dot(x, w10)), np.zeros((3, M10)), output_bounds=(0.0, 6.0))
    res10 = eng10.explain(np.ones(M10), max_budget=100)
    assert np.allclose(res10['shapley_values'], w10, atol=0.2)
    print("✓ Test 10 Passed: Surrogate analytical attributability recovers additive ground truth.")

    print("\nALL 10 VERIFICATION TESTS COMPLETED WITH 100% PASS RATE.")


if __name__ == "__main__":
    test_all()
