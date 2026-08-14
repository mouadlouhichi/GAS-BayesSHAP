#!/usr/bin/env python
"""Build notebooks/SHAP_WINE_GAS.ipynb — Tier-A wine exact-ground-truth
verification with GAS-BayesSHAP certified attributions.

Structure (per the audit's recommended upgrade):
  1. Load Wine Dataset (local CSV -> UCI URL -> synthetic fallback)
  2. Preprocess / Scale
  3. PCA visualization
  4. K-Means cluster selection (n_init=10, random_state=1301)
  5. Final K-Means clustering
  6. Train/test split for the LightGBM cluster surrogate
  7. Surrogate validation accuracy (held-out)
  8. Baseline TreeSHAP explanations (logit-space, non-certified)
  9. Exact Shapley enumeration for M=11 (2^11 = 2048 coalitions)
 10. GAS-BayesSHAP certified explanations (probability game)
 11. RMSE / coverage / query-count comparison
 12. Certified waterfall plot with error bars

The scientific algorithms (exact Shapley, GAS-BayesSHAP, Monte-Carlo
baseline) are imported from the gas_bayesshap package — never duplicated
inside the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


def md(text: str) -> dict:
    return cell(text, "markdown")


C = []
A = C.append

A(md("# Wine Quality — Tier-A Exact Ground-Truth Verification with GAS-BayesSHAP\n\n"
     "Continuation of `SHAP_WINE_CODE.ipynb` (previous-work baseline).  This notebook turns the "
     "wine clustering pipeline into a **certified estimator validation**:\n\n"
     "- cluster-membership probability game $v_{x,c}(S) = \\mathbb{E}[g_c(x_S, Z_{\\bar S})]$, "
     "$g_c(x) = P(\\text{cluster}=c\\mid x)$ from a LightGBM surrogate;\n"
     "- exact Shapley ground truth by enumerating $2^{11} = 2048$ coalitions;\n"
     "- GAS-BayesSHAP certified attributions with post-projection widths "
     "$W_i^{\\text{proj}}$ (Theorem C / Corollary C.1);\n"
     "- RMSE / coverage / query-count comparison vs exact, TreeSHAP, KernelSHAP and SamplingSHAP.\n\n"
     "> Data loading: local `../data/winequality-white.csv` → UCI URL → synthetic fallback "
     "(offline sandbox).  All scientific algorithms are imported from the `gas_bayesshap` package."))

A(md("## 1. Load Wine Dataset"))
A(cell("""import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, f1_score
import lightgbm as lgb
import shap

sys.path.insert(0, "..")
from gas_bayesshap import GASBayesSHAP
from gas_bayesshap.game.oracle import CoalitionOracle
from gas_bayesshap.game.brute_force import exact_game_values, exact_shapley_from_values
from gas_bayesshap.benchmarking.metrics import rmse, mae, max_abs_error
from gas_bayesshap.benchmarking.monte_carlo import monte_carlo_shapley

DATA_SOURCE = "unknown"

def load_wine():
    global DATA_SOURCE
    local = os.path.join("..", "data", "winequality-white.csv")
    if os.path.exists(local):
        DATA_SOURCE = f"local:{local}"
        return pd.read_csv(local, sep=";")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    try:
        df = pd.read_csv(url, sep=";")
        DATA_SOURCE = f"url:{url}"
        return df
    except Exception as exc:
        print("Network unavailable -> using SYNTHETIC wine-like fallback (offline sandbox).", exc)
        DATA_SOURCE = "synthetic-fallback"
        rng = np.random.RandomState(1301)
        n = 1500
        cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"]
        X = rng.randn(n, len(cols))
        df = pd.DataFrame(X, columns=cols)
        df["quality"] = rng.randint(3, 9, n)
        return df

df = load_wine()
print("Dataset shape:", df.shape, "| source:", DATA_SOURCE)
X = df.drop(columns="quality")
y = df["quality"]
feature_names = list(X.columns)
print("Features:", feature_names)"""))

A(md("## 2. Preprocess / Scale"))
A(cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("X_scaled:", X_scaled.shape)"""))

A(md("## 3. PCA visualization"))
A(cell("""pca = PCA().fit(X_scaled)
plt.figure(figsize=(6, 4))
plt.plot(1 + np.arange(X.shape[1]), np.cumsum(pca.explained_variance_ratio_), "o-")
plt.xlabel("Component"); plt.ylabel("Cumulative explained variance"); plt.grid(alpha=0.3)
plt.title("PCA cumulative explained variance")
plt.show()

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=4, alpha=0.5)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Wine (scaled) in 2D PCA space")
plt.show()"""))

A(md("## 4. K-Means cluster selection (deterministic: n_init=10, random_state=1301)"))
A(cell("""sils, dbs = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=1301, n_init=10)
    lab = km.fit_predict(X_scaled)
    sils.append(silhouette_score(X_scaled, lab))
    dbs.append(davies_bouldin_score(X_scaled, lab))
    print(f"k={k}: silhouette={sils[-1]:.3f}  davies_bouldin={dbs[-1]:.3f}")

best_k = int(2 + np.argmax(sils))
print("Chosen k (best silhouette):", best_k)"""))

A(md("## 5. Final K-Means clustering"))
A(cell("""kmeans = KMeans(n_clusters=best_k, random_state=1301, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
print("Cluster sizes:", np.bincount(cluster_labels))"""))

A(md("## 6. Train/test split for the LightGBM cluster surrogate"))
A(cell("""X_tr, X_te, y_tr, y_te = train_test_split(
    X, cluster_labels, test_size=0.3, random_state=1301, stratify=cluster_labels)
print("train:", X_tr.shape, "test:", X_te.shape)

params = {
    "objective": "multiclass",
    "num_class": best_k,
    "random_state": 1301,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
}
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_tr, y_tr)"""))

A(md("## 7. Surrogate validation accuracy (held-out)"))
A(cell("""pred = lgb_model.predict(X_te)
print("Accuracy :", round(accuracy_score(y_te, pred), 4))
print("Macro-F1 :", round(f1_score(y_te, pred, average="macro"), 4))
print("Classes  :", sorted(set(y_te)))"""))

A(md("## 8. Baseline TreeSHAP explanations (logit space, NON-certified)"))
A(cell("""# TreeSHAP explains the LightGBM margin/logit, NOT the probability game
# v_{x,c}(S) = E[g_c(x_S, Z_\\bar S)] with g_c = P(cluster = c | x).  We report
# it only as a non-certified baseline and flag the space mismatch.
explainer_tree = shap.TreeExplainer(lgb_model)
sv_tree_raw = explainer_tree.shap_values(X_te)
cluster_id = 0

def class_shap(sv, n_test, n_classes, c):
    # Robust per-class SHAP extraction (multiclass list OR (n, f, c) array).
    a = np.asarray(sv)
    if a.ndim == 3:
        if a.shape[0] == n_test and a.shape[2] == n_classes:
            return a[:, :, c]
        return a[c]
    return a

sv_tree = class_shap(sv_tree_raw, len(X_te), best_k, cluster_id)
tree_imp = np.abs(sv_tree).mean(axis=0)
print("TreeSHAP (logit) mean |importance| for cluster", cluster_id, ":")
print(pd.Series(tree_imp, index=feature_names).sort_values(ascending=False).round(4))"""))

A(md("## 9. Exact Shapley enumeration for M=11 (2^11 = 2048 coalitions)"))
A(cell("""def make_proba_model(cluster_id):
    def model_fn(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(lgb_model.predict_proba(x)[0, cluster_id])
    return model_fn

B = 64
rng = np.random.RandomState(1301)
background = X.sample(B, random_state=1301).values
x0 = X_te.iloc[0].values
M = len(x0)
print("M =", M, "| 2^M =", 2 ** M, "| background B =", B)

model_fn = make_proba_model(cluster_id)
oracle = CoalitionOracle(model_fn, background, output_bounds=(0.0, 1.0),
                         model_tag=f"wine-cluster-{cluster_id}")

values = exact_game_values(oracle, x0, M)
phi_exact = exact_shapley_from_values(values, M)
delta_total = values[(1 << M) - 1] - values[0]
print("exact coalition evals:", oracle.total_coalition_evals, "(= 2^M)")
print("efficiency: sum(phi) =", round(float(phi_exact.sum()), 6),
      "| v(N)-v(empty) =", round(float(delta_total), 6))"""))

A(md("## 10. GAS-BayesSHAP certified explanations"))
A(cell("""engine = GASBayesSHAP(
    model_fn=model_fn,
    background=background,
    output_bounds=(0.0, 1.0),
    rng=np.random.RandomState(1301),
    config={
        "domain_game": "membership",
        "checkpoint_enabled": False,
        "cache_enabled": True,
        "persist_cache": False,
        "log_level": "NONE",
        "results_dir": "../results/runs",
        "checkpoints_dir": "../checkpoints",
        "run_id": f"wine-tierA-cluster{cluster_id}",
    },
)
result = engine.explain(x0, epsilon=0.05, delta=0.05, max_budget=1000,
                        n_pilot=3, n_active_steps=15)
phi_gas = np.asarray(result["shapley_values"])
W_proj = np.asarray(result["certified_projected_widths"])
print("status:", result["status"], "| converged:", result["converged"],
      "| rigorous:", result["certificate_is_rigorous"])
print("coalition evals (this call):", result["num_coalition_evals_this_call"],
      "| model evals:", result["num_model_evals_this_call"])
print("surrogate+residual:", np.round(phi_gas, 5))
print("W_proj:", np.round(W_proj, 5))"""))

A(md("## 11. RMSE / coverage / query-count comparison"))
A(cell("""# --- TreeSHAP (logit space, baseline) ---
tree_phi = sv_tree[0]   # first test instance (class extractor applied in cell 8)

# --- KernelSHAP (probability game, non-certified baseline) ---
def proba_matrix(Xmat):
    return lgb_model.predict_proba(np.asarray(Xmat))[:, cluster_id]

kexplainer = shap.KernelExplainer(proba_matrix, background[:32])
kernel_phi = kexplainer.shap_values(x0, nsamples=128)

# --- SamplingSHAP (Monte-Carlo baseline on the probability game) ---
mc = monte_carlo_shapley(oracle, x0, n_samples=150, rng=np.random.RandomState(0))
mc_phi = np.asarray(mc["shapley_values"])

rows = {
    "GAS-BayesSHAP (certified)": dict(
        rmse=rmse(phi_gas, phi_exact), mae=mae(phi_gas, phi_exact),
        max_err=max_abs_error(phi_gas, phi_exact),
        coalition_evals=result["num_coalition_evals_this_call"],
        model_evals=result["num_model_evals_this_call"],
        coverage=float(np.mean(np.abs(phi_gas - phi_exact) <= W_proj))),
    "TreeSHAP (logit)": dict(
        rmse=rmse(tree_phi, phi_exact), mae=mae(tree_phi, phi_exact),
        max_err=max_abs_error(tree_phi, phi_exact), coalition_evals="-", model_evals="-", coverage="-"),
    "KernelSHAP": dict(
        rmse=rmse(kernel_phi, phi_exact), mae=mae(kernel_phi, phi_exact),
        max_err=max_abs_error(kernel_phi, phi_exact),
        coalition_evals=128, model_evals=128 * B, coverage="-"),
    "SamplingSHAP (MC)": dict(
        rmse=rmse(mc_phi, phi_exact), mae=mae(mc_phi, phi_exact),
        max_err=max_abs_error(mc_phi, phi_exact),
        coalition_evals=mc["num_coalition_evals"], model_evals=mc["num_model_evals"], coverage="-"),
    "Exact (ground truth)": dict(
        rmse=0.0, mae=0.0, max_err=0.0,
        coalition_evals=2 ** M, model_evals=oracle.total_model_evals, coverage="1.0"),
}
df_cmp = pd.DataFrame(rows).T
print(df_cmp.round(5))
print("\\nCoverage (GAS certified):", df_cmp.loc["GAS-BayesSHAP (certified)", "coverage"])"""))

A(md("## 12. Certified waterfall plot with error bars"))
A(cell("""certified = np.abs(phi_gas) > W_proj
order = np.argsort(-np.abs(phi_gas))
feats = [feature_names[i] for i in order]
ph = phi_gas[order]
wd = W_proj[order]
cc = certified[order]
colors = ["tab:blue" if c else "lightgray" for c in cc]

plt.figure(figsize=(11, 5))
plt.bar(feats, ph, yerr=wd, color=colors, capsize=4)
plt.axhline(0, color="black", lw=0.8)
plt.xticks(rotation=45, ha="right")
plt.ylabel("certified attribution $\\phi_i^*$")
plt.title(f"GAS-BayesSHAP certified attributions, cluster {cluster_id} "
          f"(blue = sign-certified, grey = 0 in interval)")
plt.tight_layout()
plt.show()

print("sign-certified features:", [feature_names[i] for i in range(M) if certified[i]])
print("(grey bars contain 0 within W_proj and are NOT sign-certified)")"""))

A(md("## Summary\n\n"
     "- **Exact ground truth** available at $2^{11}=2048$ coalitions (efficiency holds exactly).\n"
     "- **GAS-BayesSHAP** returns certified widths $W_i^{\\text{proj}}$; coverage vs exact is reported "
     "in the table above.\n"
     "- TreeSHAP explains the logit, KernelSHAP/SamplingSHAP explain the probability game — only "
     "GAS-BayesSHAP carries the anytime certification guarantee.\n"
     "- To run on the real wine data: place `winequality-white.csv` at `data/winequality-white.csv` "
     "or allow network access; the loader prefers local → URL → synthetic fallback."))

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "SHAP_WINE_GAS.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(C)} cells")
