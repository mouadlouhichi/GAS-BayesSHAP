#!/usr/bin/env python
"""Build notebooks/AIR_QUALITY_GAS.ipynb — Beijing air-quality Tier-A/Tier-B
experiment with certified GAS-BayesSHAP attributions.

Tier A (exact ground truth, M = 11 static features, 2^11 = 2048 coalitions):
  1. Load Beijing air-quality dataset (local CSV -> UCI zip URL -> synthetic
     regime-structured fallback for offline sandboxes)
  2. Preprocess / scale
  3. PCA visualization
  4. K-Means regime cluster selection (n_init=10, random_state=1301)
  5. Final K-Means clustering
  6. Regime characterization (photochemical / winter smog / stagnant
     inversion / clean air) from cluster feature profiles
  7. Train/test split for the LightGBM regime surrogate
  8. Surrogate validation accuracy (held-out)
  9. Baseline TreeSHAP explanations (logit space, non-certified)
 10. Exact Shapley enumeration for M=11 (2^11 = 2048 coalitions)
 11. GAS-BayesSHAP certified regime attributions
 12. RMSE / coverage / query-count comparison
 13. Certified waterfall plot with error bars

Tier B (high-dimensional scaling, M = 66 lagged features -> M_group = 11
macro-players, exact 2^11 ground truth at group level):
 14. Group-lag spatiotemporal game (lags t, t-1, t-3, t-6, t-12, t-24)
 15. LightGBM surrogate on 66 lagged features
 16. Exact group Shapley (2^11) via the package's group-lag game
 17. GAS-BayesSHAP certified macro attributions
 18. Tier-B comparison + macro waterfall

All scientific algorithms are imported from the gas_bayesshap package.
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

A(md("# Beijing Air-Quality — Tier A Exact Ground-Truth + Certified GAS-BayesSHAP Regime Attribution\n\n"
     "This notebook runs the paper's air-quality regime-monitoring experiments on the "
     "**Beijing multi-site air-quality dataset** (11 static pollutant/meteorological features):\n\n"
     "- **Tier A**: K-Means regime discovery (photochemical smog, winter smog, stagnant inversion, "
     "clean air), a LightGBM cluster-membership surrogate $g_c(x) = P(\\text{regime}=c \\mid x)$, "
     "exact Shapley ground truth at $2^{11} = 2048$ coalitions, and **GAS-BayesSHAP certified** "
     "attributions with post-projection widths $W_i^{\\text{proj}}$ (Theorem C / Corollary C.1).\n"
     "- **Tier B**: the group-lag spatiotemporal game — 66 lagged features "
     "($t, t-1, t-3, t-6, t-12, t-24$) grouped into 11 macro-players, exact group Shapley at "
     "$2^{11} = 2048$, and certified macro attributions.\n\n"
     "> Data loading: local `../data/Beijing_MultiSite_AirQuality.csv` → UCI zip URL → synthetic "
     "regime-structured fallback (offline sandbox).  All scientific algorithms are imported from "
     "the `gas_bayesshap` package — never duplicated here."))

A(md("## Tier A — Exact Ground-Truth Verification (M = 11)\n\n## 1. Load Air-Quality dataset"))
A(cell("""import os, sys, io, zipfile
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
from gas_bayesshap.game.domain_games import group_lag_game, build_group_lags

FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
DATA_DIR = os.path.join("..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_SOURCE = "unknown"

# Download url to dest (cached for later runs). Returns True on success.
def _download(url, dest, timeout=40):
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GAS-BayesSHAP notebook)"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  downloaded {os.path.getsize(dest) / 1e6:.2f} MB -> {dest}")
        return True
    except Exception as exc:
        print(f"  download failed ({url}): {type(exc).__name__} {str(exc)[:80]}")
        return False



def _synthetic_air(n=3000, seed=1301):
    \"\"\"Synthetic hourly Beijing-like series with regime switching.\"\"\"
    rng = np.random.RandomState(seed)
    hour = np.arange(n) % 24
    # daily + seasonal temperature cycle
    temp = 18 + 10 * np.sin(2 * np.pi * (np.arange(n) % 365) / 365) + 4 * np.sin(2 * np.pi * hour / 24)
    # regime states (means over the 11 features)
    regimes = {
        "clean":        [15, 30, 3, 15, 0.5, 60,  20, 1010, -5,  0.0, 5.5],
        "photochemical":[45, 80, 8, 45, 0.8, 140, 30, 1008, -15, 0.0, 2.5],
        "winter_smog":  [180, 220, 40, 90, 2.5, 25,  0,  1025, -20, 0.0, 1.2],
        "stagnant":     [120, 150, 20, 70, 1.8, 35,  5,  1030, -10, 0.0, 0.6],
    }
    names = list(regimes.keys())
    X = np.zeros((n, 11))
    state = "clean"
    for i in range(n):
        if rng.rand() < 0.03:
            state = names[rng.randint(len(names))]
        base = np.array(regimes[state], dtype=float)
        X[i] = base + rng.randn(11) * np.array([8, 12, 2, 6, 0.15, 8, 1.5, 2, 3, 0.1, 0.5])
        X[i, 6] += temp[i]  # TEMP
        X[i, 2] = max(X[i, 2], 0.0)   # SO2 >= 0
        X[i, 3] = max(X[i, 3], 0.0)   # NO2 >= 0
        X[i, 0] = max(X[i, 0], 0.0)   # PM2.5 >= 0
        X[i, 1] = max(X[i, 1], 0.0)   # PM10 >= 0
        X[i, 4] = max(X[i, 4], 0.05)  # CO > 0
    df = pd.DataFrame(X, columns=FEATURES)
    return df


# Load the Beijing multi-site air-quality dataset:
# cached data/ CSV -> cached ZIP (download from mirrors) -> synthetic.
def load_air():
    global DATA_SOURCE
    combined = os.path.join(DATA_DIR, "Beijing_MultiSite_AirQuality.csv")
    zip_path = os.path.join(DATA_DIR, "Beijing_MultiSite_AirQuality.zip")

    def _valid(df):
        return set(FEATURES).issubset(df.columns)

    def _quarantine(path):
        bad = path + ".bad"
        try:
            os.replace(path, bad)
            print(f"  quarantined invalid file -> {os.path.basename(bad)}")
        except OSError:
            try: os.remove(path)
            except OSError: pass

    # 1) cached combined CSV (previous download or user-provided) -- VALIDATED;
    #    a stale wrong file (e.g. a stock CSV cached by an older loader) is
    #    quarantined and ignored instead of crashing.
    if os.path.exists(combined):
        try:
            cand = pd.read_csv(combined)
            if _valid(cand):
                DATA_SOURCE = f"cache:{combined}"
                return cand
            print(f"WARNING: cached {os.path.basename(combined)} is NOT the air-quality "
                  f"dataset (cols={list(cand.columns)[:6]}...) -> quarantining")
            _quarantine(combined)
        except Exception as exc:
            print("cached combined CSV unreadable:", type(exc).__name__, "- quarantining")
            _quarantine(combined)

    # 1b) directory of per-station CSVs (e.g. PRSA_Data_*.csv extracted from the
    #     Kaggle/UCI archive into data/beijing/).  Merged like the standard
    #     multi-file recipe, with a station column and air-quality validation.
    for sub in ("beijing", "PRSA_Data_20130301-20170228"):
        subdir = os.path.join(DATA_DIR, sub)
        if os.path.isdir(subdir):
            frames = []
            for fn in sorted(os.listdir(subdir)):
                if fn.lower().endswith(".csv"):
                    tmp = pd.read_csv(os.path.join(subdir, fn))
                    if not set(FEATURES).intersection(tmp.columns):
                        print(f"  skipping {fn} (not an air-quality file)")
                        continue
                    if "station" not in tmp.columns:
                        tmp["station"] = os.path.splitext(fn)[0]
                    frames.append(tmp)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                if _valid(df):
                    df.to_csv(combined, index=False)
                    DATA_SOURCE = f"dir-merge:{subdir} (cached to {combined})"
                    return df
                print(f"WARNING: merged CSVs in {subdir} lack the air-quality features "
                      f"({list(df.columns)[:6]}...) -> quarantining combined")

    # 2) cached ZIP -- validated; a wrong archive is quarantined and re-downloaded
    def _zip_is_valid(zpath):
        try:
            with zipfile.ZipFile(zpath) as z:
                names = [n for n in z.namelist() if n.endswith(".csv")]
                if not names:
                    return False
                frames = []
                for nm in names:
                    tmp = pd.read_csv(z.open(nm))
                    if set(FEATURES).intersection(tmp.columns):
                        if "station" not in tmp.columns:
                            tmp["station"] = os.path.splitext(os.path.basename(nm))[0]
                        frames.append(tmp)
                if not frames:
                    return False
                probe = pd.concat(frames, ignore_index=True)
                return set(FEATURES).issubset(probe.columns)
        except Exception:
            return False

    if os.path.exists(zip_path) and not _zip_is_valid(zip_path):
        print(f"WARNING: cached {os.path.basename(zip_path)} is an INVALID archive -> quarantining")
        _quarantine(zip_path)

    if not os.path.exists(zip_path):
        print("Downloading Beijing multi-site air-quality dataset ...")
        # CSV mirrors (single combined table) tried first; ZIP mirrors (per-station
        # archives) extracted and combined.  Each is validated to contain the 11
        # air-quality features before being accepted.
        csv_mirrors = [
            # 1) stable GitHub mirror of the UCI dataset (combined data.csv)
            "https://raw.githubusercontent.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/main/data/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata/data.csv",
        ]
        zip_mirrors = [
            # 2) Kaggle-signed GCS mirror (PRSA_Data_* station CSVs, 12 sites).
            #    NOTE: this signed URL EXPIRES (~3 days from issue); if it 404s
            #    the loader falls through to the official UCI mirrors.
            "https://storage.googleapis.com/kaggle-data-sets/409180/783762/bundle/archive.zip"
            "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
            "&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260814%2Fauto%2Fstorage%2Fgoog4_request"
            "&X-Goog-Date=20260814T184332Z"
            "&X-Goog-Expires=259200"
            "&X-Goog-SignedHeaders=host"
            "&X-Goog-Signature=4e88c0c92fe6c7bccc6d20988e45443705be421025b41ef54866c8bd31358c1a93babbcd41d78e0e5577e77d78a1660e1f180ad4da76849cf8467874e711e9377a09d09d9280803587466528ee25587e14752da8f74def2d2cb860d0a630455d5e20b6bc47ea49c4ee152fff3ba5b5ded44f9b1b685928d2ec2e81763b637dbf33ccff081ff3d2f43af2fec4bc000a3c97e0f4d49ba9f3b4e039ad5f1eb5974ba26ce07b81619f2866b7d035cc0de2e025fe509317bb886ca35f1d9cf65a2f7919dd7df6ea2d83294103f35e4cb2355156c13991fc5bf66169bd674e3633044b502214bce941478ea0b853cab4bf9eb558a6d4ccd94b9e82e9634e616d8261b3",
            # 3) official UCI locations (may redirect to a wrong archive)
            "https://archive.ics.uci.edu/static/public/501/beijing+multi-site+air-quality+data.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/Beijing_MultiSite_AirQuality.zip",
        ]
        ok = False
        for u in csv_mirrors:
            tmp = combined + ".tmp"
            if _download(u, tmp):
                try:
                    probe = pd.read_csv(tmp)
                    if _valid(probe):
                        os.replace(tmp, combined)
                        ok = True
                        break
                except Exception:
                    pass
                if os.path.exists(tmp):
                    os.remove(tmp)
                print("  CSV mirror invalid (wrong content) - skipping")
        if not ok:
            for u in zip_mirrors:
                tmp = zip_path + ".tmp"
                if _download(u, tmp):
                    if _zip_is_valid(tmp):
                        os.replace(tmp, zip_path)
                        ok = True
                        break
                    print("  ZIP mirror returned an INVALID archive (wrong content) - skipping")
                    if os.path.exists(tmp):
                        os.remove(tmp)
        if not ok:
            print("All downloads failed or were invalid -> using SYNTHETIC regime-structured fallback (offline sandbox).")
            DATA_SOURCE = "synthetic-fallback"
            return _synthetic_air()

    # a CSV mirror caches directly into the combined CSV -> return it now
    # (VALIDATED: never return a stale wrong file)
    if os.path.exists(combined):
        cand = pd.read_csv(combined)
        if _valid(cand):
            DATA_SOURCE = "url-csv (cached to " + combined + ")"
            return cand
        print("post-download combined CSV is INVALID -> quarantining and falling back")
        _quarantine(combined)
        DATA_SOURCE = "synthetic-fallback"
        return _synthetic_air()

    # 3) extract + combine ALL station CSVs (multi-site), then sort by station/time
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        frames = []
        for nm in csv_names:
            tmp = pd.read_csv(z.open(nm))
            if not set(FEATURES).intersection(tmp.columns):
                print(f"  skipping {nm} (not an air-quality file)")
                continue
            if "station" not in tmp.columns:
                tmp["station"] = os.path.splitext(os.path.basename(nm))[0]
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)

    # validate the combined table once more before proceeding
    if not _valid(df):
        print("combined download does NOT contain the 11 air-quality features:",
              list(df.columns)[:12])
        print("-> using SYNTHETIC regime-structured fallback.")
        DATA_SOURCE = "synthetic-fallback"
        return _synthetic_air()

    time_cols = ["year", "month", "day", "hour"]
    if all(c in df.columns for c in time_cols):
        if "station" in df.columns:
            df = df.sort_values(["station"] + time_cols).reset_index(drop=True)
        else:
            df = df.sort_values(time_cols).reset_index(drop=True)

    # cache the combined CSV for fast re-runs
    df.to_csv(combined, index=False)
    DATA_SOURCE = f"url-zip (cached to {combined})"
    return df


LOADER_VERSION = 4
print("air loader version:", LOADER_VERSION)
df = load_air()
print("Dataset shape:", df.shape, "| source:", DATA_SOURCE)
if not set(FEATURES).issubset(df.columns):
    print("Available columns:", list(df.columns)[:20])
    print("ERROR: the air-quality features are missing from the loaded data.")
    print("Fix: delete any stale files in ../data/ and rerun:")
    print("    rm -f ../data/Beijing_MultiSite_AirQuality.csv* ../data/Beijing_MultiSite_AirQuality.zip*")
    print("    python scripts/build_air_quality_gas_notebook.py   # rebuild the notebook")
    raise ValueError("air-quality loader returned a non-air-quality dataset (see message above); "
                     "delete stale ../data/ files and rebuild the notebook")

X_full = df[FEATURES].copy()
# keep one station for a clean hourly series if the multi-site file was loaded
for c in ["station", "wd"]:
    if c in df.columns and c not in X_full.columns:
        X_full[c] = df[c].values
X = X_full.dropna().reset_index(drop=True)
X = X[FEATURES]
if "station" in df.columns and df["station"].nunique() > 1:
    st = df["station"].value_counts().index[0]
    print("Multi-site data loaded -> using station", st, "for a coherent series")
    X = X_full[X_full["station"] == st][FEATURES].dropna().reset_index(drop=True)
print("Rows after dropna:", len(X))"""))

A(md("## 2. Preprocess / Scale"))
A(cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("X_scaled:", X_scaled.shape)"""))

A(md("## 3. PCA visualization"))
A(cell("""pca = PCA().fit(X_scaled)
plt.figure(figsize=(6, 4))
plt.plot(1 + np.arange(X.shape[1]), np.cumsum(pca.explained_variance_ratio_), "o-")
plt.xlabel("Component"); plt.ylabel("Cumulative explained variance"); plt.grid(alpha=0.3)
plt.title("PCA cumulative explained variance (11 features)")
plt.show()

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=4, alpha=0.5)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Beijing air quality in 2D PCA space")
plt.show()"""))

A(md("## 4. K-Means regime cluster selection (deterministic: n_init=10, random_state=1301)"))
A(cell("""sils, dbs = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=1301, n_init=10)
    lab = km.fit_predict(X_scaled)
    sils.append(silhouette_score(X_scaled, lab))
    dbs.append(davies_bouldin_score(X_scaled, lab))
    print(f"k={k}: silhouette={sils[-1]:.3f}  davies_bouldin={dbs[-1]:.3f}")

best_k = int(2 + np.argmax(sils))
print("Chosen k (best silhouette):", best_k)"""))

A(md("## 5. Final K-Means clustering"))
A(cell("""kmeans = KMeans(n_clusters=best_k, random_state=1301, n_init=10)
regime_labels = kmeans.fit_predict(X_scaled)
print("Regime cluster sizes:", np.bincount(regime_labels))"""))

A(md("## 6. Regime characterization (photochemical / winter smog / stagnant / clean)"))
A(cell("""profiles = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)
# de-standardize for interpretability
profiles_raw = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES).round(2)
print("Cluster mean profiles (raw units):")
print(profiles_raw)

g = X.describe().loc[["25%", "75%"]]

def regime_name(means):
    if means["O3"] > g.loc["75%", "O3"] and means["TEMP"] > X["TEMP"].mean():
        return "photochemical_smog"
    if means["CO"] > g.loc["75%", "CO"] or means["SO2"] > g.loc["75%", "SO2"]:
        return "winter_smog"
    if means["WSPM"] < g.loc["25%", "WSPM"] and means["PRES"] > X["PRES"].mean():
        return "stagnant_inversion"
    return "clean_air"

regime_names = [regime_name(profiles_raw.iloc[c]) for c in range(best_k)]
print("\\nRegime labels:", regime_names)

# pick the photochemical regime as the certified attribution target (paper RQ3)
target_cluster = regime_names.index("photochemical_smog") if "photochemical_smog" in regime_names else 0
print("Target regime cluster:", target_cluster, "->", regime_names[target_cluster])"""))

A(md("## 7. Train/test split for the LightGBM regime surrogate"))
A(cell("""X_tr, X_te, y_tr, y_te = train_test_split(
    X, regime_labels, test_size=0.3, random_state=1301, stratify=regime_labels)
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

A(md("## 8. Surrogate validation accuracy (held-out)"))
A(cell("""pred = lgb_model.predict(X_te)
print("Accuracy :", round(accuracy_score(y_te, pred), 4))
print("Macro-F1 :", round(f1_score(y_te, pred, average="macro"), 4))
print("Classes  :", sorted(set(y_te)))"""))

A(md("## 9. Baseline TreeSHAP explanations (logit space, NON-certified)"))
A(cell("""# TreeSHAP explains the LightGBM margin/logit, NOT the probability game
# v_{x,c}(S) = E[g_c(x_S, Z_\\bar S)] with g_c = P(regime = c | x).  Reported only
# as a non-certified baseline (space mismatch flagged).
explainer_tree = shap.TreeExplainer(lgb_model)
sv_tree_raw = explainer_tree.shap_values(X_te)
cluster_id = target_cluster

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
print("TreeSHAP (logit) mean |importance| for regime", regime_names[cluster_id], ":")
print(pd.Series(tree_imp, index=FEATURES).sort_values(ascending=False).round(4))"""))

A(md("## 10. Exact Shapley enumeration for M=11 (2^11 = 2048 coalitions)"))
A(cell("""def make_proba_model(cluster_id):
    def model_fn(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        # pass a DataFrame with the training feature names so LightGBM does not
        # emit "X does not have valid feature names" on every call
        x_df = pd.DataFrame(x, columns=FEATURES)
        return float(lgb_model.predict_proba(x_df)[0, cluster_id])
    return model_fn

B = 64
rng = np.random.RandomState(1301)
background = X.sample(B, random_state=1301).values
x0 = X_te.iloc[0].values
M = len(x0)
print("M =", M, "| 2^M =", 2 ** M, "| background B =", B)

model_fn = make_proba_model(cluster_id)
oracle = CoalitionOracle(model_fn, background, output_bounds=(0.0, 1.0),
                         model_tag=f"beijing-regime-{cluster_id}")

values = exact_game_values(oracle, x0, M)
phi_exact = exact_shapley_from_values(values, M)
delta_total = values[(1 << M) - 1] - values[0]
print("exact coalition evals:", oracle.total_coalition_evals, "(= 2^M)")
print("efficiency: sum(phi) =", round(float(phi_exact.sum()), 6),
      "| v(N)-v(empty) =", round(float(delta_total), 6))"""))

A(md("## 11. GAS-BayesSHAP certified regime attributions"))
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
        "run_id": f"beijing-tierA-regime{cluster_id}",
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

A(md("## 12. RMSE / coverage / query-count comparison"))
A(cell("""# --- TreeSHAP (logit space, baseline) ---
tree_phi = sv_tree[0]   # first test instance (class extractor applied above)

# --- KernelSHAP (probability game, non-certified baseline) ---
def proba_matrix(Xmat):
    Xdf = pd.DataFrame(np.asarray(Xmat), columns=FEATURES)
    return lgb_model.predict_proba(Xdf)[:, cluster_id]

kexplainer = shap.KernelExplainer(proba_matrix, background)
  # matched background
kernel_phi = kexplainer.shap_values(x0, nsamples=128)

# --- SamplingSHAP (Monte-Carlo baseline on the probability game) ---
mc = monte_carlo_shapley(oracle, x0, n_samples=150, rng=np.random.RandomState(0))
mc_phi = np.asarray(mc["shapley_values"])

# The theorem guarantees SIMULTANEOUS coverage: P(forall i, |phi_i - phi_i| <= W_i) >= 1 - delta.
err_gas = np.abs(phi_gas - phi_exact)
gas_simultaneous = float(np.all(err_gas <= W_proj))
gas_marginal = float(np.mean(err_gas <= W_proj))

rows = {
    "GAS-BayesSHAP (certified)": dict(
        rmse=rmse(phi_gas, phi_exact), mae=mae(phi_gas, phi_exact),
        max_err=max_abs_error(phi_gas, phi_exact),
        coalition_evals=result["num_coalition_evals_this_call"],
        model_evals=result["num_model_evals_this_call"],
        simultaneous_coverage=gas_simultaneous, marginal_coverage=gas_marginal),
    "KernelSHAP": dict(
        rmse=rmse(kernel_phi, phi_exact), mae=mae(kernel_phi, phi_exact),
        max_err=max_abs_error(kernel_phi, phi_exact),
        coalition_evals=128, model_evals=128 * B,
        simultaneous_coverage="-", marginal_coverage="-"),
    "SamplingSHAP (MC)": dict(
        rmse=rmse(mc_phi, phi_exact), mae=mae(mc_phi, phi_exact),
        max_err=max_abs_error(mc_phi, phi_exact),
        coalition_evals=mc["num_coalition_evals"], model_evals=mc["num_model_evals"],
        simultaneous_coverage="-", marginal_coverage="-"),
    "Exact (ground truth)": dict(
        rmse=0.0, mae=0.0, max_err=0.0,
        coalition_evals=2 ** M, model_evals=oracle.total_model_evals,
        simultaneous_coverage="1.0", marginal_coverage="1.0"),
}
df_cmp = pd.DataFrame(rows).T
print("DATA_SOURCE:", DATA_SOURCE)
print(df_cmp.round(5))
print("GAS simultaneous coverage (all features in interval):", gas_simultaneous)
print("GAS marginal coverage (per-feature):", round(gas_marginal, 4))

# --- TreeSHAP is logit-space: NOT a same-game estimator, shown separately ---
print("TreeSHAP (logit space, model-specific baseline, NOT same-game):")
print(pd.Series(tree_imp, index=FEATURES).sort_values(ascending=False).round(4))

import os as _os
_out = os.path.join("..", "results", "air_quality_tierA")
os.makedirs(_out, exist_ok=True)
df_cmp.to_csv(os.path.join(_out, "comparison.csv"))
print("saved results/air_quality_tierA/comparison.csv")"""))

A(md("## 13. Certified waterfall plot with error bars"))
A(cell("""certified = np.abs(phi_gas) > W_proj
order = np.argsort(-np.abs(phi_gas))
feats = [FEATURES[i] for i in order]
ph = phi_gas[order]
wd = W_proj[order]
cc = certified[order]
colors = ["tab:blue" if c else "lightgray" for c in cc]

plt.figure(figsize=(11, 5))
plt.bar(feats, ph, yerr=wd, color=colors, capsize=4)
plt.axhline(0, color="black", lw=0.8)
plt.xticks(rotation=45, ha="right")
plt.ylabel("certified attribution $\\\\phi_i^*$")
plt.title(f"GAS-BayesSHAP certified attributions for {regime_names[cluster_id]} "
          f"(blue = sign-certified, grey = 0 in interval)")
plt.tight_layout()
plt.show()

print("sign-certified features:", [FEATURES[i] for i in range(M) if certified[i]])"""))

A(md("# Tier B — Group-Lag Spatiotemporal Game (M = 66 → M_group = 11)\n\n"
     "Expand each of the 11 variables into 6 lag blocks ($t, t-1, t-3, t-6, t-12, t-24$) = "
     "66 raw lagged features, grouped into 11 macro-players.  Exact ground truth is available "
     "at $2^{11} = 2048$ **group** coalitions.\n\n## 14. Build lagged features"))

A(cell("""LAGS = (0, 1, 3, 6, 12, 24)
n_lag = 1500  # consecutive hours
series = X.iloc[:n_lag].copy()  # treat rows as an hourly series

def make_lagged(df, lags):
    cols = {}
    for var in df.columns:
        for lag in lags:
            cols[f"{var}_t-{lag}"] = df[var].shift(lag)
    out = pd.DataFrame(cols)
    valid = out.dropna().index
    return out.loc[valid].reset_index(drop=True), valid

X_lag, valid_index = make_lagged(series, LAGS)
print("lagged feature matrix:", X_lag.shape, "| expected 11 x 6 =", 11 * len(LAGS))
groups = build_group_lags(n_vars=11, lags=LAGS)
print("macro-players:", len(groups), "| members per group:", [len(g) for g in groups][:3], "...")"""))

A(md("## 15. LightGBM surrogate on the 66 lagged features"))
A(cell("""# regime label aligned to the lagged rows (drop first max(lags) rows)
# regime label aligned to the lagged rows: row i of X_lag corresponds to
# the original row at valid_index[i] (dropna removes max(LAGS) rows)
lag_target = np.asarray(regime_labels)[valid_index]
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
    X_lag, lag_target, test_size=0.3, random_state=1301, stratify=lag_target)

lgb_lag = lgb.LGBMClassifier(
    objective="multiclass", num_class=best_k, random_state=1301,
    n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced")
lgb_lag.fit(Xl_tr, yl_tr)
pred_lag = lgb_lag.predict(Xl_te)
print("lagged-surrogate Accuracy:", round(accuracy_score(yl_te, pred_lag), 4),
      "| Macro-F1:", round(f1_score(yl_te, pred_lag, average="macro"), 4))"""))

A(md("## 16. Exact group Shapley (2^11 = 2048 group coalitions)"))
A(cell("""def make_lag_proba_model(cluster_id):
    def model_fn(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        x_df = pd.DataFrame(x, columns=list(X_lag.columns))
        return float(lgb_lag.predict_proba(x_df)[0, cluster_id])
    return model_fn

B_lag = 32
bg_lag = X_lag.sample(B_lag, random_state=1301).values
x0_lag = Xl_te.iloc[0].values
group_oracle, group_spec = group_lag_game(
    model_fn=make_lag_proba_model(cluster_id), background=bg_lag,
    n_vars=11, lags=LAGS, output_bounds=(0.0, 1.0))

print("group oracle M (macro):", group_spec.M, "| raw features:", group_spec.extra["M_feat"])
values_g = exact_game_values(group_oracle, x0_lag, group_spec.M)
phi_exact_g = exact_shapley_from_values(values_g, group_spec.M)
print("exact group coalition evals:", group_oracle.total_coalition_evals, "(= 2^11)")
print("group efficiency: sum =", round(float(phi_exact_g.sum()), 6))"""))

A(md("## 17. GAS-BayesSHAP certified macro attributions"))
A(cell("""eng_g = GASBayesSHAP(
    oracle=group_oracle,
    rng=np.random.RandomState(1301),
    config={
        "domain_game": "group_lag",
        "checkpoint_enabled": False,
        "cache_enabled": True,
        "persist_cache": False,
        "log_level": "NONE",
        "results_dir": "../results/runs",
        "checkpoints_dir": "../checkpoints",
        "run_id": "beijing-tierB-grouplag",
    },
)
res_g = eng_g.explain(x0_lag, epsilon=0.06, delta=0.05, max_budget=900,
                      n_pilot=3, n_active_steps=12)
phi_gas_g = np.asarray(res_g["shapley_values"])
W_proj_g = np.asarray(res_g["certified_projected_widths"])
print("status:", res_g["status"], "| converged:", res_g["converged"],
      "| rigorous:", res_g["certificate_is_rigorous"])
print("macro coalition evals (this call):", res_g["num_coalition_evals_this_call"])
macro_names = [f"var_{j} (6 lags)" for j in range(11)]
print("macro RMSE vs exact:", round(float(rmse(phi_gas_g, phi_exact_g)), 6))
err_g = np.abs(phi_gas_g - phi_exact_g)
macro_sim = float(np.all(err_g <= W_proj_g))
macro_mar = float(np.mean(err_g <= W_proj_g))
print("macro SIMULTANEOUS coverage:", macro_sim)
print("macro marginal coverage:", round(macro_mar, 4))"""))

A(md("## 18. Tier-B comparison + macro waterfall"))
A(cell("""cert_g = np.abs(phi_gas_g) > W_proj_g
order_g = np.argsort(-np.abs(phi_gas_g))
plt.figure(figsize=(10, 5))
plt.bar([macro_names[i] for i in order_g], phi_gas_g[order_g],
        yerr=W_proj_g[order_g],
        color=["tab:blue" if cert_g[i] else "lightgray" for i in order_g], capsize=4)
plt.axhline(0, color="black", lw=0.8)
plt.xticks(rotation=45, ha="right")
plt.ylabel("certified macro attribution")
plt.title("GAS-BayesSHAP certified macro-player attributions (M_group = 11)")
plt.tight_layout()
plt.show()

print("sign-certified macro players:", [macro_names[i] for i in range(11) if cert_g[i]])
print("\\nExact group Shapley (ground truth):", np.round(phi_exact_g, 5))
print("GAS group Shapley:", np.round(phi_gas_g, 5))

import os as _os
_outB = os.path.join("..", "results", "air_quality_tierB")
os.makedirs(_outB, exist_ok=True)
pd.DataFrame({"macro": macro_names, "gas": phi_gas_g, "exact": phi_exact_g,
              "W_proj": W_proj_g}).to_csv(os.path.join(_outB, "group_lag_comparison.csv"), index=False)
plt.savefig(os.path.join(_outB, "macro_waterfall.png"), dpi=120, bbox_inches="tight")
print("saved results/air_quality_tierB/group_lag_comparison.csv + macro_waterfall.png")"""))

A(md("## Summary\n\n"
     "- **Tier A** ($M=11$ static features): exact ground truth at $2^{11}=2048$ coalitions "
     "(efficiency holds exactly); GAS-BayesSHAP returns certified widths $W_i^{\\text{proj}}$ and "
     "sign-certified regime attributions; coverage vs exact reported above.\n"
     "- **Tier B** ($M=66 \\to M_{\\text{group}}=11$): exact group Shapley at $2^{11}=2048$ group "
     "coalitions; GAS-BayesSHAP certifies the 11 macro-players (per-variable lag blocks).\n"
     "- TreeSHAP explains the logit, KernelSHAP/SamplingSHAP explain the probability game — only "
     "GAS-BayesSHAP carries the anytime certification guarantee.\n"
     "- **Data download:** the loader downloads the UCI Beijing multi-site ZIP into `data/`, "
     "combines ALL station CSVs (multi-site, station column added), sorts by station/time, "
     "and caches the combined CSV for fast re-runs.  Place a local copy at "
     "`data/Beijing_MultiSite_AirQuality.csv` to bypass download; synthetic fallback only when offline."))

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "AIR_QUALITY_GAS.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(C)} cells")
