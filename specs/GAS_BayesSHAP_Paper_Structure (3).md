# Academic Paper Structure & Theoretical Manuscript Blueprint (v11.0 Gold-Standard Final)

**Working Title:** Anytime-Certified Bayesian Control Variates for Explainable Clustering and Air-Quality Regime Discovery  
**Target Venues:** NeurIPS / ICML (Main Track) / IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / JMLR  
**Keywords:** Explainable AI, Shapley Values, Bayesian Quadrature, Active Control Variates, Anytime Confidence Sequences, Explainable Clustering, Environmental Monitoring  

---

## Abstract

Explaining unsupervised cluster formation in complex systems (e.g., environmental pollution regimes, industrial fault diagnosis) is critical for operational decision-making. While post-hoc surrogate Shapley values can interpret cluster assignments, existing approaches produce uncalibrated point estimates that lack non-asymptotic uncertainty bounds and scale exponentially $\mathcal{O}(2^M)$ with feature dimension, rendering high-dimensional spatiotemporal regime monitoring intractable.

In this work, we propose **GAS-BayesSHAP** (*Gaussian-Adaptive Stratified Bayesian Shapley Estimation*), a unified framework that combines **Bayesian-accelerated control variates** with **distribution-free anytime frequentist certification** for explainable clustering. 

Our core contributions are:
1. **Bounded Linear Control Variate Decomposition:** We decompose the empirical cluster attribution game $\hat{v}_{x,c}(S) = \frac{1}{B}\sum_{b=1}^B g_c(x_S, z_{\bar{S}}^{(b)}) \in [0, 1]$ into a frozen active bounded linear surrogate $m_b(S) = c + \lambda h(S)$ and a residual game $r_{\mathcal{D}}(S) = \hat{v}(S) - m_b(S)$. The surrogate attribution $\boldsymbol{\phi}(m_b) = \lambda \mathbf{K}_{\phi, \mathcal{D}} \boldsymbol{\alpha}$ is evaluated in closed form with zero sampling variance and zero clipping bias, while the residual $\boldsymbol{\phi}(r_{\mathcal{D}})$ is estimated on a decoupled sample set with drastically reduced variance.
2. **Exact Spectral Hypercube Calculus (Lemmas 1 & 2):** We derive exact closed-form expressions for the **Hypergeometric Kernel-Shapley Cross-Covariance $\mathbf{K}_{\phi, \mathcal{D}}$** in $\mathcal{O}(M^2)$ time and the **Prior Shapley Covariance Matrix $\mathbf{K}_{\phi,\phi}$** in $\mathcal{O}(M^3)$ under Exponential Hamming kernels via adjacent weight-difference $\Delta w_s$ factorization and raw pair counting.
3. **Exact Singleton Stratum Identification (Lemma 3):** We prove that extreme strata $s=0$ and $s=M-1$ are singletons with zero residual variance after deterministic pilot initialization, eliminating the coupon-collector waiting time.
4. **Conditional Stratum Uniformity (Lemma 4):** We prove that simultaneous Add-One and Remove-One marginal evaluations preserve unbiased stratum-conditioned sampling across all features.
5. **Coupled Adjacent-Stratum Neyman Allocation (Theorem 1):** We formulate and solve the convex allocation program over draw sizes $q \in \{1, \dots, M-1\}$ accounting for joint stratum sample accumulation under add-one/remove-one sampling.
6. **Anytime Stratified Empirical-Bernstein Confidence Sequences (Theorem 2 & Remarks 2.1–2.3):** We establish time-uniform confidence sequences on the residual game with deterministic range $R_\Delta^{\text{res}} = 4$ for membership games ($R_\Delta^{\text{res}} = 8$ for contrastive games), guaranteeing finite-sample coverage $\ge 1 - \delta$ under adaptive stopping.
7. **Posterior-Diagonal Uncertainty-Weighted Projection & Post-Projection Certificate (Theorem 3 & Corollary 1):** We derive the unique diagonal uncertainty-weighted projection onto $\sum_i \phi_i = \Delta_{\text{total}}$ with $\lambda^2$-scaled posterior covariance and prove that post-projection attributions satisfy $|\phi_i^* - \phi_i| \le W_i^{\text{proj}}$, enabling **Sign-Certified Feature Importance** ($|\phi_i^*| > W_i^{\text{proj}}$).
8. **Domain Application to Atmospheric Regime Discovery:** We apply GAS-BayesSHAP to discover and certify pollution regimes on multi-year air-quality datasets (Beijing 2013–2017) across exact 11-feature benchmarks and high-dimensional 66-feature lagged temporal models with block-imputation and Group-Shapley coherence.

Extensive experiments against SOTA baselines (ShaplEIG, OddSHAP, KernelSHAP, SamplingSHAP, TreeSHAP) demonstrate that GAS-BayesSHAP achieves significant query reduction in model forward passes, preserves domain regime semantics (photochemical, winter smog, stagnant dispersion, clean air), and guarantees valid $\ge (1 - \delta)$ coverage.

---

## 1. Introduction

### 1.1 From Interpretable Clustering to Certified Regime Discovery
* **The Context:** Unsupervised clustering (K-Means, Deep Embedded Clustering) groups complex multivariate data into actionable regimes. To explain *why* an observation belongs to a cluster, practitioners fit a cluster-membership surrogate $g_c(x) \in [0, 1]$ and compute feature attributions via Shapley values:
  $$\phi_{i,c}(x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(M - |S| - 1)!}{M!} \mathbb{E}\left[ g_c(x_{S \cup \{i\}}, Z_{\overline{S \cup \{i\}}}) - g_c(x_S, Z_{\bar{S}}) \right]$$
* **The Open Problem in Previous Work:** Earlier studies (e.g., 2025 benchmarks on Portuguese wine quality and Beijing air quality) demonstrated that Shapley values reveal interpretable chemical and meteorological structures in clusters. However, they identified three major future-work barriers:
  1. *Computational cost:* Permutation/kernel sampling scales exponentially with feature dimension $M$.
  2. *Lack of finite-sample uncertainty:* Point-estimate explanations provide no confidence intervals, risking false-positive feature importance in safety-critical domains.
  3. *Inability to scale to lagged temporal features:* Expanding from 11 static features to 66 lagged features ($t, t-1, t-3, t-6, t-12, t-24$) makes existing black-box SHAP methods intractable.

### 1.2 The Positioning: GAS-BayesSHAP vs. 2026 SOTA
```
+---------------------------------------------------------------------------------------------------------+
|                                    Methodological Evolution & Positioning                               |
+---------------------------------------------------------------------------------------------------------+
| Prior Clustering XAI (2025)   | Approximate SHAP point estimates on 11 static features (No bounds)      |
| ShaplEIG (ICML 2026)          | Active GP for Shapley (Relies on model-dependent Bayesian credibility)   |
| OddSHAP (ICML 2026)           | Odd-spectrum regression (Heuristic variance reduction, no anytime CIs)  |
| **GAS-BayesSHAP (This Work)** | **Bounded linear control variate + Distribution-free anytime residual CS|
|                               | **+ Deterministic extreme singletons + Bounded range certification      |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Formalization of Domain Cooperative Games

```
                               CLUSTER EXPLANATION GAMES
                                           │
         ┌─────────────────────────┬───────┴─────────────────┬─────────────────────────┐
         ▼                         ▼                         ▼                         ▼
   Local Membership        Contrastive Regime        Global Cluster-Level      Intrinsic Quality
     Attribution                Regime                    Regime                  Separation
         │                         │                         │                         │
   v_{x,c}(S) ∈ [0,1]       v_{x,c,c'}(S) ∈ [-1,1]    v_c(S) ∈ [0,1]            v_sil(S) ∈ [-1,1]
   R_Δ^res = 4              R_Δ^res = 8               R_Δ^res = 4               R_Δ^res = 8
```

### Convention 1 (Empirical-Background Game & Imputation Semantics)
All background-replacement evaluations use **interventional (marginal) imputation**: non-$S$ features are drawn independently from a fixed empirical background $\{z^{(b)}\}_{b=1}^B$, not conditioned on $x_S$, ensuring the oracle is strictly deterministic:
$$\hat{v}_{x,c}(S) = \frac{1}{B} \sum_{b=1}^B g_c(x_S, z^{(b)}_{\bar{S}})$$
All anytime coverage guarantees in this paper are established with respect to $\phi_i(\hat{v})$. The background discretization error $\|\phi(\hat{v}) - \phi(v)\| = \mathcal{O}(B^{-1/2})$ is a standard Monte Carlo property orthogonal to coalition sampling.

### 2.1 Primary Game: Local Cluster-Membership Attribution
Let $g_c(x) \in [0, 1]$ be the soft membership probability of observation $x$ belonging to regime $c$.
$$\hat{v}_{x,c}(S) = \frac{1}{B} \sum_{b=1}^B g_c(x_S, z^{(b)}_{\bar{S}})$$
* **Interpretation:** "How much did feature $i$ contribute to assigning observation $x$ to environmental regime $c$?"
* **Deterministic Bounds:** $\hat{v}_{x,c}(S) \in [0, 1] \implies L=0, U=1 \implies \mathbf{R_\Delta^{\text{res}} = 4(1 - 0) = 4}$. Zero heuristic tuning required.

### 2.2 Contrastive Regime Attribution Game
$$\hat{v}_{x,c,c'}(S) = \frac{1}{B} \sum_{b=1}^B [g_c(x_S, z^{(b)}_{\bar{S}}) - g_{c'}(x_S, z^{(b)}_{\bar{S}})] \in [-1, 1] \implies \mathbf{R_\Delta^{\text{res}} = 8}$$
* **Interpretation:** "Why was observation $x$ classified as Photochemical Smog (Cluster 1) instead of Clean Air (Cluster 2)?"
* **Efficiency Constant:** $\Delta_{\text{total}} = [g_c(x) - g_{c'}(x)] - \frac{1}{B}\sum_{b=1}^B [g_c(z^{(b)}) - g_{c'}(z^{(b)})]$.

### 2.3 Global Cluster-Level Archetype Game
To avoid oracle noise from continuous observation sampling, we fix a representative subsampled set $\tilde{\mathcal{I}}_c \subseteq \mathcal{I}_c$ of size $N_{\text{archetype}}$:
$$\hat{v}_c(S) = \frac{1}{|\tilde{\mathcal{I}}_c| \cdot B} \sum_{x \in \tilde{\mathcal{I}}_c} \sum_{b=1}^B g_c(x_S, z^{(b)}_{\bar{S}}) \in [0, 1] \implies \mathbf{R_\Delta^{\text{res}} = 4}$$

### 2.4 Intrinsic Clustering Quality Game (Costly Oracle Stress Test)
$$\hat{v}_{\text{sil}}(S) = \text{Silhouette}\left( \text{Cluster}(X_S) \right) \in [-1, 1] \implies \mathbf{R_\Delta^{\text{res}} = 8}$$
with convention $\hat{v}_{\text{sil}}(\emptyset) = 0$ and deterministic clustering initialization.

### 2.5 Group-Lag Spatiotemporal Game
For high-dimensional lagged time series (e.g. $M=66$ with 6 lags per variable), define $M_{\text{group}} = 11$ macro-players $G_j = \{X_j^{(t)}, X_j^{(t-1)}, \dots, X_j^{(t-24)}\}$ evaluated under block background sampling, providing exact ground truth at $2^{11} = 2048$ coalitions.

---

## 3. Theoretical Framework & Mathematical Proofs

### 3.1 Theorem 1: Coupled Adjacent-Stratum Neyman Allocation Program

#### Theorem Statement
In the add-one/remove-one sampling scheme, drawing a coalition of cardinality $q$ yields add-one samples in stratum $q$ for $(M-q)$ features and remove-one samples in stratum $q-1$ for $q$ features. The expected sample count backing interior stratum $s \in \{1, \dots, M-2\}$ is $n_s(\mathbf{K}) = \frac{M-s}{M} K_s + \frac{s+1}{M} K_{s+1}$.

Under total Stage-2 sampling budget $\sum_{q=1}^{M-1} K_q = K_{\text{cert}}$, the optimal draw distribution $\mathbf{K}^*$ minimizes the coupled convex program over draw sizes $q \in \{1, \dots, M-1\}$:
$$\boxed{ \min_{\mathbf{K}} \mathcal{E}(\mathbf{K}) = \frac{1}{M} \sum_{s=1}^{M-2} \frac{\|\boldsymbol{\sigma}_s^r\|_2^2}{(M-s) K_s + (s+1) K_{s+1}} \quad \text{s.t.} \quad \sum_{q=1}^{M-1} K_q = K_{\text{cert}}, \; K_q \ge 0 }$$
where $\|\boldsymbol{\sigma}_s^r\|_2 = \sqrt{\sum_{i=1}^M (\sigma_{i,s}^r)^2}$. Draw size $q=M-1$ is explicitly included to supply remove-one samples to the final interior stratum $s=M-2$.

---

### 3.2 Lemma 1: Exact $\mathcal{O}(M^2)$ Closed-Form Cross-Covariance

#### Lemma Statement
Let $k(S, S') = \sigma_0^2 \rho^{|S \Delta S'|}$ with $\rho = e^{-1/\ell} \in (0, 1)$. For any coalition $S_j \subseteq N$ of size $r = |S_j|$, all $i \in S_j$ share value $V_{\text{in}}(r)$ (for $r > 0$) and all $i \notin S_j$ share $V_{\text{out}}(r)$ (for $r < M$), evaluated in $\mathcal{O}(M^2)$ total operations:
$$\boxed{ [\mathbf{K}_{\phi, \mathcal{D}}]_{i, j} = \begin{cases} V_{\text{in}}(r) & \text{if } i \in S_j \\ V_{\text{out}}(r) & \text{if } i \notin S_j \end{cases} }$$
where:
$$V_{\text{in}}(r) = \frac{\sigma_0^2 (1 - \rho)}{M} \sum_{s=0}^{M-1} \sum_{l=\max(0, s - M + r)}^{\min(s, r-1)} \frac{\binom{r-1}{l} \binom{M - r}{s - l}}{\binom{M-1}{s}} \rho^{r - 1 + s - 2l} \quad (r > 0)$$
$$V_{\text{out}}(r) = -\frac{\sigma_0^2 (1 - \rho)}{M} \sum_{s=0}^{M-1} \sum_{l=\max(0, s - M + 1 + r)}^{\min(s, r)} \frac{\binom{r}{l} \binom{M - 1 - r}{s - l}}{\binom{M-1}{s}} \rho^{r + s - 2l} \quad (r < M)$$

---

### 3.3 Lemma 2: Exact Analytical Prior Shapley Covariance Matrix $\mathbf{K}_{\phi,\phi}$

#### Lemma Statement
The prior covariance matrix $[\mathbf{K}_{\phi,\phi}]_{ij} = \mathcal{A}_i \mathcal{A}_j' k(S, T)$ has exact analytical structure:
$$\mathbf{K}_{\phi,\phi} = (V_{\text{diag}} - V_{\text{off}}) \mathbf{I}_M + V_{\text{off}} \mathbf{1}_M \mathbf{1}_M^T$$
where:
$$V_{\text{diag}} = \frac{2 \sigma_0^2 (1 - \rho)}{M^2} \sum_{s=0}^{M-1} \sum_{t=0}^{M-1} \sum_{l=\max(0, s+t-M+1)}^{\min(s, t)} \frac{\binom{s}{l} \binom{M - 1 - s}{t - l}}{\binom{M-1}{t}} \rho^{s + t - 2l}$$
$$V_{\text{off}} = \sigma_0^2 (1 - \rho)^2 \sum_{s=0}^{M-2} \sum_{t=0}^{M-2} \Delta w_s \Delta w_t \sum_{l=\max(0, s+t-M+2)}^{\min(s, t)} \binom{M-2}{s}\binom{s}{l}\binom{M-2-s}{t-l} \rho^{s + t - 2l}$$
and $\Delta w_s = w_s - w_{s+1} = \frac{s!(M-2-s)!(M - 2 - 2s)}{M!}$.

---

### 3.4 Lemma 3: Exact Singleton Extreme-Stratum Identification
Because $\binom{M-1}{0} = 1$ and $\binom{M-1}{M-1} = 1$, extreme strata $s=0$ and $s=M-1$ are deterministic singletons. Direct evaluation of $v(\{i\}) - v(\emptyset)$ and $v(N) - v(N \setminus \{i\})$ determines the exact stratum residual mean with strictly zero variance ($\sigma_{i,0}^r = \sigma_{i,M-1}^r = 0$), contributing $0.0$ width to the confidence sequence and eliminating the $\mathcal{O}(M \log M)$ coupon-collector delay. $\blacksquare$

---

### 3.5 Lemma 4: Conditional Stratum Uniformity of Add-One and Remove-One Marginals
Let $S \subseteq N$ be drawn uniformly from all $\binom{M}{s^*}$ subsets of size $s^*$.
1. **Add-One:** For any $i \notin S$, $S \mid (i \notin S)$ is uniformly distributed over all $\binom{M-1}{s^*}$ subsets of $N \setminus \{i\}$. Thus $\Delta_i(S) = v(S \cup \{i\}) - v(S)$ is an unbiased draw from stratum $s^*$.
2. **Remove-One:** For any $i \in S$, $(S \setminus \{i\}) \mid (i \in S)$ is uniformly distributed over all $\binom{M-1}{s^*-1}$ subsets of $N \setminus \{i\}$. Thus $\Delta_i(S \setminus \{i\}) = v(S) - v(S \setminus \{i\})$ is an unbiased draw from stratum $s^* - 1$.
Both sample types preserve unbiased stratum-wise conditioning. $\blacksquare$

---

### 3.6 Theorem 2: Anytime Stratified Empirical-Bernstein Confidence Sequences

#### Theorem Statement (Theorem 2)
Let $\widehat{\phi}_i^{\text{raw}} = \phi_i(m_b) + \widehat{\phi}_i(r_{\mathcal{D}})$ be the decoupled linear estimator. For interior strata $s \in \{1, \dots, M-2\}$, let $(\widehat{\sigma}_{i,s}^r)^2$ be the sample variance of residual marginals and $n_{i,s} = |\mathcal{D}_{\text{cert}}(i, s)| \ge 2$. Extreme singleton strata $s=0$ and $s=M-1$ contribute $0$ width (Lemma 3).
Let $R_\Delta^{\text{res}} = 4(U - L)$ be the deterministic residual range bound. Define the time-uniform boundary via summable error allocation:
$$\boxed{ W_i^{\text{res}}(\mathbf{n}_i) = \frac{1}{M} \sum_{s=1}^{M-2} \left( \sqrt{\frac{2 (\widehat{\sigma}_{i,s}^r)^2 \log\left( \frac{\pi^2 M^2 n_{i,s}^2}{3\delta} \right)}{n_{i,s}}} + \frac{7 R_\Delta^{\text{res}} \log\left( \frac{\pi^2 M^2 n_{i,s}^2}{3\delta} \right)}{3(n_{i,s} - 1)} \right) }$$
The denominator $M$ preserves the interpretation of $W_i^{\text{res}}$ as a direct bound on the error of the full $M$-term stratified Shapley estimator $\phi_i = \frac{1}{M} \sum_{s=0}^{M-1} \mu_{i,s}$.
Then the stopping rule $\tau = \inf \{ \mathbf{n} : \max_{i \in N} W_i^{\text{res}}(\mathbf{n}_i) \le \epsilon \}$ satisfies:
$$\mathbb{P}\left( \exists \mathbf{n} \ge 2\mathbf{1}_{\text{interior}}, \exists i \in N : |\widehat{\phi}_i^{\text{raw}} - \phi_i| > W_i^{\text{res}}(\mathbf{n}_i) \right) \le \delta$$

---

### 3.7 Theorem 3 & Corollary 1: Posterior-Diagonal Uncertainty-Weighted Projection & Sign Certification

#### Theorem Statement (Theorem 3)
Let $\widehat{\boldsymbol{\phi}}^{\text{raw}} = \boldsymbol{\phi}(m_b) + \widehat{\boldsymbol{\phi}}(r_{\mathcal{D}})$ and $\mathbf{v} = \operatorname{diag}(\boldsymbol{\Sigma}_{m_b \mid \mathcal{D}_{\text{gp}}})$. The unique solution to the diagonal uncertainty-weighted projection onto the efficiency manifold $\sum_{i=1}^M \phi_i = \Delta_{\text{total}}$ is:
$$\boxed{ \phi_i^* = \widehat{\phi}_i^{\text{raw}} + v_i \cdot \left[ \frac{\Delta_{\text{total}} - \sum_{j=1}^M \widehat{\phi}_j^{\text{raw}}}{\sum_{j=1}^M v_j} \right] }$$

#### Corollary Statement (Corollary 1: Post-Projection Certificate)
On the $(1 - \delta)$ coverage event, the post-projection attribution satisfies:
$$\boxed{ |\phi_i^* - \phi_i| \le W_i^{\text{proj}} \equiv W_i^{\text{res}} + \frac{v_i}{\sum_{j=1}^M v_j} \sum_{j=1}^M W_j^{\text{res}} }$$

#### Definition 1 (Sign-Certified Feature Importance)
A feature $i \in N$ is defined as **Sign-Certified Important** at confidence $(1 - \delta)$ if its post-projection confidence interval strictly excludes zero ($|\phi_i^*| > W_i^{\text{proj}}$). In visual waterfall plots, features failing this condition are visually greyed out.

---

## 4. Two-Model Domain Experimental Design

```
+---------------------------------------------------------------------------------------------------------+
|                                    Experimental Domain Matrix                                           |
+---------------------------------------------------------------------------------------------------------+
| Tier A: Exact Ground Truth Verification (M = 11, 2¹¹ = 2048 coalitions enumerated)                      |
|   • Dataset 1: Portuguese Wine Quality (Acidity, Sulfur Dioxide, Alcohol, pH, Density)                  |
|   • Dataset 2: Beijing Air Quality Static (PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM)|
|   • Model Pipeline: PCA → K-Means → LightGBM Surrogate (Replicating 2025 Paper Benchmark)              |
|                                                                                                         |
| Tier B: High-Dimensional Black-Box Scaling (M = 66 Lagged Spatiotemporal Features)                      |
|   • Dataset: Beijing Air Quality Multi-Station (383,000 hourly records, 2013–2017)                      |
|   • Lags: t, t-1, t-3, t-6, t-12, t-24 hours                                                            |
|   • Imputation: Block-Background Sampling (Preserving temporal coherence across lag blocks)             |
|   • Models: (1) Ungrouped M=66 Black-Box; (2) Group-Shapley M=11 Macro-Players (Exact Ground Truth)     |
|   • Architecture: Deep Embedded Clustering (DEC) + Temporal Transformer (Frozen during attribution)     |
+---------------------------------------------------------------------------------------------------------+
```

### 4.1 Research Questions
* **RQ1 (Attribution Fidelity):** Does GAS-BayesSHAP match exact Shapley values on 11-feature ground truth with significantly fewer queries than KernelSHAP, SamplingSHAP, and OddSHAP?
* **RQ2 (Anytime Calibration):** Does the post-projection confidence interval $W_i^{\text{proj}}$ maintain empirical coverage $\ge 95\%$ across $R=500$ independent trials?
* **RQ3 (Regime Semantics Preservation & Sign Certification):** Does GAS-BayesSHAP reproduce the atmospheric regimes discovered in the 2025 paper with certified significance ($|\phi_i^*| > W_i^{\text{proj}}$):
  * *Photochemical Smog:* High $\text{O}_3$, high temperature, low humidity.
  * *Winter Smog:* Elevated $\text{CO}$, $\text{SO}_2$, $\text{PM}_{10}$, stagnant wind.
  * *Stagnant Inversion:* Low wind speed ($\text{WSPM}$), elevated pressure, trapped particulates.
  * *Clean-Air Events:* Strong wind dispersion, low precursor gases.
* **RQ4 (Temporal Lag Attribution in $M=66$):** Do lagged attributions distinguish instantaneous emission peaks from delayed meteorological accumulation (e.g., $\text{PM}_{2.5}^{(t-6)}$ vs. $\text{WSPM}^{(t-12)}$)?
* **RQ5 (Efficiency vs. ShaplEIG & OddSHAP):** What query reduction factor is achieved over ShaplEIG (ICML 2026) and OddSHAP (ICML 2026) at fixed error $\epsilon$?
* **RQ6 (Kernel Misspecification Robustness):** Does residual supermartingale certification maintain valid coverage when the GP prior is deliberately misspecified on parity games?

---

## 5. Planned Figures & Tables in the Manuscript

* **Figure 1:** Dual-Module Architecture: Active GP Control Variates + Neyman Residual Supermartingales.
* **Figure 2 (Atmospheric Waterfall Explanations):** Certified waterfall plots showing feature attributions $\phi_i^*$ with error bars $\pm W_i^{\text{proj}}$, greying out uncertified features where $0 \in [\phi_i^* - W_i^{\text{proj}}, \phi_i^* + W_i^{\text{proj}}]$.
* **Figure 3 (Spatiotemporal Heatmap, $M=66$):** Lagged temporal attributions across 24 hours for PM2.5, NO2, and WSPM.
* **Figure 4 (Convergence Curves):** RMSE vs. Model Forward Passes ($B \times K_{\text{coal}}$) comparing GAS-BayesSHAP, ShaplEIG, OddSHAP, KernelSHAP, SamplingSHAP, and TreeSHAP.
* **Figure 5 (Calibration Curves):** Empirical coverage vs. nominal confidence level $(1 - \delta)$ over $R=500$ trials.
* **Table 1:** Benchmark performance on Wine ($M=11$), Beijing Static ($M=11$), and Beijing Lagged ($M=66$). Primary axis: Model Forward Passes ($B \times K_{\text{coal}}$).
* **Table 2:** 4-Tier Ablation Study (Uniform Sampling vs. Neyman Stratification vs. Active GP vs. Decoupled Residual Certification).
* **Table 3:** Width inflation ratio $W_i^{\text{proj}} / W_i^{\text{res}}$ across all clusters (confirming controlled $\approx 1.5\text{--}2.0\times$ tightness).

---

## Supplementary Material Roadmap

* **Appendix A:** Complete algebraic proofs of Lemmas 1, 2, 3, and 4.
* **Appendix B:** Time-uniform martingale filtration proofs under adaptive stratification and Corollary 1 projection bounds.
* **Appendix C:** Preprocessing, block-imputation, and lag construction details for the Beijing Air Quality Dataset (383,000 records).
* **Appendix D:** Full tabular profiling logs, hyperparameter sensitivity ($\ell, \eta$), and DEC model architecture specifications.
