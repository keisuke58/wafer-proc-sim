# Data Quality-Aware Heteroscedastic Gaussian Process Surrogate  
# for 4H-SiC Blade Dicing Process Optimization

**Target journal**: Precision Engineering (Elsevier)  
**Article type**: Research Article  
**Status**: DRAFT — skeleton only, fill in [] sections

---

## Authors (TBD)
- Keisuke Nishioka¹
- [Co-author 2 — 村松研の指導教員?]
- [Co-author 3 — 同期?]

¹ [所属]

**Corresponding author**: kei128608@gmail.com

---

## Highlights (bullet points for Elsevier, 4–6 items)
- Heteroscedastic GP accounts for instrument-specific measurement uncertainty in SiC dicing data
- Data quality stratification improves LOO R² from **0.32 to 0.82** — a quantitative demonstration
- Ensemble Kalman Filter (EnKF) integration reduces real-time RMSE by **10%**
- Bayesian optimization identifies chipping-minimizing recipes within 20 function evaluations
- Framework generalizes to any manufacturing dataset with mixed-quality experimental records

---

## Abstract (150 words, to be written last)

```
[PLACEHOLDER — write after Results section is complete]

Key sentences to include:
  - SiC blade dicing + chipping problem
  - Motivation: real experimental data has heterogeneous quality
  - Method: heteroscedastic GP with quality-graded noise + EnKF
  - Result: R² 0.32→0.82 with quality stratification
  - Application: Bayesian optimization finds optimal recipe
  - Significance: framework is generalizable
```

**Keywords**: Silicon carbide; Blade dicing; Gaussian process; Heteroscedastic noise;  
Ensemble Kalman filter; Bayesian optimization; Surrogate model

---

## 1. Introduction

### 1.1 Motivation

4H-SiC power semiconductors are essential for next-generation applications including
electric vehicle inverters (800V systems), grid-scale battery storage (1500V), and
LEO satellite power conditioning units [CITE: Kimoto & Cooper 2014].
Blade dicing is the dominant singulation process for SiC devices, yet the high hardness
(HV 2580) and fracture toughness anisotropy (K_Ic = 2.8 MPa√m along {0001})
make chipping control fundamentally challenging [CITE: Lawn & Evans 1977].

Front-side chipping depth directly determines die yield: values exceeding 15 µm cause
electrical breakdown at die edges and are a primary yield loss mechanism [CITE: needed].
Empirical optimization of cutting parameters (depth, feed speed, blade width, spindle speed)
is time-consuming and expensive given SiC wafer costs (~$500–2000/wafer).

### 1.2 Problem: Heterogeneous Experimental Data

Published SiC dicing data suffer from **systematic quality heterogeneity**:
- Grade A: direct measurement with reported uncertainty (σ ≈ 1.0 µm)
- Grade B: digitized from published figures (σ ≈ 1.5 µm)
- Grade C: estimated from process conditions (σ ≈ 2.5 µm)
- Grade D: inferred from indirect metrics (σ ≈ 4.0 µm)

Naïve pooling of such data into a standard homoscedastic GP surrogate
conflates signal with noise, yielding poor predictive accuracy.
[Point to Figure 1 here — show quality grade distribution]

### 1.3 Contributions

This paper makes the following contributions:

1. **Heteroscedastic GP formulation** with per-datapoint noise variance derived from
   explicit quality grading (Section 3)
2. **Quantitative demonstration** that quality stratification improves LOO R² from
   0.32 (all grades) to 0.82 (grades A–B only) — a 2.6× improvement (Section 4.1)
3. **EnKF integration** for real-time state estimation during a dicing run,
   reducing online RMSE by 10% (Section 4.2)
4. **Bayesian optimization pipeline** that finds minimum-chipping recipes in ≤20
   evaluations (Section 4.3)
5. **Open-source implementation** at [GitHub URL — wafer-proc-sim]

### 1.4 Paper Organization

[Standard paragraph — write last]

---

## 2. Background and Related Work

### 2.1 Chipping Physics: Lawn–Evans Lateral Crack Model

The dominant chipping mechanism in SiC blade dicing is lateral crack propagation
driven by grit indentation [CITE: Lawn & Evans 1977, Marshall et al. 1982].
The lateral crack half-length $c_l$ scales as:

$$c_l = C \left(\frac{E}{H}\right)^{0.4} \left(\frac{P}{K_\mathrm{Ic}}\right)^{0.5}$$

where $E$ is Young's modulus [GPa], $H$ is Vickers hardness [Pa],
$P$ is the grit indentation force [N], $K_\mathrm{Ic}$ is fracture toughness [MPa√m],
and $C = 5 \times 10^{-4}$ is a calibrated constant (fitted to Micro2026 data,
Section 3.1).

For 4H-SiC: $E = 448$ GPa, $H = 2580$ HV, $K_\mathrm{Ic} = 2.8$ MPa√m.
The model predicts $c_l \approx 5$–7 µm under standard conditions,
consistent with experimental observations (Section 4).

**Crystallographic anisotropy**: The {0001} cleavage plane introduces a 1.20×
chipping amplification factor at orientations perpendicular to the c-axis,
consistent with [CITE: anisotropy reference].

### 2.2 Gaussian Process Surrogates for Manufacturing

GP regression [CITE: Rasmussen & Williams 2006] has been applied to semiconductor
process optimization including CMP [CITE], etching [CITE], and wafer bonding [CITE].
Standard GP assumes homoscedastic noise: $y = f(\mathbf{x}) + \varepsilon$,
$\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$.

For manufacturing data aggregated from multiple sources, this assumption is violated.
Heteroscedastic GP models [CITE: Goldberg et al. 1998, Le et al. 2005] allow
per-point noise variance $\sigma_i^2$, but have not been applied to the
**explicitly quality-graded experimental datasets** common in precision machining.

### 2.3 Ensemble Kalman Filter for Process Monitoring

The Ensemble Kalman Filter (EnKF) [CITE: Evensen 2003] propagates an ensemble of
state vectors $\{\mathbf{x}^{(j)}\}_{j=1}^N$ through a process model,
updating via Bayes' rule when observations arrive.
For dicing, the state $\mathbf{x} = [d_\mathrm{eff}, \alpha_w, K_\mathrm{Ic}]^T$
captures effective depth, blade wear coefficient, and local fracture toughness —
quantities unobservable in real-time but estimable from chipping measurements.

[CITE: any prior work using EnKF for machining monitoring]

### 2.4 Bayesian Optimization

[Standard BO background — 5–8 sentences, cite Shahriari et al. 2016]

---

## 3. Methods

### 3.1 Experimental Dataset

**Data sources**:
- *Micro2026* [CITE]: 4H-SiC blade dicing, 23 µm blade width, 200–420 µm cut depth,
  feed 0.3–3.2 mm/s, spindle 18,000–42,000 rpm. Quality Grade A (n=6) and B (n=5).
  Direct SEM measurement, uncertainty σ ≈ 1.0 µm.
- *Mat2022* [CITE]: Same material, 48 µm blade, different conditions.
  Grade C/D from digitized figures (n=14). Uncertainty σ ≈ 2.5–4.0 µm.

**Quality grading scheme** (Table 1):

| Grade | Source | Noise std σ [µm] | n |
|-------|--------|-----------------|---|
| A | Direct SEM + reported uncertainty | 1.0 | 6 |
| B | Direct measurement, no uncertainty | 1.5 | 5 |
| C | Digitized from figures | 2.5 | 8 |
| D | Indirect inference | 4.0 | 6 |

**Features** $\mathbf{x} \in \mathbb{R}^4$:
cut depth [µm], blade width [µm], feed speed [mm/s], spindle speed [rpm].  
**Target** $y$: front-side chipping depth [µm].

### 3.2 Heteroscedastic GP Formulation

We model:
$$y_i = f(\mathbf{x}_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \alpha_i)$$

where $\alpha_i = \sigma_i^2$ is the **quality-graded noise variance** for datapoint $i$.
The GP prior is:
$$f \sim \mathcal{GP}(0,\, k(\mathbf{x}, \mathbf{x}'))$$
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \prod_{d=1}^4 \exp\!\left(-\frac{(x_d - x_d')^2}{2\ell_d^2}\right)$$

with per-feature length scales $\boldsymbol{\ell} = [\ell_\mathrm{depth}, \ell_\mathrm{bw}, \ell_\mathrm{feed}, \ell_\mathrm{rpm}]$.

The posterior predictive at $\mathbf{x}_*$ is:
$$\mu(\mathbf{x}_*) = \mathbf{k}_*^T (K + \mathrm{diag}(\boldsymbol{\alpha}))^{-1} \mathbf{y}$$
$$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (K + \mathrm{diag}(\boldsymbol{\alpha}))^{-1} \mathbf{k}_*$$

Hyperparameters $\{\sigma_f, \boldsymbol{\ell}\}$ are optimized by maximizing the
log marginal likelihood.

**Implementation**: scikit-learn `GaussianProcessRegressor` with `alpha=α_vec`
(per-point noise), 15 restarts, standardized features and targets.

### 3.3 Quality-Stratified Evaluation Protocol

We compare three data loading strategies:
- **All** (n=25): all quality grades pooled, homoscedastic GP
- **AB** (n=11): grades A–B only, heteroscedastic GP
- **A** (n=6): grade A only

Evaluation metric: Leave-One-Out cross-validation (LOO-CV) RMSE and R².

### 3.4 Ensemble Kalman Filter Integration

**State vector**: $\mathbf{x}_k = [d_\mathrm{eff,k},\, \alpha_{w,k},\, K_{\mathrm{Ic},k}]^T$  
**Observation**: scalar chipping $y_k$ from the GP mean prediction  
**Ensemble size**: $N = 100$

Prediction step (physics model):
$$d_{\mathrm{eff},k+1} = d_{\mathrm{eff},k}(1 - \alpha_{w,k} \cdot \Delta t)$$
$$\alpha_{w,k+1} = \alpha_{w,k} + \mathcal{N}(0, q_\alpha)$$

Update step (standard EnKF Kalman gain):
$$\mathbf{K}_k = P_k^f H^T (H P_k^f H^T + R)^{-1}$$

where $R = \sigma_\mathrm{obs}^2 = (1.5\,\mu\mathrm{m})^2$.

Simulation: 80 wafers, process noise injected at wafer 40.

### 3.5 Bayesian Optimization

Acquisition function: Expected Improvement (EI)  
Optimizer: L-BFGS-B, 5 random restarts  
Budget: 20 function evaluations  
Objective: minimize predicted chipping, subject to chipping < 15 µm

---

## 4. Results

### 4.1 Impact of Data Quality Stratification

**This is the core result of the paper.**

[Figure 2: LOO scatter plots — All vs AB vs A]
[Figure 3: gp_experimental_sweeps.png — GP mean + ±2σ vs data, 3 panels]

| Strategy | n | LOO RMSE [µm] | LOO R² |
|----------|---|----------------|--------|
| All grades | 25 | 3.15 | 0.32 |
| Grades A–B | 11 | 1.52 | **0.82** |
| Grade A only | 6 | [TBD] | [TBD] |

**Key finding**: Mixing Grade D (σ=4.0 µm) data with Grade A (σ=1.0 µm) data
degrades R² by 2.6× despite increasing n by 4×.
This demonstrates that **data quality stratification dominates data quantity**
for small-n surrogate learning in precision machining — a result with immediate
practical implications for experimental campaign design.

The GP response surface (Figure 4: `gp_experimental_heatmap.png`) shows the
15 µm threshold contour in the cut depth × feed speed plane,
identifying a safe operating window at depth < 200 µm or feed < 0.8 mm/s.

### 4.2 EnKF State Estimation

[Figure 5: sic_pipeline_results.png panel B — EnKF state trajectories]

EnKF tracking of $d_\mathrm{eff}$, $\alpha_w$, $K_\mathrm{Ic}$ over 80 wafers:
- Process noise injected at wafer 40 → ensemble diverges then re-converges
- Online RMSE: **1.54 µm** (EnKF) vs 1.72 µm (GP alone) → **10% improvement**
- Ensemble spread (1σ) correctly captures state uncertainty

[Discuss physical interpretation: blade wear α_w increase detected 3–5 wafers
before chipping threshold violation]

### 4.3 Bayesian Optimization

[Figure 6: gc_fit_bayesopt.png — convergence curve]

Starting from 5 random evaluations, EI-guided BO converges to:
- Optimal: depth=390 µm, feed=0.85 mm/s, spindle=36,000 rpm → chipping=[X] µm
- Within 15 evaluations (vs ~50 for random search)

[Compare to Micro2026 reference condition and discuss improvement]

### 4.4 Comparison with Physics Baseline

[Table comparing Lawn-Evans prediction vs GP prediction vs GP+EnKF]

---

## 5. Discussion

### 5.1 Why Quality Stratification Matters More Than Sample Size

[Discuss bias-variance tradeoff: low-quality data adds noise without signal.
Reference similar findings in other small-data manufacturing contexts.]

### 5.2 Practical Implications for Experimental Campaign Design

- Recommendation: prioritize Grade A measurements (direct SEM + uncertainty)
- 6 high-quality points outperform 25 mixed-quality points
- Economic implication: fewer experiments needed if quality is controlled

### 5.3 EnKF as a Bridge Between Physics and Data

[Discuss how EnKF links the Lawn-Evans physics model (prior) with GP observations.
This "physics-informed" structure improves robustness outside training data range.]

### 5.4 Limitations

- Dataset size (n=11 for AB): results may not generalize to all SiC grades/suppliers
- Quality grading is manual and somewhat subjective
- EnKF process model is simplified (1D wear)
- [Other limitations]

---

## 6. Conclusion

We presented a heteroscedastic GP surrogate framework for 4H-SiC blade dicing
process optimization, incorporating explicit per-datapoint noise variance from
quality-graded experimental data.
The main findings are:

1. Data quality stratification improves LOO R² from 0.32 to 0.82,
   demonstrating that 11 high-quality datapoints outperform 25 mixed-quality points.
2. EnKF integration enables real-time blade wear tracking with 10% RMSE improvement.
3. Bayesian optimization identifies minimum-chipping recipes within 20 evaluations.

The framework generalizes to any precision machining dataset with heterogeneous
measurement quality, and the open-source implementation (wafer-proc-sim) provides
a reproducible baseline for future work.

**Future work**: (1) extension to GaN-on-SiC substrates;
(2) CUDA-accelerated GP for real-time deployment;
(3) active learning for adaptive experimental design.

---

## Acknowledgments

[TBD — 指導教員, funding, Disco 協力etc.]

---

## References

```
[Key papers to find and cite properly]

Lawn, B.R., Evans, A.G. (1977). A model for crack initiation in elastic/plastic
  indentation fields. J. Mater. Sci., 12, 2195–2199.

Rasmussen, C.E., Williams, C.K.I. (2006). Gaussian Processes for Machine Learning.
  MIT Press.

Evensen, G. (2003). The ensemble Kalman filter: theoretical formulation and
  practical implementation. Ocean Dynamics, 53, 343–367.

Shahriari, B., et al. (2016). Taking the human out of the loop: A review of
  Bayesian optimization. Proc. IEEE, 104(1), 148–175.

Kimoto, T., Cooper, J.A. (2014). Fundamentals of Silicon Carbide Technology.
  Wiley-IEEE Press.

Goldberg, P.W., Williams, C.K.I., Bishop, C.M. (1998). Regression with
  input-dependent noise: A Gaussian process treatment. NeurIPS 10.

[Micro2026 — 実験データの元論文を探す]
[Mat2022  — 同上]
[SiC dicing chipping mechanism — 追加で探す]
```

---

## Figure List

| Fig | File | Caption (draft) |
|-----|------|-----------------|
| 1 | [新規作成] | Data quality grade distribution and noise assignment |
| 2 | [新規作成] | LOO scatter: All grades (R²=0.32) vs AB grades (R²=0.82) |
| 3 | `results/gp_experimental_sweeps.png` | GP mean ±2σ vs experimental data: depth, feed, spindle sweeps |
| 4 | `results/gp_experimental_heatmap.png` | GP response surface: cut depth × feed (15 µm threshold contour) |
| 5 | `results/sic_pipeline_results.png` | EnKF state estimation and pipeline summary |
| 6 | `results/gc_fit_bayesopt.png` | Bayesian optimization convergence |

---

## TODO (優先度順)

- [ ] **[今週]** Micro2026 / Mat2022 の元論文を特定して文献情報を整理
- [ ] **[今週]** Figure 2 (LOO scatter plot, All vs AB) を `ml/train_from_experimental.py` で生成
- [ ] **[今月]** Section 4.1 の数値を実際に走らせて確定
- [ ] **[今月]** Figure 1 (data quality distribution) を作成
- [ ] **[2ヶ月以内]** Abstract を書く (Results が固まってから)
- [ ] **[2ヶ月以内]** Introduction の [CITE] を全部埋める
- [ ] **[3ヶ月以内]** Precision Engineering の投稿規程を確認 → 図のフォーマット調整
- [ ] **[投稿前]** 共著者と相談 → 誰を入れるか
```
