# Cover Letter — Precision Engineering (Elsevier)

---

[Date]

**To the Editor-in-Chief,**  
*Precision Engineering*  
Elsevier

Dear Editor,

We submit for your consideration the manuscript entitled:

**"Data Quality-Aware Heteroscedastic Gaussian Process Surrogate  
for 4H-SiC Blade Dicing Process Optimization"**

---

## Why Precision Engineering

This work addresses a core challenge in precision machining: how to build accurate
process surrogates when experimental data comes from multiple sources with different
measurement quality. We demonstrate the problem and solution in the context of
4H-SiC blade dicing — a high-impact process for power semiconductor manufacturing —
making it directly relevant to the scope of *Precision Engineering*.

---

## Novel Contributions

**The central finding is counter-intuitive and practically significant:**

> 11 high-quality datapoints (direct SEM measurement, Grade A)
> outperform 25 mixed-quality datapoints (LOO R² = 0.80 vs 0.34)
> in predicting front-side chipping depth.

This quantifies, for the first time in the SiC dicing literature, that
**data quality stratification dominates data quantity** for small-n
Gaussian process surrogates. The implication for industrial practice is direct:
investing in a smaller number of carefully measured experiments is more valuable
than pooling data from the literature without quality control.

The paper further contributes:
- A heteroscedastic GP formulation with externally-prescribed per-point noise
  (as opposed to learned noise, which fails at small n)
- An Ensemble Kalman Filter for real-time blade wear estimation, reducing
  online RMSE by 10% compared to GP alone
- Bayesian optimization identifying a sub-15 µm chipping window within 20 evaluations
- Full open-source implementation (github.com/keisuke58/wafer-proc-sim)
  enabling reproducible research in SiC dicing process optimization

---

## Broader Impact

4H-SiC power devices are essential for electric vehicle inverters, grid-scale energy
storage, and LEO satellite power conditioning — markets with combined projected size
exceeding $28B by 2030 (Yole Développement, 2024). Our surrogate framework reduces
the experimental cost of process optimization for these high-value devices.

The quality-aware GP methodology is not specific to SiC dicing and generalizes
immediately to any precision manufacturing process where data is aggregated
from heterogeneous sources — CMP, etching, grinding, and laser dicing among others.

---

## Suggested Reviewers

1. **Prof. [Name]** — [University] — expertise: GP surrogates for manufacturing
   [find from Google Scholar: GP surrogate manufacturing]
2. **Prof. [Name]** — [University] — expertise: SiC dicing / brittle material machining
   [find from: Micro2026 or Mat2022 citing papers]
3. **Dr. [Name]** — expertise: Bayesian optimization for semiconductor processes
   [find from: arXiv:2511.23141 authors]

*(Fill in actual names before submission — use Google Scholar to identify 3 active reviewers)*

---

## Statement

This manuscript is original work, has not been published previously,
and is not under consideration for publication elsewhere.
All authors have approved the manuscript for submission.
The code and data supporting this work are publicly available at
github.com/keisuke58/wafer-proc-sim.

We believe this manuscript makes a focused and reproducible contribution
to precision manufacturing science and is well-suited for *Precision Engineering*.

Sincerely,

**Keisuke Nishioka**  
[Department], [University]  
kei128608@gmail.com

---

*Enclosures: Manuscript (PDF), Figures 1–6 (TIFF 300 dpi), Highlights*
