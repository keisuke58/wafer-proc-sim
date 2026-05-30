# wafer-proc-sim

**Physics-informed ML for SiC wafer dicing process optimisation**

Keisuke Nishioka · Keio University / Leibniz Universität Hannover

> End-to-end pipeline from ABAQUS FEM → Gaussian Process surrogate → TMCMC Bayesian inference. Validated against open-access experimental data (Micro2026, Mat2022). Targeting DISCO / KABRA® process engineering roles.

---

## Status

| Component | Status | Key result |
|-----------|--------|------------|
| Experimental data (Micro2026 + Mat2022) | ✅ Done | 18 data points, 4 features |
| 4-feature GP surrogate | ✅ Done | LOO-RMSE = 2.81 µm, R² = 0.58 |
| Sensitivity analysis (Sobol) | ✅ Done | depth > feed >> blade_W >> spindle |
| Pareto optimisation (chipping vs MRR) | ✅ Done | 97.5% of parameter space safe |
| TMCMC calibration | ✅ Done | MAP error < 2% vs ground truth |
| 2D FEM — shallow cuts (20–60 µm) | ✅ Done | Drucker-Prager, 15-job sweep |
| 2D FEM — deep cuts (80–360 µm) | 🔄 Running | Contact fix + fracture model enabled |
| GP re-training with FEM data | ⏳ After FEM | FEM + experimental data fusion |

---

## Key Results

### Sensitivity Analysis (Sobol indices, N = 4096)

| Parameter | First-order S_i | Total-effect S_Ti |
|-----------|-----------------|-------------------|
| Cut depth | **0.779** | **0.778** |
| Feed speed | 0.210 | 0.251 |
| Blade width | 0.011 | 0.018 |
| Spindle speed | 0.000 | 0.000 |

→ Depth dominates chipping (78% of variance). Feed is the secondary lever. Spindle speed is negligible.

### TMCMC Calibration

Given observed chipping = 10.0 µm (Micro2026 reference: depth=390 µm, feed=1.0 mm/s):

| | Ground truth | Posterior MAP |
|--|---|---|
| Cut depth | 390 µm | 382 µm (–2%) |
| Feed speed | 1.0 mm/s | 1.06 mm/s (+6%) |

### Safe Operating Region

97.5% of the depth × feed parameter space satisfies chipping < 15 µm (production threshold) at blade_W = 23 µm, spindle = 30 krpm. Pareto front reaches depth = 380 µm at max feed while staying below the threshold.

---

## Pipeline

```
Experimental data          ABAQUS FEM
(Micro2026, Mat2022)   ←→  2D plane-strain, CPE4R
        │                   Drucker-Prager plasticity
        ▼                   DuctileDamageInitiation
 GP Surrogate               STATUS + RF field outputs
 [cut_depth, blade_W,           │
  feed, spindle] → chipping     ▼
        │               parametric_summary.csv
        ├── Sensitivity analysis (Sobol)
        ├── Pareto front (chipping vs MRR)
        └── TMCMC inference
            [depth, feed] ← observed chipping
```

---

## Repository Structure

```
wafer-proc-sim/
├── data/materials/
│   └── material_properties.py   # Si, SiC (4H), GaN — elastic, DP, fracture
├── fem/
│   ├── dicing_blade_2d.py       # ABAQUS/Explicit 2D parametric study
│   │                             # Drucker-Prager + DuctileDamageInitiation
│   ├── dicing_blade_3d.py       # 3D coarse validation model
│   └── kabra_thermal_2d.py      # TAIKO® back-grinding thermal model
├── ml/
│   ├── surrogate_gp.py          # GP surrogate (FEM output: deletion_fraction)
│   ├── train_from_experimental.py  # GP trained on experimental chipping data
│   ├── sensitivity_analysis.py  # Sobol indices + gradient sensitivity
│   └── surrogate_fno.py         # FNO stress-field surrogate (planned)
├── optimization/
│   ├── bayesian_opt.py          # Expected Improvement + constraint
│   ├── pareto_front.py          # Chipping vs MRR Pareto curve
│   └── tmcmc_dicing.py          # TMCMC: infer process params from observation
├── validation/
│   └── experimental_data.py     # Digitised Micro2026 + Mat2022 chipping data
├── notebooks/
│   └── demo_sic_dicing.ipynb    # End-to-end demo: data → GP → TMCMC
└── results/                     # Trained models + plots (committed)
    ├── gp_experimental.pkl
    ├── gp_experimental_sweeps.png
    ├── gp_experimental_heatmap.png
    ├── sensitivity_analysis.png
    ├── pareto_front.png
    └── tmcmc_exp_calibrate_posterior.png
```

---

## Quickstart

```bash
git clone https://github.com/keisuke58/wafer-proc-sim.git
cd wafer-proc-sim
pip install scikit-learn joblib numpy pandas matplotlib scipy

# Train GP on experimental data
python ml/train_from_experimental.py --loo --plot

# Sensitivity analysis
python ml/sensitivity_analysis.py --plot

# Pareto front (chipping vs MRR)
python optimization/pareto_front.py --plot

# TMCMC calibration: infer (depth, feed) from observed chipping
python -c "
from optimization.tmcmc_dicing import calibrate_experimental
calibrate_experimental(observed_chip_um=10.0, n_samples=1000)
"

# Run the demo notebook
jupyter notebook notebooks/demo_sic_dicing.ipynb
```

### ABAQUS FEM (requires licence)

```bash
# Deep-cut parametric study (80–360 µm, blade_W=23 µm, Micro2026 conditions)
cd runs/extended_sic
bash submit_jobs.sh

# Extract results after completion
abaqus python ../../runs/parametric_sic/run_extract.py
```

---

## Materials

| Material | E [GPa] | K_Ic [MPa√m] | G_c [J/m²] | σ_t [MPa] |
|----------|---------|--------------|-----------|-----------|
| 4H-SiC   | 400     | 2.8          | 19.6      | 350       |
| Si       | 130     | 0.83         | 5.3       | 150       |
| GaN      | 295     | 0.9          | 2.7       | 100       |

Fracture model: Drucker-Prager pressure-dependent plasticity + energy-based damage evolution calibrated from K_Ic (Irwin: G_c = K_Ic²/E).

---

## References

| # | Citation |
|---|---------|
| 1 | Huang et al., *Micromachines* 17(2):187, 2026 — [DOI:10.3390/mi17020187](https://doi.org/10.3390/mi17020187) — experimental data source |
| 2 | Zhang et al., *Materials* 15(22):8083, 2022 — [DOI:10.3390/ma15228083](https://doi.org/10.3390/ma15228083) — experimental data source |
| 3 | Ching & Chen, *J. Eng. Mech.* 133(7):816–832, 2007 — TMCMC algorithm |
| 4 | Saltelli et al., *Comp. Phys. Comm.* 181(2):259–270, 2010 — Sobol estimator |
| 5 | Rasmussen & Williams, *Gaussian Processes for ML*, MIT Press, 2006 |
| 6 | DISCO Corporation, TAIKO® process technical notes |

---

*Contact: k.nishioka@stud.uni-hannover.de*
