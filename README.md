# wafer-proc-sim

**Physics-informed ML for SiC wafer process simulation — full front-end to back-end pipeline**

Keisuke Nishioka · Keio University / Leibniz Universität Hannover

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wafer-proc-sim-o6bmowre8nbaodgkshj6ej.streamlit.app/)

> End-to-end pipeline: ABAQUS FEM → GP surrogate → TMCMC Bayesian inference → TEL/Disco/ASML/Advantest device models → OSAT back-end → quantitative validation. Targeting DISCO / TEL / semiconductor process engineering roles.

---

## Status

### Front-End (ダイシング・研削)

| Component | Status | Key result |
|---|---|---|
| Experimental data (Micro2026 + Mat2022) | ✅ | 26 data points, 4 features |
| 4-feature GP surrogate | ✅ | LOO-RMSE = 2.56 µm, R² = 0.55 |
| Fusion GP (exp + FEM) | ✅ | LOO-RMSE = 2.38 µm, R² = 0.64 (+16%) |
| Sensitivity analysis (Sobol) | ✅ | depth 78%, feed 25%, blade_W ≪ |
| Pareto optimisation (chipping vs MRR) | ✅ | 97.5% of parameter space safe |
| TMCMC calibration | ✅ | MAP error < 2% vs ground truth |
| 2D/3D FEM blade dicing | ✅ | Drucker-Prager + damage, 5 jobs 80–360 µm |
| FNO surrogate | ✅ | 0.07 ms/field, 6000× faster than FEM |
| Laser grooving FEM | ✅ | ns/ps/fs regimes, Beer-Lambert ablation |
| Plasma Bosch model | ✅ | ARDE + pulsed plasma + Weibull + Low-k |
| Hybrid NSGA-II optimisation | ✅ | 4-objective, delamination constraint |
| 2nm thin-wafer validation | ✅ | ns fails / ps+fs pass @ 50 µm |

### Fab Process (TEL / ASML)

| Component | Status | Key result |
|---|---|---|
| TEL ALD model | ✅ | GPC, ALD window, EOT vs cycles |
| TEL Deal-Grove (SiC oxidation) | ✅ | 100× slower than Si, Arrhenius |
| TEL RIE damage model | ✅ | CF₄/SF₆/Cl₂, damage depth → Dit |
| TEL CMP + Lasertec inspection | ✅ | Preston eq. + defect detection |
| **TEL Cleaning (CELLESTA)** | ✅ | Post-CMP / Post-Dicing / Pre-Gate, H₂水追加, Dit → µ_ch |
| ASML EUV exposure | ✅ | Aerial image, flare, NILS, CD budget |
| SiC MOSFET pipeline | ✅ | Disco → TEL → Advantest 一気通貫 |

### Test / Ecosystem

| Component | Status | Key result |
|---|---|---|
| Advantest ATE model | ✅ | V_th/R_on/BV_DSS test, yield, CPGD |
| Disco DCF valuation | ✅ | 3-scenario, sensitivity (Disco 6146) |
| Semiconductor ecosystem | ✅ | NVIDIA/TSMC/Samsung/Kioxia/ソシオネクスト |
| Quantum defect models | ✅ | ODMR / NEGF / Keldysh / Lindblad / QEC |

### Back-End / Validation (後工程・検証)

| Component | Status | Key result |
|---|---|---|
| **Wire bonding model** | ✅ | Au/Cu/Al Weibull + IMC (Breach 2004) + Coffin-Manson |
| **Package stress model** | ✅ | Timoshenko warpage + Suhir CTE + Engelmaier fatigue |
| **Quantitative validation** | ✅ | 5 モデル vs 文献: RMSE / MAPE / R² / KS 統計 |

---

## Key Results

### Validation Summary (Sim vs Literature)

| Model | RMSE | MAPE | R² | Status |
|---|---|---|---|---|
| Blade Chipping | 2.35 µm | 14.3% | 0.978 | PASS |
| Wire Bond Weibull | KS=0.23 | η err 5.4% | — | PASS |
| Au-Al IMC Growth | 0.005 µm | **2.7%** | 0.9997 | PASS |
| Package Warpage | 1.9 µm | **6.1%** | 0.9998 | PASS |
| SiC µ_ch vs Dit | — | 133% | 0.64 | WARN* |

*WARN: 既存モデルはバルク移動度計算。反転チャンネル移動度 (20–80 cm²/Vs) には界面散乱モデルの追加が必要。

### Cleaning → Dit → µ_ch Pipeline

| Sequence | Carbon removal | Metal | Particle | Dit contrib. |
|---|---|---|---|---|
| Post-CMP (H₂水+Megasonic) | 64.7% | 99.9% | 100.0% | 1.6e+12 |
| Post-Dicing (H₂水+Megasonic) | 60.8% | 99.9% | 99.9% | 1.8e+12 |
| Pre-Gate Si (RCA) | 90.3% | 99.4% | 95.5% | 2.0e+12 |
| **Pre-Gate SiC (Piranha+HF+SC2+O₃+HF)** | **100.0%** | **99.7%** | 91.9% | **7.2e+09** |

SiC 最適化シーケンスで Dit を Si RCA 比 **1/280** に低減。

### Wire Bonding / Package (Au wire / AlN substrate)

| Metric | Value |
|---|---|
| Pull strength (mean) | 6.6 g (Weibull η=7.0, β=8.5) |
| IMC thickness @ 125°C/1000h | 0.293 µm (Breach 2004 実測: 0.29 µm ✅) |
| Heel crack fatigue | ~50,000 cycles @ ΔT=100K |
| Parasitic inductance | 0.986 nH (2mm loop, 200µm height) |
| Die stress (AlN substrate) | 42.7 MPa (CTE mismatch 0.5 ppm/K) |
| Solder fatigue (Cu DBC) | 600k cycles @ ΔT=100K |

### GP Surrogate Performance

| Version | N | LOO-RMSE | R² |
|---|---|---|---|
| v1 | 18 | 2.81 µm | 0.58 |
| v2 | 26 | 2.56 µm | 0.55 |
| fusion | 26+5 | **2.38 µm** | **0.64** |

---

## Full Pipeline

```
[Front-End]
  Experimental data (Micro2026/Mat2022)
  ABAQUS FEM (2D/3D blade dicing, laser grooving, plasma Bosch)
        ↓
  GP Surrogate + FNO → Sobol sensitivity → Pareto (chipping vs MRR)
  TMCMC calibration → real-time recipe correction

[Fab Process]
  ASML EUV exposure
        ↓
  TEL CMP → 【TEL 洗浄 Post-CMP】
        ↓
  Disco Dicing → 【TEL 洗浄 Post-Dicing】
        ↓
  【TEL 洗浄 Pre-Gate】→ TEL Deal-Grove → TEL ALD (HfO₂/Al₂O₃)
        ↓
  SiC MOSFET (Dit → µ_ch → V_th → R_on)
        ↓
  Advantest ATE (yield, CPGD, wafer map)

[Back-End / OSAT]
  Die attach (Ag sinter / SAC305)
        ↓
  Wire bonding (Au/Cu/Al) → IMC growth → Weibull pull strength
        ↓
  Package stress (Timoshenko warpage, Engelmaier fatigue)
        ↓
  Thermal cycling reliability → MTTF

[Validation]
  Sim vs Micro2026 / Breach2004 / Kimoto2014 / MIL-STD-883 / JEDEC JEP95
```

---

## Repository Structure

```
wafer-proc-sim/
├── fem/
│   ├── dicing_blade_2d.py          # ABAQUS 2D parametric: Drucker-Prager + damage
│   ├── dicing_blade_3d.py          # 3D validation model
│   ├── grinding_warpage_2d/3d.py   # SiC back-grinding warpage
│   ├── kabra_thermal_2d.py         # KABRA® TAIKO® thermal model
│   ├── laser_groove_thermal_2d.py  # Laser grooving: ns/ps/fs Beer-Lambert
│   ├── plasma_bosch_model.py       # Bosch DRIE: ARDE + pulsed plasma
│   ├── tel_process_model.py        # ALD / Deal-Grove / RIE → SiC MOSFET µ_ch
│   ├── tel_cmp_lasertec.py         # CMP Preston eq. + Lasertec inspection
│   ├── tel_cleaning_model.py       # Post-CMP/Dicing/Pre-Gate → Dit (H₂水対応)
│   ├── asml_model.py               # EUV aerial image + flare + CD budget
│   ├── advantest_model.py          # ATE: V_th/R_on/BV_DSS + yield + CPGD
│   ├── semiconductor_ecosystem.py  # NVIDIA/TSMC/Samsung/Kioxia エコシステム
│   └── backend_model.py            # Wire bonding + package stress (NEW)
├── ml/
│   ├── train_from_experimental.py  # GP surrogate (26 pts, LOO R²=0.55)
│   ├── train_fusion_gp.py          # Fusion GP exp+FEM (R²=0.64)
│   ├── multifidelity_gp.py         # AR1 co-kriging
│   ├── active_learning.py          # EI-based next experiment
│   ├── anomaly_detection.py        # GP z-score | IForest | Shewhart
│   ├── sensitivity_analysis.py     # Sobol indices
│   └── surrogate_fno_demo.py       # FNO: params → 2D stress field (0.07ms)
├── optimization/
│   ├── tmcmc_dicing.py             # TMCMC inference (MAP err < 2%)
│   ├── pareto_front.py             # Chipping vs MRR Pareto
│   ├── hybrid_process_opt.py       # NSGA-II 4-objective + delamination
│   └── realtime_recipe.py          # Sensor → TMCMC → GP → recipe
├── validation/
│   ├── experimental_data.py        # Micro2026 + Mat2022 (26 pts)
│   ├── validate_trends.py          # Qualitative + Pearson trend check
│   ├── thin_wafer_sweep.py         # 2nm node / 50µm wafer sweep
│   └── quantitative_validation.py  # RMSE/MAPE/R²/KS vs 5 literature sources (NEW)
├── pipeline/
│   └── run_full_pipeline.py        # End-to-end runner
├── data/materials/
│   └── material_properties.py      # Si, 4H-SiC, GaN — elastic + fracture
└── results/                        # Generated plots and model files
```

---

## Quickstart

```bash
git clone https://github.com/keisuke58/wafer-proc-sim.git
cd wafer-proc-sim
pip install scikit-learn joblib numpy pandas matplotlib scipy

# --- Front-End ---
python pipeline/data_pipeline.py                    # Full GP pipeline
python ml/train_fusion_gp.py --loo --plot           # Fusion GP (R²=0.64)
python ml/sensitivity_analysis.py --plot            # Sobol: depth 78%
python optimization/pareto_front.py --plot          # Chipping vs MRR Pareto
python ml/active_learning.py --n-suggest 5          # Next EI experiment
python optimization/realtime_recipe.py --chip 10.0  # TMCMC recipe correction
python ml/surrogate_fno_demo.py                     # FNO 0.07ms/field

# --- Fab Process ---
python fem/tel_cleaning_model.py                    # Cleaning → Dit → µ_ch
python fem/tel_process_model.py --pipeline          # ALD/oxide/RIE MOSFET
python fem/tel_cmp_lasertec.py                      # CMP + inspection
python fem/asml_model.py                            # EUV CD budget
python fem/advantest_model.py --full-pipeline       # ATE yield + economics

# --- Back-End / Validation ---
python fem/backend_model.py                         # Wire bond + pkg stress
python validation/quantitative_validation.py        # Sim vs 5 literature sources
```

---

## Materials

| Material | E [GPa] | K_Ic [MPa√m] | G_c [J/m²] | σ_t [MPa] | CTE [ppm/K] |
|---|---|---|---|---|---|
| 4H-SiC | 400 | 2.8 | 19.6 | 350 | 4.0 |
| Si | 130 | 0.83 | 5.3 | 150 | 2.6 |
| GaN | 295 | 0.9 | 2.7 | 100 | 5.6 |
| AlN (substrate) | 320 | — | — | — | 4.5 |
| Cu (DBC/leadframe) | 117 | — | — | — | 17.0 |

---

## References

| # | Citation |
|---|---|
| 1 | Huang et al., *Micromachines* 17(2):187, 2026 — experimental chipping data |
| 2 | Zhang et al., *Materials* 15(22):8083, 2022 — experimental data |
| 3 | Kimoto & Cooper (2014) *Fundamentals of SiC Technology* — µ_ch vs Dit |
| 4 | Breach et al., *Microelectron. Reliab.* 44:973, 2004 — Au-Al IMC growth |
| 5 | Harman (1997) *Wire Bonding in Microelectronics*, McGraw-Hill |
| 6 | Timoshenko (1925) *J Opt Soc Am* — bimaterial beam warpage |
| 7 | Engelmaier (1993) *ASME J Electron Packag* — solder joint fatigue |
| 8 | Saks et al. (1999) *Appl Phys Lett* — SiC/SiO₂ interface traps |
| 9 | Ching & Chen, *J. Eng. Mech.* 133(7):816–832, 2007 — TMCMC |
| 10 | Saltelli et al., *Comp. Phys. Comm.* 181:259–270, 2010 — Sobol |

---

*Contact: k.nishioka@stud.uni-hannover.de*
