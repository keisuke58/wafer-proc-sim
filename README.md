# wafer-proc-sim

**Physics-informed surrogate modeling and process optimization for semiconductor wafer dicing**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`wafer-proc-sim` is an open-source framework for **4H-SiC blade dicing process optimization**, combining physics-based simulation, heteroscedastic Gaussian process surrogates, and Bayesian optimization.

Developed to support:

> **"Data Quality-Aware Heteroscedastic Gaussian Process Surrogate  
> for 4H-SiC Blade Dicing Process Optimization"**  
> K. Nishioka et al. — *Precision Engineering* (submitted)

---

## Key Results

| Method | LOO RMSE | LOO R² | n |
|--------|----------|--------|---|
| Homoscedastic GP (all grades) | 3.12 µm | 0.34 | 25 |
| **Heteroscedastic GP (Grade A)** | **1.62 µm** | **0.80** | **11** |
| GP + EnKF (real-time) | 1.54 µm | — | 80 wafers sim |

**Core finding**: 11 high-quality datapoints outperform 25 mixed-quality datapoints.
Data quality stratification dominates data quantity for small-n process surrogates.

---

## Repository Structure

```
wafer-proc-sim/
├── fem/                        # Physics simulation models
│   ├── dicing_blade_2d.py      # 2D FEM blade dicing (fracture/chipping)
│   ├── dicing_blade_3d.py      # 3D FEM blade dicing
│   ├── crystal_anisotropy.py   # 4H-SiC anisotropy
│   ├── blade_thermal_wear_2d.py    # Cutting heat + Taylor/Archard tool wear
│   ├── stage_vibration_modal.py   # Chuck stage modal analysis (f1 vs f_blade)
│   ├── stage_cad_geometry.py      # Parametric CAD geometry + DXF/JSON export
│   ├── grinding_warpage_2d.py     # Backside grinding → residual stress → warpage
│   ├── kabra_thermal_2d.py        # KABRA® laser slicing thermo-mechanical FEM
│   ├── cmp_dry_polish_model.py    # CMP/Dry polish: Preston eq. + Arrhenius chem
│   ├── stealth_dicing_crack_model.py  # Stealth dicing: K_I / XFEM crack propagation
│   ├── plasma_etch_profile_model.py   # Plasma DRIE: Bosch ARDE + sidewall geometry
│   ├── dbg_sdbg_model.py          # DBG/SDBG: Weibull die strength improvement
│   ├── spindle_motor_control.py   # PMSM + FOC spindle motor electrical model
│   └── wafer_thickness_sensor.py  # Eddy current + capacitive in-process sensor
│   ├── cfrp_cutting_model.py   # CFRP cutting (milling/drilling/AWJ)
│   ├── cfrp_defect_model.py    # CFRP NDT physics (5 methods)
│   └── [30+ industry models]   # Market and competitive analysis
├── ml/
│   ├── train_from_experimental.py  # Heteroscedastic GP surrogate
│   ├── recipe_optimizer.py         # Multi-objective BO recipe control (SW layer)
│   ├── cfrp_defect_detection.py    # CFRP anomaly + classification + sizing
│   └── quantum_kernel_advanced.py  # Quantum kernel GP (experimental)
├── pipeline/
│   └── sic_dicing_pipeline.py  # End-to-end: GP + EnKF + BO
├── validation/
│   └── experimental_data.py    # Curated SiC dicing dataset (quality-graded)
├── paper/
│   └── sic_gp_quality/         # Manuscript draft + reproducible figures
│       ├── draft.md            # Full paper draft
│       ├── gen_figures.py      # Reproduce Figures 1–2
│       └── figures/tiff_300dpi/  # 300 dpi TIFF for journal submission
└── results/                    # Generated figures
```

---

## Quick Start

```bash
git clone https://github.com/keisuke58/wafer-proc-sim.git
cd wafer-proc-sim
pip install numpy scipy matplotlib pandas scikit-learn joblib
```

**Reproduce paper figures:**
```bash
python paper/sic_gp_quality/gen_figures.py
```

**Run full GP + EnKF + BO pipeline:**
```bash
python pipeline/sic_dicing_pipeline.py
```

**Train heteroscedastic GP with quality filter:**
```bash
python ml/train_from_experimental.py --loo --quality A
# --quality: all | AB | A
```

---

## Dataset

Curated from two open-access publications:

| Key | Reference | DOI | n | Grade |
|-----|-----------|-----|---|-------|
| Micro2026 | Wang Y. et al., *Micromachines* 17(2):187, 2026 | [10.3390/mi17020187](https://doi.org/10.3390/mi17020187) | 19 | A/C |
| Mat2022 | Feng Y. et al., *Materials* 15(22):8083, 2022 | [10.3390/ma15228083](https://doi.org/10.3390/ma15228083) | 10 | D |

Quality grades: A = direct SEM + reported σ, C = interpolated, D = estimated.
Full specification: `validation/experimental_data.py`.

---

## Physics Models

**SiC Dicing**
- Lawn–Evans lateral crack model: $c_l = C (E/H)^{0.4} (P/K_\text{Ic})^{0.5}$
- 4H-SiC crystallographic anisotropy (K_Ic direction dependence, 1.20× on {0001})
- Ensemble Kalman Filter for real-time blade wear state estimation
- Expected Improvement Bayesian optimization

**CFRP Processing**
- NDT simulation: pulse-echo UT, thermography, ECT, acoustic emission, X-ray
- Cutting: Merchant–Zhang milling, Hocheng-Dharan drilling, Lawn-Evans dicing, Shanmugam AWJ

**Semiconductor Industry Models** (30+ companies/markets in `fem/`)
- Power device markets: humanoid robots (GaN), drones (GaN), BESS (SiC), LEO (SiC), EV (SiC)
- Competitive risk simulation: blade → laser dicing transition (Bass diffusion model)
- NVIDIA GPU, TSMC, TEL, Disco, Lasertec, Advantest, and more

---

## Citation

```bibtex
@article{nishioka2025sic,
  title   = {Data Quality-Aware Heteroscedastic {Gaussian} Process Surrogate
             for {4H-SiC} Blade Dicing Process Optimization},
  author  = {Nishioka, Keisuke},
  journal = {Precision Engineering},
  year    = {2025},
  note    = {submitted},
  url     = {https://github.com/keisuke58/wafer-proc-sim}
}
```

---

## Career Context

This repository was developed as a research portfolio targeting **DISCO Corporation — all five engineering tracks**.

| DISCO 職種 | カバレッジ | 主要ファイル |
|---|---|---|
| **メカ** (機械設計・構造FEM) | ◎ | `fem/disco_machine_struct.py` — スピンドルハウジング剛性・軸受選定・フレームモーダル・振動絶縁・熱膨張・公差積み上げ<br>`fem/dicing_blade_2d.py`, `3d.py` — ブレードFEM破壊解析<br>`fem/stage_vibration_modal.py`, `fem/stage_cad_geometry.py` — ステージ動特性・CAD |
| **エレキ** (電気・制御・センサ) | ◎ | `fem/disco_elec_system.py` — DCバス設計・スピンドルインバータ熱計算・XYリニアモーター電流/位置制御・ビジョンLED同期・EMI予算・安全PLC(IEC 62061 SIL-2)<br>`fem/spindle_motor_control.py` — PMSM + FOC<br>`fem/wafer_thickness_sensor.py` — 渦電流/容量センサ + EnKF |
| **プロセス** (プロセス研究・条件最適化) | ◎ | `fem/kabra_thermal_2d.py`, `fem/cmp_dry_polish_model.py`, `fem/stealth_dicing_crack_model.py`, `fem/plasma_etch_profile_model.py`, `fem/dbg_sdbg_model.py`<br>`optimization/bayesian_opt.py`, `optimization/tmcmc_dicing.py` |
| **R&D** (先行研究・代理モデル) | ◎ | `ml/surrogate_gp.py`, `ml/train_from_experimental.py` — ヘテロセダスティックGP LOO-CV<br>`ml/recipe_optimizer.py` — 多目的BO<br>`ml/multifidelity_gp.py`, `ml/active_learning.py` — 次世代手法 |
| **ソフト** (機械制御ソフト・パイプライン) | ◎ | `pipeline/disco_sw_stack.py` — 状態機械(IEC 61131-3準拠)・レシピ管理・1msRTループ・データヒストリアン・オペレータAPI・SW-FMEA<br>`pipeline/sic_dicing_pipeline.py`, `pipeline/keyence_apc.py` — APCパイプライン |

**面接戦略**: 2次面接まで。各面接官の職種に応じて上記ファイルをデモとして提示し、「理論→実装→検証」の一貫性をアピール。

---

## License

MIT — see [LICENSE](LICENSE).  
Contact: Keisuke Nishioka · kei128608@gmail.com
