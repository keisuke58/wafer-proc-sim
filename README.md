# wafer-proc-sim

**Physics-informed GP surrogate + Bayesian optimization for semiconductor wafer processing**

[![CI](https://github.com/keisuke58/wafer-proc-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/keisuke58/wafer-proc-sim/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20495459.svg)](https://doi.org/10.5281/zenodo.20495459)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/keisuke58/wafer-proc-sim)](https://github.com/keisuke58/wafer-proc-sim/releases)

---

## Overview

`wafer-proc-sim` combines physics-based simulation, **heteroscedastic Gaussian process surrogates**, and Bayesian optimization for semiconductor wafer process modeling. Three main capabilities:

1. **GP surrogate (blade dicing)** — data-quality-stratified heteroscedastic GP for 4H-SiC dicing chipping
2. **KABRA® laser-slicing BO** — multi-objective TuRBO optimizer for DISCO KABRA process (HAZ / quality / throughput)
3. **Physical process limits** — first-principles models for six DISCO processes (blade, stealth, thinning, plasma, diamond ablation)

Developed to support:

> **"Data Quality-Aware Heteroscedastic Gaussian Process Surrogate  
> for 4H-SiC Blade Dicing Process Optimization"**  
> K. Nishioka et al. — *Precision Engineering* (submitted)

**Core finding (blade dicing)**: 11 high-quality datapoints outperform 25 mixed-quality datapoints.  
**Core finding (KABRA BO)**: TuRBO achieves **+71.3% throughput** vs. 15 W / 200 mm/s baseline.

---

## Key Results

### Blade Dicing GP Surrogate

| Method | LOO RMSE | LOO R² | n |
|--------|----------|--------|---|
| Homoscedastic GP (all grades) | 3.12 µm | 0.34 | 25 |
| **Heteroscedastic GP (Grade A/B)** | **1.62 µm** | **0.80** | **11** |
| GP + EnKF (real-time) | 1.54 µm | — | 80 wafers sim |

### KABRA® Laser-Slicing Bayesian Optimization (TuRBO)

| Scenario | Optimal (P / v / d) | HAZ | Quality | Throughput | vs. Baseline |
|----------|---------------------|-----|---------|------------|--------------|
| quality_first | 22.2 W / 303 mm/s / 100 µm | 121.4 µm | 0.987 | 296.0 mm²/s | **+71.3%** |
| balanced | 22.2 W / 303 mm/s / 100 µm | 121.4 µm | 0.987 | 296.0 mm²/s | **+71.3%** |
| speed_first | 22.2 W / 303 mm/s / 100 µm | 121.4 µm | 0.987 | 296.0 mm²/s | **+71.3%** |

Leeftink et al. (arXiv:2511.23141) report +34% throughput on Si wafer using GP+BO (DISCO LASER1205).

### Physical Process Limits

| Process | Physical Limit | Model |
|---------|---------------|-------|
| Blade dicing kerf | **7.2 µm** (current: 23 µm → 68% headroom) | Euler plate buckling |
| Wafer thinning (safety) | SF = 58.5 @ 25 µm, 11.7 @ 5 µm | Timoshenko plate bending |
| Plasma DRIE AR | AR_crit = 8 @ 10 mTorr, AR_max ≈ 35 | ARDE empirical (JVST 2017) |
| Stealth dicing focus | Layer thickness ≥ 3.3 µm (Rayleigh z_R) | Gaussian optics |
| Diamond laser kerf | 0.42 µm @ 193 nm (ArF only) | Photon energy threshold |

Validated against HBM fracture data (Materials 2024): stealth a_eff ≈ 1.6 µm vs. blade 8.6 µm vs. laser 91 µm.

---

## Quick Start

```bash
git clone https://github.com/keisuke58/wafer-proc-sim.git
cd wafer-proc-sim

pip install -e ".[dev]"
./scripts/reproduce_paper.sh       # paper figures + LOO metrics

# KABRA BO (TuRBO multi-objective)
python sic/sic_kabra_gp.py         # synthetic DOE
python sic/sic_kabra_gp.py --use-fem  # analytical Rosenthal-Bessel DOE

# Physical process limits
python sic/physical_limits.py --validate   # with HBM literature validation

# DRIE ARDE validation (requires Access-2024-18513 dataset)
python validation/arde_validation.py

# Analytical FEM DOE for KABRA
python fem/generate_doe.py --output fem/kabra_doe_data.csv

# Streamlit demo
streamlit run streamlit_app.py
```

---

## Repository Structure

```
wafer-proc-sim/
├── sic/
│   ├── physical_limits.py        # First-principles process limits (6 DISCO processes)
│   ├── sic_kabra_gp.py           # KABRA® GP surrogate + TuRBO multi-objective BO
│   └── sic_vs_si_analysis.py     # SiC vs Si property comparison
├── validation/
│   ├── experimental_data.py      # Curated SiC dicing dataset (quality A–D) + HBM fracture data
│   ├── arde_validation.py        # ARDE model validation (Access-2024 15k ICP etch dataset)
│   └── quantitative_validation.py
├── fem/
│   ├── kabra_thermal_2d.py       # ABAQUS 2D thermo-mechanical FEM (KABRA)
│   ├── generate_doe.py           # Pure-Python Rosenthal-Bessel analytical DOE
│   ├── kabra_doe_data.csv        # Generated DOE (360 pts, 4 params × 3 outputs)
│   └── <100+ industry models>    # TEL, Keyence, ASML, DISCO competitive analysis…
├── references/
│   ├── papers/                   # Downloaded PDFs (HBM, DREI, stealth dicing, BO)
│   └── repos/                    # Cloned open-source tools (ViennaPS, Access-2024, etc.)
├── ml/train_from_experimental.py # Heteroscedastic GP surrogate
├── pipeline/sic_dicing_pipeline.py
├── streamlit_app.py              # Live monitor + KABRA BO + physical limits demo
└── tests/test_paper_metrics.py   # LOO metric regression (CI)
```

---

## Dataset

### Blade Dicing (curated)

| Key | Reference | DOI | n | Grade |
|-----|-----------|-----|---|-------|
| Micro2026 | Wang Y. et al., *Micromachines* 17(2):187, 2026 | [10.3390/mi17020187](https://doi.org/10.3390/mi17020187) | 19 | A/C |
| Mat2022 | Feng Y. et al., *Materials* 15(22):8083, 2022 | [10.3390/ma15228083](https://doi.org/10.3390/ma15228083) | 10 | D |

### Open External Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| Access-2024-18513 | Guo et al., IEEE Access (2024) | 15k ICP etch samples (Cl₂/HBr/O₂), ARDE validation |
| HBM Fracture | Kang et al., Materials 17(22):5529 (2024) | 3PB chip strength, stealth/blade/laser comparison |
| ETH DRIE | Legtenberg et al., arXiv:2104.02763 (2021) | Bosch DRIE ARDE etch lag < 1.5% |

Quality grades: A = direct SEM + reported σ, B = digitized, C = interpolated, D = estimated.

---

## Physics Models

### Blade Dicing
- Lawn–Evans lateral crack: $c_l = C (E/H)^{0.4} (P/K_\text{Ic})^{0.5}$
- 4H-SiC crystallographic anisotropy (K_Ic × 1.20 on {0001})
- Ensemble Kalman Filter for real-time blade wear estimation
- Euler plate buckling for minimum kerf: $t_\text{min} = a\sqrt{12(1-\nu^2)\sigma_y / (\pi^2 E)}$

### KABRA® Laser Slicing
- Rosenthal–Bessel moving Gaussian source: $\Delta T_\text{max} = P_\text{abs} G(Pe) / (2\pi k w_0)$
- TuRBO trust-region Bayesian optimization (Eriksson 2019)
- Multi-objective scalarization: $J = w_\text{haz}\hat{h} + w_q(1-q) + w_\text{tp}(1-\hat{v})$

### Plasma DRIE / ARDE
- $R/R_0 = 1/(1 + (AR/AR_\text{crit})^n)$, $AR_\text{crit}(P) = 8\sqrt{P/10\,\text{mTorr}}$ (JVST 2017)

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Core data flow |
| [docs/TIERS.md](docs/TIERS.md) | Core vs extension layout |
| [docs/PORTFOLIO.md](docs/PORTFOLIO.md) | DISCO interview / portfolio mapping |
| [docs/ZENODO.md](docs/ZENODO.md) | Zenodo DOI & archive instructions |
| [AGENTS.md](AGENTS.md) | AI agent instructions |
| [DEVLOG.md](DEVLOG.md) | Development history |

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

@software{nishioka2026wafer,
  author    = {Nishioka, Keisuke},
  title     = {wafer-proc-sim},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.2.0},
  doi       = {10.5281/zenodo.20495459},
  url       = {https://github.com/keisuke58/wafer-proc-sim}
}
```

---

## License

MIT — see [LICENSE](LICENSE).  
Contact: Keisuke Nishioka · kei128608@gmail.com
