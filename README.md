# wafer-proc-sim

**Data-quality-aware heteroscedastic GP surrogate for 4H-SiC blade dicing process optimization**

[![CI](https://github.com/keisuke58/wafer-proc-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/keisuke58/wafer-proc-sim/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`wafer-proc-sim` combines physics-based simulation, **heteroscedastic Gaussian process surrogates**, and Bayesian optimization for small-n semiconductor process datasets.

Developed to support:

> **"Data Quality-Aware Heteroscedastic Gaussian Process Surrogate  
> for 4H-SiC Blade Dicing Process Optimization"**  
> K. Nishioka et al. — *Precision Engineering* (submitted)

**Core finding**: 11 high-quality datapoints outperform 25 mixed-quality datapoints. Data quality stratification dominates data quantity for small-n process surrogates.

---

## Key Results

| Method | LOO RMSE | LOO R² | n |
|--------|----------|--------|---|
| Homoscedastic GP (all grades) | 3.12 µm | 0.34 | 25 |
| **Heteroscedastic GP (Grade A/B)** | **1.62 µm** | **0.80** | **11** |
| GP + EnKF (real-time) | 1.54 µm | — | 80 wafers sim |

Metrics are regression-tested in CI (`tests/test_paper_metrics.py`).

---

## Quick Start

```bash
git clone https://github.com/keisuke58/wafer-proc-sim.git
cd wafer-proc-sim

# Core (paper reproduction)
pip install -e ".[dev]"
./scripts/reproduce_paper.sh

# Full (Streamlit demos, quantum experiments)
pip install -e ".[full,dev]"
```

**Individual steps:**

```bash
python paper/sic_gp_quality/gen_figures.py          # Figures 1–2
python ml/train_from_experimental.py --loo --quality AB
python pipeline/sic_dicing_pipeline.py --no-plots
pytest tests/ -v
streamlit run demos/paper_app.py                    # minimal paper demo
```

---

## Core Repository Structure

```
wafer-proc-sim/
├── validation/experimental_data.py   # Curated SiC dicing dataset (quality A–D)
├── ml/train_from_experimental.py     # Heteroscedastic GP surrogate
├── pipeline/sic_dicing_pipeline.py   # GP + EnKF + recipe optimization
├── paper/sic_gp_quality/             # Manuscript + reproducible figures
├── scripts/reproduce_paper.sh        # One-command reproduction
├── tests/test_paper_metrics.py       # LOO metric regression
└── docs/ARCHITECTURE.md              # Data-flow diagram
```

**Extensions** (TEL, Keyence, DISCO portfolio, 100+ industry models) live in `fem/`, `pipeline/`, and `demos/`. See [docs/TIERS.md](docs/TIERS.md) and [fem/README.md](fem/README.md).

---

## Dataset

Curated from two open-access publications:

| Key | Reference | DOI | n | Grade |
|-----|-----------|-----|---|-------|
| Micro2026 | Wang Y. et al., *Micromachines* 17(2):187, 2026 | [10.3390/mi17020187](https://doi.org/10.3390/mi17020187) | 19 | A/C |
| Mat2022 | Feng Y. et al., *Materials* 15(22):8083, 2022 | [10.3390/ma15228083](https://doi.org/10.3390/ma15228083) | 10 | D |

Quality grades: A = direct SEM + reported σ, B = digitized, C = interpolated, D = estimated.  
Full specification: `validation/experimental_data.py`.

---

## Physics Models (core)

- Lawn–Evans lateral crack model: $c_l = C (E/H)^{0.4} (P/K_\text{Ic})^{0.5}$
- 4H-SiC crystallographic anisotropy (K_Ic direction dependence, 1.20× on {0001})
- Ensemble Kalman Filter for real-time blade wear state estimation
- Expected Improvement Bayesian optimization

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Core data flow |
| [docs/TIERS.md](docs/TIERS.md) | Core vs extension layout |
| [docs/PORTFOLIO.md](docs/PORTFOLIO.md) | DISCO interview / portfolio mapping |
| [AGENTS.md](AGENTS.md) | AI agent instructions |
| [DEVLOG.md](DEVLOG.md) | Development history |
| [demos/README.md](demos/README.md) | Streamlit & pipeline demos |

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

## License

MIT — see [LICENSE](LICENSE).  
Contact: Keisuke Nishioka · kei128608@gmail.com