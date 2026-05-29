# wafer-proc-sim

**FEM × Bayesian Optimization for Wafer Dicing/Grinding Process Simulation**

Keisuke Nishioka (Keio University / Leibniz Universität Hannover)

> The first open-source simulation toolkit for semiconductor wafer dicing and grinding process optimization, combining ABAQUS FEM with Bayesian inference and machine learning.

## Research Roadmap

| Phase | Timeline | Content |
|-------|----------|---------|
| **Phase 1** | 2027 Mar–Jun | 2D/3D FEM of blade dicing (Si, SiC, GaN) |
| **Phase 2** | 2027 Jul–Oct | Surrogate + Bayesian optimization of process parameters |
| **Phase 3** | 2027 Nov–2028 Feb | GNN residual stress prediction for TAIKO® grinding |

## Structure

```
wafer-proc-sim/
├── fem/
│   ├── dicing_blade_2d.py     # Phase 1: 2D blade dicing FEM (ABAQUS)
│   └── grinding_taiko.py      # Phase 3: TAIKO® back-grinding FEM
├── ml/
│   ├── surrogate_gp.py        # Phase 2: Gaussian process surrogate
│   └── gnn_stress.py          # Phase 3: GNN residual stress prediction
├── optimization/
│   └── bayesian_opt.py        # Phase 2: Bayesian optimization
└── data/materials/
    └── material_properties.py # Si, SiC, GaN constants
```

## References
1. [Automated Laser Dicing × Bayesian (arXiv 2025)](https://arxiv.org/abs/2511.23141)
2. [FEM Grinding Force + Warpage (Oxford 2023)](https://academic.oup.com/jom/article/doi/10.1093/jom/ufad018/7227339)
3. [ML Chipping Prediction (MDPI 2024)](https://www.mdpi.com/2079-9292/13/10/1802)
4. [Digital Twin Stealth Dicing (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1369800124009065)
