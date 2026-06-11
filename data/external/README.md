# External Data & Resources

## General Semiconductor ML Datasets (downloaded 2026-06-12)

These are the canonical open benchmarks for semiconductor wafer-defect /
process-control ML. Not SiC-dicing-specific, but useful for pretraining,
APC fault-detection prototyping, and as held-out generalization sets.

### WM-811K — wafer map defect dataset (`WM-811K/`)
- **Source**: MIR Lab, NTU — `http://mirlab.org/dataset/public/MIR-WM811K.zip`
  (mirror of the original WM-811K / LSWMD). Also on Kaggle (`qingyi/wm811k-wafer-map`).
- **Contents**: 811,457 real-fab wafer maps; ~20% expert-labeled with 9 classes
  (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none).
- **Files**: `MIR-WM811K/Python/WM811K.pkl` (Python), `MATLAB/WM811K.mat` (MATLAB) + examples.
- **Use**: Canonical wafer-defect classification benchmark → CNN/ViT pretraining,
  transfer to our dicing chipping-map detection.
- **Note**: `.pkl` is a Python pickle — load only in a trusted env (`pickle.load`
  executes arbitrary code; source is an external mirror).

### MixedWM38 — mixed-type wafer defect dataset (`MixedWM38/`)
- **Source**: Junliangwangdhu/WaferMap (Donghua Univ.), Google Drive id `1M59pX-lPqL9APBIbp2AKQRTvngeUK8Va`.
  Also Kaggle (`co1d7era/mixedtype-wafer-defect-datasets`).
- **Contents**: `Wafer_Map_Datasets.npz` — `arr_0` = 38,015 × 52 × 52 wafer maps (int32),
  `arr_1` = 38,015 × 8 one-hot defect labels (1 normal + 8 single + 29 mixed = 38 patterns).
- **Use**: Multi-label / mixed-defect recognition; harder than WM-811K, good OOD/robustness probe.

### SECOM — semiconductor process sensor data (`SECOM/`)
- **Source**: UCI ML Repo (DOI 10.24432/C54305) — `https://archive.ics.uci.edu/static/public/179/secom.zip`. CC BY 4.0.
- **Contents**: `secom.data` = 1567 × 590 sensor features, `secom_labels.data` = pass(-1)/fail(1)
  + timestamp (104 fails / 1463 pass). `secom.names` = description.
- **Use**: Directly matches our APC/fault-detection suite — high-dim noisy fab sensor
  signals, imbalanced yield classification, feature selection / change-point detection.

### How these are wired into the codebase

| Dataset | Entry point | What it does |
|---------|-------------|--------------|
| SECOM | `python ml/anomaly_detection.py --secom` | Runs Layer-2 IsolationForest (semi-supervised, fit on pass-only) + Layer-3 Shewhart on real fab sensors; reports ROC-AUC / recall / FPR. Loader: `data/load_secom.py`. |
| WM-811K | `python vision/pretrain_wm811k.py --trust` | Pretrains the shared CNN backbone (9-class) → `results/wm811k_backbone.pt`. `--trust` required (unpickles). |
| (transfer) | `python vision/kerf_quality_classifier.py --pretrained results/wm811k_backbone.pt` | Warm-starts the kerf chipping-grade CNN from the WM-811K backbone (size-agnostic via global avg pool). |
| MixedWM38 | `python vision/mixedwm38_benchmark.py [--split ood] [--init-backbone ...]` | 8-way multilabel benchmark; `--split ood` trains on pure single-defect maps, tests on mixed. |

Shared backbone: `vision/wafer_backbone.py` (`WaferBackbone` / `WaferClassifier`).
All three pipelines smoke-tested 2026-06-12 (`--quick`). WM-811K pretrain itself
not auto-run (pickle gated behind `--trust`).

## TMCMC Implementations

### transitional-mcmc (Ramancha et al. 2022)
- **Source**: github.com/mukeshramancha/transitional-mcmc
- **Ref**: Ramancha et al., Mech. Syst. Signal Process. 167, 108517 (2022)
- **Use**: Compare with our TMCMC implementation in optimization/tmcmc_dicing.py

### TEMCMC — Transitional Ensemble MCMC (Adolphus8)
- **Source**: github.com/Adolphus8/Transitional_Ensemble_MCMC
- **Contains**: Python + MATLAB, Coupled Oscillator / Aluminium Frame benchmarks
- **Use**: TEMCMC is more efficient than TMCMC for high-D — consider for future upgrade

## Cutting Simulation Reference

### DiSECt Cutting Dataset (NVIDIA, Huang et al. 2022)
- **Source**: github.com/NVlabs/DiSECt  DOI: 10.48550/arXiv.2105.12244
- **Contents**: Real-world knife cutting force CSVs (cylinder/prism geometries, soft materials)
- **Use**: Reference for FEM + Bayesian inference framework (not SiC, but methodology)
- **Note**: Soft-material cutting only (potato/apple); not SiC. Useful for surrogate-GP architecture ideas.

## SiC Dicing Data Sources (Papers — no open raw data found)

| Source | DOI | Data status |
|--------|-----|-------------|
| Micro2026 (4H-SiC, Ni-bond blade) | 10.3390/mi17020187 | Digitized → validation/experimental_data.py |
| Mat2022 (SiC, resin blade) | 10.3390/ma15228083 | Digitized → validation/experimental_data.py |
| AIP2021 (precision dicing) | 10.1063/5.0055498 | Digitized → validation/experimental_data.py |

Raw force/chipping datasets for SiC blade dicing are not publicly deposited on Zenodo/Figshare. Contact authors for raw data.

## Oral Microbiome Data (Masterarbeit / NIFE)

### NCBI SRA — Joshi et al. 2025 (peri-implantitis severity, Szafrański/MHH)
- **Accession**: PRJNA1192962 (SRA IDs: 37705620–37705624)
- **Samples**: 49 implants, 34 patients; 16S full-length + metatranscriptomics
- **Download**: `fasterq-dump PRJNA1192962` (requires SRA Toolkit)

### NCBI SRA — Anuntakarun et al. 2025 (longitudinal peri-implantitis 0/3/6m)
- **Accession**: PRJNA1215005
- **Samples**: Longitudinal 3-timepoint 16S V3-V4 (Illumina MiSeq)
- **Ref**: DOI 10.1016/j.identj.2025.100951

### BEEM — Bayesian estimation of gLV from microbiome time-series (R)
- **Source**: github.com/CSB5/BEEM
- **Use**: Competing method to Hamilton replicator for microbiome interaction inference
