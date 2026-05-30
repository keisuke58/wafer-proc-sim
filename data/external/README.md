# External Data & Resources

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
