# NEU-CLS — real-image defect classification

The kerf classifier (`vision/kerf_quality_classifier.py`) runs on *synthetic*
dicing images. This page is the **real-photograph** counterpart: the same
BatchNorm CNN backbone trained and evaluated on the NEU Surface Defect Database
— actual grayscale photographs, not simulation.

- Script:  `vision/neu_real_classifier.py`
- Loader:  `vision/neu_steel_adapter.py`
- Figures: `results/neu_real_classifier.png`, `results/neu_real_defects.png`
- Stats:   `results/neu_real_stats.json`

![NEU real defect gallery](../results/neu_real_defects.png)

## Dataset

NEU-CLS (Northeastern University, surface-inspection lab): **1,800 real
grayscale images, 200×200 px, 6 defect classes × 300**. Hot-rolled steel-strip
surface defects — a real-image analog for wafer/kerf surface inspection.

| NEU class | Kerf / wafer inspection analog |
|---|---|
| scratches | kerf-edge linear chipping (closest analog) |
| crazing | distributed chipping (surface cracks) |
| pitted_surface | point chipping |
| inclusion | hard-particle inclusion in Si substrate |
| patches | surface stain / contamination |
| rolled-in_scale | scale / debris defect |

Downloaded from Figshare (public NEU-CLS, ~27 MB) into `data/external/NEU-CLS/`
(gitignored), reorganized into one directory per class so `NeuSteelAdapter`
discovers them.

## Results (6-class, 25% held-out test, measured)

| Model | accuracy | macro-F1 | balanced-acc |
|---|---:|---:|---:|
| RandomForest on pixels (baseline) | 0.771 | 0.771 | 0.771 |
| CNN (BatchNorm, cold start) | 0.867 | 0.867 | 0.867 |
| **CNN + WM-811K warm start** | **0.962** | **0.963** | **0.962** |

6-class chance = 0.167.

![NEU real-image classifier (warm start)](../results/neu_real_classifier.png)

## The headline result: real-fab pretraining transfers to real photos

Warm-starting the backbone from the WM-811K real-fab wafer-map pretraining
(`results/wm811k_backbone.pt`, see [wm811k_dataset.md](wm811k_dataset.md))
lifts real-image accuracy **0.867 → 0.962 (+9.5 pts)**, and the weakest class
(`inclusion`) F1 from 0.685 (RF) to 0.919.

This is a clean, honest transfer story: features learned on 172k real fab
wafer maps generalize to a *different* real-image defect domain (steel
surface). It is the real-image complement to the synthetic-kerf transfer
finding, where the same pretraining mainly reduced variance
(`vision/transfer_eval_multiseed.py`).

Note: a 32-feature 3-layer backbone is deliberately small (it is shared across
wafer maps, kerf, and these photos); published ResNet-class models reach ~99%
on NEU-CLS. The point here is **architectural coherence + demonstrable
transfer**, not leaderboard SOTA.

## Reproduce

```bash
# one-time download (Figshare public mirror, ~27 MB) into data/external/NEU-CLS/<class>/
python vision/neu_real_classifier.py                                   # cold start
python vision/neu_real_classifier.py --pretrained results/wm811k_backbone.pt  # warm start
python vision/neu_real_classifier.py --quick                           # fast smoke
```

Evaluation follows the repo multi-metric policy (accuracy + macro-F1 +
balanced-acc + per-class F1 + confusion matrix + a pixel baseline), never
accuracy alone.
