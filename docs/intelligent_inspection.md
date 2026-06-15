# 検査の知能化 — intelligent inspection, one story

This ties three inspection pieces of the repo into a single narrative:
**from optical metrology to learned visual inspection.**

- Pipeline: `vision/intelligent_inspection.py`
- Figure:   `results/intelligent_inspection.png`
- Stats:    `results/intelligent_inspection_stats.json`

![intelligent inspection](../results/intelligent_inspection.png)

## The three arms

| Arm | Module | Role | Result |
|---|---|---|---|
| ① Metrology (physics) | `ml/lasertec_inspection.py` | confocal/Weibull chipping measurement → grade A/B/C against `INSPECTION_SPEC` | instrument ground truth |
| ② Vision (CNN) | `vision/intelligent_inspection.py` + shared backbone | kerf image → predicts the **same** grade from pixels | agreement **acc 0.80**, **reject-recall 0.97**, reject-FPR 0.02 |
| ③ Real-image transfer | `vision/neu_real_classifier.py` | same backbone on real NEU photos, WM-811K warm start | **0.96** (generalizes to real data) |

## How ① and ② connect

1. The Lasertec-style metrology model measures per-die chipping (Weibull over
   each process) and grades it: **A** < 1 µm, **B** 1–5 µm, **C** ≥ 5 µm
   (aligned with `INSPECTION_SPEC` `chip_um_max = 5.0` and the 2 nm-node 0.5 µm
   target). This grade is the **label**.
2. For each measured die we render a kerf image whose chipping severity is
   calibrated to that chipping size (`vision.generate_synthetic_kerf`).
3. The shared BatchNorm CNN is trained to predict the metrology grade **from
   the image alone**. On held-out dies it reproduces the optical grader with
   0.80 accuracy and catches **97% of reject-grade dies** at a 2% false-alarm
   rate.

So the CNN *learns the optical metrology rule from pixels* — fast inline
grading with no instrument in the loop at inference time.

## The honest, sophisticated point (use this in interview)

The vision arm sees **chipping** — a surface-visible defect. The metrology arm
additionally measures **subsurface crack depth** and **PL defect density**,
which an optical image fundamentally cannot show. So the two are
**complementary, not redundant**:

> "The CNN replaces routine chipping grading at line speed; optical/PL
> metrology stays for what pixels can't see. The win is a fast image triage
> that flags 97% of rejects, with metrology reserved for borderline and
> subsurface cases."

This is the "intelligent inspection" pitch for DISCO's inspection line — and
it shows awareness of where learned vision helps and where physics metrology
(the Lasertec/Hitachi-SEM domain) is still required.

## Reproduce

```bash
python vision/intelligent_inspection.py                                   # cold
python vision/intelligent_inspection.py --pretrained results/wm811k_backbone.pt
python vision/intelligent_inspection.py --quick                           # smoke
```

Related: [neu_real_classifier.md](neu_real_classifier.md),
[wm811k_dataset.md](wm811k_dataset.md).
