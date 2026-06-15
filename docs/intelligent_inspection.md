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

## Fusion as a two-stage triage (quantified)

`vision/inspection_triage.py` turns the "complementary, not redundant" claim
into a measured system. Every die is screened fast by the CNN; only
low-confidence dies (softmax < τ) are routed to physics metrology, which also
measures the **subsurface crack depth the image cannot see**.

Subsurface cracks are modelled per die from the process crack mean (blade 3.0,
laser_ns 2.0 µm exceed the 2.0 µm limit) — these are reject dies that look
clean to the camera. A die is a true reject if chipping ≥ 5 µm **or** crack
≥ 2 µm.

Measured (900 dies, 254 true rejects of which **73 are crack-only =
image-invisible**), operating point chosen at ≤1% escape budget:

| System | fast-path (throughput) | reject-recall | subsurface-escape |
|---|---:|---:|---:|
| CNN only (image) | 1.00 | 0.93 | **0.07** |
| **Fusion (τ=0.70)** | **0.76** | **1.00** | **0.00** |
| Metrology all | 0.00 | 1.00 | 0.00 |

![inspection triage](../results/inspection_triage.png)

So the fusion keeps **metrology-grade safety (100% reject-recall, zero
subsurface escapes) while handling 76% of dies at line speed** — the slow
instrument sees only the uncertain 24%. Pure-image inspection would let 7% of
rejects (the crack-only ones) escape.

> Interview line: "A confidence-gated triage — the CNN clears 76% at line
> speed, metrology takes the uncertain 24% and catches the subsurface cracks
> the camera can't see. Zero reject escapes at a quarter of the metrology load."

## Reproduce

```bash
python vision/intelligent_inspection.py                                   # cold
python vision/intelligent_inspection.py --pretrained results/wm811k_backbone.pt
python vision/intelligent_inspection.py --quick                           # smoke
python vision/inspection_triage.py                                        # fusion triage
```

Related: [neu_real_classifier.md](neu_real_classifier.md),
[wm811k_dataset.md](wm811k_dataset.md).
