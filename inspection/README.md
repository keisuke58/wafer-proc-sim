# inspection — advanced mask/wafer defect inspection

A demonstrator of the algorithms real **actinic mask / wafer inspection** tools
use (Lasertec ABICS/ACTIS-class), one level above a plain die-to-die subtraction.
Pure Python (`numpy` + `scipy` + `PyTorch` + `matplotlib`), self-validated on a
synthetic die array with injected defects.

```bash
python3 defect_inspect.py   # writes figures/*.png + defect_inspect_results.json
```

## The pipeline

1. **Die-to-die (D2D) with a robust golden reference** — the reference is the
   pixelwise *median* of many sub-pixel-aligned dies (phase-correlation with a
   parabolic sub-pixel peak), not one noisy neighbour.
2. **Adaptive statistical detection** — the difference is scored in units of the
   *per-pixel* noise measured from the spread of the reference dies (robust MAD).
   Pattern edges left by residual misalignment vary across dies → high noise →
   automatically de-weighted. This is what turns ">90 % of flagged sites are
   nuisance" into a usable capture rate.
3. **Phase-defect channel** — a multilayer/phase defect is invisible in the
   in-focus amplitude image; using the Transport-of-Intensity relation
   `I(+z) − I(−z) ∝ ∇²φ`, a through-focus phase channel reveals it. This is the
   whole reason actinic (13.5 nm) tools exist.
4. **Deep-learning anomaly detection** — a convolutional autoencoder trained only
   on defect-free dies flags whatever it cannot reconstruct (unsupervised, PyTorch).
5. **Fusion + evaluation** — combine the DL amplitude detector with the phase
   channel, and score everything as a **capture-rate vs nuisance (false-count)
   curve** — the KPI an inspection engineer actually optimizes.

## Result (synthetic array: 20 dies, 6 amplitude + 2 phase defects)

Capture-rate vs nuisance, best method at each nuisance level:

| method | what it is | ceiling |
|---|---|---|
| fixed threshold | global threshold on raw difference | ~0.50 (edge nuisance swamps it) |
| adaptive (MAD z) | per-pixel statistical D2D | ~0.63 |
| DL autoencoder | unsupervised reconstruction residual | **0.75 — all amplitude defects** |
| **fused (amp + phase)** | DL + phase channel | **1.0 — breaks the amplitude ceiling** |

The DL curve hits a hard ceiling at 0.75 (= 6/8): **no amplitude-only method can
capture the phase defects.** Only adding the phase channel reaches full capture —
the quantitative case for actinic phase inspection. Phase-defect capture at a
fixed nuisance budget: amplitude channel **0.0**, phase channel **1.0**.

Figures: `insp_scene.png` (array + phase channel), `insp_detect.png` (fixed vs
adaptive vs DL vs fused maps), `insp_phase.png` (phase-defect reveal),
`insp_curve.png` (capture vs nuisance).
