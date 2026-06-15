# WM-811K — dataset report

Real-fab wafer-map defect-pattern dataset wired into this repo as the
pretraining source for the shared CNN backbone
(`vision/pretrain_wm811k.py` → `results/wm811k_backbone.pt`), which warm-starts
the kerf-quality classifier and the MixedWM38 benchmark.

All numbers below are measured from the local copy with
`python vision/wm811k_explore.py --trust` (figure + `results/wm811k_stats.json`),
**not** quoted from the paper.

![WM-811K overview](../results/wm811k_overview.png)

## What it is

WM-811K is the de-facto benchmark for **wafer-map defect-pattern
recognition**. It was collected from a **real semiconductor fab** and released
by the NTU MIR Lab. Each datapoint is one wafer's *bin map*: a 2-D grid where
every cell is a die, colored by its wafer-sort (electrical test) result.

| Field | Measured value |
|---|---|
| Total wafer maps | **811,457** |
| Lots | **46,293** |
| Expert-labelled maps | **172,950** (21.3%) |
| Unlabelled maps | 638,507 |
| Classes | 9 — 8 defect patterns + `none` |
| Cell encoding | `0` = no die / outside wafer · `1` = pass die · `2` = fail die |
| Map size | variable — height 10–300, width 12–205 die (median ≈ 36×35) |
| Unique map shapes (in 60k sample) | 523 |
| Die per wafer | 34 – 48,099 (median 953) |

The **variable map size** is why the backbone ends in a global-average-pool:
that makes it input-size agnostic, so a backbone learned on ~36×36 wafer maps
drops straight into the 128×128 kerf classifier without reshaping.

## The 9 classes and what they mean

The *spatial pattern* of failing die is a fingerprint of the upstream process
fault — this is exactly why auto-classification is "intelligent inspection".

| Class | Labelled count | Typical root-cause signature |
|---|---:|---|
| none | 147,431 | normal wafer (no systematic pattern) |
| Edge-Ring | 9,680 | etch / CMP edge non-uniformity |
| Edge-Loc | 5,189 | localized edge defect, handling at the rim |
| Center | 4,294 | spin-coat / deposition center non-uniformity |
| Loc | 3,593 | localized cluster (equipment / particle) |
| Scratch | 1,193 | mechanical scratch from handling / polishing |
| Random | 866 | random particle contamination |
| Donut | 555 | concentric (annular) thickness non-uniformity |
| Near-full | 149 | near-total failure (catastrophic) |

## Key takeaways

1. **Severe class imbalance.** Of the labelled set, 147,431 are `none` and only
   **25,519 are defects** (≈14.8%). Within defects the rarest class
   (Near-full, 149) is ~65× rarer than the most common (Edge-Ring, 9,680).
   → inverse-frequency class weights are mandatory; macro-F1 / balanced-acc /
   per-class F1 must be reported, never accuracy alone
   (matches the repo's multi-metric policy).

2. **Most data is unlabelled** (638,507 / 811,457 = 78.7%). This is a textbook
   case for self-/semi-supervised pretraining — a future axis beyond the
   current supervised 9-class pretrain.

3. **Real-fab, not synthetic.** This is why it is valuable as a pretraining
   source. In this repo the measured transfer effect (WM-811K → kerf) is
   **variance reduction, not a mean-accuracy win** (multi-seed paired test:
   cold-start accuracy SD 0.068 → warm SD 0.016, ~4× tighter); see
   `vision/transfer_eval_multiseed.py`. Honest framing for interviews:
   pretraining stabilizes deployment, but the safety-critical reject class
   still needs task-specific validation.

## DISCO relevance

The problem class — *classify defects on a structured field* — is the same one
that drives kerf-chipping grading (`vision/kerf_quality_classifier.py`) and
CFRP defect localization (the GNN work). Wafer-map pattern recognition is the
canonical "intelligent inspection" task: turn a pass/fail die map into an
automatic process-root-cause diagnosis. That is the bridge from the published
GNN/CNN defect research to DISCO's inspection / yield-analysis line.

## Reproduce

```bash
# SECURITY: WM811K.pkl is an external-mirror pickle (executes code on load).
python vision/wm811k_explore.py --trust            # → results/wm811k_overview.png + stats.json
python vision/pretrain_wm811k.py --trust --quick   # → results/wm811k_backbone.pt
```

Source: NTU MIR Lab mirror — `http://mirlab.org/dataset/public/MIR-WM811K.zip`.
Stored locally under `data/external/WM-811K/` (gitignored, ~2 GB).
